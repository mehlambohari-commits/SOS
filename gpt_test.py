"""
AISA - Hand Gesture Emergency Assistant  (PC VERSION - no extra hardware)
=================================================

Gesture -> emergency call -> AISA voice assistant (Gemini)

Keys (click the AISA window first):
    ESC     quit
    ENTER   confirm the call was answered (manual_key mode)
    X       end the current call session
    F       toggle fullscreen
    D       show live finger numbers (for tuning)
    B       panic call: 3 second countdown, press B again to cancel
    mouse   hold the on-screen panic button for 3 seconds
    0-5     TEACH MODE (for building the training data, see below)

Dataset (for the ML step later)
    Every frame that contains a hand is converted into DATA:
    21 hand landmarks -> 63 numbers, normalised so they do not
    depend on where the hand is or how far it is from the camera.
    Saved to  aisa_gesture_dataset.csv  together with the label
    (what the user wanted) so a model can be trained on it.

Everything runs on this PC:
    siren   = the laptop speaker while the call is ringing
    red LED = the camera frame flashes red while calling
    button  = the on-screen panic button (hold 3 s) or the B key
    answer  = detected by watching the Phone Link "Call on PC" window

ML (what does the user want?)
    At start-up a Random Forest is trained from the dataset of EARLIER
    sessions. It looks at the hand shape and predicts the service
    (Police / Fire / Ambulance / Women Helpline / General Helpline /
    none). It is advisory: it never starts or blocks a call.
    This session's rows are saved but not used until the next start.

TEACH MODE (developer / data collection only)
    Press 1-5 and show that gesture: every frame is saved with that
    true label. Press 0 and show other hand shapes: label "none".
    Press the same key again to stop. Calls are disabled while
    teaching. Once the model is trained, normal use needs no keys.
"""

import cv2
import time
import os
import math
import csv
import random
import threading
import logging
from collections import deque
from functools import lru_cache

import numpy as np
import pygame
import pyautogui
import mediapipe as mp
import speech_recognition as sr
import pyttsx3
import pythoncom

from dotenv import load_dotenv
from google import genai

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:
    PIL_OK = False

# silences the harmless "automatic function calling" notice
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# ============================================================
# SETTINGS
# ============================================================

# True  = no real phone call
# False = actually opens Windows Phone and calls
TEST_MODE = False

# How long the same gesture must be held (seconds)
HOLD_TIME = 5.0

CAMERA_INDEX = 0

WINDOW_NAME = "AISA - Hand Gesture Emergency App"

# windowed by default so the Phone Link "Call on PC" window stays visible
# (the answer is detected by looking at it). F toggles fullscreen.
FULLSCREEN_ON_START = False

SIREN_SOUND = "C:\\Users\\TARMAL COMPUTERS\\siren.wav.mp3"

# Read aloud to the responder once the call is answered.
# EDIT THIS before using outside TEST_MODE.
EMERGENCY_ADDRESS = "REPLACE_WITH_YOUR_ACTUAL_ADDRESS, Pune, Maharashtra"

# --- how AISA decides the call was answered -----------------
#   "screen"      - watches the Phone Link "Call on PC" window: the
#                   Mute / Keypad / Hold call buttons light up when the
#                   other person answers. ENTER also works as a backup
#                   (click the AISA window first, then press ENTER).
#                   needs: pip install pygetwindow
#   "manual_key"  - only ENTER in the AISA window
#   "timer"       - wait CALL_CONNECT_WAIT seconds
#   "auto_window" - older window-title check, falls back to timer
CALL_ANSWER_MODE = "screen"
CALL_CONNECT_WAIT = 4
CALL_ANSWER_TIMEOUT = 30
CALL_WINDOW_TITLE = "Call on PC"
ANSWER_CONFIRM_POLLS = 2          # must look answered this many checks in a row
NO_ANSWER_WARNING_SECONDS = 45    # then: show a warning, keep waiting
TEST_AUTO_ANSWER_SECONDS = 8      # TEST_MODE only: pretend it was answered
                                  # after this long (0 = wait for ENTER)

# --- alarm and panic button (all on this PC) ------------------
SIREN_ENABLED = True              # laptop speaker sounds while the call rings
PANIC_HOLD_SECONDS = 3.0          # hold the on-screen button / B countdown
BUTTON_SERVICE = 5                # panic button calls this: 5 = General Helpline

# --- speech ---------------------------------------------------
SPEECH_LANGUAGE = "en-IN"

# --- reinforcement learning files -----------------------------
Q_TABLE_FILE = "aisa_q_table.npy"

# --- DATASET ---------------------------------------------------
# New dataset (landmark features + labels, ready for ML)
DATASET_FILE = "aisa_gesture_dataset.csv"

# Old dataset(s) from earlier versions: deleted automatically
LEGACY_DATASET_FILES = ["aisa_rl_dataset.csv"]

# True  = also wipe the NEW dataset every time the app starts
# False = keep adding rows across runs (recommended while collecting)
RESET_DATASET_ON_START = False

# Save one row every N frames that contain a hand
SAMPLE_EVERY_N_FRAMES = 2

# --- ML: "what does the user want?" -------------------------------
# Needs:  pip install scikit-learn
ML_ENABLED = True
ML_MODEL_FILE = "aisa_gesture_model.joblib"
ML_MIN_ROWS_PER_CLASS = 40      # a gesture needs at least this many rows
ML_MIN_TRUE_ROWS = 150          # teach-mode rows needed to prefer them
ML_MAX_TRAIN_ROWS = 40000       # keeps start-up training fast
ML_TREES = 80
ML_PREDICT_EVERY = 3            # predict every N frames (keeps FPS up)
ML_MIN_PROBABILITY = 0.70       # below this the HUD shows "unsure"


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set.")
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

gemini_model = None


def find_gemini_model():
    global gemini_model

    if client is None:
        return None

    try:
        print("\nSearching for available Gemini models...")

        available = []

        for model in client.models.list():
            name = getattr(model, "name", "")
            actions = getattr(model, "supported_actions", []) or []

            if "generateContent" in actions:
                if name.startswith("models/"):
                    name = name[len("models/"):]
                available.append(name)

        if not available:
            print("No generateContent Gemini model found.")
            return None

        print("Available Gemini models:")
        for name in available:
            print("  -", name)

        preferred = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        for name in preferred:
            if name in available:
                gemini_model = name
                break

        if gemini_model is None:
            for name in available:
                if "flash" in name.lower():
                    gemini_model = name
                    break

        if gemini_model is None:
            gemini_model = available[0]

        print("\nAISA Gemini model:", gemini_model)
        return gemini_model

    except Exception as error:
        print("Could not list Gemini models:")
        print(error)
        return None


if client is not None:
    find_gemini_model()


# ============================================================
# EMERGENCY NUMBERS
# ============================================================

emergency_numbers = {
    1: "9359953694",
    2: "9335587271",
    3: "8878010010",
    4: "8793796952",
    5: "6232481125",
}

emergency_labels = {
    1: "Police",
    2: "Fire",
    3: "Ambulance",
    4: "Women Helpline",
    5: "General Helpline",
}


# ============================================================
# PYGAME / MEDIAPIPE / CAMERA
# ============================================================

pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# GLOBAL STATE
# ============================================================

last_call_time = 0
call_cooldown = 3

recording = False
record_start_time = 0
out = None

previous_fingers = 0
gesture_stable_time = 0.0

confirming = False
confirmed_fingers = 0

assistant_running = False
assistant_thread = None
conversation = []

# True from the moment a call is triggered until its AISA session
# ends. While True, new gestures are ignored (no second call).
call_in_progress = False
current_call_label = ""

# "idle" -> "dialing" -> "announcing" -> "assisting" -> "idle"
call_phase = "idle"

# set by pressing ENTER in the AISA window
call_answered_event = threading.Event()

# everything shown in the assistant panel: (speaker, text)
# speakers: "AISA", "YOU", "CALL" (said to responder), "SYSTEM"
chat_log = deque(maxlen=40)

# --- call detection / panic button ---
call_watcher = None         # watches the Phone Link call window
button_requested = False    # set when the panic button was held / counted down
no_answer_warning = False   # True after NO_ANSWER_WARNING_SECONDS without an answer
panic_start = None          # when the panic hold / countdown began
panic_mode = None           # "mouse" (hold) or "key" (B countdown)

# ============================================================
# RL ACTIONS
# ============================================================

ACTION_IGNORE = 0
ACTION_WAIT = 1
ACTION_CONFIRM = 2
ACTION_TRIGGER = 3

actions = ["IGNORE", "WAIT", "CONFIRM", "TRIGGER"]


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):
        self.num_actions = 4
        self.q_table = {}
        self.alpha = 0.15
        self.gamma = 0.90
        self.epsilon = 0.05

    def get_state(self, hand_detected, fingers, confidence, stable, hold_time):
        confidence_level = int(min(confidence * 10, 10))
        hold_level = min(int(hold_time), 5)
        return (
            int(hand_detected),
            int(fingers),
            confidence_level,
            int(stable),
            hold_level,
        )

    def initialize_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float32)
            self.q_table[state][ACTION_TRIGGER] = -5.0   # safe start
            self.q_table[state][ACTION_WAIT] = 1.0

    def choose_action(self, state):
        self.initialize_state(state)

        if random.random() < self.epsilon:
            return random.choice([ACTION_IGNORE, ACTION_WAIT, ACTION_CONFIRM])

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        self.initialize_state(state)
        self.initialize_state(next_state)

        current_q = self.q_table[state][action]
        best_next_q = np.max(self.q_table[next_state])

        self.q_table[state][action] = current_q + self.alpha * (
            reward + self.gamma * best_next_q - current_q
        )

    def save(self, filename=Q_TABLE_FILE):
        np.save(filename, self.q_table, allow_pickle=True)
        print("RL Q-table saved:", filename)

    def load(self, filename=Q_TABLE_FILE):
        if not os.path.exists(filename):
            print("No previous Q-table found.")
            return

        try:
            self.q_table = np.load(filename, allow_pickle=True).item()
            print("RL Q-table loaded:", filename)
        except Exception as error:
            print("Could not load Q-table:", error)


rl_agent = QLearningAgent()
rl_agent.load()


def calculate_reward(hand_detected, fingers, confidence, stable, hold_time, action):
    reward = 0

    if not hand_detected:
        if action == ACTION_IGNORE:
            reward += 5
        elif action == ACTION_WAIT:
            reward += 1
        elif action == ACTION_CONFIRM:
            reward -= 5
        elif action == ACTION_TRIGGER:
            reward -= 15
        return reward

    if fingers < 1 or fingers > 5:
        reward += 3 if action == ACTION_IGNORE else -3
        return reward

    if confidence < 0.60:
        if action in (ACTION_IGNORE, ACTION_WAIT):
            reward += 2
        else:
            reward -= 5
        return reward

    if stable:
        if action == ACTION_WAIT:
            reward += 2
        elif action == ACTION_CONFIRM:
            reward += 4
        elif action == ACTION_TRIGGER:
            reward += 10 if hold_time >= HOLD_TIME else -8
        elif action == ACTION_IGNORE:
            reward -= 1
    else:
        if action == ACTION_WAIT:
            reward += 4
        elif action == ACTION_IGNORE:
            reward += 1
        elif action == ACTION_TRIGGER:
            reward -= 10

    return reward


# ============================================================
# HAND  ->  DATA   (this is what the ML model will learn from)
# ============================================================
# DATASET BEGIN

# ---- finger detection tuning ---------------------------------
# Press D in the AISA window to see the live numbers for each
# finger, then adjust these if one finger is read wrongly.
FINGER_STRAIGHT_ANGLE = 145.0   # degrees at the middle knuckle (bigger = stricter)
THUMB_REACH_RATIO = 1.15        # bigger = thumb must be spread further out
THUMB_STRAIGHT_ANGLE = 125.0

FINGER_SMOOTH_FRAMES = 7        # majority vote over the last N frames
HAND_LOST_GRACE = 0.4           # seconds a lost hand may flicker without
                                # resetting the hold timer


def _landmark_points(landmarks, w, h):
    # pixel units (not 0..1) so angles are not stretched by the 4:3 image
    return np.array(
        [[p.x * w, p.y * h, p.z * w] for p in landmarks.landmark],
        dtype=np.float32,
    )


def _angle(a, b, c):
    """angle at point b (degrees) between the lines b->a and b->c"""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)

    if n < 1e-6:
        return 0.0

    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def finger_states_debug(landmarks, w=640, h=480):
    """
    Returns ([thumb, index, middle, ring, pinky] extended?, debug lines).

    The old rule compared x / y positions ("tip is above the joint",
    "thumb tip is left of the thumb joint"). That only works for an
    upright right hand with the palm facing the camera.

    This one only uses angles and distances between joints, which do
    not change when the hand is rotated, mirrored (left hand) or
    turned around (back of the hand):

      finger  extended = the finger is straight (angle at the middle
                         knuckle is large) AND the tip is farther
                         from the wrist than that knuckle
      thumb   extended = the tip is far from the base of the little
                         finger (spread out, not tucked over the palm)
                         AND the thumb is not sharply bent
    """
    p = _landmark_points(landmarks, w, h)

    def dist(i, j):
        return float(np.linalg.norm(p[i] - p[j]))

    states, info = [], []

    reach = dist(4, 17) / max(dist(2, 17), 1e-6)
    angle = _angle(p[2], p[3], p[4])
    up = reach >= THUMB_REACH_RATIO and angle >= THUMB_STRAIGHT_ANGLE

    states.append(up)
    info.append(f"Thumb   reach {reach:4.2f}  angle {angle:3.0f}   {'UP' if up else '--'}")

    joints = (("Index", 5, 6, 8), ("Middle", 9, 10, 12),
              ("Ring", 13, 14, 16), ("Pinky", 17, 18, 20))

    for name, mcp, pip, tip in joints:
        angle = _angle(p[mcp], p[pip], p[tip])
        reach = dist(0, tip) / max(dist(0, pip), 1e-6)
        up = angle >= FINGER_STRAIGHT_ANGLE and reach > 1.0

        states.append(up)
        info.append(f"{name:7s} reach {reach:4.2f}  angle {angle:3.0f}   {'UP' if up else '--'}")

    return states, info


def finger_states(landmarks, w=640, h=480):
    return finger_states_debug(landmarks, w, h)[0]


class FingerSmoother:
    """
    A finger count only changes when the new value wins a majority of the
    last few frames, so a single misread frame cannot reset the 5 second
    hold (and therefore cannot stop the call from ever triggering).
    """

    def __init__(self, size=FINGER_SMOOTH_FRAMES):
        self.buffer = deque(maxlen=size)
        self.current = 0

    def update(self, value):
        self.buffer.append(value)

        counts = {}
        for v in self.buffer:
            counts[v] = counts.get(v, 0) + 1

        best, votes = max(counts.items(), key=lambda kv: kv[1])

        if votes > len(self.buffer) // 2 or self.current not in counts:
            self.current = best

        return self.current

    def clear(self):
        self.buffer.clear()
        self.current = 0


def calculate_hand_confidence(landmarks):
    xs = [p.x for p in landmarks.landmark]
    ys = [p.y for p in landmarks.landmark]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width < 0.05 or height < 0.05:
        return 0.35

    if width > 0.20 and height > 0.20:
        return 0.95

    return 0.75


def landmark_features(landmarks):
    """
    21 landmarks (x, y, z) -> 63 numbers.

    1. move the wrist to the origin   -> position independent
    2. divide by the hand size        -> distance independent
    So the same gesture gives the same numbers anywhere in the frame.
    """
    pts = np.array(
        [[p.x, p.y, p.z] for p in landmarks.landmark],
        dtype=np.float32,
    )

    pts -= pts[0]

    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))

    if scale < 1e-6:
        scale = 1.0

    pts /= scale

    return pts.flatten()


DATASET_META_COLUMNS = [
    "timestamp", "session_id", "frame", "event", "handedness",
    "fingers", "thumb", "index", "middle", "ring", "pinky",
    "hand_conf", "stable", "hold_time",
    "label", "true_label", "valid_gesture",
    "rl_action", "rl_reward",
    "ml_label", "ml_prob",
    "call_in_progress", "test_mode",
]

DATASET_LM_COLUMNS = [
    f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")
]

DATASET_COLUMNS = DATASET_META_COLUMNS + DATASET_LM_COLUMNS


class DatasetLogger:

    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.rows = 0
        self.total_rows = 0
        self.session_id = int(time.time())

    def _upgrade_columns(self):
        """older files miss the newer columns (true_label, ml_*):
        add them and keep every existing row"""
        with open(self.path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            if header is None or header == DATASET_COLUMNS:
                return

            old = [dict(zip(header, r)) for r in reader]

        tmp = self.path + ".tmp"

        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(DATASET_COLUMNS)

            for row in old:
                writer.writerow([row.get(c, "") for c in DATASET_COLUMNS])

        os.replace(tmp, self.path)

        print(f"Upgraded {self.path} to the new column layout ({len(old)} rows kept).")

    def open(self, reset=False):
        # remove datasets from older versions
        for old in LEGACY_DATASET_FILES:
            if os.path.exists(old):
                try:
                    os.remove(old)
                    print("Deleted old dataset:", old)
                except Exception as error:
                    print("Could not delete", old, "-", error)

        if reset and os.path.exists(self.path):
            try:
                os.remove(self.path)
                print("Deleted previous dataset:", self.path)
            except Exception as error:
                print("Could not delete", self.path, "-", error)

        exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0

        if exists:
            try:
                self._upgrade_columns()
            except Exception as error:
                print("Could not upgrade the dataset format:", error)
                self.path = f"aisa_gesture_dataset_{self.session_id}.csv"
                exists = False

        if exists:
            try:
                with open(self.path, "r", newline="") as f:
                    self.total_rows = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                self.total_rows = 0

        try:
            self.file = open(self.path, "a" if exists else "w", newline="")
        except PermissionError:
            # file is open in Excel -> use a new name instead
            self.path = f"aisa_gesture_dataset_{self.session_id}.csv"
            print("Dataset is locked, writing to:", self.path)
            self.file = open(self.path, "w", newline="")
            exists = False
            self.total_rows = 0

        self.writer = csv.writer(self.file)

        if not exists:
            self.writer.writerow(DATASET_COLUMNS)
            self.file.flush()

        print("Dataset:", self.path, f"({self.total_rows} existing rows)")

    def log(self, event, frame_index, handedness, states, fingers,
            confidence, stable, hold_time, label, valid_gesture,
            rl_action, rl_reward, features,
            true_label="", ml_label="", ml_prob=0.0):

        if self.writer is None:
            return

        states = list(states) + [False] * (5 - len(states))

        row = [
            round(time.time(), 3),
            self.session_id,
            frame_index,
            event,
            handedness,
            fingers,
            *[int(v) for v in states[:5]],
            round(confidence, 3),
            int(stable),
            round(hold_time, 3),
            label,
            true_label,
            int(valid_gesture),
            actions[rl_action],
            rl_reward,
            ml_label,
            round(float(ml_prob), 3),
            int(call_in_progress),
            int(TEST_MODE),
            *[round(float(v), 5) for v in features],
        ]

        self.writer.writerow(row)

        self.rows += 1
        self.total_rows += 1

        if self.rows % 25 == 0:
            self.file.flush()

    def close(self):
        if self.file is not None:
            try:
                self.file.flush()
                self.file.close()
            except Exception:
                pass
            self.file = None

# DATASET END


dataset = DatasetLogger(DATASET_FILE)
dataset.open(reset=RESET_DATASET_ON_START)


# ============================================================
# ML  -  WHAT DOES THE USER WANT?
# ============================================================
#
# Predicts WHICH SERVICE the user is asking for (Police, Fire,
# Ambulance, Women Helpline, General Helpline, or "none") from the
# shape of the hand: the 63 landmark numbers + a left/right flag.
#
# Session flow:
#   start-up   -> train from the dataset of EARLIER sessions only
#   running    -> use that model (advisory: it never blocks a call)
#                 and keep SAVING new rows, which are NOT used yet
#   next start -> those rows are now "earlier sessions" and are used
#
# Labels it learns from:
#   - your teach-mode labels (keys 0-5) if there are enough of them
#   - otherwise the rule-made "label" column. The model then mostly
#     copies the rule, so use teach mode once to give it real answers.
#
# Columns used:   lm0_x ... lm20_z (63) + handedness   -> the input
#                 true_label (or label)                -> the answer
#                 event == "frame"                      -> which rows
#                 session_id                            -> honest test split
# Not used:       fingers/thumb..pinky, stable, hold_time, valid_gesture,
#                 rl_*: these come from the rule or from timing, so the
#                 model would just copy them instead of learning the hand.
# ============================================================
# ML BEGIN

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

ml_status = "OFF"        # OFF | NO DATA | TRAINING | READY 94% | ERROR
ml_predictor = None


def load_previous_sessions(path, current_session):
    """Read the CSV once. Rows written by THIS session are ignored."""
    if not os.path.exists(path):
        return None

    X, left, true, pseudo, fingers, sessions = [], [], [], [], [], []

    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):

            if row.get("event") != "frame":
                continue

            if row.get("session_id") == str(current_session):
                continue

            try:
                feats = [float(row[c]) for c in DATASET_LM_COLUMNS]
                finger_count = int(float(row.get("fingers") or 0))
            except (ValueError, KeyError, TypeError):
                continue

            X.append(feats)
            left.append(1.0 if row.get("handedness") == "Left" else 0.0)
            true.append(row.get("true_label") or "")
            pseudo.append(row.get("label") or "none")
            fingers.append(finger_count)
            sessions.append(row.get("session_id") or "")

    if not X:
        return None

    return {
        "X": np.hstack([np.array(X, np.float32), np.array(left, np.float32)[:, None]]),
        "true": np.array(true),
        "pseudo": np.array(pseudo),
        "fingers": np.array(fingers),
        "sessions": np.array(sessions),
    }


def _choose_target(data):
    """returns (labels, usable-row mask or None, description)"""
    true = data["true"]
    true_mask = true != ""

    if true_mask.sum() >= ML_MIN_TRUE_ROWS and len(np.unique(true[true_mask])) >= 2:
        y, mask = true, true_mask.copy()
        source = "your teach-mode labels"
    else:
        y, mask = data["pseudo"], np.ones(len(true), dtype=bool)
        source = ("rule-made labels (the model will mostly copy the rule - "
                  "use teach mode for real labels)")

    classes, counts = np.unique(y[mask], return_counts=True)
    keep = [c for c, n in zip(classes, counts) if n >= ML_MIN_ROWS_PER_CLASS]

    if len(keep) < 2:
        return y, None, source

    mask &= np.isin(y, keep)

    return y, mask, source


class GesturePredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features, handedness):
        """returns (label, probability)"""
        x = np.concatenate(
            [features, [1.0 if handedness == "Left" else 0.0]]
        ).astype(np.float32)[None, :]

        probs = self.model.predict_proba(x)[0]
        best = int(np.argmax(probs))

        return str(self.model.classes_[best]), float(probs[best])


def _make_forest():
    return RandomForestClassifier(
        n_estimators=ML_TREES,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=1,
        random_state=0,
    )


def _train_ml(data):
    global ml_status, ml_predictor

    try:
        y_all, mask, source = _choose_target(data)

        if mask is None:
            ml_status = "NO DATA"
            print("\nML: not enough labelled rows yet "
                  f"(need {ML_MIN_ROWS_PER_CLASS}+ per gesture, 2+ gestures).")
            return

        idx = np.where(mask)[0]

        if len(idx) > ML_MAX_TRAIN_ROWS:
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(idx, ML_MAX_TRAIN_ROWS, replace=False))

        X = data["X"][idx]
        y = y_all[idx]
        sessions = data["sessions"][idx]
        rule = np.array([emergency_labels.get(int(f), "none") for f in data["fingers"][idx]])

        # honest check: test on sessions the model has NOT seen
        # (neighbouring frames are near-copies, a random split would cheat)
        if len(set(sessions)) >= 2:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                          .split(X, y, groups=sessions))
            how = f"{len(set(sessions[te]))} unseen session(s)"
        else:
            cut = int(len(idx) * 0.75)
            tr, te = np.arange(cut), np.arange(cut, len(idx))
            how = "the last 25% of the only session"

        acc = rule_acc = None

        if len(te) and len(set(y[tr])) >= 2:
            check = _make_forest().fit(X[tr], y[tr])
            acc = float(np.mean(check.predict(X[te]) == y[te]))
            rule_acc = float(np.mean(rule[te] == y[te]))

        model = _make_forest().fit(X, y)

        try:
            joblib.dump(model, ML_MODEL_FILE)
        except Exception:
            pass

        ml_predictor = GesturePredictor(model)
        ml_status = f"READY {acc:.0%}" if acc is not None else "READY"

        classes, counts = np.unique(y, return_counts=True)

        print("\n" + "-" * 50)
        print("ML model trained from earlier sessions")
        print("  labels used :", source)
        print("  rows        :", len(idx), " ", dict(zip(classes.tolist(), counts.tolist())))

        if acc is not None:
            print(f"  accuracy    : {acc:.1%} on {how}")
            print(f"  rule alone  : {rule_acc:.1%} on the same rows")

        print("-" * 50)

    except Exception as error:
        ml_status = "ERROR"
        print("ML training failed:", error)


def start_ml(path, session_id):
    global ml_status

    if not ML_ENABLED:
        ml_status = "OFF"
        return

    if not SKLEARN_OK:
        ml_status = "OFF"
        print("\nML is off: run  pip install scikit-learn")
        return

    print("\nML: loading earlier sessions...")

    data = load_previous_sessions(path, session_id)

    if data is None:
        ml_status = "NO DATA"
        print("ML: no earlier data yet. This session collects it; "
              "the model trains next time.")
        return

    ml_status = "TRAINING"

    # in the background, so the camera opens straight away
    threading.Thread(target=_train_ml, args=(data,), daemon=True).start()

# ML END


start_ml(dataset.path, dataset.session_id)


# ============================================================
# FRONTEND  (drawn with OpenCV shapes + Pillow text)
# ============================================================
# UI BEGIN

UI_W, UI_H = 1280, 720

# colours are RGB here and converted when drawing with OpenCV
C = {
    "text":   (236, 241, 250),
    "text2":  (148, 160, 188),
    "muted":  (94, 106, 134),
    "panel":  (17, 24, 41),
    "card":   (27, 36, 59),
    "border": (50, 63, 98),
    "dark":   (9, 13, 24),
    "cyan":   (34, 211, 238),
    "green":  (52, 211, 153),
    "amber":  (251, 191, 36),
    "red":    (248, 113, 113),
}

SERVICE_COLORS = {
    1: (76, 141, 255),    # police      - blue
    2: (255, 107, 53),    # fire        - orange
    3: (255, 77, 109),    # ambulance   - rose
    4: (178, 107, 255),   # women help  - violet
    5: (46, 230, 166),    # general     - teal
}


def bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def mix(a, b, t):
    """blend colour a towards b by t (0..1)"""
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


# ---------------- fonts ----------------

_FONTS = {}

_FONT_FILES = {
    False: [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    True: [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def get_font(size, bold=False):
    key = (size, bold)

    if key not in _FONTS:
        font = None

        if PIL_OK:
            for path in _FONT_FILES[bold]:
                if os.path.exists(path):
                    try:
                        font = ImageFont.truetype(path, size)
                        break
                    except Exception:
                        pass

        _FONTS[key] = font

    return _FONTS[key]


@lru_cache(maxsize=4096)
def measure(text, size, bold=False):
    font = get_font(size, bold)

    if font is not None:
        return int(font.getlength(text))

    (w, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, 1
    )
    return w


@lru_cache(maxsize=1024)
def wrap(text, size, max_w, bold=False, max_lines=9):
    words = str(text).split()
    lines, line = [], ""

    for word in words:
        trial = (line + " " + word).strip()

        if measure(trial, size, bold) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word

    if line:
        lines.append(line)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,") + "..."

    return tuple(lines)


# ---------------- shapes ----------------

_MASKS = {}


def rr_mask(w, h, r):
    """anti-aliased rounded-rectangle alpha mask (cached)"""
    r = max(1, min(r, w // 2, h // 2))
    key = (w, h, r)

    if key not in _MASKS:
        s = 4
        big = np.zeros((h * s, w * s), np.uint8)
        R = r * s
        W, H = w * s - 1, h * s - 1

        cv2.rectangle(big, (R, 0), (W - R, H), 255, -1)
        cv2.rectangle(big, (0, R), (W, H - R), 255, -1)

        for cx, cy in ((R, R), (W - R, R), (R, H - R), (W - R, H - R)):
            cv2.circle(big, (cx, cy), R, 255, -1)

        small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
        _MASKS[key] = small.astype(np.float32) / 255.0

    return _MASKS[key]


def fill_rrect(img, x, y, w, h, r, color, alpha=1.0):
    x, y, w, h = int(x), int(y), int(w), int(h)

    roi = img[y:y + h, x:x + w]
    m = (rr_mask(w, h, r) * alpha)[:, :, None]
    col = np.array(bgr(color), np.float32)

    roi[:] = (roi * (1.0 - m) + col * m).astype(np.uint8)


def rr_outline(img, x, y, w, h, r, color, th=1):
    x, y, w, h = int(x), int(y), int(w), int(h)
    r = max(1, min(r, w // 2, h // 2))
    c = bgr(color)
    aa = cv2.LINE_AA

    cv2.line(img, (x + r, y), (x + w - r, y), c, th, aa)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), c, th, aa)
    cv2.line(img, (x, y + r), (x, y + h - r), c, th, aa)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), c, th, aa)

    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, c, th, aa)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, c, th, aa)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, c, th, aa)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, c, th, aa)


def glow(img, x, y, w, h, color, radius=16, strength=0.55):
    pad = radius * 2

    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])

    layer = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.rectangle(layer, (x - x0, y - y0), (x - x0 + w, y - y0 + h), 1.0, -1)
    layer = cv2.GaussianBlur(layer, (0, 0), radius / 2.0)[:, :, None]

    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi += layer * np.array(bgr(color), np.float32) * strength
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


def paste_rounded(canvas, img, x, y, r):
    """paste the camera image with rounded corners (cheap: only corners are blended)"""
    h, w = img.shape[:2]
    m = rr_mask(w, h, r)

    corners = [(0, 0), (0, w - r), (h - r, 0), (h - r, w - r)]
    saved = [canvas[y + cy:y + cy + r, x + cx:x + cx + r].copy() for cy, cx in corners]

    canvas[y:y + h, x:x + w] = img

    for (cy, cx), bg in zip(corners, saved):
        mm = m[cy:cy + r, cx:cx + r, None]
        patch = canvas[y + cy:y + cy + r, x + cx:x + cx + r]
        patch[:] = (patch * mm + bg * (1.0 - mm)).astype(np.uint8)


def make_background():
    top = np.array(bgr((9, 13, 24)), np.float32)
    bot = np.array(bgr((19, 27, 52)), np.float32)

    t = np.linspace(0, 1, UI_H, dtype=np.float32)[:, None, None]
    col = top * (1 - t) + bot * t
    bg = np.repeat(col, UI_W, axis=1)

    # soft colour glows in two corners
    layer = np.zeros((UI_H, UI_W, 3), np.float32)
    cv2.circle(layer, (180, 90), 260, bgr(C["cyan"]), -1)
    cv2.circle(layer, (1120, 650), 300, bgr((139, 92, 246)), -1)
    layer = cv2.GaussianBlur(layer, (0, 0), 140) * 0.20

    return np.clip(bg + layer, 0, 255).astype(np.uint8)


# ---------------- cached text bitmaps ----------------
# Each distinct string is rendered with Pillow once, then reused as
# an alpha mask. Most HUD text never changes, so this is much faster
# than redrawing every string on every frame.

_TEXT_CACHE = {}


def text_bitmap(text, size, bold):
    key = (text, size, bold)

    if key not in _TEXT_CACHE:
        if len(_TEXT_CACHE) > 1500:
            _TEXT_CACHE.clear()

        font = get_font(size, bold)
        asc, desc = font.getmetrics()
        pad = 3

        w = int(font.getlength(text)) + pad * 2 + 2
        h = asc + desc + pad * 2

        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).text((pad, pad + (asc + desc) / 2.0), text, font=font, fill=255, anchor="lm")

        alpha = (np.asarray(img, dtype=np.float32) / 255.0)[:, :, None]

        _TEXT_CACHE[key] = (alpha, (pad, pad + (asc + desc) // 2), int(font.getlength(text)))

    return _TEXT_CACHE[key]


# ---------------- layout ----------------

CAM_X, CAM_Y, CAM_W, CAM_H = 320, 100, 640, 480
PANIC_X, PANIC_Y, PANIC_W, PANIC_H = 20, 534, 280, 46     # on-screen panic button
LEFT_X, RIGHT_X, PANEL_W = 20, 980, 280
STATUS_Y, STATUS_H = 590, 110


class UI:

    def __init__(self):
        self.canvas = None
        self.texts = []
        self.static = self._build_static()

    # ---------------- text layer ----------------

    def text(self, s, x, y, size=14, color=None, bold=False, anchor="lm"):
        self.texts.append(
            (str(s), int(x), int(y), size, color or C["text"], bold, anchor)
        )

    def flush(self):
        if not self.texts:
            return

        cv = self.canvas
        H, W = cv.shape[:2]

        if PIL_OK and get_font(14) is not None:

            for s, x, y, size, color, bold, anchor in self.texts:
                alpha, (ax, ay), text_w = text_bitmap(s, size, bold)

                if anchor[0] == "m":
                    x -= text_w // 2
                elif anchor[0] == "r":
                    x -= text_w

                x0, y0 = int(x - ax), int(y - ay)
                h, w = alpha.shape[:2]

                # clip to the canvas
                cx0, cy0 = max(x0, 0), max(y0, 0)
                cx1, cy1 = min(x0 + w, W), min(y0 + h, H)

                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
                roi = cv[cy0:cy1, cx0:cx1]

                roi[:] = (roi * (1.0 - a) + np.array(bgr(color), np.float32) * a).astype(np.uint8)

        else:
            for s, x, y, size, color, bold, anchor in self.texts:
                scale = size / 28.0
                (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)

                if anchor[0] == "m":
                    x -= w // 2
                elif anchor[0] == "r":
                    x -= w

                cv2.putText(
                    cv, s, (x, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, bgr(color),
                    2 if bold else 1, cv2.LINE_AA,
                )

        self.texts = []

    # ---------------- small widgets ----------------

    def chip(self, x, y, text, color, size=12, align="l", h=24, pad=11, dot=False):
        w = measure(text, size, True) + pad * 2 + (14 if dot else 0)

        if align == "r":
            x -= w

        fill_rrect(self.canvas, x, y, w, h, h // 2, color, 0.16)
        rr_outline(self.canvas, x, y, w, h, h // 2, mix(color, C["dark"], 0.35), 1)

        tx = x + pad

        if dot:
            cv2.circle(self.canvas, (x + pad + 3, y + h // 2), 4, bgr(color), -1, cv2.LINE_AA)
            tx += 14

        self.text(text, tx, y + h // 2, size, color, True, "lm")

        return w

    def bar(self, x, y, w, h, frac, color):
        fill_rrect(self.canvas, x, y, w, h, h // 2, C["border"], 0.55)

        frac = max(0.0, min(1.0, frac))

        if frac > 0:
            fw = max(h, int(w * frac))
            fill_rrect(self.canvas, x, y, fw, h, h // 2, color, 1.0)

    # ---------------- static parts (drawn once) ----------------

    def _build_static(self):
        self.canvas = make_background()
        cv = self.canvas
        self.texts = []

        # header bar
        cv[0:68] = (cv[0:68] * 0.45).astype(np.uint8)
        cv2.line(cv, (0, 68), (UI_W, 68), bgr(C["border"]), 1)

        # logo
        cv2.circle(cv, (50, 34), 21, bgr(C["cyan"]), -1, cv2.LINE_AA)
        cv2.circle(cv, (50, 34), 21, bgr(mix(C["cyan"], (255, 255, 255), 0.5)), 1, cv2.LINE_AA)
        cv2.rectangle(cv, (46, 24), (54, 44), bgr(C["dark"]), -1)
        cv2.rectangle(cv, (40, 30), (60, 38), bgr(C["dark"]), -1)

        self.text("AISA", 84, 27, 25, C["text"], True)
        self.text("Gesture Emergency Assistant", 85, 51, 12, C["text2"])

        # panels
        fill_rrect(cv, RIGHT_X, CAM_Y, PANEL_W, CAM_H, 18, C["panel"], 0.82)
        rr_outline(cv, RIGHT_X, CAM_Y, PANEL_W, CAM_H, 18, C["border"], 1)

        for px, pw in ((LEFT_X, PANEL_W), (CAM_X, CAM_W), (RIGHT_X, PANEL_W)):
            fill_rrect(cv, px, STATUS_Y, pw, STATUS_H, 18, C["panel"], 0.82)
            rr_outline(cv, px, STATUS_Y, pw, STATUS_H, 18, C["border"], 1)

        # camera backdrop
        fill_rrect(cv, CAM_X - 6, CAM_Y - 6, CAM_W + 12, CAM_H + 12, 24, C["dark"], 0.55)

        # controls
        self.text("CONTROLS", LEFT_X + 18, STATUS_Y + 18, 10, C["muted"], True)

        keys = [("ESC", "Quit"), ("ENTER", "Answered"),
                ("X", "End session"), ("F", "Fullscreen"),
                ("D", "Finger debug"), ("0-5", "Teach labels"),
                ("B", "Panic call")]

        for i, (k, d) in enumerate(keys):
            kx = LEFT_X + 18 + (i % 2) * 130
            ky = STATUS_Y + 36 + (i // 2) * 21

            kw = measure(k, 11, True) + 14
            fill_rrect(cv, kx, ky - 9, kw, 18, 6, C["card"], 1.0)
            rr_outline(cv, kx, ky - 9, kw, 18, 6, C["border"], 1)

            self.text(k, kx + kw / 2, ky, 11, C["text"], True, "mm")
            self.text(d, kx + kw + 7, ky, 11, C["text2"])

        self.flush()

        return self.canvas.copy()

    # ---------------- per-frame drawing ----------------

    def render(self, cam, s):
        pulse = 0.5 + 0.5 * math.sin(time.time() * 4.5)

        self.canvas = self.static.copy()
        self.texts = []

        self._header(s, pulse)
        self._cards(s, pulse)
        self._panic(s)
        self._camera(cam, s, pulse)
        self._status(s)
        self._assistant(s, pulse)
        self._system(s)

        self.flush()

        return self.canvas

    def _header(self, s, pulse):
        right = UI_W - 24

        label, col = ("TEST MODE", C["amber"]) if s["test_mode"] else ("LIVE", C["red"])
        right -= self.chip(right, 22, label, col, 12, "r", dot=True) + 10

        if s["recording"]:
            m, sec = divmod(int(s["rec_elapsed"]), 60)
            right -= self.chip(right, 22, f"REC {m:02d}:{sec:02d}", C["red"], 12, "r", dot=True) + 14

        self.text(time.strftime("%H:%M:%S"), right, 28, 19, C["text"], True, "rm")
        self.text(time.strftime("%a, %d %b"), right, 51, 11, C["text2"], False, "rm")

        self.text(
            f"Show 1-5 fingers and hold for {s['hold_target']:.0f} seconds",
            UI_W // 2, 34, 14, C["text2"], False, "mm",
        )

    def _cards(self, s, pulse):
        cv = self.canvas
        phase = s["call_phase"]

        for i, n in enumerate(range(1, 6)):
            x, y, w, h = LEFT_X, CAM_Y + i * 86, PANEL_W, 78

            color = SERVICE_COLORS[n]
            label = s["labels"][n]

            calling = phase != "idle" and s["call_label"] == label
            active = (
                phase == "idle" and s["hand_detected"] and s["fingers"] == n
            )
            dim = phase != "idle" and not calling

            hot = active or calling

            if hot:
                glow(cv, x, y, w, h, color, 16, 0.30 + 0.12 * pulse)

            fill_rrect(cv, x, y, w, h, 16, C["card"], 0.5 if dim else 0.95)

            if hot:
                fill_rrect(cv, x, y, w, h, 16, color, 0.15)
                rr_outline(cv, x, y, w, h, 16, color, 2)
            else:
                rr_outline(cv, x, y, w, h, 16, C["border"], 1)

            # number badge
            cx, cy = x + 46, y + 39

            if hot:
                cv2.circle(cv, (cx, cy), 25, bgr(color), -1, cv2.LINE_AA)
                digit_col = C["dark"]
            else:
                cv2.circle(cv, (cx, cy), 25, bgr(mix(color, C["card"], 0.78)), -1, cv2.LINE_AA)
                cv2.circle(cv, (cx, cy), 25, bgr(mix(color, C["card"], 0.40)), 1, cv2.LINE_AA)
                digit_col = color

            self.text(str(n), cx, cy - 1, 25, digit_col, True, "mm")

            name_col = C["muted"] if dim else C["text"]
            self.text(label, x + 88, y + 27, 18, name_col, True)

            if calling:
                sub = {"dialing": "Dialing...",
                       "announcing": "Connected",
                       "assisting": "On call"}.get(phase, "")
                self.text(sub, x + 88, y + 50, 12, color, True)
                cv2.circle(cv, (x + w - 20, y + 18), int(4 + pulse * 3), bgr(color), -1, cv2.LINE_AA)

            elif active:
                remaining = max(0.0, s["hold_target"] - s["stable_time"])
                self.text(f"Hold... {remaining:.1f}s", x + 88, y + 47, 12, color, True)
                self.bar(x + 88, y + 59, 170, 6, s["stable_time"] / s["hold_target"], color)

            else:
                hint = f"Show {n} finger" + ("s" if n > 1 else "")
                self.text(hint, x + 88, y + 50, 12, C["muted"] if dim else C["text2"])

    def _panic(self, s):
        """on-screen panic button: hold 3 s (mouse) or B for a countdown"""
        cv = self.canvas
        x, y, w, h = PANIC_X, PANIC_Y, PANIC_W, PANIC_H
        cx, cy = x + w // 2, y + h // 2

        blocked = s["panic_blocked"]
        frac = s["panic_frac"]
        col = C["amber"]

        if blocked:
            fill_rrect(cv, x, y, w, h, 14, C["card"], 0.5)
            rr_outline(cv, x, y, w, h, 14, C["border"], 1)
            self.text(blocked.upper(), cx, cy, 12, C["muted"], True, "mm")
            return

        fill_rrect(cv, x, y, w, h, 14, C["card"], 0.95)

        if frac is not None:
            fill_rrect(cv, x, y, max(int(w * frac), 28), h, 14, col, 0.45)
            rr_outline(cv, x, y, w, h, 14, col, 2)

            if s["panic_mode"] == "key":
                left = max(0.0, PANIC_HOLD_SECONDS * (1.0 - frac))
                text = f"CALLING IN {left:.1f}s  -  B TO CANCEL"
            else:
                text = "KEEP HOLDING..."

            self.text(text, cx, cy, 12, C["text"], True, "mm")

        else:
            fill_rrect(cv, x, y, w, h, 14, col, 0.10)
            rr_outline(cv, x, y, w, h, 14, mix(col, C["dark"], 0.3), 1)
            self.text(f"HOLD {PANIC_HOLD_SECONDS:.0f} s: " + s["labels"][BUTTON_SERVICE].upper(),
                      cx, y + 16, 12, col, True, "mm")
            self.text("click and hold, or press B", cx, y + 33, 10, C["text2"], False, "mm")

    def _camera(self, cam, s, pulse):
        cv = self.canvas

        if cam.shape[1] != CAM_W or cam.shape[0] != CAM_H:
            cam = cv2.resize(cam, (CAM_W, CAM_H))

        paste_rounded(cv, cam, CAM_X, CAM_Y, 18)

        phase = s["call_phase"]
        fingers = s["fingers"]

        if phase == "dialing":
            ring = C["red"] if pulse > 0.5 else mix(C["red"], C["dark"], 0.55)   # flashing red
        elif phase in ("announcing", "assisting"):
            ring = C["green"]
        elif s["hand_detected"] and s["stable"] and fingers in SERVICE_COLORS:
            ring = SERVICE_COLORS[fingers]
        elif s["hand_detected"]:
            ring = C["cyan"]
        else:
            ring = C["border"]

        rr_outline(cv, CAM_X - 3, CAM_Y - 3, CAM_W + 6, CAM_H + 6, 21, ring, 2)

        # viewfinder corners
        L, inset = 30, 14
        x0, y0 = CAM_X + inset, CAM_Y + inset
        x1, y1 = CAM_X + CAM_W - inset, CAM_Y + CAM_H - inset
        rc = bgr(ring)

        for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(cv, (px, py), (px + dx * L, py), rc, 3, cv2.LINE_AA)
            cv2.line(cv, (px, py), (px, py + dy * L), rc, 3, cv2.LINE_AA)

        # chips
        self.chip(CAM_X + 26, CAM_Y + 24, f"CAMERA  {s['fps']:.0f} FPS", C["cyan"], 11, dot=True, h=22)

        if s["hand_detected"]:
            self.chip(CAM_X + CAM_W - 26, CAM_Y + 24, "HAND DETECTED", C["green"], 11, "r", dot=True, h=22)
        else:
            self.chip(CAM_X + CAM_W - 26, CAM_Y + 24, "NO HAND", C["muted"], 11, "r", dot=True, h=22)

        # teach mode banner
        if s.get("teach") is not None:
            t_text = f"TEACHING: {s['teach']}"
            t_w = measure(t_text, 11, True) + 22 + 14
            self.chip(CAM_X + (CAM_W - t_w) // 2, CAM_Y + 24, t_text, C["amber"], 11, dot=True, h=22)

        # live finger numbers (D key)
        dbg = s.get("debug")

        if dbg:
            bh = 16 + len(dbg) * 18
            bx, by = CAM_X + 28, CAM_Y + CAM_H - bh - 28

            fill_rrect(cv, bx, by, 330, bh, 12, C["dark"], 0.85)
            rr_outline(cv, bx, by, 330, bh, 12, C["border"], 1)

            for k, line in enumerate(dbg):
                col = C["green"] if line.rstrip().endswith("UP") else C["text2"]
                self.text(line, bx + 14, by + 16 + k * 18, 11, col)

        # call banner
        label = s["call_label"]
        banner = None

        if phase == "dialing":
            if s.get("call_warning"):
                banner = ("No answer detected  -  click here, press ENTER if answered", C["amber"])
            elif CALL_ANSWER_MODE == "manual_key":
                banner = (f"Calling {label}  -  press ENTER when answered", C["amber"])
            elif CALL_ANSWER_MODE == "screen":
                banner = (f"Calling {label}  -  waiting for the call to be answered", C["amber"])
            else:
                banner = (f"Calling {label}...", C["amber"])
        elif phase == "announcing":
            banner = (f"Connected  -  telling {label} where you are", C["green"])
        elif phase == "assisting":
            banner = (f"On call with {label}  -  AISA is helping you", C["green"])

        if banner:
            text, col = banner
            bw = measure(text, 15, True) + 64
            bx = CAM_X + (CAM_W - bw) // 2
            by = CAM_Y + 60

            fill_rrect(cv, bx, by, bw, 38, 19, C["dark"], 0.82)
            rr_outline(cv, bx, by, bw, 38, 19, col, 2)
            cv2.circle(cv, (bx + 24, by + 19), int(4 + pulse * 2), bgr(col), -1, cv2.LINE_AA)
            self.text(text, bx + 40, by + 19, 15, col, True)

        elif not s["hand_detected"]:
            self.text("Show your hand to the camera", CAM_X + CAM_W // 2, CAM_Y + CAM_H - 44,
                      16, C["text2"], False, "mm")

    def _status(self, s):
        x, y = CAM_X + 24, STATUS_Y

        size = 20

        while measure(s["status"], size, True) > CAM_W - 48 and size > 13:
            size -= 1

        self.text(s["status"], x, y + 24, size, s["status_color"], True)

        frac = s["stable_time"] / s["hold_target"] if s["hand_detected"] else 0.0
        self.bar(x, y + 44, CAM_W - 48, 12, frac, s["status_color"])

        agree = s["ml_agree"]
        ml_col = C["green"] if agree is True else (C["amber"] if agree is False else C["text2"])

        tiles = [
            (0,   "FINGERS", str(s["fingers"]), C["text"], 16),
            (88,  "CONFIDENCE", f"{s['confidence']:.2f}", C["text"], 16),
            (184, "HOLD", f"{s['stable_time']:.1f}s / {s['hold_target']:.1f}s", C["text"], 16),
            (300, "RL ACTION", s["rl_action"], C["text"], 16),
            (404, "ML SAYS", s["ml_pred"], ml_col, 15),
        ]

        for off, label, value, col, size in tiles:
            while measure(value, size, True) > 184 and size > 11:
                size -= 1

            self.text(label, x + off, y + 72, 10, C["muted"], True)
            self.text(value, x + off, y + 92, size, col, True)

    def _assistant(self, s, pulse):
        cv = self.canvas
        x, y, w = RIGHT_X, CAM_Y, PANEL_W
        phase = s["call_phase"]

        self.text("AISA ASSISTANT", x + 18, y + 24, 13, C["text"], True)

        if phase == "assisting":
            self.chip(x + w - 16, y + 12, "ACTIVE", C["green"], 10, "r", dot=True, h=22)
        elif phase == "dialing":
            self.chip(x + w - 16, y + 12, "WAITING", C["amber"], 10, "r", dot=True, h=22)
        elif phase == "announcing":
            self.chip(x + w - 16, y + 12, "SPEAKING", C["cyan"], 10, "r", dot=True, h=22)
        else:
            self.chip(x + w - 16, y + 12, "STANDBY", C["muted"], 10, "r", dot=True, h=22)

        cv2.line(cv, (x + 14, y + 46), (x + w - 14, y + 46), bgr(C["border"]), 1)

        top, bottom = y + 56, y + CAM_H - 14
        inner_x, inner_w = x + 16, w - 32

        msgs = list(s["chat"])[-14:]

        if not msgs:
            self.text("AISA is standing by", x + w // 2, y + 190, 15, C["text"], True, "mm")

            for k, line in enumerate(wrap(
                "Hold a gesture for a few seconds to start an emergency call. "
                "AISA will talk to the responder and stay with you.",
                12, inner_w - 20,
            )):
                self.text(line, x + w // 2, y + 222 + k * 18, 12, C["text2"], False, "mm")
            return

        cursor = bottom

        for who, message in reversed(msgs):

            if who == "SYSTEM":
                lines = wrap(message, 11, inner_w - 10, max_lines=3)
                height = len(lines) * 15 + 6

                if cursor - height < top:
                    break

                cursor -= height

                for k, line in enumerate(lines):
                    self.text(line, x + w // 2, cursor + 9 + k * 15, 11, C["muted"], False, "mm")

                cursor -= 8
                continue

            fs = 12 if who == "CALL" else 13
            lh = 16 if who == "CALL" else 17

            max_text_w = inner_w - 40 if who != "CALL" else inner_w - 24
            lines = wrap(message, fs, max_text_w, max_lines=5 if who == "CALL" else 8)

            text_w = max(measure(line, fs) for line in lines)
            bw = inner_w if who == "CALL" else min(inner_w - 12, text_w + 22)
            bh = 22 + len(lines) * lh + 6

            if cursor - bh < top:
                break

            cursor -= bh

            if who == "YOU":
                bx, col, name = inner_x + inner_w - bw, C["cyan"], "YOU"
                fill_rrect(cv, bx, cursor, bw, bh, 14, C["cyan"], 0.14)
                rr_outline(cv, bx, cursor, bw, bh, 14, mix(C["cyan"], C["dark"], 0.45), 1)

            elif who == "CALL":
                bx, col, name = inner_x, C["amber"], "TO RESPONDER"
                fill_rrect(cv, bx, cursor, bw, bh, 14, C["amber"], 0.12)
                rr_outline(cv, bx, cursor, bw, bh, 14, mix(C["amber"], C["dark"], 0.45), 1)

            else:
                bx, col, name = inner_x, C["green"], "AISA"
                fill_rrect(cv, bx, cursor, bw, bh, 14, C["card"], 1.0)
                rr_outline(cv, bx, cursor, bw, bh, 14, C["border"], 1)

            self.text(name, bx + 12, cursor + 11, 9, col, True)

            for k, line in enumerate(lines):
                self.text(line, bx + 12, cursor + 28 + k * lh, fs, C["text"])

            cursor -= 8

    def _system(self, s):
        x, y = RIGHT_X + 16, STATUS_Y + 12

        model = s["gemini_model"]

        if model:
            gem, gem_col = model.replace("gemini-", ""), C["green"]
        else:
            gem, gem_col = "OFFLINE", C["amber"]

        tiles = [
            ("GEMINI", gem, gem_col),
            ("CALL ANSWER", s["answer_mode"].replace("_", " "), C["text"]),
            ("DATASET", f"{s['dataset_rows']:,} rows", C["cyan"]),
            ("ML MODEL", s["ml_status"],
             C["green"] if s["ml_status"].startswith("READY")
             else (C["amber"] if s["ml_status"] == "TRAINING" else C["muted"])),
        ]

        for i, (label, value, col) in enumerate(tiles):
            tx = x + (i % 2) * 126
            ty = y + (i // 2) * 48

            fill_rrect(self.canvas, tx, ty, 118, 42, 10, C["card"], 0.9)
            self.text(label, tx + 11, ty + 12, 9, C["muted"], True)
            self.text(value, tx + 11, ty + 29, 13, col, True)

# UI END


ui = UI()


# ============================================================
# DRAWING ON THE CAMERA IMAGE
# ============================================================

LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(238, 211, 34), thickness=2, circle_radius=3)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(235, 235, 235), thickness=2)


def draw_hand(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        LANDMARK_SPEC, CONNECTION_SPEC,
    )


def draw_finger_points(frame, landmarks, states):
    h, w, _ = frame.shape

    for tip_index, up in zip((4, 8, 12, 16, 20), states):
        p = landmarks.landmark[tip_index]
        x, y = int(p.x * w), int(p.y * h)

        color = (153, 211, 52) if up else (110, 110, 110)

        cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 13, (255, 255, 255), 2, cv2.LINE_AA)


# ============================================================
# AISA TEXT TO SPEECH
# ============================================================

def speak(text, speaker="AISA"):
    print()
    print(speaker + ":", text)

    chat_log.append((speaker, str(text)))

    try:
        pythoncom.CoInitialize()

        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)
        engine.say(str(text))
        engine.runAndWait()
        engine.stop()

    except Exception as error:
        print("TTS ERROR:", error)

    finally:
        pythoncom.CoUninitialize()


# ============================================================
# SPEECH TO TEXT
# ============================================================

recognizer = sr.Recognizer()

# wait longer before deciding a sentence is finished
recognizer.pause_threshold = 1.5
recognizer.non_speaking_duration = 0.8
recognizer.dynamic_energy_threshold = True

# filled by listen() so Gemini knows when a transcript may be wrong
last_speech_confidence = None
last_speech_alternatives = []


def calibrate_microphone():
    # once, in a quiet moment (not before every listen)
    try:
        with sr.Microphone() as source:
            print("\nCalibrating microphone (stay quiet)...")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)

        print("Mic energy threshold:", int(recognizer.energy_threshold))

    except Exception as error:
        print("Mic calibration failed:", error)


def listen():
    global last_speech_confidence, last_speech_alternatives

    try:
        time.sleep(0.4)   # don't catch the tail of AISA's own voice

        with sr.Microphone() as source:
            print("\nAISA Listening...")

            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)

        print("Processing speech...")

        result = recognizer.recognize_google(
            audio, language=SPEECH_LANGUAGE, show_all=True
        )

        if not result or "alternative" not in result:
            raise sr.UnknownValueError()

        alternatives = result["alternative"]

        text = alternatives[0].get("transcript", "")
        last_speech_confidence = alternatives[0].get("confidence")
        last_speech_alternatives = [
            a.get("transcript", "") for a in alternatives if a.get("transcript")
        ]

        if last_speech_confidence is not None:
            print(f"(speech confidence: {last_speech_confidence:.2f})")

        print("YOU:", text)

        if text:
            chat_log.append(("YOU", text))

        return text

    except sr.WaitTimeoutError:
        print("No speech detected.")
        return ""

    except sr.UnknownValueError:
        print("Could not understand your voice.")
        return ""

    except sr.RequestError as error:
        print("Speech recognition error:", error)
        return ""

    except Exception as error:
        print("Microphone error:", error)
        return ""


# ============================================================
# MESSAGE FOR THE RESPONDER  (fixed template, never AI-written)
# ============================================================

def build_responder_message(emergency_type):
    return (
        f"This is an automated emergency alert from AISA. "
        f"{emergency_type} assistance is needed at {EMERGENCY_ADDRESS}. "
        f"This call was triggered by a hand gesture emergency system. "
        f"Please stay on the line, AISA will now assist the caller "
        f"while you are connected."
    )


# ============================================================
# CALL WINDOW WATCHER  (is the call answered?)
# ============================================================

# score = how "switched on" the Mute / Keypad / Hold call buttons look,
# compared with the End button (1.0 = as bright as End)
ANSWERED_MIN_SCORE = 0.85       # measured on a real call: about 1.2
RINGING_SCORE_RANGE = (0.25, 0.70)   # measured on a real call: about 0.5


def find_end_button(img):
    """Locate the red, round End button. Returns (cx, cy, radius) or None."""
    b, g, r = cv2.split(img)
    mask = ((r > 170) & (g < 100) & (b < 100)).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    best = None

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if w < 12 or h < 12:
            continue

        aspect = w / float(h)
        fill = area / float(w * h)

        # a filled circle with a white phone icon cut out of it
        if 0.8 <= aspect <= 1.25 and 0.6 <= fill <= 0.9:
            if best is None or area > best[0]:
                best = (area, x + w / 2.0, y + h / 2.0, (w + h) / 4.0)

    if best is None:
        return None

    return best[1], best[2], best[3]


def _contrast(gray, cx, cy, half_w, half_h):
    """how strongly text / icons stand out from the background here"""
    x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, gray.shape[1]))
    y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, gray.shape[0]))

    crop = gray[y0:y1, x0:x1]

    if crop.size < 20:
        return 0.0

    med = float(np.median(crop))

    # max of both directions so light-theme (dark text) works too
    return max(float(np.percentile(crop, 97)) - med,
               med - float(np.percentile(crop, 3)))


def classify_call_window(img):
    """
    img = BGR picture of the call window (or of the whole screen).
    Returns (state, score) with state "ringing", "answered" or "unknown".
    """
    found = find_end_button(img)

    if found is None:
        return "unknown", 0.0

    cx, cy, r = found
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # the "End" label is always fully bright: use it as the reference
    reference = _contrast(gray, cx, cy + 1.6 * r, 1.5 * r, 0.45 * r)

    if reference < 25:
        return "unknown", 0.0

    labels, icons = [], []

    # Hold call, Keypad, Mute sit 3.76 radii apart, to the left of End
    for k in (1, 2, 3):
        x = cx - k * 3.76 * r
        labels.append(_contrast(gray, x, cy + 1.6 * r, 1.5 * r, 0.45 * r) / reference)
        icons.append(_contrast(gray, x, cy, 0.6 * r, 0.6 * r) / reference)

    score = 0.5 * float(np.mean(labels)) + 0.5 * float(np.mean(icons))

    if score >= ANSWERED_MIN_SCORE:
        return "answered", score

    if RINGING_SCORE_RANGE[0] <= score <= RINGING_SCORE_RANGE[1]:
        return "ringing", score

    return "unknown", score


class CallWatcher:
    """
    check() returns one of
        "no_window"  the call window is not on screen
        "ringing"    window found, buttons still greyed out
        "answered"   window found, buttons lit up
        "unknown"    window found but could not be read
    find_window / grab can be replaced (used by the tests).
    """

    def __init__(self, title="Call on PC", find_window=None, grab=None):
        self.title = title
        self.available = True
        self.last_score = 0.0
        self._warned = False

        self._find_window = find_window or self._default_find_window
        self._grab = grab or self._default_grab

        if find_window is None or grab is None:
            try:
                import pygetwindow  # noqa: F401
                import pyautogui    # noqa: F401
            except ImportError:
                self.available = False

    def _default_find_window(self):
        import pygetwindow as gw

        for win in gw.getWindowsWithTitle(self.title):
            if getattr(win, "isMinimized", False):
                continue

            if win.width > 100 and win.height > 100:
                return win

        return None

    def _default_grab(self, win, whole_screen=False):
        import pyautogui

        if whole_screen:
            shot = pyautogui.screenshot()
        else:
            shot = pyautogui.screenshot(
                region=(max(win.left, 0), max(win.top, 0), win.width, win.height)
            )

        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    def check(self):
        try:
            win = self._find_window()

            if win is None:
                return "no_window"

            state, score = classify_call_window(self._grab(win))
            self.last_score = score

            # window rectangle can be off on a scaled display:
            # fall back to looking at the whole screen once
            if state == "unknown":
                try:
                    state, score = classify_call_window(self._grab(win, True))
                    self.last_score = score
                except TypeError:
                    pass

            return state

        except Exception as error:
            if not self._warned:
                print("CallWatcher error:", error)
                self._warned = True

            return "unknown"


# the watcher is only needed for real calls (test mode simulates the answer)
if CALL_ANSWER_MODE == "screen" and not TEST_MODE:
    call_watcher = CallWatcher(CALL_WINDOW_TITLE)

    if not call_watcher.available:
        print("Screen check needs:  pip install pygetwindow")
        call_watcher = None


# ============================================================
# CALL ANSWER DETECTION
# ============================================================

def _wait_seconds(seconds):
    """sleep, but stop early on ENTER or when the session is cancelled"""
    end = time.time() + seconds

    while time.time() < end:
        if call_answered_event.is_set():
            return True
        if not assistant_running:
            return False
        time.sleep(0.1)

    return True


def wait_for_call_window(timeout=CALL_ANSWER_TIMEOUT):
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow not installed - cannot auto-detect call state.")
        print("Run: pip install pygetwindow")
        return False

    start = time.time()

    while time.time() - start < timeout:
        if call_answered_event.is_set():
            return True
        if not assistant_running:
            return False

        for title in gw.getAllTitles():
            lowered = title.lower()

            # heuristic: a connected call shows a running timer like 0:07
            if "phone" in lowered and ":" in lowered and any(ch.isdigit() for ch in lowered):
                return True

        time.sleep(1)

    return False


def _wait_for_answer_on_screen():
    """
    Watches the Phone Link "Call on PC" window. While it rings, the
    Mute / Keypad / Hold call buttons are greyed out; when the other
    person answers they light up. ENTER (in the AISA window) also counts.
    """
    global no_answer_warning

    start = time.time()
    seen = 0
    warned = False
    last = ""

    if TEST_MODE:
        if TEST_AUTO_ANSWER_SECONDS:
            print(f"\nTEST MODE: pretending the call is answered after "
                  f"{TEST_AUTO_ANSWER_SECONDS} s (ENTER answers sooner).")
        else:
            print("\nTEST MODE: press ENTER in the AISA window to answer.")

        chat_log.append(("SYSTEM", "Test mode: waiting for the simulated answer"))

    elif call_watcher is None:
        print("\nScreen check unavailable (pip install pygetwindow): "
              "press ENTER when the responder answers.")
        chat_log.append(("SYSTEM", "Press ENTER when the responder answers"))

    else:
        print("\nWatching the Phone Link call window for the answer...")
        chat_log.append(("SYSTEM", "Waiting for the call to be answered"))

    while assistant_running:

        if call_answered_event.is_set():
            return True

        if TEST_MODE:
            if TEST_AUTO_ANSWER_SECONDS and time.time() - start >= TEST_AUTO_ANSWER_SECONDS:
                return True

        elif call_watcher is not None:
            status = call_watcher.check()

            if status != last:
                print("Call window:", status)
                last = status

            seen = seen + 1 if status == "answered" else 0

            if seen >= ANSWER_CONFIRM_POLLS:
                return True

        if not warned and time.time() - start >= NO_ANSWER_WARNING_SECONDS:
            warned = True
            no_answer_warning = True

            chat_log.append(("SYSTEM", "No answer detected yet - click AISA and press "
                                       "ENTER if it was answered, or X to end"))

        time.sleep(0.4)

    return False


def wait_for_call_answered():
    """True = go ahead and announce.  False = session was cancelled."""

    if CALL_ANSWER_MODE == "screen":
        return _wait_for_answer_on_screen()

    if CALL_ANSWER_MODE == "manual_key":
        print("\nWaiting for the call to be answered.")
        print("Press ENTER in the AISA window when the responder picks up.")

        chat_log.append(("SYSTEM", "Press ENTER when the responder answers"))

        while assistant_running and not call_answered_event.is_set():
            call_answered_event.wait(0.2)

        return call_answered_event.is_set()

    if CALL_ANSWER_MODE == "auto_window":
        if wait_for_call_window():
            return True

        print("Could not auto-detect the call - falling back to a timed wait.")

    return _wait_seconds(CALL_CONNECT_WAIT)


# ============================================================
# GEMINI WITH RETRY + BACKUP MODELS
# ============================================================

GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
GEMINI_TIME_BUDGET = 12


def generate_with_fallback(prompt):
    candidates = []

    for name in [gemini_model] + GEMINI_FALLBACK_MODELS:
        if name and name not in candidates:
            candidates.append(name)

    deadline = time.time() + GEMINI_TIME_BUDGET

    for name in candidates:
        for _ in range(2):
            if time.time() > deadline:
                return None

            try:
                response = client.models.generate_content(model=name, contents=prompt)
                text = response.text

                if text:
                    if name != gemini_model:
                        print("(used backup model:", name + ")")
                    return text.strip()

            except Exception as error:
                message = str(error)
                print(f"GEMINI ERROR ({name}):", message[:140])

                if "404" in message or "NOT_FOUND" in message:
                    break

                time.sleep(1.0)

    return None


# ============================================================
# OFFLINE FALLBACK  (if every Gemini attempt fails)
# ============================================================

OFFLINE_STEPS = [
    "My AI connection is unstable, but I am still here. "
    "Stay with the person and keep them still. Are they awake and breathing?",

    "If there is bleeding, press firmly on the wound with a clean cloth "
    "and keep pressing.",

    "Do not move the person unless there is immediate danger. "
    "Keep them warm and keep talking to them.",

    "Tell the responder on the call exactly where you are and what happened.",

    "Stay calm and stay on the line. Follow the responder's instructions.",
]

offline_step = 0


def reset_offline_state():
    global offline_step
    offline_step = 0


def offline_reply(user_text):
    global offline_step

    text = user_text.lower()

    if "not breathing" in text or "isn't breathing" in text or "no breathing" in text:
        reply = ("Tell the responder on the call right now that the person is "
                 "not breathing, and follow their instructions.")

    elif "bleed" in text or "blood" in text:
        reply = "Press firmly on the wound with a clean cloth and keep pressing. Do not remove the cloth."

    elif "unconscious" in text or "not awake" in text or "not responding" in text:
        reply = ("Tell the responder the person is not responding. "
                 "Do not move them unless there is danger.")

    else:
        reply = OFFLINE_STEPS[min(offline_step, len(OFFLINE_STEPS) - 1)]
        offline_step += 1

    conversation.append("AISA: " + reply)

    return reply


# ============================================================
# GEMINI AI
# ============================================================

def ask_ai(user_text, emergency_type):

    if client is None or gemini_model is None:
        conversation.append("User: " + user_text)
        return offline_reply(user_text)

    conversation.append("User: " + user_text)

    history = "\n".join(conversation[-12:])

    speech_note = ""

    if last_speech_confidence is not None:
        speech_note += f"Speech recognition confidence: {last_speech_confidence:.2f}\n"

    if len(last_speech_alternatives) > 1:
        speech_note += (
            "Other possible transcriptions of the last user message: "
            + " | ".join(last_speech_alternatives[1:4]) + "\n"
        )

    prompt = f"""
You are AISA, an emergency voice assistant.

Emergency type:
{emergency_type}

The emergency call has been initiated by the gesture emergency system.
A short automated message with the emergency type and address has
already been read out to the responder on the call.

Your job now is to help the person communicate and stay calm while
emergency professionals are being contacted.

IMPORTANT RULES:

1. Keep responses short.
2. Responses must sound natural when spoken aloud.
3. Ask only useful questions.
4. Stay calm and professional.
5. Never invent information.
6. Never claim emergency services contacted something that the
   program did not actually do.
7. You are not a doctor, police officer, firefighter, or paramedic.
8. Do not provide dangerous medical instructions.
9. If appropriate, ask whether the person is conscious and breathing.
10. If appropriate, ask about severe bleeding or serious injuries.
11. Encourage the user to follow instructions from emergency professionals.
12. Do not overwhelm the user with information.
13. Ask only ONE question per reply. Never combine two questions.
14. Never repeat a question you already asked in the conversation.
    If the user answered it, even briefly ("yes", "no", "they are"),
    accept the answer and move on to something new.
15. If the user's message seems cut off or unclear, use what you did
    understand instead of repeating your last question.
16. If the user gives a place name or address, repeat it back clearly
    so they can tell the responder on the call.
17. Always move forward: each reply must either give ONE safe, concrete
    next step (for example: stay with the person, apply firm pressure to
    bleeding with a clean cloth, do not move them unless there is danger,
    keep them warm) or ask ONE new useful question.
18. Maximum 40 words per reply.
19. The user's words come from speech recognition and may contain
    mistakes (for example "fastest fish injured" may really be "my
    friend is injured"). Use the alternative transcriptions and the
    emergency context to work out the most likely meaning and respond
    to that. Do not mention the recognition error unless the message
    truly makes no sense; then ask them to repeat it in a few simple words.
20. You cannot send anything to the responder. Never say you "noted",
    "recorded", "told" or "passed on" something. If the user gives a
    location, say it back and remind them to tell the responder themselves.
21. If the user's reply is a yes/no plus one odd, unrelated word, accept
    the yes/no and ignore the odd word. Only treat a word as a place if
    the user clearly said they are at that place.

Conversation so far (the last "User:" line is the message you are
replying to now):

{history}

{speech_note}
Reply now with a concise spoken response that follows the rules above.
"""

    answer = generate_with_fallback(prompt)

    if answer:
        conversation.append("AISA: " + answer)
        return answer

    print("All Gemini attempts failed - using offline help.")

    return offline_reply(user_text)


# ============================================================
# CALL SESSION  (runs in its own thread)
# ============================================================
#
#   dial  ->  wait until answered  ->  tell the responder  ->  help the caller
#
# The responder message and the caller conversation are sequential
# (one mic / one speaker). Doing both at the same time needs two audio
# devices, e.g. a virtual audio cable.
# ============================================================

def _two_tone_siren():
    """a siren made in code, used when the siren file is missing"""
    rate = 44100
    t = np.linspace(0, 0.25, int(rate * 0.25), endpoint=False)

    one = np.sin(2 * np.pi * 900 * t)
    two = np.sin(2 * np.pi * 700 * t)

    wave = (np.concatenate([one, two]) * 0.35 * 32767).astype(np.int16)

    import pygame.sndarray
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))


def start_siren():
    """the laptop speaker is the alarm while the call is ringing"""
    try:
        pygame.mixer.music.load(SIREN_SOUND)
        pygame.mixer.music.play(-1)          # loops until stop_siren()
        return
    except Exception:
        pass

    try:
        _two_tone_siren().play(loops=-1)
    except Exception as error:
        print("Siren error:", error)


def stop_siren():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

    try:
        pygame.mixer.stop()
    except Exception:
        pass


def dial_number(number):
    if TEST_MODE:
        print("\nTEST MODE: real call disabled. Would dial:", number)
        return

    os.system("start ms-phone://")
    time.sleep(3)

    pyautogui.write(number)
    time.sleep(1)
    pyautogui.press("enter")


def run_call_session(emergency_type, number):
    global assistant_running, call_in_progress, current_call_label, call_phase
    global no_answer_warning

    assistant_running = True
    call_phase = "dialing"
    no_answer_warning = False

    print("\n" + "=" * 60)
    print("AISA EMERGENCY VOICE ASSISTANT")
    print("=" * 60)

    conversation.clear()
    reset_offline_state()

    try:
        chat_log.append(("SYSTEM", f"Dialing {emergency_type}..."))

        if SIREN_ENABLED:
            start_siren()

        dial_number(number)

        if not wait_for_call_answered():
            chat_log.append(("SYSTEM", "Call session cancelled"))
            return

        call_phase = "announcing"
        no_answer_warning = False
        stop_siren()                      # silence before the address is read out
        chat_log.append(("SYSTEM", "Call answered"))

        responder_message = build_responder_message(emergency_type)

        if TEST_MODE:
            print("\nTEST MODE: would announce to responder:")
            print(responder_message)

        speak(responder_message, speaker="CALL")

        call_phase = "assisting"

        speak(
            f"{emergency_type} emergency activated. "
            "I am AISA, your emergency voice assistant. "
            "Please tell me what happened."
        )

        listen_failures = 0

        while assistant_running:

            user_text = listen()

            if not user_text:
                listen_failures += 1

                print("Mic energy threshold:", int(recognizer.energy_threshold))

                # say something instead of looping silently
                if listen_failures % 2 == 0:
                    recognizer.energy_threshold = max(100, recognizer.energy_threshold * 0.7)

                    speak(
                        "I could not hear you clearly. Please speak a "
                        "little louder, in a few simple words."
                    )

                continue

            listen_failures = 0

            command = user_text.lower().strip()

            if command in ("exit", "stop", "stop assistant", "quit", "end conversation"):
                speak("AISA assistant shutting down.")
                assistant_running = False
                break

            try:
                speak(ask_ai(user_text, emergency_type))

            except Exception as error:
                print("AISA ERROR:", error)
                speak("I am having trouble processing that right now.")

    finally:
        assistant_running = False
        call_in_progress = False
        current_call_label = ""
        call_phase = "idle"
        no_answer_warning = False

        stop_siren()

        chat_log.append(("SYSTEM", "Session ended - gestures re-enabled"))

        print("\nCall session ended - gestures re-enabled.")


def start_call_thread(emergency_type, number):
    global assistant_thread

    assistant_thread = threading.Thread(
        target=run_call_session,
        args=(emergency_type, number),
        daemon=True,
    )

    assistant_thread.start()


# ============================================================
# EMERGENCY CALL  (called from the main loop - must be quick)
# ============================================================

def call_emergency(fingers):
    global last_call_time, recording, record_start_time, out
    global call_in_progress, current_call_label

    now = time.time()

    if fingers not in emergency_numbers:
        return "Invalid gesture"

    if now - last_call_time < call_cooldown:
        return "Emergency cooldown active"

    number = emergency_numbers[fingers]
    label = emergency_labels[fingers]

    # lock out new gestures until this session ends
    call_in_progress = True
    current_call_label = label
    call_answered_event.clear()

    print("\n" + "=" * 50)
    print("EMERGENCY:", label)
    print("NUMBER:", number)
    print("=" * 50)

    last_call_time = now

    # start recording the camera
    if out is not None:
        out.release()

    out = cv2.VideoWriter(
        "call_recording.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (frame_width, frame_height),
    )

    recording = True
    record_start_time = time.time()

    # dialing + AISA happen in the background thread
    # so the camera and the screen never freeze
    start_call_thread(label, number)

    return f"CALL TRIGGERED: {label}"


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 60)
print("AISA HAND GESTURE EMERGENCY")
print("MEDIAPIPE + REINFORCEMENT LEARNING + GEMINI")
print("=" * 60)
print("\nTEST MODE:", TEST_MODE)
print("HOLD TIME:", HOLD_TIME, "seconds")
print("Q-table:", Q_TABLE_FILE)
print("Dataset:", dataset.path)
print("Gemini:", gemini_model if gemini_model else "NOT AVAILABLE")
print("ML:", ml_status)
print("\nShow 1-5 fingers. ESC to exit.")
print("Teach mode (building the dataset): press 1-5 / 0 to record a label, "
      "same key again to stop. Calls are disabled while teaching.\n")

calibrate_microphone()


def on_mouse(event, x, y, flags, param):
    """hold the on-screen panic button for PANIC_HOLD_SECONDS"""
    global panic_start, panic_mode

    inside = (PANIC_X <= x <= PANIC_X + PANIC_W) and (PANIC_Y <= y <= PANIC_Y + PANIC_H)

    if event == cv2.EVENT_LBUTTONDOWN and inside:
        if not call_in_progress and teach_label is None:
            panic_start = time.time()
            panic_mode = "mouse"

    elif event == cv2.EVENT_LBUTTONUP:
        if panic_mode == "mouse":
            panic_start = None
            panic_mode = None


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

is_fullscreen = FULLSCREEN_ON_START

if is_fullscreen:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)

frame_index = 0
last_frame_time = time.time()
fps = 0.0

smoother = FingerSmoother()
lost_time = 0.0          # how long the hand has been missing
debug_hud = False        # toggled with the D key
ml_label = ""            # what the ML model thinks the user wants
ml_prob = 0.0
teach_label = None       # None, or the label being taught (keys 0-5)


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Camera frame error.")
        break

    # real time between frames (so HOLD_TIME is real seconds)
    now = time.time()
    dt = min(now - last_frame_time, 0.1)
    last_frame_time = now

    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

    frame_index += 1

    frame = cv2.flip(frame, 1)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    hand_detected = False
    fingers = 0
    raw_fingers = 0
    finger_info = []
    states = [False] * 5
    confidence = 0.0
    landmarks = None
    handedness = ""
    features = None

    if results.multi_hand_landmarks:
        hand_detected = True
        landmarks = results.multi_hand_landmarks[0]

        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        states, finger_info = finger_states_debug(
            landmarks, frame.shape[1], frame.shape[0]
        )
        raw_fingers = min(sum(states), 5)
        fingers = smoother.update(raw_fingers)
        confidence = calculate_hand_confidence(landmarks)
        features = landmark_features(landmarks)

        draw_hand(frame, landmarks)
        draw_finger_points(frame, landmarks, states)

    # --------------------------------------------------------
    # GESTURE STABILITY (real seconds)
    # --------------------------------------------------------

    lost_time = 0.0 if hand_detected else lost_time + dt

    if hand_detected:
        if fingers == previous_fingers:
            gesture_stable_time += dt
        else:
            gesture_stable_time = 0.0
            previous_fingers = fingers

    elif lost_time > HAND_LOST_GRACE:
        # hand really gone (not just one missed frame)
        gesture_stable_time = 0.0
        previous_fingers = 0
        smoother.clear()

    stable = gesture_stable_time >= 0.4

    # --------------------------------------------------------
    # ML: what does the user want?  (advisory only)
    # --------------------------------------------------------

    if (
        ml_predictor is not None
        and features is not None
        and frame_index % ML_PREDICT_EVERY == 0
    ):
        try:
            ml_label, ml_prob = ml_predictor.predict(features, handedness)
        except Exception as error:
            print("ML predict error:", error)
            ml_label, ml_prob = "", 0.0

    if not hand_detected and lost_time > HAND_LOST_GRACE:
        ml_label, ml_prob = "", 0.0

    rule_label = emergency_labels.get(fingers, "none") if hand_detected else ""

    if ml_label and ml_prob >= ML_MIN_PROBABILITY:
        ml_text = f"{ml_label}  {ml_prob:.0%}"
        ml_agree = (ml_label == rule_label) if rule_label else None
    elif ml_label:
        ml_text, ml_agree = "unsure", None
    else:
        ml_text, ml_agree = "-", None

    # --------------------------------------------------------
    # REINFORCEMENT LEARNING
    # --------------------------------------------------------

    state = rl_agent.get_state(hand_detected, fingers, confidence, stable, gesture_stable_time)
    action = rl_agent.choose_action(state)

    valid_gesture = (
        hand_detected
        and fingers in emergency_numbers
        and confidence >= 0.60
        and stable
        and gesture_stable_time >= HOLD_TIME
    )

    reward = calculate_reward(hand_detected, fingers, confidence, stable, gesture_stable_time, action)

    next_state = rl_agent.get_state(hand_detected, fingers, confidence, stable, gesture_stable_time)
    rl_agent.update(state, action, reward, next_state)

    # --------------------------------------------------------
    # DATASET: one row per N frames that contain a hand
    # --------------------------------------------------------

    if hand_detected and frame_index % SAMPLE_EVERY_N_FRAMES == 0:

        # what the user wants: the service they hold up, once stable
        label = emergency_labels[fingers] if (stable and fingers in emergency_labels) else "none"

        dataset.log(
            "frame", frame_index, handedness, states, fingers,
            confidence, stable, gesture_stable_time, label, valid_gesture,
            action, reward, features,
            true_label=teach_label or "", ml_label=ml_label, ml_prob=ml_prob,
        )

    # --------------------------------------------------------
    # STATUS TEXT
    # --------------------------------------------------------

    if not hand_detected:
        status, status_color = "No hand detected", C["muted"]

    elif fingers == 0:
        status, status_color = "Fist detected", C["text2"]

    elif not stable:
        status, status_color = f"Detecting {fingers} finger{'s' if fingers != 1 else ''}...", C["cyan"]

    elif gesture_stable_time < HOLD_TIME:
        remaining = max(0.0, HOLD_TIME - gesture_stable_time)
        status = f"{emergency_labels.get(fingers, 'Gesture')}  -  hold {remaining:.1f}s"
        status_color = SERVICE_COLORS.get(fingers, C["cyan"])

    else:
        status, status_color = f"Valid {fingers}-finger gesture", C["green"]

    if call_in_progress:
        status = f"Call in progress  -  {current_call_label}  (gestures paused)"
        status_color = C["amber"]

    elif teach_label is not None:
        status = f"TEACH MODE  -  recording '{teach_label}'  (calls disabled)"
        status_color = C["amber"]

    # --------------------------------------------------------
    # SAFETY TRIGGER (RL does not decide whether a valid gesture may call)
    # --------------------------------------------------------

    if (
        valid_gesture
        and not confirming
        and not call_in_progress
        and teach_label is None
        and time.time() - last_call_time >= call_cooldown
    ):
        confirming = True
        confirmed_fingers = fingers

        print("\nValid emergency gesture:", emergency_labels[fingers])

    if confirming:

        label = emergency_labels[confirmed_fingers]

        status = f"Confirm {label}  -  keep {confirmed_fingers} finger{'s' if confirmed_fingers != 1 else ''} up"
        status_color = C["green"]

        if lost_time > HAND_LOST_GRACE or (hand_detected and fingers != confirmed_fingers):
            confirming = False
            confirmed_fingers = 0
            gesture_stable_time = 0.0

        elif gesture_stable_time >= HOLD_TIME:

            status = call_emergency(confirmed_fingers)
            status_color = C["amber"]

            dataset.log(
                "call_trigger", frame_index, handedness, states, fingers,
                confidence, True, gesture_stable_time, label, True,
                ACTION_TRIGGER, 10,
                features if features is not None else [0.0] * 63,
                ml_label=ml_label, ml_prob=ml_prob,
            )

            confirming = False
            confirmed_fingers = 0
            gesture_stable_time = 0.0

    # --------------------------------------------------------
    # PANIC BUTTON: hold 3 s (mouse) or B countdown = General Helpline
    # --------------------------------------------------------

    panic_frac = None

    if panic_start is not None:
        if call_in_progress or teach_label is not None:
            panic_start = None
            panic_mode = None
        else:
            held = time.time() - panic_start
            panic_frac = min(held / PANIC_HOLD_SECONDS, 1.0)

            if held >= PANIC_HOLD_SECONDS:
                button_requested = True
                panic_start = None
                panic_frac = None

                if panic_mode == "mouse":
                    panic_mode = "mouse-done"      # ignore until the mouse is released
                else:
                    panic_mode = None

    if button_requested:
        button_requested = False

        if (
            not call_in_progress
            and teach_label is None
            and time.time() - last_call_time >= call_cooldown
        ):
            status = call_emergency(BUTTON_SERVICE)
            status_color = C["amber"]
            confirming = False
            confirmed_fingers = 0
            gesture_stable_time = 0.0
        else:
            print("Panic button ignored (call in progress, teach mode or cooldown).")

    # --------------------------------------------------------
    # RECORDING (raw camera + hand skeleton)
    # --------------------------------------------------------

    rec_elapsed = 0.0

    if recording and out is not None:
        out.write(frame)

        rec_elapsed = time.time() - record_start_time

        if rec_elapsed >= 600:
            recording = False
            out.release()
            out = None
            print("Recording saved.")

    # --------------------------------------------------------
    # DRAW THE INTERFACE
    # --------------------------------------------------------

    canvas = ui.render(frame, {
        "hand_detected": hand_detected,
        "fingers": fingers,
        "confidence": confidence,
        "stable": stable,
        "stable_time": gesture_stable_time,
        "hold_target": HOLD_TIME,
        "status": status,
        "status_color": status_color,
        "rl_action": actions[action],
        "call_phase": call_phase,
        "call_label": current_call_label,
        "chat": list(chat_log),
        "labels": emergency_labels,
        "gemini_model": gemini_model,
        "answer_mode": CALL_ANSWER_MODE,
        "dataset_rows": dataset.total_rows,
        "ml_status": ml_status,
        "ml_pred": ml_text,
        "ml_agree": ml_agree,
        "teach": teach_label,
        "test_mode": TEST_MODE,
        "recording": recording,
        "rec_elapsed": rec_elapsed,
        "fps": fps,
        "call_warning": no_answer_warning,
        "panic_frac": panic_frac,
        "panic_mode": panic_mode,
        "panic_blocked": ("call in progress" if call_in_progress
                          else ("teach mode" if teach_label is not None else "")),
        "debug": (
            (finger_info + [f"raw count {raw_fingers}  ->  stable count {fingers}"])
            if (debug_hud and hand_detected) else None
        ),
    })

    cv2.imshow(WINDOW_NAME, canvas)

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:                                   # ESC
        assistant_running = False
        break

    elif key in (13, 10):                           # ENTER = call answered
        if call_phase == "dialing":
            call_answered_event.set()

    elif key in (ord("x"), ord("X")):               # end session
        if call_in_progress:
            assistant_running = False

    elif ord("0") <= key <= ord("5"):               # teach mode (data collection)
        if not call_in_progress:
            chosen = "none" if key == ord("0") else emergency_labels[key - ord("0")]
            teach_label = None if teach_label == chosen else chosen
            print("TEACH MODE:", teach_label if teach_label else "off")

            # the hold must be earned again after teaching, otherwise
            # switching teach mode off with the hand still up would
            # start a real call at once
            gesture_stable_time = 0.0
            confirming = False
            confirmed_fingers = 0

    elif key in (ord("b"), ord("B")):               # panic countdown
        if panic_mode == "key":
            panic_start = None
            panic_mode = None
            print("Panic countdown cancelled.")
        elif not call_in_progress and teach_label is None and panic_start is None:
            panic_start = time.time()
            panic_mode = "key"
            print(f"Panic countdown: calling in {PANIC_HOLD_SECONDS:.0f} s (press B to cancel).")

    elif key in (ord("d"), ord("D")):               # debug overlay
        debug_hud = not debug_hud

    elif key in (ord("f"), ord("F")):               # fullscreen
        is_fullscreen = not is_fullscreen

        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)


# ============================================================
# CLEANUP
# ============================================================

assistant_running = False

print("\nSaving RL Q-table...")
rl_agent.save()

dataset.close()

if out is not None:
    out.release()

cap.release()
hands.close()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("\nAISA stopped.")
print("Q-table saved to:", Q_TABLE_FILE)
print("Dataset saved to:", dataset.path, f"({dataset.total_rows} rows)")