"""
AISA - Hand Gesture Emergency Assistant  (FINAL)
=================================================

Gesture -> emergency call -> AISA voice assistant (Gemini)

Keys (click the AISA window first):
    ESC     quit
    ENTER   confirm the call was answered (manual_key mode)
    X       end the current call session
    F       toggle fullscreen

Dataset (for the ML step later)
    Every frame that contains a hand is converted into DATA:
    21 hand landmarks -> 63 numbers, normalised so they do not
    depend on where the hand is or how far it is from the camera.
    Saved to  aisa_gesture_dataset.csv  together with the label
    (what the user wanted) so a model can be trained on it later.
    The ML code is at the bottom of the file, commented out.
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

FULLSCREEN_ON_START = True

SIREN_SOUND = "C:\\Users\\TARMAL COMPUTERS\\siren.wav.mp3"

# Read aloud to the responder once the call is answered.
# EDIT THIS before using outside TEST_MODE.
EMERGENCY_ADDRESS = "REPLACE_WITH_YOUR_ACTUAL_ADDRESS, Pune, Maharashtra"

# --- how AISA decides the call was answered -----------------
#   "manual_key"  - press ENTER in the AISA window when the
#                   responder picks up (most reliable)
#   "timer"       - wait CALL_CONNECT_WAIT seconds
#   "auto_window" - best-effort window-title check
#                   (pip install pygetwindow), falls back to timer
CALL_ANSWER_MODE = "manual_key"
CALL_CONNECT_WAIT = 4
CALL_ANSWER_TIMEOUT = 30

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

# --- ML (the code is commented out at the bottom of the file) --
ML_ENABLED = False


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

# ML status shown in the HUD (see ML block at the bottom)
ml_status = "OFF"
ml_prediction = ""


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

def finger_states(landmarks):
    """[thumb, index, middle, ring, pinky] -> True when extended."""
    lm = landmarks.landmark

    states = [lm[4].x < lm[3].x]

    for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18)):
        states.append(lm[tip].y < lm[joint].y)

    return states


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
    "label", "valid_gesture",
    "rl_action", "rl_reward",
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

        exists = os.path.exists(self.path)

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
            rl_action, rl_reward, features):

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
            int(valid_gesture),
            actions[rl_action],
            rl_reward,
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
                ("X", "End session"), ("F", "Fullscreen")]

        for i, (k, d) in enumerate(keys):
            kx = LEFT_X + 18 + (i % 2) * 130
            ky = STATUS_Y + 44 + (i // 2) * 34

            kw = measure(k, 11, True) + 14
            fill_rrect(cv, kx, ky - 11, kw, 22, 6, C["card"], 1.0)
            rr_outline(cv, kx, ky - 11, kw, 22, 6, C["border"], 1)

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
            x, y, w, h = LEFT_X, CAM_Y + i * 98, PANEL_W, 88

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
            cx, cy = x + 48, y + 44

            if hot:
                cv2.circle(cv, (cx, cy), 25, bgr(color), -1, cv2.LINE_AA)
                digit_col = C["dark"]
            else:
                cv2.circle(cv, (cx, cy), 25, bgr(mix(color, C["card"], 0.78)), -1, cv2.LINE_AA)
                cv2.circle(cv, (cx, cy), 25, bgr(mix(color, C["card"], 0.40)), 1, cv2.LINE_AA)
                digit_col = color

            self.text(str(n), cx, cy - 1, 25, digit_col, True, "mm")

            name_col = C["muted"] if dim else C["text"]
            self.text(label, x + 88, y + 32, 18, name_col, True)

            if calling:
                sub = {"dialing": "Dialing...",
                       "announcing": "Connected",
                       "assisting": "On call"}.get(phase, "")
                self.text(sub, x + 88, y + 56, 12, color, True)
                cv2.circle(cv, (x + w - 20, y + 18), int(4 + pulse * 3), bgr(color), -1, cv2.LINE_AA)

            elif active:
                remaining = max(0.0, s["hold_target"] - s["stable_time"])
                self.text(f"Hold... {remaining:.1f}s", x + 88, y + 54, 12, color, True)
                self.bar(x + 88, y + 66, 170, 7, s["stable_time"] / s["hold_target"], color)

            else:
                hint = f"Show {n} finger" + ("s" if n > 1 else "")
                self.text(hint, x + 88, y + 56, 12, C["muted"] if dim else C["text2"])

    def _camera(self, cam, s, pulse):
        cv = self.canvas

        if cam.shape[1] != CAM_W or cam.shape[0] != CAM_H:
            cam = cv2.resize(cam, (CAM_W, CAM_H))

        paste_rounded(cv, cam, CAM_X, CAM_Y, 18)

        phase = s["call_phase"]
        fingers = s["fingers"]

        if phase == "dialing":
            ring = C["amber"]
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

        # call banner
        label = s["call_label"]
        banner = None

        if phase == "dialing":
            if CALL_ANSWER_MODE == "manual_key":
                banner = (f"Calling {label}  -  press ENTER when answered", C["amber"])
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

        tiles = [
            ("FINGERS", str(s["fingers"])),
            ("CONFIDENCE", f"{s['confidence']:.2f}"),
            ("HOLD", f"{s['stable_time']:.1f}s / {s['hold_target']:.1f}s"),
            ("RL ACTION", s["rl_action"]),
        ]

        for i, (label, value) in enumerate(tiles):
            tx = x + i * 150
            self.text(label, tx, y + 72, 10, C["muted"], True)
            self.text(value, tx, y + 92, 16, C["text"], True)

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
            ("ML MODEL", s["ml_text"], C["muted"] if s["ml_text"] == "OFF" else C["green"]),
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


def wait_for_call_answered():
    """True = go ahead and announce.  False = session was cancelled."""

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

def dial_number(number):
    if TEST_MODE:
        print("\nTEST MODE: real call disabled. Would dial:", number)
        return

    try:
        pygame.mixer.music.load(SIREN_SOUND)
        pygame.mixer.music.play()
    except Exception as error:
        print("Siren error:", error)

    os.system("start ms-phone://")
    time.sleep(3)

    pyautogui.write(number)
    time.sleep(1)
    pyautogui.press("enter")


def run_call_session(emergency_type, number):
    global assistant_running, call_in_progress, current_call_label, call_phase

    assistant_running = True
    call_phase = "dialing"

    print("\n" + "=" * 60)
    print("AISA EMERGENCY VOICE ASSISTANT")
    print("=" * 60)

    conversation.clear()
    reset_offline_state()

    try:
        chat_log.append(("SYSTEM", f"Dialing {emergency_type}..."))

        dial_number(number)

        if not wait_for_call_answered():
            chat_log.append(("SYSTEM", "Call session cancelled"))
            return

        call_phase = "announcing"
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
# ML  -  GESTURE INTENT MODEL        (DISABLED FOR NOW)
# ============================================================
#
# Plan: use the dataset above (63 landmark numbers -> label) to teach
# a model what the user wants, then use it next to the rule-based
# finger counter.
#
# To enable:
#   1. collect data by using the app normally (several minutes per
#      gesture, different distances / angles / lighting / people)
#   2. uncomment this block
#   3. run  train_gesture_model()  once  (python -c, or temporarily
#      call it before the main loop)
#   4. set ML_ENABLED = True at the top
#   5. uncomment the two "ML:" lines inside the main loop
#
# ------------------------------------------------------------
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# ML_MODEL_FILE = "aisa_gesture_model.joblib"
# ML_MIN_PROBABILITY = 0.80
#
#
# def train_gesture_model():
#     df = pd.read_csv(DATASET_FILE)
#
#     # only the per-frame rows, not the call-trigger events
#     df = df[df["event"] == "frame"]
#
#     feature_cols = DATASET_LM_COLUMNS
#     X = df[feature_cols].values
#     y = df["label"].values            # "Police", "Fire", ..., "none"
#
#     print("Rows:", len(df))
#     print(df["label"].value_counts())
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     model = RandomForestClassifier(
#         n_estimators=200,
#         random_state=42,
#         class_weight="balanced",
#     )
#
#     model.fit(X_train, y_train)
#
#     print(classification_report(y_test, model.predict(X_test)))
#
#     joblib.dump(model, ML_MODEL_FILE)
#     print("Model saved:", ML_MODEL_FILE)
#
#
# class GesturePredictor:
#
#     def __init__(self, path=ML_MODEL_FILE):
#         self.model = None
#
#         if os.path.exists(path):
#             self.model = joblib.load(path)
#             print("ML model loaded:", path)
#         else:
#             print("No ML model found - run train_gesture_model() first.")
#
#     def predict(self, features):
#         """returns (label, probability)"""
#         if self.model is None:
#             return "", 0.0
#
#         probs = self.model.predict_proba([features])[0]
#         best = int(np.argmax(probs))
#
#         return self.model.classes_[best], float(probs[best])
#
#
# ml_predictor = GesturePredictor() if ML_ENABLED else None
#
# if ML_ENABLED and ml_predictor and ml_predictor.model is not None:
#     ml_status = "READY"
# ------------------------------------------------------------


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
print("\nShow 1-5 fingers. ESC to exit.\n")

calibrate_microphone()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

is_fullscreen = FULLSCREEN_ON_START

if is_fullscreen:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow(WINDOW_NAME, UI_W, UI_H)

frame_index = 0
last_frame_time = time.time()
fps = 0.0


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

        states = finger_states(landmarks)
        fingers = min(sum(states), 5)
        confidence = calculate_hand_confidence(landmarks)
        features = landmark_features(landmarks)

        draw_hand(frame, landmarks)
        draw_finger_points(frame, landmarks, states)

    # --------------------------------------------------------
    # GESTURE STABILITY (real seconds)
    # --------------------------------------------------------

    if hand_detected:
        if fingers == previous_fingers:
            gesture_stable_time += dt
        else:
            gesture_stable_time = 0.0
            previous_fingers = fingers
    else:
        gesture_stable_time = 0.0
        previous_fingers = 0

    stable = gesture_stable_time >= 0.4

    # --------------------------------------------------------
    # ML (disabled)
    # --------------------------------------------------------
    # ML: if ML_ENABLED and features is not None:
    # ML:     ml_label, ml_prob = ml_predictor.predict(features)
    # ML:     ml_prediction = f"{ml_label} {ml_prob:.0%}"

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

    # --------------------------------------------------------
    # SAFETY TRIGGER (RL does not decide whether a valid gesture may call)
    # --------------------------------------------------------

    if (
        valid_gesture
        and not confirming
        and not call_in_progress
        and time.time() - last_call_time >= call_cooldown
    ):
        confirming = True
        confirmed_fingers = fingers

        print("\nValid emergency gesture:", emergency_labels[fingers])

    if confirming:

        label = emergency_labels[confirmed_fingers]

        status = f"Confirm {label}  -  keep {confirmed_fingers} finger{'s' if confirmed_fingers != 1 else ''} up"
        status_color = C["green"]

        if not hand_detected or fingers != confirmed_fingers:
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
            )

            confirming = False
            confirmed_fingers = 0
            gesture_stable_time = 0.0

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
        "ml_text": ml_prediction or ml_status,
        "test_mode": TEST_MODE,
        "recording": recording,
        "rec_elapsed": rec_elapsed,
        "fps": fps,
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