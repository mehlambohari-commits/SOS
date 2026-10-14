import cv2
import time
import os
import pygame
import numpy as np
import pyautogui
import mediapipe as mp
import random
import csv
import threading
import speech_recognition as sr
import pyttsx3
import pythoncom

from dotenv import load_dotenv
from google import genai

import logging

# silences the harmless "automatic function calling" notice
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


# ============================================================
# SETTINGS
# ============================================================

# True  = no real phone call
# False = actually opens Windows Phone and calls
TEST_MODE = False

# How long the same gesture must be held
HOLD_TIME = 5.0

# Camera
CAMERA_INDEX = 0

WINDOW_NAME = "AISA - Hand Gesture Emergency App"

SIREN_SOUND = (
    "C:\\Users\\TARMAL COMPUTERS\\siren.wav.mp3"
)

# RL files
Q_TABLE_FILE = "aisa_q_table.npy"

DATASET_FILE = "aisa_rl_dataset.csv"

# ------------------------------------------------------------
# FIXED / PRESET ADDRESS
# ------------------------------------------------------------
# This is read aloud to the responder as part of the automated
# emergency message. Edit this before using outside TEST_MODE.
# ------------------------------------------------------------

EMERGENCY_ADDRESS = (
    "REPLACE_WITH_YOUR_ACTUAL_ADDRESS, Pune, Maharashtra"
)

# How many seconds to wait after dialing before AISA speaks the
# responder message (gives the call time to connect in real mode).
CALL_CONNECT_WAIT = 4

# ------------------------------------------------------------
# HOW AISA DECIDES THE CALL WAS ANSWERED
# ------------------------------------------------------------
# There is no public API to read call state from Windows'
# Phone Link app, so "auto" detection below is best-effort
# (window-title heuristic) and may not work on every version.
#
#   "timer"       - just wait CALL_CONNECT_WAIT seconds, then
#                    speak (simplest, least reliable, no
#                    dependency on the phone app's UI)
#
#   "manual_key"  - AISA waits for you to press ENTER in the
#                    console the moment the responder picks up
#                    (most reliable, needs a human watching)
#
#   "auto_window" - best-effort: polls window titles for a
#                    running call-duration timer (e.g. "0:07")
#                    requires: pip install pygetwindow
#                    falls back to "timer" if it can't detect
#                    anything within the timeout
# ------------------------------------------------------------

CALL_ANSWER_MODE = "manual_key"

CALL_ANSWER_TIMEOUT = 30


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    print(
        "WARNING: GEMINI_API_KEY is not set."
    )

    client = None

else:

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# AUTOMATIC GEMINI MODEL SELECTION
# ============================================================

gemini_model = None


def find_gemini_model():

    global gemini_model

    if client is None:

        return None

    try:

        print()
        print(
            "Searching for available Gemini models..."
        )

        available_models = []

        for model in client.models.list():

            name = getattr(
                model,
                "name",
                ""
            )

            supported_actions = getattr(
                model,
                "supported_actions",
                []
            )

            if (
                "generateContent"
                in supported_actions
            ):

                clean_name = name

                if clean_name.startswith(
                    "models/"
                ):

                    clean_name = clean_name[
                        len("models/"):

                    ]

                available_models.append(
                    clean_name
                )

        if not available_models:

            print(
                "No generateContent Gemini model found."
            )

            return None

        print(
            "Available Gemini models:"
        )

        for model_name in available_models:

            print(
                "  -",
                model_name
            )

        # ----------------------------------------------------
        # Prefer Flash models
        # ----------------------------------------------------

        preferred = [

            "gemini-3.6-flash",

            "gemini-3.5-flash",

            "gemini-2.5-flash",

            "gemini-2.0-flash"

        ]

        for preferred_model in preferred:

            if preferred_model in available_models:

                gemini_model = preferred_model

                break

        # ----------------------------------------------------
        # Otherwise select any flash model
        # ----------------------------------------------------

        if gemini_model is None:

            for model_name in available_models:

                if (
                    "flash"
                    in model_name.lower()
                ):

                    gemini_model = model_name

                    break

        # ----------------------------------------------------
        # Last fallback
        # ----------------------------------------------------

        if gemini_model is None:

            gemini_model = available_models[0]

        print()
        print(
            "AISA Gemini model:",
            gemini_model
        )

        return gemini_model

    except Exception as error:

        print(
            "Could not list Gemini models:"
        )

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

    5: "6232481125"

}


emergency_labels = {

    1: "Police",

    2: "Fire",

    3: "Ambulance",

    4: "Women Helpline",

    5: "General Helpline"

}


# ============================================================
# PYGAME
# ============================================================

pygame.mixer.init()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = (
    mp.solutions.drawing_styles
)


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    model_complexity=0,

    min_detection_confidence=0.60,

    min_tracking_confidence=0.60

)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# FULLSCREEN
# ============================================================

cv2.namedWindow(

    WINDOW_NAME,

    cv2.WND_PROP_FULLSCREEN

)

cv2.setWindowProperty(

    WINDOW_NAME,

    cv2.WND_PROP_FULLSCREEN,

    cv2.WINDOW_FULLSCREEN

)


frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

last_call_time = 0

call_cooldown = 3

recording = False

record_start_time = 0

out = None

previous_fingers = 0

gesture_stable_time = 0

confirming = False

confirmed_fingers = 0

assistant_running = False

assistant_thread = None

conversation = []

# True from the moment a call is triggered until the AISA
# session for that call fully ends. While True, the main loop
# ignores new gestures so a second call can't be dialed on top
# of an active one.
call_in_progress = False

current_call_label = ""


# ============================================================
# RL ACTIONS
# ============================================================

ACTION_IGNORE = 0

ACTION_WAIT = 1

ACTION_CONFIRM = 2

ACTION_TRIGGER = 3


actions = [

    "IGNORE",

    "WAIT",

    "CONFIRM",

    "TRIGGER"

]


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


    # --------------------------------------------------------
    # GET STATE
    # --------------------------------------------------------

    def get_state(

        self,

        hand_detected,

        fingers,

        confidence,

        stable,

        hold_time

    ):

        confidence_level = int(

            min(

                confidence * 10,

                10

            )

        )

        hold_level = min(

            int(hold_time),

            5

        )

        state = (

            int(hand_detected),

            int(fingers),

            confidence_level,

            int(stable),

            hold_level

        )

        return state


    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def initialize_state(
        self,
        state
    ):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(

                self.num_actions,

                dtype=np.float32

            )

            # Safe initial values

            self.q_table[state][
                ACTION_TRIGGER
            ] = -5.0

            self.q_table[state][
                ACTION_WAIT
            ] = 1.0


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    def choose_action(
        self,
        state
    ):

        self.initialize_state(
            state
        )

        # Exploration

        if random.random() < self.epsilon:

            return random.choice([

                ACTION_IGNORE,

                ACTION_WAIT,

                ACTION_CONFIRM

            ])

        return int(

            np.argmax(
                self.q_table[state]
            )

        )


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(

        self,

        state,

        action,

        reward,

        next_state

    ):

        self.initialize_state(
            state
        )

        self.initialize_state(
            next_state
        )

        current_q = (
            self.q_table[state][action]
        )

        best_next_q = np.max(

            self.q_table[next_state]

        )

        new_q = (

            current_q

            + self.alpha * (

                reward

                + self.gamma
                * best_next_q

                - current_q

            )

        )

        self.q_table[state][action] = (
            new_q
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        filename=Q_TABLE_FILE
    ):

        np.save(

            filename,

            self.q_table,

            allow_pickle=True

        )

        print(
            "RL Q-table saved:",
            filename
        )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(
        self,
        filename=Q_TABLE_FILE
    ):

        if not os.path.exists(
            filename
        ):

            print(
                "No previous Q-table found."
            )

            return

        try:

            self.q_table = np.load(

                filename,

                allow_pickle=True

            ).item()

            print(
                "RL Q-table loaded:",
                filename
            )

        except Exception as error:

            print(
                "Could not load Q-table:",
                error
            )


rl_agent = QLearningAgent()

rl_agent.load()


# ============================================================
# DATASET
# ============================================================

dataset_columns = [

    "timestamp",

    "hand_detected",

    "finger_count",

    "confidence",

    "stable",

    "hold_time",

    "action",

    "action_name",

    "reward",

    "next_finger_count",

    "next_confidence",

    "next_stable",

    "next_hold_time",

    "valid_gesture",

    "emergency_type",

    "called",

    "test_mode"

]


def initialize_dataset():

    if not os.path.exists(
        DATASET_FILE
    ):

        with open(

            DATASET_FILE,

            "w",

            newline=""

        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                dataset_columns
            )


initialize_dataset()


def save_dataset_row(

    hand_detected,

    fingers,

    confidence,

    stable,

    hold_time,

    action,

    reward,

    next_fingers,

    next_confidence,

    next_stable,

    next_hold_time,

    valid_gesture,

    emergency_type="",

    called=False

):

    with open(

        DATASET_FILE,

        "a",

        newline=""

    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([

            time.time(),

            int(hand_detected),

            fingers,

            round(
                confidence,
                3
            ),

            int(stable),

            round(
                hold_time,
                3
            ),

            action,

            actions[action],

            reward,

            next_fingers,

            round(
                next_confidence,
                3
            ),

            int(next_stable),

            round(
                next_hold_time,
                3
            ),

            int(valid_gesture),

            emergency_type,

            int(called),

            int(TEST_MODE)

        ])


# ============================================================
# TEXT
# ============================================================

def draw_text(

    frame,

    text,

    position,

    scale=0.8,

    color=(255, 255, 255)

):

    x, y = position

    cv2.putText(

        frame,

        text,

        (x + 2, y + 2),

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        (0, 0, 0),

        4,

        cv2.LINE_AA

    )

    cv2.putText(

        frame,

        text,

        (x, y),

        cv2.FONT_HERSHEY_SIMPLEX,

        scale,

        color,

        2,

        cv2.LINE_AA

    )


# ============================================================
# FINGER COUNT
# ============================================================

def count_fingers(
    landmarks
):

    lm = landmarks.landmark

    fingers = 0

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    if lm[4].x < lm[3].x:

        fingers += 1


    # --------------------------------------------------------
    # INDEX / MIDDLE / RING / PINKY
    # --------------------------------------------------------

    finger_pairs = [

        (8, 6),

        (12, 10),

        (16, 14),

        (20, 18)

    ]


    for tip, joint in finger_pairs:

        if lm[tip].y < lm[joint].y:

            fingers += 1


    return min(
        fingers,
        5
    )


# ============================================================
# HAND CONFIDENCE
# ============================================================

def calculate_hand_confidence(
    landmarks
):

    xs = [

        point.x

        for point in landmarks.landmark

    ]

    ys = [

        point.y

        for point in landmarks.landmark

    ]

    width = (
        max(xs) - min(xs)
    )

    height = (
        max(ys) - min(ys)
    )


    if (
        width < 0.05
        or height < 0.05
    ):

        return 0.35


    if (
        width > 0.20
        and height > 0.20
    ):

        return 0.95


    return 0.75


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(

    frame,

    hand_landmarks

):

    mp_drawing.draw_landmarks(

        frame,

        hand_landmarks,

        mp_hands.HAND_CONNECTIONS,

        mp_drawing_styles.get_default_hand_landmarks_style(),

        mp_drawing_styles.get_default_hand_connections_style()

    )


# ============================================================
# DRAW FINGER POINTS
# ============================================================

def draw_finger_points(

    frame,

    landmarks,

    fingers

):

    h, w, _ = frame.shape

    tips = [

        4,

        8,

        12,

        16,

        20

    ]


    for i, tip_index in enumerate(tips):

        point = landmarks.landmark[
            tip_index
        ]

        x = int(
            point.x * w
        )

        y = int(
            point.y * h
        )


        if i < fingers:

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                100,
                100,
                100
            )


        cv2.circle(

            frame,

            (x, y),

            10,

            color,

            -1

        )

        cv2.circle(

            frame,

            (x, y),

            13,

            (255, 255, 255),

            2

        )


# ============================================================
# EMERGENCY LEGEND
# ============================================================

def draw_emergency_legend(
    frame
):

    legend = [

        (
            "1  POLICE",
            (0, 0, 255)
        ),

        (
            "2  FIRE",
            (255, 120, 100)
        ),

        (
            "3  AMBULANCE",
            (255, 0, 255)
        ),

        (
            "4  WOMEN HELPLINE",
            (0, 255, 255)
        ),

        (
            "5  GENERAL HELPLINE",
            (0, 255, 0)
        )

    ]


    y = 250


    for text, color in legend:

        draw_text(

            frame,

            text,

            (40, y),

            0.65,

            color

        )

        y += 35


# ============================================================
# RL REWARD
# ============================================================

def calculate_reward(

    hand_detected,

    fingers,

    confidence,

    stable,

    hold_time,

    action

):

    reward = 0


    # --------------------------------------------------------
    # NO HAND / EXTERNAL OBJECT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # INVALID FINGER COUNT
    # --------------------------------------------------------

    if fingers < 1 or fingers > 5:

        if action == ACTION_IGNORE:

            reward += 3

        else:

            reward -= 3

        return reward


    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    if confidence < 0.60:

        if action in [

            ACTION_IGNORE,

            ACTION_WAIT

        ]:

            reward += 2

        else:

            reward -= 5

        return reward


    # --------------------------------------------------------
    # VALID HAND
    # --------------------------------------------------------

    if stable:

        if action == ACTION_WAIT:

            reward += 2

        elif action == ACTION_CONFIRM:

            reward += 4

        elif action == ACTION_TRIGGER:

            if hold_time >= HOLD_TIME:

                reward += 10

            else:

                reward -= 8

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
# AISA TEXT TO SPEECH
# ============================================================

def speak(text):

    print()
    print(
        "AISA:",
        text
    )

    try:

        pythoncom.CoInitialize()

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            165
        )

        engine.setProperty(
            "volume",
            1.0
        )

        engine.say(
            str(text)
        )

        engine.runAndWait()

        engine.stop()

    except Exception as error:

        print(
            "TTS ERROR:",
            error
        )

    finally:

        pythoncom.CoUninitialize()


# ============================================================
# SPEECH TO TEXT
# ============================================================

recognizer = sr.Recognizer()

# Wait longer before deciding the sentence is finished, so
# natural pauses ("my friend is injured... what can I do")
# are not cut off mid-sentence.
recognizer.pause_threshold = 1.5

recognizer.non_speaking_duration = 0.8

# Let the energy threshold adapt while listening.
recognizer.dynamic_energy_threshold = True

# Indian English recognises Indian accents much better than
# the default US English model.
SPEECH_LANGUAGE = "en-IN"

# Filled in by listen() so Gemini can be told when the
# transcript is probably wrong, and what else it might have been.
last_speech_confidence = None

last_speech_alternatives = []


def calibrate_microphone():

    # Calibrate ONCE, in a quiet moment. Doing it before every
    # listen (old behaviour) ate the first half second of the
    # reply and, if the person started talking during it, set
    # the threshold too high -> "could not understand".

    try:

        with sr.Microphone() as source:

            print()
            print(
                "Calibrating microphone (stay quiet)..."
            )

            recognizer.adjust_for_ambient_noise(

                source,

                duration=1.5

            )

        print(
            "Mic energy threshold:",
            int(recognizer.energy_threshold)
        )

    except Exception as error:

        print(
            "Mic calibration failed:",
            error
        )


def listen():

    global last_speech_confidence

    global last_speech_alternatives

    try:

        # tiny gap so the tail of AISA's own voice
        # is not picked up by the mic

        time.sleep(0.4)

        with sr.Microphone() as source:

            print()
            print(
                "AISA Listening..."
            )

            audio = recognizer.listen(

                source,

                timeout=10,

                phrase_time_limit=20

            )


        print(
            "Processing speech..."
        )

        result = recognizer.recognize_google(

            audio,

            language=SPEECH_LANGUAGE,

            show_all=True

        )

        if (
            not result
            or "alternative" not in result
        ):

            raise sr.UnknownValueError()

        alternatives = result["alternative"]

        text = alternatives[0].get(
            "transcript",
            ""
        )

        last_speech_confidence = alternatives[0].get(
            "confidence"
        )

        last_speech_alternatives = [

            alt.get("transcript", "")

            for alt in alternatives

            if alt.get("transcript")

        ]

        if last_speech_confidence is not None:

            print(
                f"(speech confidence: "
                f"{last_speech_confidence:.2f})"
            )

        print(
            "YOU:",
            text
        )

        return text


    except sr.WaitTimeoutError:

        print(
            "No speech detected."
        )

        return ""


    except sr.UnknownValueError:

        print(
            "Could not understand your voice."
        )

        return ""


    except sr.RequestError as error:

        print(
            "Speech recognition error:",
            error
        )

        return ""


    except Exception as error:

        print(
            "Microphone error:",
            error
        )

        return ""


# ============================================================
# RESPONDER MESSAGE
# ============================================================
#
# This is the short, fixed-format announcement AISA delivers
# FIRST when a call is placed. It targets the person who was
# called (e.g. the ambulance number) so they immediately get
# the essentials, before AISA switches over to helping the
# caller (the person in front of the camera).
#
# Kept as a template (not Gemini) on purpose: the message that
# goes to the responder should be predictable and never
# hallucinated.
# ============================================================

def build_responder_message(
    emergency_type
):

    message = (

        f"This is an automated emergency alert from AISA. "

        f"{emergency_type} assistance is needed "

        f"at {EMERGENCY_ADDRESS}. "

        f"This call was triggered by a hand gesture "

        f"emergency system. "

        f"Please stay on the line, AISA will now assist "

        f"the caller while you are connected."

    )

    return message


# ============================================================
# CALL ANSWER DETECTION
# ============================================================

def wait_for_call_window(
    timeout=CALL_ANSWER_TIMEOUT
):

    try:

        import pygetwindow as gw

    except ImportError:

        print(
            "pygetwindow not installed -- "
            "cannot auto-detect call state."
        )

        print(
            "Run: pip install pygetwindow"
        )

        return False


    start_time = time.time()


    while (

        time.time()
        - start_time

        < timeout

    ):

        for title in gw.getAllTitles():

            lowered = title.lower()

            if (

                "phone" in lowered

            ):

                # Heuristic only: a connected call often shows a
                # running duration like "0:07" in the window
                # title. This is fragile and may need tuning for
                # your specific Phone Link version.

                has_digit = any(

                    character.isdigit()

                    for character in lowered

                )

                if (
                    has_digit
                    and ":" in lowered
                ):

                    return True

        time.sleep(1)


    return False


def wait_for_call_answered():

    if CALL_ANSWER_MODE == "manual_key":

        print()

        print(
            "Waiting for confirmation the call was answered."
        )

        print(
            "Press ENTER in this console the moment "
            "the responder picks up."
        )

        try:

            input()

        except Exception as error:

            print(
                "Input error, continuing anyway:",
                error
            )

        return


    if CALL_ANSWER_MODE == "auto_window":

        detected = wait_for_call_window(

            timeout=CALL_ANSWER_TIMEOUT

        )

        if detected:

            return

        print(
            "Could not auto-detect the call connecting, "
            "falling back to a timed wait."
        )


    # ----------------------------------------------------
    # "timer" mode, and the fallback for "auto_window"
    # ----------------------------------------------------

    time.sleep(
        CALL_CONNECT_WAIT
    )


# ============================================================
# GEMINI WITH RETRY + BACKUP MODELS
# ============================================================
#
# A 503 ("high demand") is temporary and per-model. Instead of
# giving up on the first failure, try again, then try other
# models, all within a short time budget (this is an emergency,
# the caller can't wait a minute for an answer).
# ============================================================

GEMINI_FALLBACK_MODELS = [

    "gemini-2.5-flash",

    "gemini-2.0-flash",

    "gemini-2.5-flash-lite"

]

GEMINI_TIME_BUDGET = 12


def generate_with_fallback(
    prompt
):

    candidates = []

    for name in (
        [gemini_model]
        + GEMINI_FALLBACK_MODELS
    ):

        if name and name not in candidates:

            candidates.append(name)


    deadline = (
        time.time()
        + GEMINI_TIME_BUDGET
    )


    for name in candidates:

        for attempt in range(2):

            if time.time() > deadline:

                return None

            try:

                response = client.models.generate_content(

                    model=name,

                    contents=prompt

                )

                text = response.text

                if text:

                    if name != gemini_model:

                        print(
                            "(used backup model:",
                            name + ")"
                        )

                    return text.strip()

            except Exception as error:

                message = str(error)

                print(
                    f"GEMINI ERROR ({name}):",
                    message[:140]
                )

                # model does not exist -> next model now

                if (
                    "404" in message
                    or "NOT_FOUND" in message
                ):

                    break

                time.sleep(1.0)


    return None


# ============================================================
# OFFLINE FALLBACK
# ============================================================
#
# If every Gemini attempt fails, AISA must still help. These are
# basic, widely-taught first-aid steps. They are intentionally
# simple and never claim anything about the responder.
# ============================================================

OFFLINE_STEPS = [

    "My AI connection is unstable, but I am still here. "
    "Stay with the person and keep them still. "
    "Are they awake and breathing?",

    "If there is bleeding, press firmly on the wound "
    "with a clean cloth and keep pressing.",

    "Do not move the person unless there is immediate "
    "danger. Keep them warm and keep talking to them.",

    "Tell the responder on the call exactly where you "
    "are and what happened.",

    "Stay calm and stay on the line. "
    "Follow the responder's instructions."

]

offline_step = 0


def reset_offline_state():

    global offline_step

    offline_step = 0


def offline_reply(
    user_text
):

    global offline_step

    text = user_text.lower()

    if (
        "not breathing" in text
        or "isn't breathing" in text
        or "no breathing" in text
    ):

        reply = (
            "Tell the responder on the call right now "
            "that the person is not breathing, and "
            "follow their instructions."
        )

    elif (
        "bleed" in text
        or "blood" in text
    ):

        reply = (
            "Press firmly on the wound with a clean "
            "cloth and keep pressing. "
            "Do not remove the cloth."
        )

    elif (
        "unconscious" in text
        or "not awake" in text
        or "not responding" in text
    ):

        reply = (
            "Tell the responder the person is not "
            "responding. Do not move them unless "
            "there is danger."
        )

    else:

        reply = OFFLINE_STEPS[

            min(
                offline_step,
                len(OFFLINE_STEPS) - 1
            )

        ]

        offline_step += 1


    conversation.append(
        "AISA: " + reply
    )

    return reply


# ============================================================
# GEMINI AI
# ============================================================

def ask_ai(

    user_text,

    emergency_type

):

    if client is None:

        return offline_reply(
            user_text
        )


    if gemini_model is None:

        return offline_reply(
            user_text
        )


    conversation.append(

        "User: "
        + user_text

    )


    # last 12 lines only; the newest "User:" line is already
    # in here, so it is NOT added a second time below

    history = "\n".join(
        conversation[-12:]
    )


    speech_note = ""

    if last_speech_confidence is not None:

        speech_note += (
            f"Speech recognition confidence: "
            f"{last_speech_confidence:.2f}\n"
        )

    if len(last_speech_alternatives) > 1:

        speech_note += (
            "Other possible transcriptions of the "
            "last user message: "
            + " | ".join(
                last_speech_alternatives[1:4]
            )
            + "\n"
        )


    prompt = f"""

You are AISA, an emergency voice assistant.

Emergency type:
{emergency_type}

The emergency call has been initiated by the
gesture emergency system. A short automated message
with the emergency type and address has already been
read out to the responder on the call.

Your job now is to help the person communicate
and stay calm while emergency professionals
are being contacted.

IMPORTANT RULES:

1. Keep responses short.
2. Responses must sound natural when spoken aloud.
3. Ask only useful questions.
4. Stay calm and professional.
5. Never invent information.
6. Never claim emergency services contacted
   something that the program did not actually do.
7. You are not a doctor, police officer,
   firefighter, or paramedic.
8. Do not provide dangerous medical instructions.
9. If appropriate, ask whether the person is
   conscious and breathing.
10. If appropriate, ask about severe bleeding
    or serious injuries.
11. Encourage the user to follow instructions
    from emergency professionals.
12. Do not overwhelm the user with information.
13. Ask only ONE question per reply. Never combine
    two questions in one reply.
14. Never repeat a question you already asked in the
    conversation. If the user answered it, even briefly
    ("yes", "no", "they are"), accept the answer and
    move on to something new.
15. If the user's message seems cut off or unclear, use
    what you did understand instead of repeating your
    last question.
16. If the user gives a place name or address, repeat it
    back clearly so they can tell the responder on the
    call.
17. Always move forward: each reply must either give ONE
    safe, concrete next step (for example: stay with the
    person, apply firm pressure to bleeding with a clean
    cloth, do not move them unless there is danger, keep
    them warm) or ask ONE new useful question.
18. Maximum 40 words per reply.
19. The user's words come from speech recognition and may
    contain mistakes (for example "fastest fish injured"
    may really be "my friend is injured"). Use the
    alternative transcriptions and the emergency context
    to work out the most likely meaning and respond to
    that. Do not mention the recognition error unless the
    message truly makes no sense; then ask them to repeat
    it in a few simple words.
20. You cannot send anything to the responder. Never say
    you "noted", "recorded", "told" or "passed on"
    something. If the user gives a location, say it back
    and remind them to tell the responder themselves.
21. If the user's reply is a yes/no plus one odd,
    unrelated word, accept the yes/no and ignore the odd
    word. Only treat a word as a place if the user clearly
    said they are at that place.

Conversation so far (the last "User:" line is the
message you are replying to now):

{history}

{speech_note}
Reply now with a concise spoken response that follows
the rules above.
"""


    answer = generate_with_fallback(
        prompt
    )


    if answer:

        conversation.append(

            "AISA: "
            + answer

        )

        return answer


    # every model failed -> basic offline help instead

    print(
        "All Gemini attempts failed -- using offline help."
    )

    return offline_reply(
        user_text
    )


# ============================================================
# AISA ASSISTANT
# ============================================================
#
# Flow when a call is placed:
#
#   1. (main thread) call_emergency() dials / simulates the call
#   2. this function runs in its own thread and:
#        a. waits for the call to connect
#        b. speaks the FIXED responder message once
#           (target: the person who was called)
#        c. switches into ongoing conversational support
#           for the caller (target: the person at the camera)
#
# Steps (b) and (c) are sequential, not simultaneous, because
# there is one microphone / one speaker on this machine. If you
# want (b) and (c) to run at the same time -- e.g. the responder
# hears the alert while AISA is already listening to the caller
# -- you need to route audio through two separate devices (for
# example a virtual audio cable such as VB-Audio Cable): one
# device feeds the phone call, the other is the normal mic/
# speaker AISA uses to talk with the caller. That's a bigger
# change (separate output device selection in pyttsx3 and a
# separate input device in speech_recognition) -- ask if you
# want that version built out.
# ============================================================

def start_aisa(

    emergency_type

):

    global assistant_running

    global call_in_progress

    global current_call_label

    assistant_running = True


    print()
    print(
        "=" * 60
    )

    print(
        "AISA EMERGENCY VOICE ASSISTANT"
    )

    print(
        "=" * 60
    )


    conversation.clear()

    reset_offline_state()


    try:

        # ----------------------------------------------------
        # STEP 1: wait for the call to actually be answered
        # ----------------------------------------------------
        # This runs in TEST_MODE too. No real call is placed,
        # but AISA still waits for your confirmation (or the
        # timer/auto check) before it "announces" anything --
        # otherwise you can't actually test the timing of this
        # step at all.
        # ----------------------------------------------------

        if TEST_MODE:

            print()
            print(
                "TEST MODE: no real call is being placed, "
                "but AISA will still wait per CALL_ANSWER_MODE "
                "below before it speaks."
            )

        wait_for_call_answered()


        # ----------------------------------------------------
        # STEP 2: deliver the fixed responder message
        # ----------------------------------------------------

        responder_message = build_responder_message(
            emergency_type
        )

        if TEST_MODE:

            print()
            print(
                "TEST MODE: would announce to responder:"
            )

            print(
                responder_message
            )

        speak(
            responder_message
        )


        # ----------------------------------------------------
        # STEP 3: switch to helping the caller
        # ----------------------------------------------------

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

                print(
                    "Mic energy threshold:",
                    int(recognizer.energy_threshold)
                )

                # Every 2nd failure: say something (instead of
                # silently looping) and make the mic a bit more
                # sensitive in case the threshold drifted too high.

                if listen_failures % 2 == 0:

                    recognizer.energy_threshold = max(

                        100,

                        recognizer.energy_threshold * 0.7

                    )

                    speak(

                        "I could not hear you clearly. "
                        "Please speak a little louder, "
                        "in a few simple words."

                    )

                continue


            listen_failures = 0


            command = (
                user_text
                .lower()
                .strip()
            )


            if command in [

                "exit",

                "stop",

                "stop assistant",

                "quit",

                "end conversation"

            ]:

                speak(
                    "AISA assistant shutting down."
                )

                assistant_running = False

                break


            try:

                response = ask_ai(

                    user_text,

                    emergency_type

                )

                speak(
                    response
                )

            except Exception as error:

                print(
                    "AISA ERROR:",
                    error
                )

                speak(
                    "I am having trouble processing "
                    "that right now."
                )


    finally:

        # ----------------------------------------------------
        # Whatever happened above (normal "stop", an error,
        # anything) -- this call session is over. Release the
        # lock so the main loop will accept new gestures again.
        # ----------------------------------------------------

        assistant_running = False

        call_in_progress = False

        current_call_label = ""

        print()
        print(
            "Call session ended -- gestures re-enabled."
        )


# ============================================================
# START AISA THREAD
# ============================================================

def start_aisa_thread(

    emergency_type

):

    global assistant_thread

    assistant_thread = threading.Thread(

        target=start_aisa,

        args=(emergency_type,),

        daemon=True

    )

    assistant_thread.start()


# ============================================================
# EMERGENCY CALL
# ============================================================

def call_emergency(

    fingers,

    frame

):

    global last_call_time

    global recording

    global record_start_time

    global out

    global call_in_progress

    global current_call_label


    current_time = time.time()


    if fingers not in emergency_numbers:

        return "Invalid gesture"


    if (

        current_time
        - last_call_time
        < call_cooldown

    ):

        return (
            "Emergency cooldown active"
        )


    number = emergency_numbers[
        fingers
    ]

    label = emergency_labels[
        fingers
    ]


    # Lock out new gesture-triggered calls until this one's
    # AISA session fully ends (see start_aisa's finally block).

    call_in_progress = True

    current_call_label = label


    print()
    print(
        "=" * 50
    )

    print(
        "EMERGENCY:",
        label
    )

    print(
        "NUMBER:",
        number
    )

    print(
        "=" * 50
    )


    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:

        print()
        print(
            "TEST MODE: REAL CALL DISABLED"
        )

        print(
            "Would call:",
            label
        )

        print(
            "Number:",
            number
        )


    else:

        # ----------------------------------------------------
        # SIREN
        # ----------------------------------------------------

        try:

            pygame.mixer.music.load(
                SIREN_SOUND
            )

            pygame.mixer.music.play()

        except Exception as error:

            print(
                "Siren error:",
                error
            )


        # ----------------------------------------------------
        # WINDOWS PHONE
        # ----------------------------------------------------

        os.system(
            "start ms-phone://"
        )

        time.sleep(3)


        pyautogui.write(
            number
        )

        time.sleep(1)

        pyautogui.press(
            "enter"
        )


    last_call_time = current_time


    # --------------------------------------------------------
    # VIDEO RECORDING
    # --------------------------------------------------------

    if out is not None:

        out.release()


    out = cv2.VideoWriter(

        "call_recording.mp4",

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        20.0,

        (

            frame_width,

            frame_height

        )

    )


    recording = True

    record_start_time = time.time()


    # --------------------------------------------------------
    # START AISA
    # (this now speaks the responder message first, then
    # switches into helping the caller -- see start_aisa)
    # --------------------------------------------------------

    start_aisa_thread(
        label
    )


    return (

        f"CALL TRIGGERED: {label}"

    )


# ============================================================
# MAIN
# ============================================================

print()
print(
    "=" * 60
)

print(
    "AISA HAND GESTURE EMERGENCY"
)

print(
    "MEDIAPIPE + REINFORCEMENT LEARNING + GEMINI"
)

print(
    "=" * 60
)

print()

print(
    "TEST MODE:",
    TEST_MODE
)

print(
    "HOLD TIME:",
    HOLD_TIME,
    "seconds"
)

print()

print(
    "Show 1-5 fingers."
)

print(
    "Press ESC to exit."
)

print()

print(
    "Q-table:",
    Q_TABLE_FILE
)

print(
    "Dataset:",
    DATASET_FILE
)

print()

if gemini_model:

    print(
        "Gemini:",
        gemini_model
    )

else:

    print(
        "Gemini: NOT AVAILABLE"
    )

print()


# ============================================================
# MAIN LOOP
# ============================================================

calibrate_microphone()

while cap.isOpened():

    success, frame = cap.read()


    if not success:

        print(
            "Camera frame error."
        )

        break


    # --------------------------------------------------------
    # MIRROR
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    results = hands.process(
        rgb_frame
    )


    hand_detected = False

    fingers = 0

    confidence = 0.0

    landmarks = None


    # --------------------------------------------------------
    # HAND FOUND
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_detected = True

        landmarks = (
            results.multi_hand_landmarks[0]
        )


        draw_hand(

            frame,

            landmarks

        )


        fingers = count_fingers(

            landmarks

        )


        confidence = (
            calculate_hand_confidence(
                landmarks
            )
        )


        draw_finger_points(

            frame,

            landmarks,

            fingers

        )


    # --------------------------------------------------------
    # GESTURE STABILITY
    # --------------------------------------------------------

    if hand_detected:

        if fingers == previous_fingers:

            gesture_stable_time += (
                1 / 30
            )

        else:

            gesture_stable_time = 0

            previous_fingers = fingers


    else:

        gesture_stable_time = 0

        previous_fingers = 0


    stable = (

        gesture_stable_time
        >= 0.4

    )


    # --------------------------------------------------------
    # RL STATE
    # --------------------------------------------------------

    state = rl_agent.get_state(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time

    )


    # --------------------------------------------------------
    # RL ACTION
    # --------------------------------------------------------

    action = rl_agent.choose_action(
        state
    )


    # --------------------------------------------------------
    # VALID GESTURE
    # --------------------------------------------------------

    valid_gesture = (

        hand_detected

        and fingers in emergency_numbers

        and confidence >= 0.60

        and stable

        and gesture_stable_time >= HOLD_TIME

    )


    # --------------------------------------------------------
    # RL REWARD
    # --------------------------------------------------------

    reward = calculate_reward(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time,

        action

    )


    # --------------------------------------------------------
    # NEXT STATE
    # --------------------------------------------------------

    next_state = rl_agent.get_state(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time

    )


    # --------------------------------------------------------
    # UPDATE Q TABLE
    # --------------------------------------------------------

    rl_agent.update(

        state,

        action,

        reward,

        next_state

    )


    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    save_dataset_row(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time,

        action,

        reward,

        fingers,

        confidence,

        stable,

        gesture_stable_time,

        valid_gesture

    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if not hand_detected:

        status = (
            "NO HAND - OBJECT IGNORED"
        )

        status_color = (
            0,
            165,
            255
        )


    elif fingers == 0:

        status = (
            "HAND DETECTED - FIST"
        )

        status_color = (
            255,
            255,
            0
        )


    elif not stable:

        status = (

            f"HAND DETECTED - "
            f"{fingers} FINGERS"

        )

        status_color = (
            255,
            255,
            0
        )


    elif gesture_stable_time < HOLD_TIME:

        remaining = max(

            0,

            HOLD_TIME
            - gesture_stable_time

        )


        status = (

            f"{fingers} FINGERS - "
            f"HOLD {remaining:.1f}s"

        )

        status_color = (
            0,
            255,
            255
        )


    else:

        status = (

            f"VALID {fingers}-FINGER "
            f"GESTURE"

        )

        status_color = (
            0,
            255,
            0
        )


    # --------------------------------------------------------
    # CALL IN PROGRESS OVERRIDE
    # --------------------------------------------------------
    # Camera keeps reading and RL keeps learning either way,
    # but no new call can be triggered while one is active.
    # --------------------------------------------------------

    if call_in_progress:

        status = (

            f"CALL IN PROGRESS - {current_call_label} "
            f"(gestures paused)"

        )

        status_color = (
            0,
            140,
            255
        )


    # --------------------------------------------------------
    # SAFETY TRIGGER
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # RL does NOT control whether a valid emergency
    # gesture is allowed to call.
    #
    # RL still learns from the state/reward data.
    #
    # --------------------------------------------------------

    if (

        valid_gesture

        and not confirming

        and not call_in_progress

        and (

            time.time()
            - last_call_time
            >= call_cooldown

        )

    ):

        confirming = True

        confirmed_fingers = fingers


        print()

        print(
            "Valid emergency gesture:"
        )

        print(
            emergency_labels[fingers]
        )


    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------

    if confirming:

        label = emergency_labels[
            confirmed_fingers
        ]


        status = (

            f"CONFIRM {label} - "
            f"KEEP {confirmed_fingers} "
            f"FINGERS UP"

        )

        status_color = (
            0,
            255,
            0
        )


        # ----------------------------------------------------
        # CANCEL IF HAND DISAPPEARS
        # ----------------------------------------------------

        if not hand_detected:

            confirming = False

            confirmed_fingers = 0

            gesture_stable_time = 0


        # ----------------------------------------------------
        # CANCEL IF GESTURE CHANGES
        # ----------------------------------------------------

        elif fingers != confirmed_fingers:

            confirming = False

            confirmed_fingers = 0

            gesture_stable_time = 0


        # ----------------------------------------------------
        # CALL
        # ----------------------------------------------------

        elif (

            gesture_stable_time
            >= HOLD_TIME

        ):

            status = call_emergency(

                confirmed_fingers,

                frame

            )


            # Log call event

            save_dataset_row(

                True,

                confirmed_fingers,

                confidence,

                True,

                gesture_stable_time,

                ACTION_TRIGGER,

                10,

                confirmed_fingers,

                confidence,

                True,

                gesture_stable_time,

                True,

                emergency_labels[
                    confirmed_fingers
                ],

                True

            )


            confirming = False

            confirmed_fingers = 0

            gesture_stable_time = 0


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    draw_text(

        frame,

        "AISA HAND GESTURE EMERGENCY",

        (30, 40),

        0.9,

        (255, 255, 255)

    )


    draw_text(

        frame,

        "Show 1-5 fingers",

        (30, 75),

        0.7,

        (100, 220, 255)

    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    draw_text(

        frame,

        status,

        (30, 115),

        0.70,

        status_color

    )


    # --------------------------------------------------------
    # FINGER COUNT
    # --------------------------------------------------------

    draw_text(

        frame,

        f"Fingers: {fingers}",

        (30, 155),

        0.65,

        (255, 255, 255)

    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    draw_text(

        frame,

        f"Confidence: {confidence:.2f}",

        (30, 190),

        0.55,

        (200, 200, 255)

    )


    # --------------------------------------------------------
    # HOLD TIMER
    # --------------------------------------------------------

    draw_text(

        frame,

        f"Hold: {gesture_stable_time:.1f}s / {HOLD_TIME:.1f}s",

        (30, 220),

        0.55,

        (255, 255, 255)

    )


    # --------------------------------------------------------
    # RL ACTION
    # --------------------------------------------------------

    draw_text(

        frame,

        f"RL: {actions[action]}",

        (30, 250),

        0.55,

        (200, 255, 200)

    )


    # --------------------------------------------------------
    # AISA STATUS
    # --------------------------------------------------------

    if assistant_running:

        draw_text(

            frame,

            "AISA AI: ACTIVE",

            (400, 115),

            0.65,

            (0, 255, 0)

        )

    else:

        draw_text(

            frame,

            "AISA AI: STANDBY",

            (400, 115),

            0.65,

            (150, 150, 150)

        )


    # --------------------------------------------------------
    # GEMINI MODEL
    # --------------------------------------------------------

    if gemini_model:

        draw_text(

            frame,

            "Gemini: READY",

            (400, 150),

            0.55,

            (100, 255, 200)

        )

    else:

        draw_text(

            frame,

            "Gemini: OFFLINE",

            (400, 150),

            0.55,

            (0, 165, 255)

        )


    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:

        draw_text(

            frame,

            "TEST MODE",

            (400, 185),

            0.65,

            (0, 165, 255)

        )


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    draw_emergency_legend(
        frame
    )


    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    if (

        time.time()
        - last_call_time
        < call_cooldown

    ):

        remaining = (

            call_cooldown

            - (

                time.time()
                - last_call_time

            )

        )


        draw_text(

            frame,

            f"Cooldown: {remaining:.1f}s",

            (400, 220),

            0.65,

            (0, 165, 255)

        )


    # --------------------------------------------------------
    # RECORDING
    # --------------------------------------------------------

    if (

        recording

        and out is not None

    ):

        out.write(
            frame
        )


        elapsed = (

            time.time()
            - record_start_time

        )


        draw_text(

            frame,

            f"RECORDING {elapsed:.0f}s",

            (400, 255),

            0.65,

            (0, 0, 255)

        )


        if elapsed >= 600:

            recording = False

            out.release()

            out = None

            print(
                "Recording saved."
            )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(

        WINDOW_NAME,

        frame

    )


    # --------------------------------------------------------
    # ESC
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        assistant_running = False

        break


# ============================================================
# CLEANUP
# ============================================================

assistant_running = False


print()

print(
    "Saving RL Q-table..."
)


rl_agent.save()


if out is not None:

    out.release()


cap.release()

hands.close()

cv2.destroyAllWindows()

pygame.mixer.quit()


print()

print(
    "AISA stopped."
)

print(
    "Q-table saved to:",
    Q_TABLE_FILE
)

print(
    "Dataset saved to:",
    DATASET_FILE
)