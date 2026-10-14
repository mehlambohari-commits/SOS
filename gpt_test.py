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
import json

# ============================================================
# OPTIONAL AI / SPEECH
# ============================================================

import speech_recognition as sr
import pyttsx3

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google import genai
except Exception:
    genai = None


# ============================================================
# PHONE LINK UI AUTOMATION
# ============================================================

try:
    from pywinauto import Desktop
    PYWINAUTO_AVAILABLE = True
except Exception:
    PYWINAUTO_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

# IMPORTANT:
# False = REAL CALL
TEST_MODE = False

CAMERA_INDEX = 0

WINDOW_NAME = "AISA - Hand Gesture Emergency App"

SIREN_SOUND = r"C:\Users\TARMAL COMPUTERS\siren.wav.mp3"

# ------------------------------------------------------------
# EMERGENCY LOCATION
# ------------------------------------------------------------
#
# Replace this with your actual fixed location or integrate
# GPS later.
#
EMERGENCY_LOCATION = "YOUR CURRENT LOCATION"

# ------------------------------------------------------------
# TIMERS
# ------------------------------------------------------------

HOLD_TIME = 2.5

GESTURE_STABLE_TIME = 0.4

CALL_COOLDOWN = 3

CALL_CONNECT_TIMEOUT = 120

AI_ASSISTANCE_TIME = 300       # 5 minutes

# ------------------------------------------------------------
# RL
# ------------------------------------------------------------

RL_MODEL_FILE = "aisa_q_table.npy"

RL_DATASET_FILE = "aisa_rl_dataset.csv"

# ============================================================
# EMERGENCY NUMBERS
# ============================================================

emergency_numbers = {

    1: "8793796952",
    2: "9335587271",
    3: "8878010010",
    4: "8530496089",
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

try:

    pygame.mixer.init()

except Exception as error:

    print("Pygame mixer error:", error)


# ============================================================
# TEXT TO SPEECH
# ============================================================

tts_engine = None

try:

    tts_engine = pyttsx3.init()

    tts_engine.setProperty(
        "rate",
        165
    )

except Exception as error:

    print(
        "TTS initialization error:",
        error
    )


tts_lock = threading.Lock()


def speak(text):

    """
    Speak text safely.
    """

    print()
    print("AISA:", text)

    if tts_engine is None:
        return

    try:

        with tts_lock:

            tts_engine.say(text)

            tts_engine.runAndWait()

    except Exception as error:

        print(
            "TTS error:",
            error
        )

speak("AISA voice test. Emergency assistance message.")
# ============================================================
# SPEECH RECOGNITION
# ============================================================

recognizer = sr.Recognizer()


def listen():

    """
    Listen to microphone.

    Used after the emergency call has been connected.
    """

    try:

        with sr.Microphone() as source:

            print(
                "Listening..."
            )

            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.3
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=8
            )

        try:

            text = recognizer.recognize_google(
                audio
            )

            print(
                "Caller:",
                text
            )

            return text

        except sr.UnknownValueError:

            return ""

        except sr.RequestError as error:

            print(
                "Speech recognition service error:",
                error
            )

            return ""

    except sr.WaitTimeoutError:

        return ""

    except Exception as error:

        print(
            "Microphone error:",
            error
        )

        return ""


# ============================================================
# GEMINI AI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

gemini_client = None

if genai is not None and GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini AI initialized."
        )

    except Exception as error:

        print(
            "Gemini initialization error:",
            error
        )


conversation = []


def ask_ai(user_text):

    """
    Emergency AI assistant.

    Keeps responses short because they are spoken aloud.
    """

    global conversation

    conversation.append(
        {
            "role": "user",
            "text": user_text
        }
    )

    if gemini_client is None:

        return (
            "Please stay calm. "
            "Emergency assistance has been contacted. "
            "Tell me what help you need."
        )


    recent_conversation = conversation[-10:]


    prompt = """

You are AISA, an emergency voice assistant.

The user is currently involved in an emergency.

Your responsibilities:

1. Stay calm.
2. Give short spoken responses.
3. Help the caller communicate with emergency services.
4. Ask only useful questions.
5. Never invent information.
6. Never claim that an emergency service has been contacted
   unless the program actually made the call.
7. Help with immediate general safety guidance.
8. Keep responses suitable for being spoken aloud.
9. If the caller gives a location, injury, danger,
   number of people, or other important information,
   acknowledge it clearly.
10. Do not give unnecessarily long explanations.

Conversation:

"""


    for message in recent_conversation:

        prompt += (
            f"{message['role']}: "
            f"{message['text']}\n"
        )


    try:

        response = gemini_client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt

        )

        answer = response.text.strip()

        conversation.append(
            {
                "role": "assistant",
                "text": answer
            }
        )

        return answer

    except Exception as error:

        print(
            "Gemini error:",
            error
        )

        return (
            "Stay calm. "
            "Emergency assistance is being handled. "
            "Tell me what is happening."
        )


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles


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
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


# ============================================================
# GLOBAL CALL STATE
# ============================================================

last_call_time = 0

call_in_progress = False

call_connected = False

call_start_time = 0

call_thread = None

current_emergency_label = ""

current_emergency_number = ""

current_location = EMERGENCY_LOCATION


# ============================================================
# VIDEO RECORDING
# ============================================================

recording = False

record_start_time = 0

out = None


# ============================================================
# GESTURE STATE
# ============================================================

previous_fingers = 0

gesture_stable_time = 0

confirming = False

confirmed_fingers = 0


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
    # STATE
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

            3

        )


        return (

            int(hand_detected),

            int(fingers),

            confidence_level,

            int(stable),

            hold_level

        )


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

            # Safe defaults

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


        if random.random() < self.epsilon:

            return random.choice(

                [
                    ACTION_IGNORE,
                    ACTION_WAIT,
                    ACTION_CONFIRM
                ]

            )


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


        new_q = current_q + self.alpha * (

            reward

            + self.gamma * best_next_q

            - current_q

        )


        self.q_table[state][action] = new_q


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        filename=RL_MODEL_FILE
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
        filename=RL_MODEL_FILE
    ):

        if not os.path.exists(
            filename
        ):

            print(
                "No existing Q-table found."
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


# ============================================================
# RL DATASET LOGGER
# ============================================================

class RLDataset:

    def __init__(
        self,
        filename=RL_DATASET_FILE
    ):

        self.filename = filename

        self.columns = [

            "timestamp",

            "hand_detected",

            "finger_count",

            "confidence",

            "stable",

            "hold_time",

            "state",

            "action",

            "action_name",

            "reward",

            "next_state",

            "valid_gesture",

            "emergency_type",

            "call_in_progress"

        ]


        if not os.path.exists(
            filename
        ):

            with open(
                filename,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    self.columns
                )


    def log(

        self,

        hand_detected,

        fingers,

        confidence,

        stable,

        hold_time,

        state,

        action,

        reward,

        next_state,

        valid_gesture

    ):

        emergency_type = ""

        if fingers in emergency_labels:

            emergency_type = emergency_labels[
                fingers
            ]


        row = [

            time.time(),

            int(hand_detected),

            fingers,

            round(
                confidence,
                4
            ),

            int(stable),

            round(
                hold_time,
                3
            ),

            str(state),

            action,

            actions[action],

            reward,

            str(next_state),

            int(valid_gesture),

            emergency_type,

            int(call_in_progress)

        ]


        try:

            with open(
                self.filename,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    row
                )

        except Exception as error:

            print(
                "Dataset error:",
                error
            )


rl_agent = QLearningAgent()

rl_agent.load()


rl_dataset = RLDataset()


# ============================================================
# DRAW TEXT
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


    # Thumb

    if lm[4].x < lm[3].x:

        fingers += 1


    # Other four fingers

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


    width = max(xs) - min(xs)

    height = max(ys) - min(ys)


    if width < 0.05 or height < 0.05:

        return 0.35


    if width > 0.20 and height > 0.20:

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

        ("1  POLICE", (0, 0, 255)),

        ("2  FIRE", (255, 120, 100)),

        ("3  AMBULANCE", (255, 0, 255)),

        ("4  WOMEN HELPLINE", (0, 255, 255)),

        ("5  GENERAL HELPLINE", (0, 255, 0))

    ]


    y = 270


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
# REWARD
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
    # INVALID
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
# PHONE LINK
# ============================================================

def find_phone_window():

    if not PYWINAUTO_AVAILABLE:

        return None


    try:

        windows = Desktop(
            backend="uia"
        ).windows()


        for window in windows:

            title = (
                window.window_text()
                .lower()
            )


            if (

                "phone link" in title

                or "phone" in title

            ):

                return window


    except Exception as error:

        print(
            "Phone Link search error:",
            error
        )


    return None


# ============================================================
# CHECK CALL CONNECTED
# ============================================================


def is_call_connected():
    """
    Detect an active call using Phone Link UI text.
    The exact text depends on the Phone Link version.
    """

    if not PYWINAUTO_AVAILABLE:
        return False

    try:
        window = find_phone_window()

        if window is None:
            return False

        texts = []

        try:
            texts = [
                control.window_text().strip().lower()
                for control in window.descendants()
                if control.window_text().strip()
            ]
        except Exception as error:
            print("Could not read Phone Link controls:", error)

        print("Phone Link UI:", texts)

        connected_words = [
            "end call",
            "hang up",
            "mute",
            "keypad",
            "speaker",
            "disconnect"
        ]

        return any(
            word in text
            for text in texts
            for word in connected_words
        )

    except Exception as error:
        print("Call connection detection error:", error)
        return False

# ============================================================
# WAIT FOR ANSWER
# ============================================================

def wait_for_call_connection():

    global call_connected


    print()
    print(
        "Waiting for the emergency service "
        "to answer..."
    )


    start = time.time()


    while (

        time.time() - start

        < CALL_CONNECT_TIMEOUT

    ):

        if is_call_connected():

            call_connected = True


            print()
            print(
                "================================"
            )

            print(
                "CALL CONNECTED"
            )

            print(
                "================================"
            )


            return True


        time.sleep(1)


    print(
        "Could not detect an answered call."
    )


    return False


# ============================================================
# EMERGENCY MESSAGE
# ============================================================

def speak_emergency_message(

    label,
    location

):

    if location == "YOUR CURRENT LOCATION":

        location_text = (
            "the caller's current location"
        )

    else:

        location_text = location


    message = (

        f"Emergency required. "

        f"This is an AISA emergency alert. "

        f"A {label.lower()} emergency "

        f"is required at {location_text}. "

        f"Please send assistance immediately."

    )


    print()
    print(
        "================================"
    )

    print(
        "MESSAGE TO EMERGENCY SERVICE"
    )

    print(
        "================================"
    )

    print(
        message
    )


    speak(
        message
    )


# ============================================================
# ASSIST CALLER FOR FIVE MINUTES
# ============================================================

def emergency_assistant_for_5_minutes(

    label,
    location

):

    start_time = time.time()


    conversation.clear()


    speak(

        f"The {label.lower()} emergency call "
        f"is connected. "
        f"I will stay with you and help "
        f"for the next five minutes."

    )


    while (

        time.time() - start_time

        < AI_ASSISTANCE_TIME

    ):

        remaining = (

            AI_ASSISTANCE_TIME

            - (

                time.time()
                - start_time

            )

        )


        print(
            f"AI assistance active: "
            f"{remaining:.0f} seconds remaining"
        )


        try:

            user_text = listen()


            if not user_text:

                continue


            command = (
                user_text
                .lower()
                .strip()
            )


            if command in [

                "stop assistant",

                "exit assistant",

                "quit assistant",

                "end assistant"

            ]:

                speak(
                    "Emergency assistant stopping."
                )

                break


            response = ask_ai(
                user_text
            )


            speak(
                response
            )


        except Exception as error:

            print(
                "Emergency AI error:",
                error
            )

            time.sleep(1)


    print()
    print(
        "Five-minute emergency assistance "
        "period finished."
    )


# ============================================================
# RECORDING
# ============================================================

def start_recording():

    global out
    global recording
    global record_start_time


    if out is not None:

        try:

            out.release()

        except Exception:

            pass


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


# ============================================================
# COMPLETE CALL PROCESS
# ============================================================

def make_emergency_call(

    fingers,
    location

):

    global call_in_progress

    global call_connected

    global call_start_time

    global last_call_time

    global current_emergency_label

    global current_emergency_number


    if fingers not in emergency_numbers:

        return

    if time.time() - last_call_time < CALL_COOLDOWN:
        print("Call cooldown active. Please wait.")
        return
    # --------------------------------------------------------
    # LOCK CAMERA
    # --------------------------------------------------------

    call_in_progress = True

    call_connected = False

    call_start_time = time.time()


    label = emergency_labels[
        fingers
    ]

    number = emergency_numbers[
        fingers
    ]


    current_emergency_label = label

    current_emergency_number = number


    print()
    print(
        "======================================"
    )

    print(
        "EMERGENCY CALL STARTED"
    )

    print(
        f"SERVICE : {label}"
    )

    print(
        f"NUMBER  : {number}"
    )

    print(
        f"LOCATION: {location}"
    )

    print(
        "======================================"
    )


    # --------------------------------------------------------
    # START RECORDING
    # --------------------------------------------------------

    start_recording()


    # --------------------------------------------------------
    # SIREN
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    if TEST_MODE:

        print()
        print(
            "TEST MODE ENABLED"
        )

        print(
            "No real call will be placed."
        )

        # For testing the rest of the workflow
        time.sleep(2)

        call_connected = True


    else:

        # ----------------------------------------------------
        # OPEN PHONE LINK
        # ----------------------------------------------------

        os.system(
            "start ms-phone://"
        )


        time.sleep(3)


        # ----------------------------------------------------
        # DIAL
        # ----------------------------------------------------

        pyautogui.write(

            number,

            interval=0.05

        )


        time.sleep(1)


        pyautogui.press(
            "enter"
        )


        print(
            "Number dialed."
        )


        # ----------------------------------------------------
        # WAIT FOR ACTUAL ANSWER
        # ----------------------------------------------------

        connected = (
            wait_for_call_connection()
        )


        if not connected:

            print(
                "Call connection could not "
                "be confirmed."
            )


            # IMPORTANT:
            # Do NOT speak the emergency message
            # because we don't know if somebody answered.

            call_in_progress = False

            call_connected = False

            return


    # --------------------------------------------------------
    # CALL CONNECTED
    # --------------------------------------------------------

    if call_connected:

        print()
        print(
            "CALL ANSWERED."
        )


        # ----------------------------------------------------
        # MESSAGE TO EMERGENCY SERVICE
        # ----------------------------------------------------

        speak_emergency_message(

            label,

            location

        )


        # ----------------------------------------------------
        # AI ASSISTANCE
        # ----------------------------------------------------

        emergency_assistant_for_5_minutes(

            label,

            location

        )


    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    last_call_time = time.time()

    call_in_progress = False

    call_connected = False

    print()
    print(
        "Emergency session finished."
    )


# ============================================================
# START CALL THREAD
# ============================================================

def start_emergency_call_thread(

    fingers,
    location

):

    global call_thread


    if call_in_progress:

        print(
            "Emergency call already active."
        )

        return


    call_thread = threading.Thread(

        target=make_emergency_call,

        args=(

            fingers,
            location

        ),

        daemon=True

    )


    call_thread.start()


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print(
    "=============================================="
)

print(
    "       AISA EMERGENCY ASSISTANT"
)

print(
    "       MEDIAPIPE + Q-LEARNING + AI"
)

print(
    "=============================================="
)

print()

print(
    "REAL CALL MODE:",
    not TEST_MODE
)

print(
    "Emergency assistance:",
    "5 minutes"
)

print(
    "Location:",
    EMERGENCY_LOCATION
)

print()

print(
    "Show 1-5 fingers."
)

print(
    "Press ESC to exit."
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

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
    # HAND
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


    # ========================================================
    # CAMERA LOCKED DURING CALL
    # ========================================================

    if call_in_progress:

        draw_text(

            frame,

            "EMERGENCY CALL ACTIVE",

            (30, 40),

            0.9,

            (0, 0, 255)

        )


        if call_connected:

            elapsed = (
                time.time()
                - call_start_time
            )


            remaining = max(

                0,

                AI_ASSISTANCE_TIME
                - elapsed

            )


            draw_text(

                frame,

                "CALL CONNECTED",

                (30, 80),

                0.75,

                (0, 255, 0)

            )


            draw_text(

                frame,

                f"AI ACTIVE: {remaining:.0f}s",

                (30, 115),

                0.7,

                (0, 255, 255)

            )

        else:

            draw_text(

                frame,

                "WAITING FOR ANSWER...",

                (30, 80),

                0.75,

                (0, 255, 255)

            )


        draw_text(

            frame,

            "CAMERA LOCKED - NO NEW CALL",

            (30, 150),

            0.65,

            (255, 200, 100)

        )


        draw_text(

            frame,

            f"Service: {current_emergency_label}",

            (30, 185),

            0.65,

            (255, 255, 255)

        )


        # ----------------------------------------------------
        # RECORDING
        # ----------------------------------------------------

        if recording and out is not None:

            out.write(
                frame
            )


            elapsed_recording = (

                time.time()
                - record_start_time

            )


            draw_text(

                frame,

                f"RECORDING {elapsed_recording:.0f}s",

                (400, 150),

                0.65,

                (0, 0, 255)

            )


            if elapsed_recording >= 600:

                recording = False

                out.release()

                out = None


        cv2.imshow(

            WINDOW_NAME,

            frame

        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == 27:

            break


        # VERY IMPORTANT:
        # Skip all gesture/call processing.

        continue


    # ========================================================
    # NORMAL GESTURE PROCESSING
    # ========================================================

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
        >= GESTURE_STABLE_TIME

    )


    # ========================================================
    # RL STATE
    # ========================================================

    state = rl_agent.get_state(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time

    )


    # ========================================================
    # RL ACTION
    # ========================================================

    action = rl_agent.choose_action(
        state
    )


    # ========================================================
    # VALID GESTURE
    # ========================================================

    valid_gesture = (

        hand_detected

        and fingers in emergency_numbers

        and confidence >= 0.60

        and stable

        and gesture_stable_time >= HOLD_TIME

    )


    # ========================================================
    # RL REWARD
    # ========================================================

    reward = calculate_reward(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time,

        action

    )


    # ========================================================
    # NEXT STATE
    # ========================================================

    next_state = rl_agent.get_state(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time

    )


    # ========================================================
    # UPDATE Q TABLE
    # ========================================================

    rl_agent.update(

        state,

        action,

        reward,

        next_state

    )


    # ========================================================
    # SAVE RL EXPERIENCE
    # ========================================================

    rl_dataset.log(

        hand_detected,

        fingers,

        confidence,

        stable,

        gesture_stable_time,

        state,

        action,

        reward,

        next_state,

        valid_gesture

    )


    # ========================================================
    # STATUS
    # ========================================================

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


    # ========================================================
    # TRIGGER
    # ========================================================

    if valid_gesture and not confirming:
        confirming = True
        confirmed_fingers = fingers


    # ========================================================
    # CONFIRMATION
    # ========================================================

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


        if not hand_detected:

            confirming = False

            confirmed_fingers = 0


        elif fingers != confirmed_fingers:

            confirming = False

            confirmed_fingers = 0


        elif gesture_stable_time >= (

            HOLD_TIME + 0.5

        ):

            # ------------------------------------------------
            # START REAL CALL
            # ------------------------------------------------

            start_emergency_call_thread(

                confirmed_fingers,

                current_location

            )


            confirming = False

            confirmed_fingers = 0


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # STATUS
    # ========================================================

    draw_text(

        frame,

        status,

        (30, 115),

        0.75,

        status_color

    )


    # ========================================================
    # FINGERS
    # ========================================================

    draw_text(

        frame,

        f"Fingers: {fingers}",

        (30, 155),

        0.7,

        (255, 255, 255)

    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    draw_text(

        frame,

        f"Hand confidence: {confidence:.2f}",

        (30, 190),

        0.6,

        (200, 200, 255)

    )


    # ========================================================
    # RL ACTION
    # ========================================================

    draw_text(

        frame,

        f"RL Action: {actions[action]}",

        (30, 225),

        0.6,

        (200, 255, 200)

    )


    # ========================================================
    # LEGEND
    # ========================================================

    draw_emergency_legend(
        frame
    )


    # ========================================================
    # COOLDOWN
    # ========================================================

    if (

        time.time()
        - last_call_time
        < CALL_COOLDOWN

    ):

        remaining = (

            CALL_COOLDOWN

            - (

                time.time()
                - last_call_time

            )

        )


        draw_text(

            frame,

            f"Cooldown: {remaining:.1f}s",

            (400, 115),

            0.7,

            (0, 165, 255)

        )


    # ========================================================
    # RECORDING
    # ========================================================

    if recording and out is not None:

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

            (400, 150),

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


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        frame

    )


    key = cv2.waitKey(
        1
    ) & 0xFF


    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

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


try:

    pygame.mixer.quit()

except Exception:

    pass


print()
print(
    "AISA stopped."
)