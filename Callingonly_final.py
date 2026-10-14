import cv2
import time
import os
import pygame
import numpy as np
import pyautogui
import mediapipe as mp
import random
import csv

from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

# True  = testing only, NO actual phone call
# False = actual emergency call
TEST_MODE = False

HOLD_TIME = 5

CAMERA_INDEX = 0

WINDOW_NAME = "AISA - Hand Gesture Emergency App"

SIREN_SOUND = "C:\\Users\\TARMAL COMPUTERS\\siren.wav.mp3"

Q_TABLE_FILE = "aisa_q_table.npy"

DATASET_FILE = "aisa_rl_dataset.csv"


# ============================================================
# EMERGENCY NUMBERS
# ============================================================

emergency_numbers = {

    1: "9359953694",
    2: "9335587271",
    3: "8793796952",
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

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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


# ============================================================
# GLOBAL VARIABLES
# ============================================================

last_call_time = 0

call_cooldown = 5

recording = False

record_start_time = 0

out = None

current_fingers = 0

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
# DATASET COLUMNS
# ============================================================

DATASET_COLUMNS = [

    "timestamp",

    "hand_detected",

    "finger_count",

    "confidence",

    "stable",

    "hold_time",

    "action",

    "action_name",

    "reward",

    "next_hand_detected",

    "next_finger_count",

    "valid_gesture",

    "emergency_type",

    "target_label",

    "q_value",

    "outcome"

]


# ============================================================
# DATASET INITIALIZATION
# ============================================================

def initialize_dataset():

    """
    Create CSV file and header if it does not already exist.
    """

    if not os.path.exists(DATASET_FILE):

        with open(

            DATASET_FILE,

            "w",

            newline="",

            encoding="utf-8"

        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=DATASET_COLUMNS

            )

            writer.writeheader()

        print()

        print(

            "RL dataset created:",

            DATASET_FILE

        )

    else:

        print()

        print(

            "RL dataset found:",

            DATASET_FILE

        )


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):

        self.num_actions = 4

        self.q_table = {}

        # Learning rate
        self.alpha = 0.15

        # Discount factor
        self.gamma = 0.90

        # Exploration
        self.epsilon = 0.05


    # ========================================================
    # GET STATE
    # ========================================================

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


    # ========================================================
    # INITIALIZE STATE
    # ========================================================

    def initialize_state(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(

                self.num_actions,

                dtype=np.float32

            )

            # Safe starting values

            self.q_table[state][ACTION_TRIGGER] = -5.0

            self.q_table[state][ACTION_WAIT] = 1.0


    # ========================================================
    # SELECT ACTION
    # ========================================================

    def choose_action(self, state):

        self.initialize_state(state)

        # Exploration

        if random.random() < self.epsilon:

            # NEVER randomly trigger an emergency

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


    # ========================================================
    # UPDATE Q TABLE
    # ========================================================

    def update(

        self,

        state,

        action,

        reward,

        next_state

    ):

        self.initialize_state(state)

        self.initialize_state(next_state)

        current_q = (

            self.q_table[state][action]

        )

        best_next_q = np.max(

            self.q_table[next_state]

        )

        new_q = (

            current_q

            + self.alpha *

            (

                reward

                + self.gamma * best_next_q

                - current_q

            )

        )

        self.q_table[state][action] = new_q


    # ========================================================
    # SAVE Q TABLE
    # ========================================================

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


    # ========================================================
    # LOAD Q TABLE
    # ========================================================

    def load(

        self,

        filename=Q_TABLE_FILE

    ):

        if os.path.exists(filename):

            try:

                self.q_table = np.load(

                    filename,

                    allow_pickle=True

                ).item()

                print(

                    "RL Q-table loaded:",

                    filename

                )

                print(

                    "Number of learned states:",

                    len(self.q_table)

                )

            except Exception as error:

                print(

                    "Could not load Q-table:",

                    error

                )

        else:

            print(

                "No previous Q-table found."

            )

            print(

                "A new Q-table will be created."

            )


# ============================================================
# CREATE RL AGENT
# ============================================================

rl_agent = QLearningAgent()

rl_agent.load()


# ============================================================
# CREATE DATASET
# ============================================================

initialize_dataset()


# ============================================================
# TEXT DRAWING
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
# FINGER COUNTING
# ============================================================

def count_fingers(landmarks):

    """

    Count fingers using MediaPipe landmarks.

    4  = thumb tip
    8  = index tip
    12 = middle tip
    16 = ring tip
    20 = pinky tip
    """

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

    handedness,

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

        point = landmarks.landmark[tip_index]

        x = int(point.x * w)

        y = int(point.y * h)


        if i < fingers:

            color = (0, 255, 0)

        else:

            color = (100, 100, 100)


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

def draw_emergency_legend(frame):

    legend = [

        ("1  POLICE", (0, 0, 255)),

        ("2  FIRE", (255, 120, 100)),

        ("3  AMBULANCE", (255, 0, 255)),

        ("4  WOMEN HELPLINE", (0, 255, 255)),

        ("5  GENERAL HELPLINE", (0, 255, 0))

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
# REWARD FUNCTION
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
    # INVALID FIST
    # --------------------------------------------------------

    if fingers < 1 or fingers > 5:

        if action == ACTION_IGNORE:

            reward += 3

        elif action == ACTION_WAIT:

            reward += 2

        else:

            reward -= 5

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
    # STABLE VALID HAND
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


    # --------------------------------------------------------
    # UNSTABLE HAND
    # --------------------------------------------------------

    else:

        if action == ACTION_WAIT:

            reward += 4

        elif action == ACTION_IGNORE:

            reward += 1

        elif action == ACTION_TRIGGER:

            reward -= 10


    return reward


# ============================================================
# DATASET LOGGER
# ============================================================

def save_rl_sample(

    hand_detected,

    fingers,

    confidence,

    stable,

    hold_time,

    action,

    reward,

    next_hand_detected,

    next_fingers,

    valid_gesture,

    emergency_type,

    target_label,

    outcome

):

    """

    Save one RL experience to CSV.

    """

    state = rl_agent.get_state(

        hand_detected,

        fingers,

        confidence,

        stable,

        hold_time

    )

    rl_agent.initialize_state(state)


    # Q-value of selected action

    q_value = float(

        rl_agent.q_table[state][action]

    )


    row = {

        "timestamp":

            datetime.now().strftime(

                "%Y-%m-%d %H:%M:%S.%f"

            ),

        "hand_detected":

            int(hand_detected),

        "finger_count":

            int(fingers),

        "confidence":

            round(

                float(confidence),

                3

            ),

        "stable":

            int(stable),

        "hold_time":

            round(

                float(hold_time),

                2

            ),

        "action":

            int(action),

        "action_name":

            actions[action],

        "reward":

            float(reward),

        "next_hand_detected":

            int(next_hand_detected),

        "next_finger_count":

            int(next_fingers),

        "valid_gesture":

            int(valid_gesture),

        "emergency_type":

            emergency_type,

        "target_label":

            target_label,

        "q_value":

            round(

                q_value,

                4

            ),

        "outcome":

            outcome

    }


    try:

        with open(

            DATASET_FILE,

            "a",

            newline="",

            encoding="utf-8"

        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=DATASET_COLUMNS

            )

            writer.writerow(row)

    except Exception as error:

        print(

            "Dataset write error:",

            error

        )


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


    current_time = time.time()


    if fingers not in emergency_numbers:

        return "Invalid gesture"


    if (

        current_time - last_call_time

        < call_cooldown

    ):

        return "Emergency cooldown active"


    number = emergency_numbers[fingers]

    label = emergency_labels[fingers]


    print()

    print(

        "========================================"

    )

    print(

        f"EMERGENCY: {label}"

    )

    print(

        f"NUMBER: {number}"

    )

    print(

        "========================================"

    )


    # ========================================================
    # TEST MODE
    # ========================================================

    if TEST_MODE:

        print(

            "TEST MODE: Call NOT placed."

        )

        print(

            f"Would call {label}: {number}"

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


    # ========================================================
    # VIDEO RECORDING
    # ========================================================

    if out is not None:

        out.release()


    out = cv2.VideoWriter(

        "call_recording.mp4",

        cv2.VideoWriter_fourcc(*"mp4v"),

        20.0,

        (

            frame_width,

            frame_height

        )

    )


    recording = True

    record_start_time = time.time()


    return (

        f"TRIGGERED: {label}"

    )


# ============================================================
# START
# ============================================================

print()

print(

    "============================================"

)

print(

    "      AISA HAND GESTURE EMERGENCY"

)

print(

    "      MEDIAPIPE + REINFORCEMENT LEARNING"

)

print(

    "============================================"

)

print()

print(

    "TEST MODE:",

    TEST_MODE

)

print()

print(

    "Dataset:",

    DATASET_FILE

)

print(

    "Q-table:",

    Q_TABLE_FILE

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
    # MIRROR CAMERA
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

        landmarks = results.multi_hand_landmarks[0]


        draw_hand(

            frame,

            landmarks

        )


        fingers = count_fingers(

            landmarks

        )


        if results.multi_handedness:

            confidence = calculate_hand_confidence(

                results.multi_handedness[0],

                landmarks

            )

        else:

            confidence = 0.75


        draw_finger_points(

            frame,

            landmarks,

            fingers

        )


    # ========================================================
    # GESTURE STABILITY
    # ========================================================

    if hand_detected:

        if fingers == previous_fingers:

            gesture_stable_time += 1 / 30

        else:

            gesture_stable_time = 0

            previous_fingers = fingers

    else:

        gesture_stable_time = 0

        previous_fingers = 0


    stable = (

        gesture_stable_time >= 0.4

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
    # CURRENT STATE
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
    # SAFETY OVERRIDE
    # ========================================================

    # RL is NEVER allowed to call just because it
    # selected TRIGGER.
    #
    # A call requires:
    #
    # 1. MediaPipe detects a hand
    # 2. 1-5 fingers
    # 3. Confidence >= 0.60
    # 4. Gesture stable
    # 5. Gesture held for HOLD_TIME
    # 6. RL chooses TRIGGER
    #
    # ========================================================

    if not valid_gesture:

        # Prevent accidental RL trigger

        if action == ACTION_TRIGGER:

            action = ACTION_WAIT


    # ========================================================
    # REWARD
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

    # For this frame-based implementation, the current
    # observation is used as the immediate next observation.
    #
    # The CSV still records the transition information.

    next_hand_detected = hand_detected

    next_fingers = fingers


    next_state = rl_agent.get_state(

        next_hand_detected,

        next_fingers,

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
    # DATASET LABEL
    # ========================================================

    if not hand_detected:

        target_label = (

            "NEGATIVE_EXTERNAL_OBJECT_OR_EMPTY"

        )

        emergency_type = ""

        outcome = "IGNORED"


    elif fingers == 0:

        target_label = (

            "NEGATIVE_INVALID_HAND"

        )

        emergency_type = ""

        outcome = "WAIT"


    elif valid_gesture:

        target_label = (

            "POSITIVE_VALID_HAND_GESTURE"

        )

        emergency_type = (

            emergency_labels[fingers]

        )

        if action == ACTION_TRIGGER:

            outcome = (

                "TRIGGER_CANDIDATE"

            )

        else:

            outcome = (

                "VALID_HAND_WAITING"

            )


    else:

        target_label = (

            "NEGATIVE_UNSTABLE_GESTURE"

        )

        emergency_type = (

            emergency_labels.get(

                fingers,

                ""

            )

        )

        outcome = "WAIT"


    # ========================================================
    # SAVE RL EXPERIENCE
    # ========================================================

    save_rl_sample(

        hand_detected=hand_detected,

        fingers=fingers,

        confidence=confidence,

        stable=stable,

        hold_time=gesture_stable_time,

        action=action,

        reward=reward,

        next_hand_detected=next_hand_detected,

        next_fingers=next_fingers,

        valid_gesture=valid_gesture,

        emergency_type=emergency_type,

        target_label=target_label,

        outcome=outcome

    )


    # ========================================================
    # TRIGGER / CONFIRMATION
    # ========================================================

    if (

        valid_gesture
        and not confirming

    ):

        confirming = True

        confirmed_fingers = fingers


    # ========================================================
    # CONFIRMATION
    # ========================================================

    if confirming:

        label = (

            emergency_labels[confirmed_fingers]

        )


        status = (

            f"CONFIRM {label} - "

            f"KEEP {confirmed_fingers} FINGERS UP"

        )

        status_color = (

            0,

            255,

            0

        )


        # ----------------------------------------------------
        # HAND DISAPPEARED
        # ----------------------------------------------------

        if not hand_detected:

            confirming = False

            confirmed_fingers = 0


        # ----------------------------------------------------
        # WRONG GESTURE
        # ----------------------------------------------------

        elif fingers != confirmed_fingers:

            confirming = False

            confirmed_fingers = 0


        # ----------------------------------------------------
        # FINAL HOLD
        # ----------------------------------------------------

        elif gesture_stable_time >= (

            HOLD_TIME + 0.5

        ):

            status = call_emergency(

                confirmed_fingers,

                frame

            )


            confirming = False

            confirmed_fingers = 0


    else:

        # ====================================================
        # NORMAL STATUS
        # ====================================================

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

                HOLD_TIME - gesture_stable_time

            )


            status = (

                f"{fingers} FINGERS - HOLD "

                f"{remaining:.1f}s"

            )

            status_color = (

                0,

                255,

                255

            )


        else:

            status = (

                f"VALID {fingers}-FINGER GESTURE"

            )

            status_color = (

                0,

                255,

                0

            )


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
    # FINGER COUNT
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
    # REWARD
    # ========================================================

    draw_text(

        frame,

        f"Reward: {reward}",

        (400, 190),

        0.6,

        (255, 220, 150)

    )


    # ========================================================
    # Q-VALUE
    # ========================================================

    rl_agent.initialize_state(state)

    selected_q = float(

        rl_agent.q_table[state][action]

    )


    draw_text(

        frame,

        f"Q-value: {selected_q:.2f}",

        (400, 225),

        0.6,

        (180, 255, 180)

    )


    # ========================================================
    # EMERGENCY LEGEND
    # ========================================================

    draw_emergency_legend(

        frame

    )


    # ========================================================
    # COOLDOWN
    # ========================================================

    if (

        time.time() - last_call_time

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

            (400, 115),

            0.7,

            (0, 165, 255)

        )


    # ========================================================
    # TEST MODE DISPLAY
    # ========================================================

    if TEST_MODE:

        draw_text(

            frame,

            "TEST MODE - NO REAL CALL",

            (400, 75),

            0.65,

            (0, 200, 255)

        )


    # ========================================================
    # RECORDING
    # ========================================================

    if recording and out is not None:

        out.write(frame)


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
    # SHOW
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        frame

    )


    # ========================================================
    # ESC
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

print()

print(

    "Saving RL model..."

)

rl_agent.save()


if out is not None:

    out.release()


cap.release()

hands.close()

cv2.destroyAllWindows()


try:

    pygame.mixer.quit()

except:

    pass


print()

print(

    "AISA stopped."

)

print()

print(

    "Generated files:"

)

print(

    "1.",

    Q_TABLE_FILE

)

print(

    "2.",

    DATASET_FILE

)

print()