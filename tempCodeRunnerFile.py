import cv2
import mediapipe as mp
import pyautogui
import time
import os
import pygame

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize pygame for siren sound
pygame.mixer.init()
siren_sound = "C:\\Users\\TARMAL COMPUTERS\\siren.wav.mp3"  # Updated file path

# Open Webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Gesture Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Set up video recording
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = None
recording = False
record_start_time = 0

# Function to count fingers
def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)

# Draw decorated text for visibility
def draw_decorative_text(img, text, pos, font, scale, color, bg_color):
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), font, scale, bg_color, 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, 2, cv2.LINE_AA)

# Draw emergency gesture guide
def draw_emergency_legend(frame):
    legend = [
        ("1: Police [100]", (0, 0, 255)),
        ("2: Fire [101]", (255, 100, 100)),
        ("3: Ambulance [102]", (255, 0, 255)),
        ("4: Women Helpline [1091]", (0, 255, 255)),
        ("5: General Helpline [112]", (0, 255, 0))
    ]
    y = 240
    for text, color in legend:
        draw_decorative_text(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, (20, 20, 20))
        y += 35

# Variables for call control
last_call_time = 0
call_cooldown = 2
call_confirmation_time = 0
confirming_call = False
confirmed_fingers = 0
first_hand_locked = False
first_hand_landmark_x = None

def call_emergency(fingers, frame):
    global last_call_time, call_confirmation_time, confirming_call, confirmed_fingers, recording, record_start_time, out

    current_time = time.time()
    emergency_numbers = {1: "100", 2: "101", 3: "102", 4: "1091", 5: "112"}
    emergency_labels = {
        1: "Police",
        2: "Fire",
        3: "Ambulance",
        4: "Women Helpline",
        5: "General Helpline"
    }

    if current_time - last_call_time > call_cooldown:
        if fingers in emergency_numbers and not confirming_call:
            confirming_call = True
            call_confirmation_time = current_time
            confirmed_fingers = fingers
            return f"Confirm call to {emergency_labels[fingers]} [{emergency_numbers[fingers]}]? Hold 2s"
        elif confirming_call and fingers == confirmed_fingers:
            if current_time - call_confirmation_time > 3:
                number = emergency_numbers[confirmed_fingers]
                label = emergency_labels[confirmed_fingers]

                print(f"📞 Calling Emergency Number: {label} [{number}]")

                pygame.mixer.music.load(siren_sound)
                pygame.mixer.music.play()

                os.system("start ms-phone://")
                time.sleep(3)

                pyautogui.write(number)
                time.sleep(1)
                pyautogui.press("enter")

                last_call_time = current_time
                confirming_call = False
                save_path = r"C:\Users\TARMAL COMPUTERS\Videos\call_recording.mp4"
                out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

                recording = True
                record_start_time = time.time()

                return f"Called {label} [{number}]"
            else:
                remain = 3 - int(current_time - call_confirmation_time)
                return f"Confirming: {remain}s left for {emergency_labels[fingers]}"
        else:
            confirming_call = False
            return f"Fingers detected: {fingers}"

    return f"Fingers detected: {fingers}"

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    draw_decorative_text(frame, "Show 1-5 fingers for Emergency Call", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), (0, 0, 0))

    status_text = "No hand detected"

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if not first_hand_locked:
                first_hand_locked = True
                first_hand_landmark_x = hand_landmarks.landmark[0].x
            if abs(hand_landmarks.landmark[0].x - first_hand_landmark_x) < 0.1:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = count_fingers(hand_landmarks)
                status_text = call_emergency(fingers, frame)

                for tip in [4, 8, 12, 16, 20]:
                    x = int(hand_landmarks.landmark[tip].x * frame.shape[1])
                    y = int(hand_landmarks.landmark[tip].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
                break
    else:
        first_hand_locked = False

    draw_decorative_text(frame, status_text, (50, 100),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 255, 200), (30, 30, 30))

    if time.time() - last_call_time < call_cooldown:
        remain = int(call_cooldown - (time.time() - last_call_time))
        draw_decorative_text(frame, f"Cooldown: {remain}s", (50, 150),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), (30, 30, 30))

    if recording:
        out.write(frame)
        if time.time() - record_start_time > 600:
            recording = False
            out.release()
            print("Recording saved as call_recording.mp4")

    draw_emergency_legend(frame)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()