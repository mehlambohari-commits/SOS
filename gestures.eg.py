import cv2
import pyautogui
import time
import os
import pygame
import numpy as np

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

# Simple hand detection using HSV skin color
def detect_hand(frame):
    """Detect hand using HSV skin color detection"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Also try another skin tone range
    lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask, mask2)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# Count fingers using contour analysis
def count_fingers(mask):
    """Estimate finger count from hand contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Get the largest contour (hand)
    hand_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(hand_contour) < 1000:
        return 0
    
    # Use convex hull to detect fingers
    hull = cv2.convexHull(hand_contour)
    
    # Count defects to estimate fingers (simplified)
    # Typically: closed fist = 0, open hand = 5
    if len(hull) > 20:
        return min(5, len(hull) // 4)
    
    return 0

# Function to count fingers (for old mediapipe API)
def count_fingers_old(hand_landmarks):
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
        ("1: Police [935995369]", (0, 0, 255)),
        ("2: Fire [9335587271]", (255, 100, 100)),
        ("3: Ambulance [8878010010]", (255, 0, 255)),
        ("4: Women Helpline [8530496089]", (0, 255, 255)),
        ("5: General Helpline [6232481125s]", (0, 255, 0))
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
hand_detected_frames = 0

def call_emergency(fingers, frame):
    global last_call_time, call_confirmation_time, confirming_call, confirmed_fingers, recording, record_start_time, out

    current_time = time.time()
    emergency_numbers = {1: "9359953694", 2: "9335587271", 3: "8878010010", 4: "8530496089", 5: "6232481125"}
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
            return f"Confirm call to {emergency_labels[fingers]} [{emergency_numbers[fingers]}]? Hold 3s"
        elif confirming_call and fingers == confirmed_fingers:
            if current_time - call_confirmation_time > 3:
                number = emergency_numbers[confirmed_fingers]
                label = emergency_labels[confirmed_fingers]

                print(f"📞 Calling Emergency Number: {label} [{number}]")

                try:
                    pygame.mixer.music.load(siren_sound)
                    pygame.mixer.music.play()
                except:
                    print(f"Warning: Could not play siren sound from {siren_sound}")

                os.system("start ms-phone://")
                time.sleep(3)

                pyautogui.write(number)
                time.sleep(1)
                pyautogui.press("enter")

                last_call_time = current_time
                confirming_call = False

                out = cv2.VideoWriter('call_recording.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))
                recording = True
                record_start_time = time.time()

                return f"Called {label} [{number}]"
            else:
                remain = 3 - int(current_time - call_confirmation_time)
                return f"Confirming: {remain}s left for {emergency_labels[confirmed_fingers]}"
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
    
    # Detect hand
    mask = detect_hand(frame)
    
    # Count fingers
    fingers = count_fingers(mask)
    
    draw_decorative_text(frame, "Show 1-5 fingers for Emergency Call", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), (0, 0, 0))

    status_text = "No hand detected"
    
    if cv2.countNonZero(mask) > 500:  # Hand detected if enough white pixels
        hand_detected_frames += 1
        if hand_detected_frames > 5:  # Require detection for 5 frames to confirm
            status_text = call_emergency(fingers, frame)
    else:
        hand_detected_frames = 0

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

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()