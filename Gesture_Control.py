"""
Gesture-Controlled Virtual Mouse
Author: Your Name
Description: Controls the mouse pointer using hand gestures detected by a webcam.
Libraries: OpenCV, MediaPipe, PyAutoGUI
"""

import cv2
import mediapipe as mp
import pyautogui

# ===============================
# Initialize MediaPipe and PyAutoGUI
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Variables for smoothing cursor movement
smoothening = 7
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# ===============================
# Main Loop
# ===============================
while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image horizontally for natural interaction
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
            thumb_tip = hand_landmarks.landmark[4]         # Thumb tip

            # Convert to screen coordinates
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Smooth cursor movement
            curr_x = prev_x + (x - prev_x) / smoothening
            curr_y = prev_y + (y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click detection (Thumb close to Index)
            thumb_x = int(thumb_tip.x * screen_width)
            thumb_y = int(thumb_tip.y * screen_height)

            distance = abs(thumb_x - x) + abs(thumb_y - y)
            if distance < 40:  # Threshold for click
                pyautogui.click()
                pyautogui.sleep(0.2)  # Prevent multiple clicks

            # Draw hand landmarks on webcam feed
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the video feed
    cv2.imshow("Gesture Controlled Mouse", img)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
