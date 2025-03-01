import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range (usually from 0.0 to 1.0)
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]
max_vol = volume_range[1]

# Function to detect thumbs up gesture
def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Check if the thumb is above the other fingers
    if (thumb_tip.y < index_tip.y and
        thumb_tip.y < middle_tip.y and
        thumb_tip.y < ring_tip.y and
        thumb_tip.y < pinky_tip.y):
        return True
    return False

# Function to detect pinch gesture
def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the thumb and index finger are close
    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return True
    return False

# Function to control volume based on hand position
def control_volume(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate the distance between thumb and index finger
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    # Map distance to volume level
    vol_level = np.interp(distance, [0.0, 0.2], [min_vol, max_vol])
    volume.SetMasterVolumeLevel(vol_level, None)

# Function to control scrolling
def control_scroll(hand_landmarks):
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate vertical movement
    if middle_tip.y < wrist.y - 0.1:  # Hand moved up
        pyautogui.scroll(1)  # Scroll up
    elif middle_tip.y > wrist.y + 0.1:  # Hand moved down
        pyautogui.scroll(-1)  # Scroll down

# Function to simulate mouse click
def simulate_click(hand_landmarks):
    if is_pinch(hand_landmarks):
        pyautogui.click()
        print("Click!")

# Set up MediaPipe Hands
with mp_hands.Hands(
    min_detection_confidence=0.7,  # Confidence threshold for detection
    min_tracking_confidence=0.7    # Confidence threshold for tracking
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect gestures and control actions
                if is_thumbs_up(hand_landmarks):
                    cv2.putText(frame, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Thumbs up detected!")

                if is_pinch(hand_landmarks):
                    cv2.putText(frame, "Pinch!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    print("Pinch detected!")

                # Control volume, scroll, and click
                control_volume(hand_landmarks)
                control_scroll(hand_landmarks)
                simulate_click(hand_landmarks)

                # Move the mouse cursor based on hand position
                palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                cursor_x = int(palm_base.x * screen_width)
                cursor_y = int(palm_base.y * screen_height)
                pyautogui.moveTo(cursor_x, cursor_y)

                # Visual feedback for volume control
                vol_level = volume.GetMasterVolumeLevel()
                cv2.putText(frame, f"Volume: {int(np.interp(vol_level, [min_vol, max_vol], [0, 100]))}%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Display the frame
        cv2.imshow('Hand Tracking and Gesture Control', frame)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()