import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import logging
import collections

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

# Initialize logging
logging.basicConfig(filename='gesture_control.log', level=logging.INFO)

# Initialize a deque to store hand positions for smooth mouse movement
position_history = collections.deque(maxlen=5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Function to detect wrist bend (click gesture)
def is_wrist_bend(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    angle = calculate_angle(
        [wrist.x, wrist.y],
        [index_mcp.x, index_mcp.y],
        [index_tip.x, index_tip.y]
    )

    if angle < 120:  # Adjust threshold as needed
        return True
    return False

# Function to smooth mouse movement
def smooth_mouse_movement(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    current_x = wrist.x
    current_y = wrist.y

    position_history.append((current_x, current_y))
    avg_x = sum(pos[0] for pos in position_history) / len(position_history)
    avg_y = sum(pos[1] for pos in position_history) / len(position_history)

    cursor_x = int(avg_x * screen_width)
    cursor_y = int(avg_y * screen_height)

    return cursor_x, cursor_y

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

                # Smooth mouse movement
                cursor_x, cursor_y = smooth_mouse_movement(hand_landmarks)
                pyautogui.moveTo(cursor_x, cursor_y)

                # Detect wrist bend (click gesture)
                if is_wrist_bend(hand_landmarks):
                    pyautogui.click()
                    logging.info("Wrist bend detected! Click triggered.")
                    cv2.putText(frame, "Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Tracking and Gesture Control', frame)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()