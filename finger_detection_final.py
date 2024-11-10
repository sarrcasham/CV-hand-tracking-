import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a deque to store recent finger positions
finger_positions = deque(maxlen=10)

def get_direction(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude < 5:  # Threshold for movement
        return "Stationary"
    
    angle = np.arctan2(dy, dx) * 180 / np.pi
    
    if -45 <= angle < 45:
        return "Right"
    elif 45 <= angle < 135:
        return "Down"
    elif -135 <= angle < -45:
        return "Up"
    else:
        return "Left"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)

            # Add current position to the deque
            finger_positions.append((x, y))

            # Draw a circle at the index finger tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Determine direction if we have enough positions
            if len(finger_positions) == 10:
                start_pos = finger_positions[0]
                end_pos = finger_positions[-1]
                direction = get_direction(start_pos, end_pos)

                # Display the direction
                cv2.putText(frame, f"Direction: {direction}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw a line showing the movement
                cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Finger Direction Tracking', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()