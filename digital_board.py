import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
mode = "None"  # Writing or Wiping

def count_fingers(hand_landmarks):
    """Counts how many fingers are up"""
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky tips
    finger_pip = [6, 10, 14, 18]   # corresponding lower joints
    thumb_tip, thumb_ip = 4, 3

    fingers = []

    # Thumb (compare x positions because it's sideways)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (compare y positions)
    for tip, pip in zip(finger_tips, finger_pip):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)  # total fingers up


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count fingers
            finger_count = count_fingers(hand_landmarks)

            # Get index finger tip position
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Determine mode
            if finger_count == 2:
                mode = "Writing"
            elif finger_count >= 4:
                mode = "Wiping"
            else:
                mode = "None"

            # Writing mode
            if mode == "Writing":
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), 5)
                prev_x, prev_y = x, y

            # Wiping mode
            elif mode == "Wiping":
                cv2.circle(canvas, (x, y), 60, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

            cv2.putText(frame, f"Mode: {mode}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Combine canvas and frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Digital Board", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


