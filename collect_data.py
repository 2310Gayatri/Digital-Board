import cv2
import os
import time

# Create directories for data
gestures = ["writing", "wiping"]

for gesture in gestures:
    if not os.path.exists(f"data/{gesture}"):
        os.makedirs(f"data/{gesture}")

cap = cv2.VideoCapture(0)
print("Press 'w' to collect 'writing' gesture, 'e' for 'wiping', and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 'w' = Writing | 'e' = Wiping | 'q' = Quit", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # Writing gesture
    if key == ord('w'):
        print("Collecting 'writing' gesture images...")
        for i in range(50):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 100:400]  # region of interest
            cv2.imshow("ROI - Writing", roi)
            cv2.imwrite(f"data/writing/{time.time()}.jpg", roi)
            cv2.waitKey(100)

    # Wiping gesture
    elif key == ord('e'):
        print("Collecting 'wiping' gesture images...")
        for i in range(50):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 100:400]
            cv2.imshow("ROI - Wiping", roi)
            cv2.imwrite(f"data/wiping/{time.time()}.jpg", roi)
            cv2.waitKey(100)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete ")
