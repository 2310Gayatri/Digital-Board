import cv2
import mediapipe as mp
import numpy as np
import base64
import threading
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# ── MediaPipe Setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw  = mp.solutions.drawing_utils

# ── Webcam ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── State Variables ──────────────────────────────────────────
canvas         = None
prev_x, prev_y = 0, 0
mode           = "None"
shape_points   = []
shape_drawing  = False
current_color  = (255, 255, 255)
pen_thickness  = 5
pen_tool       = "Normal"
dot_counter    = 0
lock           = threading.Lock()

# ── Color Map ─────────────────────────────────────────────────
color_map = {
    "Red":    (0, 0, 255),
    "Green":  (0, 255, 0),
    "Blue":   (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White":  (255, 255, 255),
}

# ── Thickness Map ─────────────────────────────────────────────
thickness_map = {
    "XS": 2,
    "S":  5,
    "M":  8,
    "L":  12,
    "XL": 18,
}


# ════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pip  = [6, 10, 14, 18]
    thumb_tip, thumb_ip = 4, 3
    fingers = []

    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip, pip in zip(finger_tips, finger_pip):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def detect_and_draw_shape(canvas, points, color, thickness):
    if len(points) < 10:
        return
    pts     = np.array(points, dtype=np.int32)
    hull    = cv2.convexHull(pts)
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 3:
        cv2.polylines(canvas, [approx], True, color, thickness)
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, thickness)


def draw_stroke(canvas, px, py, x, y, color, thickness, tool):
    global dot_counter
    if tool == "Normal":
        cv2.line(canvas, (px, py), (x, y), color, thickness)
    elif tool == "Dotted":
        dot_counter += 1
        if dot_counter % 4 == 0:
            cv2.circle(canvas, (x, y), thickness // 2 + 1, color, -1)
    elif tool == "Highlighter":
        overlay = canvas.copy()
        cv2.line(overlay, (px, py), (x, y), color, thickness * 4)
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
    elif tool == "Spray":
        for _ in range(25):
            ox = int(np.random.normal(0, thickness * 2))
            oy = int(np.random.normal(0, thickness * 2))
            sx = np.clip(x + ox, 0, canvas.shape[1] - 1)
            sy = np.clip(y + oy, 0, canvas.shape[0] - 1)
            canvas[sy, sx] = color


def process_frame(frame):
    global canvas, prev_x, prev_y, mode
    global shape_points, shape_drawing
    global current_color, pen_thickness, pen_tool

    frame    = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)
            ix = int(hand_landmarks.landmark[8].x * w)
            iy = int(hand_landmarks.landmark[8].y * h)

            if finger_count == 3:
                mode          = "Shape"
                shape_drawing = True
                shape_points.append((ix, iy))
                cv2.circle(frame, (ix, iy), 4, current_color, -1)
                prev_x, prev_y = 0, 0

            elif finger_count <= 1 and shape_drawing:
                with lock:
                    detect_and_draw_shape(canvas, shape_points,
                                          current_color, pen_thickness)
                shape_points  = []
                shape_drawing = False
                mode          = "None"
                prev_x, prev_y = 0, 0

            elif finger_count == 2:
                mode          = "Writing"
                shape_drawing = False
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy
                with lock:
                    draw_stroke(canvas, prev_x, prev_y, ix, iy,
                                current_color, pen_thickness, pen_tool)
                prev_x, prev_y = ix, iy

            elif finger_count >= 4:
                mode          = "Wiping"
                shape_drawing = False
                shape_points  = []
                with lock:
                    cv2.circle(canvas, (ix, iy), 60, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

            else:
                mode = "None"
                prev_x, prev_y = 0, 0

            # Show mode on frame
            cv2.putText(frame, f"Mode: {mode}", (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

    else:
        if shape_drawing and len(shape_points) >= 10:
            with lock:
                detect_and_draw_shape(canvas, shape_points,
                                      current_color, pen_thickness)
        shape_points  = []
        shape_drawing = False
        prev_x, prev_y = 0, 0

    # Merge canvas onto frame
    with lock:
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv    = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame  = cv2.bitwise_and(frame, inv)
    with lock:
        frame = cv2.bitwise_or(frame, canvas)

    return frame


def generate_camera():
    """Stream camera+drawing frames to browser."""
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


def generate_canvas():
    """Stream canvas-only frames to browser."""
    while True:
        success, _ = cap.read()
        if not success:
            break
        with lock:
            if canvas is not None:
                c = canvas.copy()
            else:
                c = np.zeros((720, 1280, 3), np.uint8)
        _, buffer = cv2.imencode('.jpg', c,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_camera(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/canvas_feed')
def canvas_feed():
    return Response(generate_canvas(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_color', methods=['POST'])
def set_color():
    global current_color
    data  = request.json
    name  = data.get('color', 'White')
    current_color = color_map.get(name, (255, 255, 255))
    return jsonify({"status": "ok", "color": name})


@app.route('/set_tool', methods=['POST'])
def set_tool():
    global pen_tool
    data     = request.json
    pen_tool = data.get('tool', 'Normal')
    return jsonify({"status": "ok", "tool": pen_tool})


@app.route('/set_thickness', methods=['POST'])
def set_thickness():
    global pen_thickness
    data          = request.json
    key           = data.get('size', 'S')
    pen_thickness = thickness_map.get(key, 5)
    return jsonify({"status": "ok", "thickness": pen_thickness})


@app.route('/clear', methods=['POST'])
def clear():
    global canvas
    with lock:
        if canvas is not None:
            canvas[:] = 0
    return jsonify({"status": "cleared"})


@app.route('/save', methods=['POST'])
def save():
    with lock:
        if canvas is not None:
            cv2.imwrite("canvas_saved.png", canvas)
    return jsonify({"status": "saved"})


@app.route('/status')
def status():
    return jsonify({
        "mode":      mode,
        "tool":      pen_tool,
        "thickness": pen_thickness,
    })


# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 45)
    print("  Digital Board Website")
    print("  Open browser → http://127.0.0.1:5000")
    print("=" * 45)
    app.run(debug=False, threaded=True)