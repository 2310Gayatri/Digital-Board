import cv2
import mediapipe as mp
import numpy as np
import pytesseract

# ✅ Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'D:\Digital Board System\Digital_Board_System\tesseract-ocr-w64-setup-5.5.0.20241111.exe'

# ── MediaPipe Setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ── Webcam ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ── State Variables ──────────────────────────────────────────
canvas         = None
prev_x, prev_y = 0, 0
mode           = "None"
shape_points   = []
current_color  = (255, 255, 255)
pen_thickness  = 5
pen_tool       = "Normal"
dot_counter    = 0

# ── Color Options (keyboard 1-5) ──────────────────────────────
colors = {
    '1': ("Red",    (0, 0, 255)),
    '2': ("Green",  (0, 255, 0)),
    '3': ("Blue",   (255, 0, 0)),
    '4': ("Yellow", (0, 255, 255)),
    '5': ("White",  (255, 255, 255)),
}

# ── Pen Tool Bar (hover finger) ───────────────────────────────
pen_tools    = ["Normal", "Dotted", "Highlighter", "Spray"]
TOOL_BOX_W   = 160
TOOL_BOX_H   = 60
TOOL_BAR_Y   = 0   # top of frame

# ── Thickness Bar (hover finger) ─────────────────────────────
thickness_levels = [2, 5, 8, 12, 18]
THICK_BOX_W  = 160
THICK_BOX_H  = 55
THICK_BAR_Y  = TOOL_BOX_H   # just below tool bar

# total UI height
UI_HEIGHT = TOOL_BOX_H + THICK_BOX_H

# ── Screen Layout ─────────────────────────────────────────────
# Change to match YOUR laptop resolution:
# 1920x1080 → half_w=960, full_h=1080
# 1536x864  → half_w=768, full_h=864
# 1366x768  → half_w=683, full_h=768
# 1280x800  → half_w=640, full_h=800
half_w = 768
full_h = 864


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
    if len(points) < 5:
        return
    pts     = np.array(points, dtype=np.int32)
    hull    = cv2.convexHull(pts)
    epsilon = 0.04 * cv2.arcLength(hull, True)
    approx  = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) == 3:
        cv2.polylines(canvas, [approx], True, color, thickness)
        print("Shape snapped: Triangle")
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        print("Shape snapped: Rectangle")
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, thickness)
        print("Shape snapped: Circle")


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


def draw_ui(frame, pen_tool, pen_thickness, mode, current_color):
    """Draw tool bar + thickness bar + status on frame."""

    # ── Pen Tool Bar (Row 1) ──────────────────────────────────
    for i, tool in enumerate(pen_tools):
        x1 = i * TOOL_BOX_W
        x2 = x1 + TOOL_BOX_W
        y1 = TOOL_BAR_Y
        y2 = TOOL_BAR_Y + TOOL_BOX_H
        bg = (50, 50, 50)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        if tool == pen_tool:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, tool, (x1 + 10, y2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # label
    cv2.putText(frame, "TOOLS", (len(pen_tools) * TOOL_BOX_W + 10, TOOL_BAR_Y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # ── Thickness Bar (Row 2) ─────────────────────────────────
    for i, t in enumerate(thickness_levels):
        x1 = i * THICK_BOX_W
        x2 = x1 + THICK_BOX_W
        y1 = THICK_BAR_Y
        y2 = THICK_BAR_Y + THICK_BOX_H
        bg = (35, 35, 35)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
        if t == pen_thickness:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        mid_y = (y1 + y2) // 2
        cv2.line(frame, (x1 + 15, mid_y), (x2 - 15, mid_y), (255, 255, 255), t)

    # label
    cv2.putText(frame, "SIZE", (len(thickness_levels) * THICK_BOX_W + 10, THICK_BAR_Y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # ── Status line ───────────────────────────────────────────
    cv2.putText(frame, f"Mode:{mode}  Tool:{pen_tool}  Size:{pen_thickness}",
                (15, UI_HEIGHT + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, current_color, 2)

    # ── Color dot (bottom left) ───────────────────────────────
    cv2.circle(frame, (30, UI_HEIGHT + 80), 18, current_color, -1)
    cv2.circle(frame, (30, UI_HEIGHT + 80), 18, (255, 255, 255), 2)
    cv2.putText(frame, "1-5", (55, UI_HEIGHT + 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ════════════════════════════════════════════════════════════
#  STARTUP MESSAGES
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("   DIGITAL BOARD  —  Final Version")
print("=" * 55)
print()
print("  GESTURES:")
print("    2 fingers   →  Writing mode")
print("    3 fingers   →  Shape drawing mode")
print("    4+ fingers  →  Wiping / Erasing")
print()
print("  HOVER FINGER on bars at top to select:")
print("    Row 1  →  Pen Tool  (Normal/Dotted/Highlighter/Spray)")
print("    Row 2  →  Thickness (2 / 5 / 8 / 12 / 18)")
print()
print("  KEYBOARD — Colors only:")
print("    1 → Red    2 → Green   3 → Blue")
print("    4 → Yellow 5 → White")
print()
print("  OTHER KEYS:")
print("    S → Save canvas    T → OCR    C → Clear    ESC → Quit")
print("=" * 55)


# ════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════
while True:
    success, frame = cap.read()
    if not success:
        break

    frame    = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    # ── Hand Detection ───────────────────────────────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)
            ix = int(hand_landmarks.landmark[8].x * w)
            iy = int(hand_landmarks.landmark[8].y * h)

            # ── Zone 1 : Pen Tool Bar ────────────────────────
            if iy < TOOL_BOX_H:
                idx = ix // TOOL_BOX_W
                if idx < len(pen_tools):
                    pen_tool = pen_tools[idx]
                    prev_x, prev_y = 0, 0

            # ── Zone 2 : Thickness Bar ───────────────────────
            elif iy < UI_HEIGHT:
                idx = ix // THICK_BOX_W
                if idx < len(thickness_levels):
                    pen_thickness = thickness_levels[idx]
                    prev_x, prev_y = 0, 0

            # ── Zone 3 : Drawing Area ────────────────────────
            else:
                prev_mode = mode

                if finger_count == 2:
                    mode = "Writing"
                elif finger_count == 3:
                    mode = "Shape"
                elif finger_count >= 4:
                    mode = "Wiping"
                else:
                    mode = "None"

                # Snap shape when leaving shape mode
                if prev_mode == "Shape" and mode != "Shape":
                    detect_and_draw_shape(canvas, shape_points,
                                          current_color, pen_thickness)
                    shape_points = []

                # Writing
                if mode == "Writing":
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = ix, iy
                    draw_stroke(canvas, prev_x, prev_y, ix, iy,
                                current_color, pen_thickness, pen_tool)
                    prev_x, prev_y = ix, iy

                # Shape
                elif mode == "Shape":
                    shape_points.append((ix, iy))
                    cv2.circle(frame, (ix, iy), 3, current_color, -1)
                    prev_x, prev_y = 0, 0

                # Wiping
                elif mode == "Wiping":
                    cv2.circle(canvas, (ix, iy), 60, (0, 0, 0), -1)
                    prev_x, prev_y = 0, 0

                else:
                    prev_x, prev_y = 0, 0

    # ── Draw UI on frame ─────────────────────────────────────
    draw_ui(frame, pen_tool, pen_thickness, mode, current_color)

    # ── Merge canvas onto frame ───────────────────────────────
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv      = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv         = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame       = cv2.bitwise_and(frame, inv)
    frame       = cv2.bitwise_or(frame, canvas)

    # ── Resize & place side by side ──────────────────────────
    display_frame  = cv2.resize(frame,  (half_w, full_h))
    display_canvas = cv2.resize(canvas, (half_w, full_h))

    cv2.imshow("Digital Board", display_frame)
    cv2.imshow("Canvas",        display_canvas)

    cv2.moveWindow("Digital Board", 0,      0)
    cv2.moveWindow("Canvas",        half_w, 0)

    # ── Keyboard Input ────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    ch  = chr(key).lower() if key != 255 else ''

    # Colors only via keyboard
    if ch in colors:
        current_color = colors[ch][1]
        print(f"Color → {colors[ch][0]}")

    # Save
    elif ch == 's':
        cv2.imwrite("canvas_saved.png", canvas)
        print("Canvas saved as canvas_saved.png")

    # OCR
    elif ch == 't':
        print("\nRunning OCR...")
        gray      = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        text      = pytesseract.image_to_string(thresh)
        print("Recognized Text:\n", text if text.strip() else "(nothing detected)")

    # Clear
    elif ch == 'c':
        canvas = np.zeros((h, w, 3), np.uint8)
        print("Canvas cleared!")

    # Quit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()