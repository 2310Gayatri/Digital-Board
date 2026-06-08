<div align="center">

# ✋ Digital Board
### AI-Powered Virtual Whiteboard Using Hand Gesture Recognition and Computer Vision

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A real-time hand gesture controlled virtual whiteboard built with
Python, OpenCV, and MediaPipe. Draw, erase, and create shapes using
just your hand — no mouse, stylus or touchscreen needed!

</div>

---

## 📸 Demo Video

> 🎥 Video Will be shared Soon!

---

## ✨ Features

- ✍️ **Real-time Drawing** — Draw on a virtual canvas using hand gestures via webcam
- 🎨 **Color Selection** — 5 colors selectable via keyboard shortcuts (1–5)
- 🖊️ **Multiple Pen Tools** — Normal, Dotted, Highlighter and Spray paint
- 📏 **Pen Thickness Control** — 5 thickness levels selectable via hover gesture
- 🔷 **Smart Shape Snapping** — Automatically snaps rough outlines to perfect circle, rectangle or triangle
- 🧹 **Erasing Mode** — Erase canvas content using 4+ finger gesture
- 💾 **Save Canvas** — Export your drawing as a PNG image
- 🌐 **Flask Web Interface** — Access the board from any browser on local network
- 🔐 **Login System** — Secure session-based login page protects access
- 📷 **Dual Stream** — Camera feed and canvas shown side by side in browser

---

## 🖐️ Gesture Guide

| Fingers | Gesture | Action |
|---|---|---|
| ✌️ 2 fingers | Index + Middle up | Writing / Drawing mode |
| 🤟 3 fingers | Index + Middle + Ring up | Shape drawing mode |
| ✊ Fist | All fingers closed | Snap and finalize shape |
| 🖐️ 4+ fingers | All fingers up | Erase / Wipe mode |

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` | Color → Red |
| `2` | Color → Green |
| `3` | Color → Blue |
| `4` | Color → Yellow |
| `5` | Color → White |
| `S` | Save canvas as PNG |
| `C` | Clear entire canvas |
| `ESC` | Quit desktop app |

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core programming language |
| OpenCV | 4.8.0 | Webcam access and image processing |
| MediaPipe | 0.10.11 | Hand landmark detection and tracking |
| Flask | 3.0.0 | Web server and browser interface |
| NumPy | 1.24+ | Image array and canvas handling |
| HTML5 / CSS3 / JS | — | Frontend web interface |

---

## 📁 Project Structure
Digital-Board/
│
├── app.py                  → Flask web server
├── digital_board_v2.py     → Desktop application
├── collect_data.py         → Gesture data collection script
├── train_model.py          → KNN model training script
├── requirements.txt        → Python dependencies
├── README.md               → Project documentation
├── .gitignore              → Git ignore rules
│
├── templates/
│   ├── login.html          → Login page
│   └── index.html          → Main board page
│
└── static/
├── style.css           → Main board styling
└── login.css           → Login page styling

---

## ⚙️ Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/2310Gayatri/Digital-Board.git
cd Digital-Board
```

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Application

**Option A — Desktop App:**
```bash
python digital_board_v2.py
```

**Option B — Web App:**
```bash
python app.py
```

Then open your browser and go to:
http://127.0.0.1:5000

---

## 📦 Requirements
flask
opencv-python==4.8.0.76
mediapipe==0.10.11
numpy
protobuf==4.25.3

---

## 🔐 Web App Login Credentials

| Username | Password |
|---|---|
| admin | 1234 |
| user1 | pass1 |
| user2 | pass2 |

> You can add more users in the `USERS` dictionary inside `app.py`

---

## 🖥️ How to Use — Desktop App
Step 1 → Run digital_board_v2.py
Step 2 → Two windows open side by side (Camera + Canvas)
Step 3 → Show your hand in front of webcam
Step 4 → Use gestures to draw, erase or create shapes
Step 5 → Hover finger on top bars to change tool or thickness
Step 6 → Press keyboard 1-5 to change colors
Step 7 → Press S to save, C to clear, ESC to quit

---

## 🌐 How to Use — Web App
Step 1 → Run app.py
Step 2 → Open http://127.0.0.1:5000 in browser
Step 3 → Login with username and password
Step 4 → Camera and canvas streams appear in browser
Step 5 → Use control panel buttons to change colors and tools
Step 6 → Use hand gestures in front of webcam to draw
Step 7 → Click Save or Clear buttons on the panel

---

## 🤖 How Shape Detection Works
Step 1 → Show 3 fingers to enter shape mode
Step 2 → Slowly trace your shape outline in the air
Step 3 → Close your fist (0 fingers) to snap the shape
Step 4 → OpenCV detects contours of your traced points
Step 5 → Shape is snapped to nearest perfect shape:
3 corners  → Triangle
4 corners  → Rectangle
Round      → Circle

---

## ⚠️ Common Issues and Fixes

| Problem | Fix |
|---|---|
| `module mediapipe has no attribute solutions` | Run `pip install mediapipe==0.10.11 protobuf==4.25.3` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Camera not opening | Change `VideoCapture(0)` to `VideoCapture(1)` |
| Windows not side by side | Change `half_w` and `full_h` to match your screen resolution |
| Shape not snapping | Switch from 3 fingers to fist to trigger snap |
| Web app not opening | Make sure `app.py` is running before opening browser |
| TensorFlow oneDNN warning | Add `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` at top of file |

---

## 📊 System Performance

| Module | Performance |
|---|---|
| Hand Detection | 20-25 FPS on standard laptop |
| Gesture Classification | Real-time with low latency |
| Shape Snapping | Accurate for clear outlines |
| Pen Tool Rendering | Smooth across all 4 tools |
| Web Streaming | 20-25 FPS on local network |
| Login System | Session-based secure access |

---

## 🚀 Future Improvements

- [ ] Undo / Redo support
- [ ] Voice command control
- [ ] Background image import
- [ ] Export canvas as PDF
- [ ] Multi-user collaborative drawing over network
- [ ] Deep learning based gesture recognition

---

## 👩‍💻 Team

| Name | Role                                                    |
|---|---------------------------------------------------------|
| **Gayatri Jagdale** | Team Leader — Core development, Data collection, Environment setup Flask, Shape detection |
| **Gayatri Dharmatti** | Pen tools, Color selection, Testing, Documentation      |



---


## 👩‍💻 Made By

**Gayatri Pratap Jagadale**

<div align="center">


⭐ If you like this project give it a star on GitHub ⭐

**Made with ❤️ using Python, OpenCV, MediaPipe and Flask**

</div>