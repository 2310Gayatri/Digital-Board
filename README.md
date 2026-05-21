<div align="center">

# ✋ Digital Board
### Hand Gesture Controlled Drawing Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A real-time hand gesture controlled digital drawing board built with
Python, OpenCV, and MediaPipe. Draw, erase, and create shapes using
just your hand — no mouse or stylus needed!

</div>

---

## 📸 Preview

> Open the web app at `http://127.0.0.1:5000` after running `app.py`

| Camera + Drawing | Canvas Only |
|---|---|
| Live webcam feed with hand tracking overlay | Clean canvas showing only your drawings |

---

## ✨ Features

- ✍️ **Real-time Drawing** — Draw on a virtual canvas using hand gestures
- 🎨 **Color Selection** — 5 colors selectable via keyboard (1–5)
- 🖊️ **Multiple Pen Tools** — Normal, Dotted, Highlighter, Spray paint
- 📏 **Pen Thickness Control** — 5 thickness levels via hover gesture
- 🔷 **Smart Shape Snapping** — Draws circle, rectangle or triangle automatically
- 🧹 **Erasing Mode** — Erase with 4+ fingers gesture
- 🔤 **OCR Text Recognition** — Reads handwritten text from canvas
- 💾 **Save Canvas** — Export drawing as PNG image
- 🌐 **Web Interface** — Access board from any browser
- 🔐 **Login System** — Secure login page to protect access
- 📷 **Dual Stream** — Camera feed and canvas shown side by side

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
| `T` | Run OCR on canvas |
| `C` | Clear entire canvas |
| `ESC` | Quit application |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand landmark detection and tracking |
| Flask | Web server and browser interface |
| NumPy | Array and image data handling |
| Pytesseract | OCR text recognition from canvas |
| HTML / CSS / JS | Frontend web interface |

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

### Step 3 — Install Tesseract OCR Engine

- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install it (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- Make sure this path is set correctly in `app.py` and `digital_board_v2.py`

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Step 4 — Run the Application

**Option A — Desktop App:**
```bash
python digital_board_v2.py
```

**Option B — Web App:**
```bash
python app.py
```
Then open your browser and go to:http://127.0.0.1:5000
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
Step 4 → Use gestures to draw, erase, or create shapes
Step 5 → Hover finger on top bars to change tool or thickness
Step 6 → Press keyboard 1-5 to change colors
Step 7 → Press S to save, C to clear, ESC to quit
## 🌐 How to Use — Web App
Step 1 → Run app.py
Step 2 → Open http://127.0.0.1:5000 in browser
Step 3 → Login with username and password
Step 4 → Camera and canvas streams appear in browser
Step 5 → Use gesture panel buttons to change colors and tools
Step 6 → Use hand gestures in front of webcam to draw
Step 7 → Click Save or Clear buttons on the panel
---

## 🤖 How Shape Detection Works
1. Show 3 fingers to enter shape mode
2. Slowly trace your shape outline in the air
3. Close your fist (0 fingers) to snap the shape
4. OpenCV detects the contours of your traced points
5. Shape is snapped to nearest perfect shape:
3 corners  → Triangle
4 corners  → Rectangle
Round      → Circle

---

## 📦 Requirements
flask
opencv-python
mediapipe
numpy
pytesseract
scikit-learn
## 🚀 Future Improvements

- [ ] Undo / Redo support
- [ ] Voice command control
- [ ] Background image import
- [ ] Export canvas as PDF
- [ ] Multi-user collaborative board
- [ ] Mobile browser support

---

## 👩‍💻 Made By

**Gayatri Pratap Jagadale**

<div align="center">

⭐ If you like this project give it a star on GitHub ⭐

</div>