# ✋ Digital Board — Hand Gesture Drawing

A real-time hand gesture controlled digital drawing board built with Python, OpenCV, and MediaPipe. Includes a Flask web interface for browser-based access.

---

## Features

- ✍️ Draw using hand gestures via webcam
- 🎨 Color selection using keyboard (1-5)
- 🖊️ Multiple pen tools — Normal, Dotted, Highlighter, Spray
- 📏 Pen thickness control
- 🔷 Shape drawing and snapping (circle, rectangle, triangle)
- 🧹 Erasing mode
- 🔤 OCR — reads text written on canvas
- 💾 Save canvas as image
- 🌐 Flask web interface with login system
- 📷 Live camera and canvas stream in browser

---

## Gesture Guide

| Gesture       | Action         |
|---------------|----------------|
| 2 fingers     | Writing mode   |
| 3 fingers     | Shape mode     |
| Fist          | Snap shape     |
| 4+ fingers    | Erase          |

---

## Keyboard Shortcuts

| Key   | Action        |
|-------|---------------|
| 1–5   | Change color  |
| S     | Save canvas   |
| T     | Run OCR       |
| C     | Clear canvas  |
| ESC   | Quit          |

---

## Installation

1. Clone the repository:
   git clone https://github.com/2310Gayatri/Digital-Board.git
   cd Digital-Board

2. Install dependencies:
   pip install -r requirements.txt

3. Install Tesseract OCR engine:
   Download from https://github.com/UB-Mannheim/tesseract/wiki

4. Run the desktop app:
   python digital_board_v2.py

5. Or run the web app:
   python app.py
   Then open http://127.0.0.1:5000
---

## Login Credentials (Web App)

| Username | Password |
|----------|----------|
| admin    | 1234     |
| user1    | pass1    |

---

## Project Structure

your_project/
├── app.py
├── digital_board_v2.py
├── collect_data.py
├── train_model.py
├── requirements.txt
├── templates/
│   ├── login.html
│   └── index.html
└── static/
    ├── style.css
    └── login.css

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Flask
- NumPy
- Pytesseract
- HTML / CSS / JavaScript

---

## Made By

Gayatri Pratap Jagdale