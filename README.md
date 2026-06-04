# Bio-Guard Smart Attendance System

A real-time, liveness-protected face recognition attendance system built with Python, OpenCV, and MediaPipe.

## Features

- **Face Recognition** — LBPH-based recognition trained on enrolled face samples
- **Liveness Detection** — Blink-based anti-spoofing (prevents photo attacks)
- **Login / Logout** — Records both check-in and check-out times per day
- **Stranger Rejection** — Negative class training to reject unrecognized faces
- **Attendance Logging** — SQLite database with one record per person per day
- **Excel Export** — Auto-exports formatted attendance sheet on app close
- **Mode Selector UI** — OpenCV-based menu (Login / Logout / Enroll)

## Project Structure

```
smart-attendance-system/
├── src/
│   ├── main.py          # App entry point — mode selector & face scan loop
│   ├── recognition.py   # MediaPipe landmarks + LBPH face recognition
│   ├── anti_spoofing.py # Eye Aspect Ratio blink detection
│   ├── database.py      # SQLite attendance logging + Excel export
│   └── enroll.py        # Face enrollment tool
├── faces/               # Face samples (excluded from Git — personal data)
├── data/                # SQLite DB + Excel exports (excluded from Git)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/smart-attendance-system.git
cd smart-attendance-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Enroll yourself
python src/enroll.py <your_id> "Your Full Name"
# Press SPACE to capture ~30 face photos, then Q to quit

# 4. Run the app
python src/main.py
```

## Usage

When the app launches, press:

| Key | Action |
|-----|--------|
| `L` | Login — mark check-in (blink to verify) |
| `O` | Logout — record check-out time (blink to verify) |
| `E` | Enroll a new person |
| `Q` | Quit + auto-export attendance to Excel |

## How It Works

```
Webcam Frame
    ↓
MediaPipe FaceMesh  →  468 landmarks
    ↓
Blink Detection (EAR)  →  Liveness confirmed
    ↓
LBPH Face Recognizer  →  Identity matched
    ↓
SQLite Log  →  attendance.db
    ↓
Excel Export  →  data/attendance.xlsx
```

## Privacy

- Face images are stored **locally only** in `faces/` and are excluded from Git
- Attendance records are stored **locally only** in `data/` and are excluded from Git
- No data is sent to any server

## Dependencies

- `opencv-python` — Camera capture, face drawing, LBPH recognizer
- `mediapipe` — Face mesh landmark detection
- `numpy` — Numerical operations
- `openpyxl` — Excel export

## License

MIT License
