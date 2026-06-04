# Bio-Guard — Smart Attendance System

> **A real-time, liveness-protected face recognition attendance system** built from scratch with Python, OpenCV, and MediaPipe — with an iterative engineering journey from pixel-matching failures to a robust biometric pipeline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange) ![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## The Engineering Journey — How Recognition Was Improved

This section documents the real technical decisions, failures, and fixes made during development. This is not a tutorial — it is an honest account of iterative problem-solving.

---

### Stage 1 — First Attempt: Pixel-Level Template Matching ❌

**Approach:** The initial version used `cv2.matchTemplate()` to compare a stored reference face image against each live webcam frame.

**Why it failed:**
- Template matching is essentially pixel subtraction — it is catastrophically sensitive to lighting changes, head rotation, and distance from camera.
- Any change in ambient light between enrollment time and scan time produced a zero-match result.
- The live crop bounding box did not align with the stored reference crop, causing consistent mismatches even for the same person.

**Lesson learned:** Pixel similarity is not face recognition. A fundamentally different feature-extraction approach was required.

---

### Stage 2 — Switched to LBPH Face Recognition ✅

**Approach:** Replaced template matching with OpenCV's `LBPHFaceRecognizer` (Local Binary Pattern Histograms).

**Why LBPH works better:**
- Instead of comparing raw pixels, LBPH divides the face into a grid of cells and computes texture patterns (which pixels are brighter than their neighbours).
- These local texture patterns are far more stable under lighting variation and minor pose shifts.
- LBPH returns a **confidence score** (0 = perfect match, higher = worse) — this gives a tunable threshold rather than a binary pass/fail.

**Threshold set to `< 100`** — faces scoring below 100 confidence are recognized; above are rejected as Unknown.

---

### Stage 3 — Single Sample Enrollment Was Not Enough ❌

**Problem:** The first enrollment script captured only 1 reference photo. The LBPH model trained on 1 image had very poor variance tolerance — any slight change in expression or angle caused it to fail.

**Fix:** Rewrote `enroll.py` to capture **30+ varied face samples** per person:
- Different angles (left, right, tilted)
- Different expressions (neutral, slight smile)
- Different distances from camera
- 0.3-second cooldown between captures to enforce variation

Result: The LBPH model now trains on 30+ samples per user and generalizes significantly better.

---

### Stage 4 — The Mirror Problem 🪞

**Problem:** The webcam feed was horizontally mirrored. The enrolled face images were captured in mirrored orientation. The live feed was captured in un-mirrored orientation. The bounding box crops between enrollment and live recognition did not match.

**Fix:** Added `cv2.flip(frame, 1)` to **both** the main recognition loop and the enrollment script, ensuring all face crops are consistently left-right corrected.

---

### Stage 5 — Single-Class Training Caused False Positives ❌

**Critical finding:** When LBPH is trained on only **one person's face**, it has no concept of what a "stranger" looks like. It will always return the single enrolled user as the closest match, regardless of whose face is in frame. The confidence threshold was the only guard — but with no negative examples, even strangers scored below 100.

**Fix — Negative Class Training:**
- Created a `_negative` enrollment class (folder prefix `_` marks it as a decoy class)
- Enrolled **12 real samples** of a different person under `faces/_negative/`
- LBPH now trains on both classes — it learns what the authorized user looks like AND what a stranger looks like
- Any match to `_negative` is explicitly returned as `"Unknown"` in `recognition.py`

**Impact:** False positive rate dropped to near zero. Strangers are now consistently rejected.

---

### Stage 6 — Crop Mismatch Between Enrollment and Recognition ❌

**Problem:** After switching to LBPH, recognition was still unreliable. Investigation revealed the face bounding box crop in `recognition.py` used different padding than `enroll.py`. The model was trained on one crop size and evaluated on a different crop size.

**Fix:** Standardized both scripts to use **20% padding** around the MediaPipe face mesh bounding box:
```python
pad_x = int((x_max - x_min) * 0.2)
pad_y = int((y_max - y_min) * 0.2)
```
Both enrollment and live recognition now produce identical-sized face ROIs, ensuring the LBPH model sees consistent input at both train and inference time.

---

### Stage 7 — Liveness Detection (Anti-Spoofing) 👁️

**Problem:** Any face recognition system can be trivially defeated by holding up a printed photo of an enrolled person.

**Solution:** Implemented **Eye Aspect Ratio (EAR)** blink detection using MediaPipe facial landmarks:

```
EAR = (vertical eye distance) / (horizontal eye distance)

Open eye  → EAR ≈ 0.30
Closed eye → EAR ≈ 0.15
```

The system requires **one confirmed blink** (EAR drops below 0.22 for ≥ 3 consecutive frames, then recovers) before marking attendance. A printed photo cannot blink — it is rejected unconditionally.

---

### Stage 8 — Repeated Checkout Spam Bug 🐛

**Problem:** Once a face was recognized and liveness confirmed, `log_checkout()` was called on **every subsequent frame** while the person remained visible. This resulted in hundreds of identical `[OK] Checked Out` log entries per second.

**Fix:** Introduced an `action_taken` boolean flag:
```python
action_taken = False  # fires only once per liveness confirmation

if is_live and current_subject != "Unknown" and not action_taken:
    db.log_checkout(current_subject)
    action_taken = True
    cv2.waitKey(2000)  # 2-second confirmation hold
    break              # auto-return to menu
```
The action now fires exactly once, shows a 2-second confirmation screen, then returns to the mode selector.

---

### Stage 9 — Database Cursor Bug After Connection Close 🐛

**Problem:** In `log_attendance()`, the `except sqlite3.IntegrityError` block (triggered on duplicate check-ins) tried to use a `cursor` variable that belonged to an already-exited `with conn:` block — a stale connection reference.

**Fix:** Opened a fresh connection inside the except block:
```python
except sqlite3.IntegrityError:
    with self._get_connection() as conn2:
        cur2 = conn2.cursor()
        cur2.execute("SELECT name FROM users WHERE user_id = ?", (user_id,))
        user = cur2.fetchone()
```

---

## Final Architecture

```
Webcam Frame
     │
     ▼
cv2.flip(frame, 1)          ← Mirror correction (consistent with enrollment)
     │
     ▼
MediaPipe FaceMesh          ← 468 landmark coordinates
     │
     ├── Eye landmarks (6 pts each eye)
     │       ↓
     │   EAR Calculation → Blink Detection → Liveness Confirmed
     │
     └── Face Bounding Box (+ 20% padding)
             ↓
         Face ROI Crop → Grayscale → 128×128 resize
             ↓
         LBPH.predict() → (label, confidence)
             │
             ├── confidence < 100 AND label ≠ "_negative"  →  Identity Matched
             │       ↓
             │   SQLite INSERT / UPDATE → attendance_logs
             │       ↓
             │   openpyxl → attendance.xlsx (auto-export on exit)
             │
             └── confidence ≥ 100 OR label == "_negative"  →  Unknown / Rejected
```

---

## Project Structure

```
smart-attendance-system/
├── src/
│   ├── main.py          # Mode selector UI + face scan loop
│   ├── recognition.py   # MediaPipe landmarks + LBPH recognition
│   ├── anti_spoofing.py # EAR blink detection
│   ├── database.py      # SQLite logging + Excel export
│   └── enroll.py        # Multi-sample face enrollment tool
├── faces/               # Face samples — excluded from Git (personal data)
├── data/                # DB + Excel exports — excluded from Git
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Enroll yourself (captures 30 face samples)
python src/enroll.py <your_id> "Your Full Name"
# Press SPACE to capture, Q to quit

# 3. (Optional) Enroll negative class for better stranger rejection
python src/enroll.py _negative

# 4. Run the app
python src/main.py
```

### Mode Selector Controls

| Key | Action |
|-----|--------|
| `L` | Login — mark check-in (blink to verify) |
| `O` | Logout — record check-out time |
| `E` | Enroll new candidate (2-step: ID + Full Name) |
| `Q` | Quit + auto-export Excel |

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| LBPH over deep learning | No GPU required; runs in real-time on CPU; sufficient for small-enrollment systems |
| MediaPipe over Haar cascades | More accurate landmarks; faster; handles partial occlusion |
| EAR blink detection | Simple, reliable, no ML model needed; defeats photo spoofing |
| Negative class training | LBPH is comparative — without negatives, every face matches the only class |
| SQLite over file-based logs | ACID-compliant; prevents duplicate entries via UNIQUE constraint; easy to query |
| 20% crop padding | Includes forehead and chin context; improves LBPH texture discrimination |

---

## Dependencies

```
opencv-python   — Camera, LBPH recognizer, UI drawing
mediapipe       — Face mesh landmark detection
numpy           — Array operations
openpyxl        — Formatted Excel export
```

---

## Privacy

- All face images are stored **locally only** and excluded from version control
- Attendance records are stored **locally only** and excluded from version control
- No data is transmitted to any external server

---

## License

MIT License — free to use, modify, and distribute.
