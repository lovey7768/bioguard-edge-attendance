# 🔒 BioGuard Edge Attendance
### Secure Biometric Tracking with Local Liveness Verification

> A high-performance, local computer vision attendance logging system built entirely from scratch using **Python**, **OpenCV**, and **MediaPipe** — engineered through iterative problem-solving, not just assembled from tutorials.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange) ![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ⚡ Key Technical Architecture

### 1. 👁️ Active Liveness Detection — Anti-Spoofing Shield

To prevent adversarial spoofing attacks (printed photos, screen playback), the system monitors **dynamic spatial geometry patterns** rather than passive structural matching. It calculates the instantaneous vertical-to-horizontal relaxation ratio of the eyelid margin via the **Eye Aspect Ratio (EAR)**:

$$EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

When a static printed image or device screen is presented to the camera, the EAR vector remains completely static. The system continuously buffers frame indices, granting verification only when an active, natural human blink velocity is confirmed — a spoof cannot blink.

### 2. 🧠 Multi-Sample LBPH Recognition with Negative Class Training

- **MediaPipe Face Mesh** configured with `refine_landmarks=True` extracts 468 precise landmark coordinates including iris contours and thin eyelid margins — far more robust than Haar cascade detectors.
- **LBPH (Local Binary Pattern Histograms)** replaces pixel-level template correlation. Each face region is divided into a grid of cells; local texture relationships (which neighbours are brighter) are histogram-encoded per cell. These patterns are stable under lighting variation and minor pose shifts — unlike raw pixel subtraction.
- **Negative class decoy training** — the recognizer is trained on both enrolled users AND a `_negative` class of stranger faces, giving LBPH a reference for what an *unauthorized* face looks like. Without this, any LBPH model trained on a single class will match every face to that class.

### 3. 💾 Idempotent Relational Storage Ledger

The persistence layer runs locally via SQLite with strict integrity constraints:
- `FOREIGN KEY` links attendance logs directly to authenticated user profiles.
- A composite `UNIQUE(user_id, date)` constraint guarantees idempotency — subsequent scans on the same calendar date are silently deduplicated rather than creating duplicate records.
- Check-out times are stored as a nullable column, updated via `UPDATE` on the existing row rather than inserting a second record.
- Auto-exports to a formatted **Excel report** (`openpyxl`) with styled headers, alternating row fills, frozen panes, and a summary sheet — generated on every clean shutdown.

---

## 🔧 The Engineering Journey — Problems Solved

*What separates this project is not just what it does, but how it got there. Every failure below was diagnosed, root-caused, and fixed.*

---

### Stage 1 — Pixel-Level Template Matching ❌

**First attempt** used `cv2.matchTemplate()` with `TM_CCOEFF_NORMED` to compare a stored reference image against live frames.

**Root cause of failure:**
- Template matching is pixel subtraction under the hood — catastrophically sensitive to lighting, distance, and head rotation.
- Crop bounding boxes between enrollment and live scan did not align — zero-match even for the correct person.

**Decision:** Abandoned pixel similarity entirely. Moved to feature-space recognition.

---

### Stage 2 — Switched to LBPH ✅

**Why LBPH over deep CNNs?**
- No GPU required — runs real-time on CPU at 30 FPS.
- Returns a continuous confidence score (0 = perfect match), enabling a tunable threshold rather than a binary decision boundary.
- Sufficient discriminative power for small-enrollment systems (< 50 users).

Threshold set at `confidence < 100` — empirically tuned by displaying live confidence scores on the webcam overlay during testing.

---

### Stage 3 — Single Sample Enrollment Was Not Enough ❌

**Problem:** One reference photo gave LBPH zero variance tolerance. Any expression change caused recognition failure.

**Fix:** Rewrote `enroll.py` to capture **30+ varied samples** per person — different angles, distances, expressions — with a 0.3-second cooldown enforcing inter-sample variation.

---

### Stage 4 — Mirror Inconsistency 🪞

**Problem:** The webcam feed was horizontally flipped. Enrollment and live recognition produced mirror-image crops that LBPH could not match.

**Fix:** Added `cv2.flip(frame, 1)` to **both** `enroll.py` and `main.py`. Consistent orientation across the full pipeline.

---

### Stage 5 — Single-Class False Positives ❌ → Negative Class Training ✅

**Critical insight:** LBPH is a *comparative* classifier. Trained on one class, it always returns that class as the closest match — strangers included. The confidence threshold was the only guard, but with no negative examples it was insufficient.

**Fix:** Enrolled a `_negative` decoy class — real human face samples of non-authorized persons. LBPH now learns *both* what the enrolled user looks like AND what a stranger looks like. Any match to `_negative` is explicitly returned as `"Unknown"`.

---

### Stage 6 — Crop Padding Mismatch ❌

**Problem:** `recognition.py` used different bounding box padding than `enroll.py`. The LBPH model was trained on one crop geometry and evaluated on another.

**Fix:** Standardized **20% padding** on all four sides in both scripts:
```python
pad_x = int((x_max - x_min) * 0.2)
pad_y = int((y_max - y_min) * 0.2)
```

---

### Stage 7 — Checkout Spam Bug 🐛

**Problem:** `log_checkout()` fired every frame while the face was visible — hundreds of DB writes per second.

**Fix:** Introduced an `action_taken` flag. The action fires exactly once after liveness confirmation, renders a 2-second confirmation overlay, then auto-returns to the mode selector menu.

---

### Stage 8 — Stale DB Cursor After Connection Close 🐛

**Problem:** The `except sqlite3.IntegrityError` block referenced a `cursor` from an already-exited `with conn:` context — a stale connection reference that could silently corrupt state.

**Fix:** Opened a fresh `self._get_connection()` inside the except block, ensuring all DB operations use a live connection.

---

## 📁 Repository Structure

```
bioguard-edge-attendance/
│
├── src/
│   ├── main.py           # Mode selector UI + webcam capture loop + overlay rendering
│   ├── recognition.py    # MediaPipe landmarks + LBPH training + face identification
│   ├── anti_spoofing.py  # EAR calculation + consecutive-frame blink state machine
│   ├── database.py       # SQLite schemas, attendance logging, Excel export
│   └── enroll.py         # Multi-sample face enrollment tool with live progress HUD
│
├── faces/                # Face samples — excluded from Git (biometric privacy)
├── data/                 # SQLite DB + Excel exports — excluded from Git
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Enroll yourself (captures 30+ face samples)
python src/enroll.py <your_id> "Your Full Name"
# SPACE = capture sample | Q = finish

# 3. (Recommended) Enroll a negative class for stranger rejection
python src/enroll.py _negative

# 4. Run
python src/main.py
```

### Mode Selector

| Key | Action |
|-----|--------|
| `L` | Login — blink to mark check-in |
| `O` | Logout — blink to record check-out |
| `E` | Enroll new candidate (2-step: ID → Full Name) |
| `Q` | Quit + auto-export Excel report |

---

## 🔑 Key Technical Decisions

| Decision | Rationale |
|---|---|
| LBPH over CNN | No GPU dependency; real-time on CPU; tunable confidence threshold |
| MediaPipe over Haar | 468-landmark precision; handles partial occlusion; iris-level accuracy |
| EAR blink detection | Mathematically simple; zero ML overhead; defeats all static spoofing |
| Negative class training | LBPH has no concept of "stranger" without a negative reference class |
| 20% crop padding | Includes forehead/chin context; improves LBPH texture discrimination |
| SQLite UNIQUE constraint | Idempotent logging without application-level duplicate checks |
| openpyxl export | Structured Excel output with styling; recruiter/HR friendly format |

---

## 📦 Dependencies

```
opencv-python   — Webcam capture, LBPH recognizer, UI rendering
mediapipe       — 468-point face mesh landmark extraction
numpy           — Array and matrix operations
openpyxl        — Formatted Excel attendance export
```

---

## 🔒 Privacy

- Face images stored **locally only** — never transmitted, never committed to Git
- Attendance records stored **locally only** — excluded from version control
- All biometric data remains fully under user control

---

## 📄 License

MIT License — Copyright (c) 2026 Lovepreet Singh
