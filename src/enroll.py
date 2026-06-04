"""
Enrollment Script - Capture your face directly from the webcam.

Usage:
    python src/enroll.py lovepreet
    
Controls:
    SPACE  - Capture current face frame
    Q      - Quit after capturing enough samples
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time

# Add src/ to path so database module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import AttendanceDatabase

FACES_DIR = "faces"
TARGET_SAMPLES = 30   # Capture 30 varied samples for robust training

def enroll_user(user_id: str, display_name: str = ""):
    save_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(save_dir, exist_ok=True)

    # Register in database (skip for negative/decoy classes)
    if not user_id.startswith("_"):
        db = AttendanceDatabase()
        name = display_name.strip() if display_name.strip() else user_id.capitalize()
        db.register_user(user_id, name)
    else:
        print(f"[INFO] Negative class '{user_id}' — skipping database registration.")

    # Count existing samples so we don't overwrite
    existing = len([f for f in os.listdir(save_dir) if f.endswith('.jpg')])
    count = existing

    # MediaPipe face mesh for landmark-based face crop
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n[ENROLL] Starting enrollment for user: '{user_id}'")
    print(f"[ENROLL] Already have {existing} samples. Need {TARGET_SAMPLES} total.")
    print("[ENROLL] SPACE = capture frame | Q = quit\n")

    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror correction (same as main app)
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_roi = None
        bbox = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = np.array([[int(l.x * w), int(l.y * h)] for l in lm])

            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)

            # Add 20% padding around the face
            pad_x = int((x_max - x_min) * 0.2)
            pad_y = int((y_max - y_min) * 0.2)
            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(w, x_max + pad_x)
            y2 = min(h, y_max + pad_y)
            bbox = (x1, y1, x2, y2)

            face_roi = frame[y1:y2, x1:x2]

            # Draw bounding box
            color = (0, 255, 0) if face_roi.size > 0 else (0, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # HUD overlay
        progress = min(count, TARGET_SAMPLES)
        bar_w    = int((progress / TARGET_SAMPLES) * 400)
        cv2.rectangle(display, (20, 20), (420, 45), (50, 50, 50), -1)
        cv2.rectangle(display, (20, 20), (20 + bar_w, 45), (0, 200, 100), -1)
        cv2.putText(display, f"Samples: {count}/{TARGET_SAMPLES}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if face_roi is not None and face_roi.size > 0:
            cv2.putText(display, "Face detected - press SPACE to capture", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
        else:
            cv2.putText(display, "No face detected - position yourself in frame", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)

        if count >= TARGET_SAMPLES:
            cv2.putText(display, "Enrollment complete! Press Q to finish.", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f"Enrollment - {user_id}", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE to capture
            now = time.time()
            if face_roi is not None and face_roi.size > 0 and (now - last_capture_time) > 0.3:
                save_path = os.path.join(save_dir, f"{count:03d}.jpg")
                cv2.imwrite(save_path, face_roi)
                count += 1
                last_capture_time = now
                print(f"[CAPTURE] Saved sample {count}/{TARGET_SAMPLES}")

                # Auto-capture flash feedback
                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
                cv2.addWeighted(flash, 0.3, display, 0.7, 0, display)
                cv2.imshow(f"Enrollment - {user_id}", display)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    print(f"\n[DONE] Enrolled {count} samples for '{user_id}' in '{save_dir}/'")
    if count < 10:
        print("[WARN] Fewer than 10 samples — recognition may be unreliable. Re-run to add more.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/enroll.py <user_id> [Full Name]")
        print("Example: python src/enroll.py lovepreet 'Lovepreet Singh'")
        sys.exit(1)

    uid          = sys.argv[1]
    display_name = sys.argv[2] if len(sys.argv) >= 3 else ""
    enroll_user(uid, display_name)
