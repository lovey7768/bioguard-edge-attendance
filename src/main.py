import cv2
import sys
import time
import subprocess
import numpy as np
from database import AttendanceDatabase
from anti_spoofing import LivenessDetector
from recognition import AttendanceRecognizer

# ─────────────────────────────────────────────────────────────────────────────
# MODE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def draw_mode_selector():
    """Renders the startup mode selection screen using OpenCV."""
    W, H = 700, 420
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Background gradient (dark navy)
    for y in range(H):
        t = y / H
        canvas[y, :] = [int(20 + 15 * t), int(24 + 20 * t), int(48 + 30 * t)]

    # Title
    cv2.putText(canvas, "BIO-GUARD ATTENDANCE SYSTEM",
                (60, 65), cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 220, 255), 2)
    cv2.line(canvas, (60, 80), (640, 80), (80, 120, 200), 1)
    cv2.putText(canvas, "Select Mode to Continue",
                (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 170, 200), 1)

    # Card helper
    def draw_card(y, key_label, key_color, title, subtitle):
        x0, x1 = 60, 640
        cv2.rectangle(canvas, (x0, y), (x1, y + 72), (35, 45, 80), -1)
        cv2.rectangle(canvas, (x0, y), (x1, y + 72), (70, 100, 180), 1)
        # Key badge
        cv2.rectangle(canvas, (x0 + 12, y + 18), (x0 + 52, y + 54), key_color, -1)
        cv2.putText(canvas, key_label, (x0 + 18, y + 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        # Text
        cv2.putText(canvas, title, (x0 + 70, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 235, 255), 2)
        cv2.putText(canvas, subtitle, (x0 + 70, y + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 155, 190), 1)

    draw_card(135, "L", (30, 140, 60),
              "Login  —  Check In",
              "Mark your attendance as Present")

    draw_card(225, "O", (180, 80, 30),
              "Logout  —  Check Out",
              "Record your departure time")

    draw_card(315, "E", (120, 40, 160),
              "Enroll New Candidate",
              "Register a new person's face")

    cv2.putText(canvas, "Press Q to Quit",
                (270, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 100, 130), 1)
    return canvas


def show_mode_selector():
    """
    Displays the mode selector and blocks until the user picks one.
    Returns: 'login' | 'logout' | 'enroll' | None (quit)
    """
    win = "Bio-Guard — Select Mode"
    cv2.namedWindow(win)

    while True:
        frame = draw_mode_selector()
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('l'):
            cv2.destroyWindow(win)
            return "login"
        elif key == ord('o'):
            cv2.destroyWindow(win)
            return "logout"
        elif key == ord('e'):
            cv2.destroyWindow(win)
            return "enroll"
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# FACE SCAN LOOP  (shared by login & logout modes)
# ─────────────────────────────────────────────────────────────────────────────

def run_face_scan(mode: str, db: AttendanceDatabase,
                  detector: LivenessDetector,
                  recognizer: AttendanceRecognizer):
    """
    Opens webcam, detects + recognizes face, and performs login or logout.
    mode: 'login' | 'logout'
    """
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Could not access the webcam device stream.")
        return

    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    action_label = "CHECK-IN" if mode == "login" else "CHECK-OUT"
    win_title    = f"Bio-Guard — {action_label}  |  Blink to Verify  |  Q to back"

    print(f"\n{'='*50}")
    print(f"[LIVE] MODE: {action_label}")
    print(f"[LOCK] Blink once to confirm identity.")
    print(f"[INFO] Press 'Q' to go back to menu.")
    print(f"{'='*50}\n")

    current_subject = "Unknown"
    consecutive_failures = 0
    MAX_FAILURES = 30
    conf = 999.0
    action_taken = False   # ← fires only once per liveness confirmation

    while True:
        ret, frame = video_capture.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print("[WARN] Dropped video frame. Retrying...")
            if consecutive_failures >= MAX_FAILURES:
                print("[ERROR] Camera stream lost. Returning to menu.")
                break
            time.sleep(0.033)
            continue
        consecutive_failures = 0

        frame    = cv2.flip(frame, 1)
        ui_frame = frame.copy()

        # Mode label watermark
        cv2.putText(ui_frame, f"[ {action_label} MODE ]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 200, 80) if mode == "login" else (0, 120, 255), 2)

        left_eye, right_eye, bbox = recognizer.extract_eye_landmarks(frame)

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            if left_eye is not None and right_eye is not None:
                ear, is_live = detector.process_liveness(left_eye, right_eye)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    identified_name, conf = recognizer.identify_face(face_roi)
                    if identified_name != current_subject:
                        detector.reset_liveness()
                        current_subject = identified_name
                        action_taken = False   # reset if a different face appears
                else:
                    conf = 999.0

                if is_live and current_subject != "Unknown" and not action_taken:
                    if mode == "login":
                        db.log_attendance(current_subject)
                        status_text  = f"Checked In: {current_subject}"
                        border_color = (0, 255, 0)
                    else:
                        result = db.log_checkout(current_subject)
                        if result and result.get("checked_out"):
                            status_text  = f"Checked Out: {current_subject}"
                            border_color = (0, 200, 255)
                        else:
                            status_text  = f"No check-in found for {current_subject}"
                            border_color = (0, 80, 200)
                    action_taken = True

                    # Show confirmation overlay for 2 seconds then auto-return
                    cv2.rectangle(ui_frame, (x1, y1), (x2, y2), border_color, 2)
                    cv2.putText(ui_frame, status_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
                    cv2.putText(ui_frame, "Returning to menu...", (20, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    cv2.imshow(win_title, ui_frame)
                    cv2.waitKey(2000)   # 2-second confirmation hold
                    break

                elif is_live and current_subject != "Unknown" and action_taken:
                    status_text  = f"Done: {current_subject}  |  Press Q to go back"
                    border_color = (0, 200, 100)
                else:
                    status_text  = f"Scanning... Blink ({detector.blink_counter}/1)"
                    border_color = (0, 165, 255)
                    if current_subject == "Unknown":
                        status_text  = "Access Denied: Unknown Identity"
                        border_color = (0, 0, 255)
            else:
                status_text  = "Isolating face..."
                border_color = (0, 255, 255)
                ear = 0.0

            cv2.rectangle(ui_frame, (x1, y1), (x2, y2), border_color, 2)
            cv2.putText(ui_frame, status_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
            cv2.putText(ui_frame, f"EAR: {ear:.2f}  |  LBPH Conf: {conf:.1f}",
                        (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (240, 240, 240), 1)
        else:
            detector.reset_liveness()
            current_subject = "Unknown"
            action_taken = False
            cv2.putText(ui_frame, "Awaiting Face...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_title, ui_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# ENROLL MODE
# ─────────────────────────────────────────────────────────────────────────────

def _text_input_screen(prompt_line1, prompt_line2, hint="Enter = confirm   Esc = cancel"):
    """Generic OpenCV single-line text input. Returns stripped string or None on cancel."""
    canvas = np.zeros((200, 560, 3), dtype=np.uint8)
    canvas[:] = (25, 30, 55)
    chars = []
    win = prompt_line1

    cv2.namedWindow(win)
    while True:
        display = canvas.copy()
        cv2.putText(display, prompt_line2, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 200, 255), 2)
        cv2.putText(display, f"> {''.join(chars)}_", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 130), 2)
        cv2.putText(display, hint, (30, 162),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 120, 160), 1)
        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF
        if key == 13:      # Enter — confirm
            cv2.destroyWindow(win)
            return "".join(chars).strip()
        elif key == 27:    # Esc — cancel
            cv2.destroyWindow(win)
            return None
        elif key == 8:     # Backspace
            if chars:
                chars.pop()
        elif 32 <= key <= 126:
            chars.append(chr(key))


def run_enroll_mode():
    """Prompts for candidate ID + full name then launches the enrollment tool."""
    # Step 1 — candidate ID (used as folder name & DB key)
    candidate_id = _text_input_screen(
        "Enroll — Step 1 of 2",
        "Candidate ID  (e.g. lovepreet):",
    )
    if not candidate_id:
        return

    # Step 2 — full display name (shown in Excel & attendance log)
    display_name = _text_input_screen(
        "Enroll — Step 2 of 2",
        "Full Name  (e.g. Lovepreet Singh):",
        hint="Enter = confirm   Esc = use ID as name",
    )
    if display_name is None:
        display_name = candidate_id.capitalize()

    print(f"\n[ENROLL] ID='{candidate_id}'  Name='{display_name}'")
    subprocess.run([sys.executable, "src/enroll.py", candidate_id, display_name])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    db         = AttendanceDatabase()
    detector   = LivenessDetector()
    recognizer = AttendanceRecognizer()

    while True:
        mode = show_mode_selector()

        if mode is None:
            break
        elif mode == "enroll":
            run_enroll_mode()
            # Reload recognizer after new enrollment
            recognizer = AttendanceRecognizer()
        else:
            run_face_scan(mode, db, detector, recognizer)
            detector.reset_liveness()

    # Export attendance on exit
    db.export_to_excel()
    print("\n[LOCK] Hardware ports closed safely. Engine offline.")


if __name__ == "__main__":
    main()