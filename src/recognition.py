import cv2
import numpy as np
import mediapipe as mp
import os

class AttendanceRecognizer:
    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        os.makedirs(self.faces_dir, exist_ok=True)

        # Initialize MediaPipe Face Mesh for fast landmark extraction
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Exact landmark coordinate mapping indices from MediaPipe schema
        self.LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # LBPH face recognizer — robust to lighting and minor pose variation
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Mapping: integer label -> user_id string
        self.label_to_user: dict[int, str] = {}

        self._train_recognizer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_recognizer(self):
        """
        Loads face images from faces_dir. Supports two formats:
          - Single image:      faces/<user_id>.jpg
          - Multi-sample dir:  faces/<user_id>/000.jpg, 001.jpg, ...
        More samples = better accuracy.
        """
        images, labels = [], []
        user_label_map: dict[str, int] = {}

        def add_image(user_id: str, path: str):
            nonlocal images, labels
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] Could not load image: {path}")
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))

            if user_id not in user_label_map:
                label = len(user_label_map)
                user_label_map[user_id] = label
                self.label_to_user[label] = user_id

            images.append(gray)
            labels.append(user_label_map[user_id])

        for entry in sorted(os.listdir(self.faces_dir)):
            entry_path = os.path.join(self.faces_dir, entry)

            if os.path.isdir(entry_path):
                # Multi-sample subdirectory: faces/<user_id>/
                user_id = entry
                for img_file in sorted(os.listdir(entry_path)):
                    if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        add_image(user_id, os.path.join(entry_path, img_file))

            elif entry.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Legacy single-image: faces/<user_id>.jpg
                user_id = os.path.splitext(entry)[0]
                add_image(user_id, entry_path)

        if images:
            self.recognizer.train(images, np.array(labels))
            sample_counts = {uid: labels.count(lbl) for uid, lbl in user_label_map.items()}
            for uid, cnt in sample_counts.items():
                print(f"[INFO] Loaded user '{uid}' with {cnt} sample(s).")
        else:
            print("[WARN] No face profiles found in 'faces/' directory.")

    # ------------------------------------------------------------------
    # Landmark extraction (unchanged)
    # ------------------------------------------------------------------

    def extract_eye_landmarks(self, frame):
        """
        Processes frame pixels to locate facial boundaries, extracting
        structural eye landmark coordinates for liveness evaluation.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, None

        face_landmarks = results.multi_face_landmarks[0].landmark

        coords = np.array([[int(l.x * w), int(l.y * h)] for l in face_landmarks])

        left_eye  = coords[self.LEFT_EYE_INDICES]
        right_eye = coords[self.RIGHT_EYE_INDICES]

        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)

        # Apply 20% padding — must match the enrollment script crop exactly
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        bbox = (
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            min(w, x_max + pad_x),
            min(h, y_max + pad_y),
        )

        return left_eye, right_eye, bbox

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def identify_face(self, face_roi):
        """
        Runs LBPH recognition on a face ROI.
        Returns (user_id, confidence) tuple.
        user_id is 'Unknown' if confidence threshold not met.

        LBPH confidence: 0 = perfect match, higher = worse.
        Threshold of 100 gives good tolerance for lighting/pose shifts.
        """
        if not self.label_to_user:
            return "Unknown", 999.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))

        try:
            label, confidence = self.recognizer.predict(gray)
        except cv2.error:
            return "Unknown", 999.0

        if confidence < 100:
            matched_id = self.label_to_user.get(label, "Unknown")
            # Labels starting with '_' are negative/decoy classes — never return them as a valid identity
            if matched_id.startswith("_"):
                return "Unknown", confidence
            return matched_id, confidence

        return "Unknown", confidence