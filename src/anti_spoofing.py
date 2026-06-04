import numpy as np

class LivenessDetector:
    def __init__(self, ear_threshold=0.22, consecutive_frames=3):
        # EAR threshold under which the eye is classified as closed
        self.ear_threshold = ear_threshold
        # Minimal consecutive video frames required to register a clean blink event
        self.consecutive_frames = consecutive_frames
        
        # Tracking frame counters for individual streams
        self.blink_counter = 0
        self.frame_counter = 0
        self.has_blinked = False

    def _calculate_ear(self, eye_points):
        """Calculates the Eye Aspect Ratio (EAR) from 6 coordinate landmark arrays."""
        # Vertical distances between upper and lower eyelids
        p2_minus_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
        p3_minus_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal width distance between outer eye corners
        p1_minus_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Compute the final normalized ratio
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear

    def process_liveness(self, left_eye, right_eye):
        """
        Evaluates real-time eye vectors to track blinks and verify active user presence.
        Expects a 6x2 numpy array for both left_eye and right_eye structures.
        """
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # Average both eyes for mathematical stability across head rotations
        avg_ear = (left_ear + right_ear) / 2.0
        
        # State machine checking for geometric blink drops
        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            # If eyes were closed for long enough, confirm a real human blink event
            if self.frame_counter >= self.consecutive_frames:
                self.blink_counter += 1
                self.has_blinked = True
            self.frame_counter = 0
            
        return avg_ear, self.has_blinked

    def reset_liveness(self):
        """Resets the state registers when a new subject enters the viewfinder window."""
        self.blink_counter = 0
        self.frame_counter = 0
        self.has_blinked = False