"""
DrowsyGuard — Real-Time Drowsiness Detection System
=====================================================
Core detection pipeline combining Eye Aspect Ratio (EAR),
PERCLOS, and Head Pose Estimation into a unified risk score.

Author: Your Name
Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using
           Facial Landmarks", CVWW 2016
"""

import cv2
import dlib
import numpy as np
import time
import logging
from collections import deque
from scipy.spatial import distance as dist
from imutils import face_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EAR_THRESHOLD       = 0.25   # Below this → eye considered closed
EAR_CONSEC_FRAMES   = 3      # Frames eye must be closed to count as blink
PERCLOS_THRESHOLD   = 0.15   # 15% closure over window → drowsy
HEAD_PITCH_THRESH   = 20     # Degrees nodding forward
HEAD_YAW_THRESH     = 30     # Degrees turning sideways
WINDOW_SECONDS      = 60     # PERCLOS rolling window length
SCORE_ALERT         = 50     # Unified risk score threshold


# ──────────────────────────────────────────────
# Facial landmark indices (dlib 68-point model)
# ──────────────────────────────────────────────
L_START, L_END = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
R_START, R_END = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 3D model points for head pose (generic face model)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,   -330.0, -65.0),    # Chin
    (-225.0,  170.0, -135.0),  # Left eye corner
    (225.0,   170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0,  -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

LANDMARK_2D_IDX = [30, 8, 36, 45, 48, 54]  # Matching 2D landmark indices


# ──────────────────────────────────────────────
# EAR Computation
# ──────────────────────────────────────────────
def compute_ear(eye_landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1..p6 are the six eye landmark coordinates in order:
    p1=outer corner, p2-p3=upper lid, p4=inner corner, p5-p6=lower lid.

    Args:
        eye_landmarks: ndarray of shape (6, 2) — (x, y) coordinates.

    Returns:
        EAR value as float. Typical open-eye range: 0.25–0.40.
    """
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)


def compute_mean_ear(shape: np.ndarray) -> float:
    """
    Compute the average EAR across both eyes.

    Args:
        shape: Full 68-point landmark array of shape (68, 2).

    Returns:
        Mean EAR of left and right eye.
    """
    left_eye  = shape[L_START:L_END]
    right_eye = shape[R_START:R_END]
    return (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0


# ──────────────────────────────────────────────
# Head Pose Estimation
# ──────────────────────────────────────────────
def estimate_head_pose(shape: np.ndarray, frame_shape: tuple) -> tuple[float, float, float]:
    """
    Estimate head rotation (pitch, yaw, roll) using solvePnP.

    Maps 6 stable 2D facial landmarks to a known 3D face model,
    then decomposes the rotation matrix into Euler angles.

    Args:
        shape:       Full 68-point landmark array of shape (68, 2).
        frame_shape: (height, width, channels) of the video frame.

    Returns:
        Tuple of (pitch, yaw, roll) in degrees.
        Pitch > 0  → nodding down.
        Yaw   > 0  → turning right.
    """
    h, w = frame_shape[:2]
    focal = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1        ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    pts_2d = np.array([shape[i] for i in LANDMARK_2D_IDX], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles
    return float(pitch), float(yaw), float(roll)


# ──────────────────────────────────────────────
# PERCLOS Calculator
# ──────────────────────────────────────────────
class PerclosCalculator:
    """
    Computes PERCLOS (PERcentage of eye CLOSure) over a rolling window.

    PERCLOS is the proportion of frames in which the eyes are more than
    70–80% closed, measured over a fixed time window (default 60 s).
    It is the most validated objective measure of drowsiness (NHTSA, 1998).
    """

    def __init__(self, fps: float = 30.0, window_seconds: float = WINDOW_SECONDS):
        self.window_size = int(fps * window_seconds)
        self._buffer: deque[int] = deque(maxlen=self.window_size)

    def update(self, ear: float) -> float:
        """
        Push a new EAR sample and return current PERCLOS.

        Args:
            ear: Current frame EAR value.

        Returns:
            PERCLOS as a float in [0, 1].
        """
        self._buffer.append(1 if ear < EAR_THRESHOLD else 0)
        if len(self._buffer) == 0:
            return 0.0
        return sum(self._buffer) / len(self._buffer)

    @property
    def value(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(self._buffer) / len(self._buffer)


# ──────────────────────────────────────────────
# Risk Score Fusion
# ──────────────────────────────────────────────
def compute_risk_score(ear: float, perclos: float, pitch: float, yaw: float) -> int:
    """
    Fuse EAR, PERCLOS, and head pose into a single drowsiness risk score.

    Weighted formula:
        score = w_ear * ear_component
              + w_perclos * perclos_component
              + w_head * head_component

    Weights: EAR=40%, PERCLOS=35%, Head=25%

    Args:
        ear:     Current mean EAR value.
        perclos: Current PERCLOS (0–1).
        pitch:   Head pitch in degrees.
        yaw:     Head yaw in degrees.

    Returns:
        Risk score as int in [0, 100]. ≥50 → alert.
    """
    ear_component     = max(0, 1 - ear / EAR_THRESHOLD) * 40
    perclos_component = min(perclos / PERCLOS_THRESHOLD, 1.0) * 35
    head_deviation    = (abs(pitch) / HEAD_PITCH_THRESH + abs(yaw) / HEAD_YAW_THRESH) / 2
    head_component    = min(head_deviation, 1.0) * 25
    return int(min(ear_component + perclos_component + head_component, 100))


# ──────────────────────────────────────────────
# Main Detector
# ──────────────────────────────────────────────
class DrowsinessDetector:
    """
    Full drowsiness detection pipeline.

    Combines:
      - dlib face detection + 68-point landmark prediction
      - EAR-based eye closure detection
      - PERCLOS rolling window
      - Head pose via solvePnP
      - Blink rate tracking
      - Fused risk score with alerting

    Usage:
        detector = DrowsinessDetector("models/shape_predictor_68_face_landmarks.dat")
        detector.run()
    """

    def __init__(self, predictor_path: str, camera_index: int = 0, fps: float = 30.0):
        """
        Initialize the detector.

        Args:
            predictor_path: Path to dlib's 68-point shape predictor .dat file.
            camera_index:   OpenCV camera index (0 = default webcam).
            fps:            Expected camera FPS for PERCLOS window sizing.
        """
        logger.info("Loading dlib face detector...")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.perclos    = PerclosCalculator(fps=fps)
        self.blink_count = 0
        self._ear_below  = 0  # consecutive frames EAR below threshold
        self.start_time  = time.time()
        self.alert_count = 0

        logger.info("DrowsyGuard initialized. Press Q to quit.")

    def _draw_overlay(self, frame: np.ndarray, ear: float, perclos: float,
                      pitch: float, yaw: float, score: int) -> np.ndarray:
        """Render HUD overlay onto the video frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        color_ok   = (0, 200, 100)
        color_warn = (0, 180, 255)
        color_bad  = (0, 60, 255)

        def score_color(val, lo, hi):
            if val < lo: return color_ok
            if val < hi: return color_warn
            return color_bad

        cv2.putText(frame, f"EAR:     {ear:.3f}", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color(ear, 0.25, 0.20)[::-1], 1)
        cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color(perclos, 0.10, 0.15)[::-1], 1)
        cv2.putText(frame, f"Pitch:   {pitch:.1f} deg",   (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color(abs(pitch), 15, 20)[::-1], 1)
        cv2.putText(frame, f"Yaw:     {yaw:.1f} deg",     (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color(abs(yaw), 20, 30)[::-1], 1)
        cv2.putText(frame, f"Blinks:  {self.blink_count}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_ok, 1)

        bar_color = score_color(score, 35, 65)
        cv2.rectangle(frame, (w-160, 10), (w-10, 40), (40,40,40), -1)
        cv2.rectangle(frame, (w-160, 10), (w-160+int(150*score/100), 40), bar_color, -1)
        cv2.putText(frame, f"RISK: {score}/100", (w-155, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        if score >= SCORE_ALERT:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 200), 4)
            cv2.putText(frame, "!! DROWSINESS DETECTED !!", (w//2 - 180, h - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 60, 255), 2)
        return frame

    def run(self) -> None:
        """
        Main detection loop.

        Reads frames from camera, detects faces, extracts landmarks,
        computes all metrics, and displays annotated output.
        Press 'Q' to quit.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to grab frame.")
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            ear, perclos_val, pitch, yaw, roll, score = 0.35, 0.0, 0.0, 0.0, 0.0, 0

            for face in faces:
                shape = face_utils.shape_to_np(self.predictor(gray, face))

                ear = compute_mean_ear(shape)
                perclos_val = self.perclos.update(ear)
                pitch, yaw, roll = estimate_head_pose(shape, frame.shape)
                score = compute_risk_score(ear, perclos_val, pitch, yaw)

                if ear < EAR_THRESHOLD:
                    self._ear_below += 1
                else:
                    if self._ear_below >= EAR_CONSEC_FRAMES:
                        self.blink_count += 1
                    self._ear_below = 0

                if score >= SCORE_ALERT:
                    self.alert_count += 1
                    logger.warning(f"ALERT #{self.alert_count} — score={score}, EAR={ear:.3f}, PERCLOS={perclos_val:.2%}")

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 212, 255), -1)

            frame = self._draw_overlay(frame, ear, perclos_val, pitch, yaw, score)
            cv2.imshow("DrowsyGuard", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self._cleanup()

    def _cleanup(self) -> None:
        elapsed = time.time() - self.start_time
        logger.info(f"Session ended. Duration: {elapsed:.0f}s | Blinks: {self.blink_count} | Alerts: {self.alert_count}")
        self.cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DrowsyGuard — Real-Time Drowsiness Detector")
    parser.add_argument("--predictor", default="models/shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib shape predictor .dat file")
    parser.add_argument("--camera",    type=int, default=0, help="Camera device index")
    parser.add_argument("--fps",       type=float, default=30.0, help="Camera FPS")
    args = parser.parse_args()

    detector = DrowsinessDetector(args.predictor, args.camera, args.fps)
    detector.run()
