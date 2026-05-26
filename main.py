"""
DrowsyGuard — main.py
Entry point. Run: python main.py --help
"""

import argparse
import logging
import sys
from src.detector import DrowsinessDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DrowsyGuard — Real-Time Drowsiness Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--predictor",
        default="models/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point shape predictor .dat file",
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera device index",
    )
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="Camera FPS (sets PERCLOS window size)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    detector = DrowsinessDetector(
        predictor_path=args.predictor,
        camera_index=args.camera,
        fps=args.fps,
    )
    detector.run()


if __name__ == "__main__":
    main()
