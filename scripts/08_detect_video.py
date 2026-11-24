"""Run the full violation pipeline on a webcam or video file and display it live (press Q to quit)."""

from __future__ import annotations

from pathlib import Path

import cv2

from helmet_detection.config import base_parser, load_config
from helmet_detection.pipeline import ViolationPipeline


def main() -> None:
    parser = base_parser("Live violation detection on a webcam or video file.")
    parser.add_argument("--source", default="0", help="webcam index (default 0) or video path")
    parser.add_argument("--save", type=Path, help="optional path to save the annotated video (.mp4)")
    args = parser.parse_args()

    pipeline = ViolationPipeline(load_config(args.config))
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Running, press Q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = pipeline.annotate(frame)
            cv2.imshow("Helmet Violation Detection", annotated)
            if writer:
                writer.write(annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved to {args.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
