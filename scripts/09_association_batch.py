"""Run the two-model association pipeline on a folder of images.

Saves annotated images and a log with per-image counts of helmets,
confirmed violations, suppressed detections and motorcycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import cv2

from helmet_detection.config import base_parser, load_config, resolve
from helmet_detection.pipeline import ViolationPipeline, draw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def log(msg: str, lf: TextIO) -> None:
    """Print to the console and append to the log file."""
    print(msg)
    lf.write(msg + "\n")
    lf.flush()


def main() -> None:
    parser = base_parser("Batch association pipeline on a folder of images.")
    parser.add_argument("--images", type=Path, default=Path("data/test_frames"),
                        help="folder of input images (default: data/test_frames)")
    parser.add_argument("--out", type=Path, help="output folder (default: <runs_dir>/association)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frames_dir = resolve(args.images)
    out_dir = resolve(args.out) if args.out else resolve(cfg["paths"]["runs_dir"]) / "association"
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in frames_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise SystemExit(f"No images found in {frames_dir}")

    pipeline = ViolationPipeline(cfg)
    totals = {"helmets": 0, "violations": 0, "suppressed": 0}

    with open(out_dir / "pipeline_log.txt", "w", encoding="utf-8") as lf:
        log(f"Helmet model: {pipeline.helmet_weights}", lf)
        log(f"Moto model:   {pipeline.moto_weights}", lf)
        log(f"Found {len(image_paths)} images\n", lf)

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                log(f"[{img_path.name}]  unreadable, skipped", lf)
                continue
            result, moto_boxes = pipeline.detect(frame)
            totals["helmets"] += len(result.helmets)
            totals["violations"] += len(result.violations)
            totals["suppressed"] += len(result.suppressed)
            log(f"[{img_path.name}]  with_helmet={len(result.helmets)}  "
                f"confirmed={len(result.violations)}  suppressed={len(result.suppressed)}  "
                f"motorcycles={len(moto_boxes)}", lf)
            cv2.imwrite(str(vis_dir / img_path.name), draw(frame, result, moto_boxes))

        raw_without = totals["violations"] + totals["suppressed"]
        supp_rate = 100.0 * totals["suppressed"] / raw_without if raw_without else 0.0
        log("", lf)
        log(f"with_helmet:   {totals['helmets']}", lf)
        log(f"violations:    {totals['violations']}", lf)
        log(f"suppressed:    {totals['suppressed']}  ({supp_rate:.1f}% of without_helmet)", lf)
        log(f"visuals:       {vis_dir}", lf)


if __name__ == "__main__":
    main()
