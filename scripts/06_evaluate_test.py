"""Evaluate a trained helmet model on the held-out test split and save metrics as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from helmet_detection.config import base_parser, load_config, resolve


def main() -> None:
    parser = base_parser("Evaluate the helmet model on the test split.")
    parser.add_argument("--weights", type=Path, help="override models.helmet_weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = args.weights or resolve(cfg["models"]["helmet_weights"])
    model = YOLO(str(weights))
    results = model.val(
        data=str(resolve(cfg["paths"]["dataset"]) / "data.yaml"),
        split="test",
        project=str(resolve(cfg["paths"]["runs_dir"]) / "test"),
    )

    metrics = {
        "weights":   str(weights),
        "save_dir":  str(results.save_dir),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
        "mAP50":     float(results.box.map50),
        "mAP50_95":  float(results.box.map),
        "per_class": {
            results.names[i]: {
                "precision": float(results.box.p[i]),
                "recall":    float(results.box.r[i]),
                "mAP50":     float(results.box.ap50[i]),
                "mAP50_95":  float(results.box.ap[i]),
            }
            for i in range(len(results.names))
        },
    }

    # 07_analyze_test.py picks up the most recent metrics.json
    out = Path(results.save_dir) / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {out}")


if __name__ == "__main__":
    main()
