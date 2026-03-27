import json
from pathlib import Path

MODEL_PATH = r"runs/detect/runs/helmet/v2/weights/best.pt"
DATA_YAML  = r"datasets/helmet_merged/data.yaml"

if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    results = model.val(data=DATA_YAML, split="test")

    metrics = {
        "save_dir": str(results.save_dir),
        "precision":  results.box.mp,
        "recall":     results.box.mr,
        "mAP50":      results.box.map50,
        "mAP50_95":   results.box.map,
        "per_class": {
            results.names[i]: {
                "precision": float(results.box.p[i]),
                "recall":    float(results.box.r[i]),
                "mAP50":     float(results.box.ap50[i]),
                "mAP50_95":  float(results.box.ap[i]),
            }
            for i in range(len(results.names))
        }
    }

    out = Path(results.save_dir) / "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {out}")
