"""Summarise the latest test-set evaluation from 06_evaluate_test.py into a report folder."""

from __future__ import annotations

import glob
import json
import os
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from helmet_detection.config import base_parser, load_config, resolve


def main() -> None:
    parser = base_parser("Summarise the latest test-set evaluation.")
    parser.add_argument("--run-name", help="label for the report folder (default: train.run_name)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_name = args.run_name or cfg["train"]["run_name"]
    report_dir = str(resolve(cfg["paths"]["reports_dir"]) / f"{run_name}_test")

    # Pick the most recently modified evaluation run
    val_jsons = glob.glob(str(resolve(cfg["paths"]["runs_dir"]) / "test" / "*" / "metrics.json"))
    if not val_jsons:
        raise FileNotFoundError("No metrics.json found. Run scripts/06_evaluate_test.py first.")
    metrics_file = max(val_jsons, key=os.path.getmtime)

    with open(metrics_file) as f:
        metrics = json.load(f)

    val_dir = metrics["save_dir"]

    print("=" * 50)
    print(f"TEST RESULTS — {run_name} on full test set")
    print("=" * 50)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  mAP50     : {metrics['mAP50']:.4f}")
    print(f"  mAP50-95  : {metrics['mAP50_95']:.4f}")

    print("\nPer-class:")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50_95']:.3f}")

    # Collect the plots that YOLO saved into the val run directory
    images = {
        "Confusion Matrix":       os.path.join(val_dir, "confusion_matrix_normalized.png"),
        "Precision-Recall Curve": os.path.join(val_dir, "BoxPR_curve.png"),
        "F1 Curve":               os.path.join(val_dir, "BoxF1_curve.png"),
        "Precision Curve":        os.path.join(val_dir, "BoxP_curve.png"),
        "Recall Curve":           os.path.join(val_dir, "BoxR_curve.png"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{run_name} on the held-out test set", fontsize=16, fontweight="bold")

    for ax, (title, path) in zip(axes.flat, images.items()):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    axes.flat[-1].axis("off")  # 5 plots in a 2x3 grid, hide the empty last cell

    os.makedirs(report_dir, exist_ok=True)
    analysis_png = os.path.join(report_dir, f"{run_name}_test_analysis.png")
    plt.tight_layout()
    plt.savefig(analysis_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {analysis_png}")

    # Copy everything into the report folder with consistent naming
    shutil.copy(metrics_file, os.path.join(report_dir, f"{run_name}_test_metrics.json"))

    for src_name, dst_name in [
        ("confusion_matrix_normalized.png", f"{run_name}_test_confusion_matrix.png"),
        ("BoxF1_curve.png",                 f"{run_name}_test_F1_curve.png"),
        ("BoxPR_curve.png",                 f"{run_name}_test_PR_curve.png"),
    ]:
        src = os.path.join(val_dir, src_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(report_dir, dst_name))

    print(f"All files saved to {report_dir}")


if __name__ == "__main__":
    main()
