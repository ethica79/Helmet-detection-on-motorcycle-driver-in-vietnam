import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob, os

# Find the most recent metrics.json across all val folders
val_jsons = glob.glob(r"C:\ProjectHate\runs\detect\val*\metrics.json")
METRICS_FILE = max(val_jsons, key=os.path.getmtime)

with open(METRICS_FILE) as f:
    metrics = json.load(f)

VAL_DIR = metrics["save_dir"]

print("=" * 50)
print("VALIDATION RESULTS — v14 on full test set")
print("=" * 50)
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  mAP50     : {metrics['mAP50']:.4f}")
print(f"  mAP50-95  : {metrics['mAP50_95']:.4f}")

print("\nPer-class:")
for cls, m in metrics["per_class"].items():
    print(f"  {cls:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50_95']:.3f}")

# --- Display curves and confusion matrix ---
images = {
    "Confusion Matrix":            rf"{VAL_DIR}\confusion_matrix_normalized.png",
    "Precision-Recall Curve":      rf"{VAL_DIR}\BoxPR_curve.png",
    "F1 Curve":                    rf"{VAL_DIR}\BoxF1_curve.png",
    "Precision Curve":             rf"{VAL_DIR}\BoxP_curve.png",
    "Recall Curve":                rf"{VAL_DIR}\BoxR_curve.png",
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("v14 Validation on Full Test Set", fontsize=16, fontweight="bold")

for ax, (title, path) in zip(axes.flat, images.items()):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

axes.flat[-1].axis("off")  # hide unused subplot

plt.tight_layout()
plt.savefig(rf"{VAL_DIR}\analysis_val.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to {VAL_DIR}\\analysis_val.png")
