import json
import os
import shutil
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# CONFIG 
RUN_NAME = "v4_M"   # change this for each new model


REPORT_DIR = rf"C:\ProjectHate\report\{RUN_NAME}_validation"

# Auto-find the most recent val*/metrics.json
val_jsons = glob.glob(r"C:\ProjectHate\runs\detect\val*\metrics.json")
if not val_jsons:
    raise FileNotFoundError("No validation metrics.json found. Run the val script first.")
METRICS_FILE = max(val_jsons, key=os.path.getmtime)

with open(METRICS_FILE) as f:
    metrics = json.load(f)

VAL_DIR = metrics["save_dir"]

print("=" * 50)
print(f"VALIDATION RESULTS — {RUN_NAME} on full test set")
print("=" * 50)
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  mAP50     : {metrics['mAP50']:.4f}")
print(f"  mAP50-95  : {metrics['mAP50_95']:.4f}")

print("\nPer-class:")
for cls, m in metrics["per_class"].items():
    print(f"  {cls:<20} P={m['precision']:.3f}  R={m['recall']:.3f}  mAP50={m['mAP50']:.3f}  mAP50-95={m['mAP50_95']:.3f}")

images = {
    "Confusion Matrix":       os.path.join(VAL_DIR, "confusion_matrix_normalized.png"),
    "Precision-Recall Curve": os.path.join(VAL_DIR, "BoxPR_curve.png"),
    "F1 Curve":               os.path.join(VAL_DIR, "BoxF1_curve.png"),
    "Precision Curve":        os.path.join(VAL_DIR, "BoxP_curve.png"),
    "Recall Curve":           os.path.join(VAL_DIR, "BoxR_curve.png"),
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"{RUN_NAME} Validation on Full Test Set", fontsize=16, fontweight="bold")

for ax, (title, path) in zip(axes.flat, images.items()):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

axes.flat[-1].axis("off")

os.makedirs(REPORT_DIR, exist_ok=True)
analysis_png = os.path.join(REPORT_DIR, f"{RUN_NAME}_val_analysis.png")
plt.tight_layout()
plt.savefig(analysis_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to {analysis_png}")

shutil.copy(METRICS_FILE, os.path.join(REPORT_DIR, f"{RUN_NAME}_val_metrics.json"))

for src_name, dst_name in [
    ("confusion_matrix_normalized.png", f"{RUN_NAME}_val_confusion_matrix.png"),
    ("BoxF1_curve.png",                 f"{RUN_NAME}_val_F1_curve.png"),
    ("BoxPR_curve.png",                 f"{RUN_NAME}_val_PR_curve.png"),
]:
    src = os.path.join(VAL_DIR, src_name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(REPORT_DIR, dst_name))

print(f"All files saved to {REPORT_DIR}")
