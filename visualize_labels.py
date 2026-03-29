import random
import cv2
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\ProjectHate\datasets\helmet_merged"
SPLIT       = "train"       # train / valid / test
SAMPLE      = 16            # number of images to show
FILTER_DS   = "abdullah"    # show only images from this dataset (set "" for random)
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}   # green=with_helmet, red=without_helmet
NAMES  = {0: "with_helmet", 1: "without_helmet"}

img_dir = Path(DATASET_DIR) / SPLIT / "images"
lbl_dir = Path(DATASET_DIR) / SPLIT / "labels"

all_images = list(img_dir.glob("*.*"))
if FILTER_DS:
    all_images = [p for p in all_images if p.stem.startswith(FILTER_DS)]

random.seed(SEED)
sample = random.sample(all_images, min(SAMPLE, len(all_images)))

cols = 4
rows = (len(sample) + cols - 1) // cols
cell_w, cell_h = 400, 300
canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

for idx, img_path in enumerate(sample):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    H, W = img.shape[:2]

    lbl_path = lbl_dir / f"{img_path.stem}.txt"
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            color = COLORS.get(cls, (255, 255, 0))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, NAMES.get(cls, str(cls)), (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    img = cv2.resize(img, (cell_w, cell_h))
    r, c = divmod(idx, cols)
    canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = img

cv2.imshow(f"Labels — {SPLIT} ({FILTER_DS or 'random'})", canvas)
print("Press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
