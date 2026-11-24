# Helmet Violation Detection for Motorcycle Riders in Vietnam

Detects motorcycle riders without a helmet in traffic images and video, using a
YOLOv8 detector and a rule that links each head to a motorcycle. Built as my
Bachelor thesis (Data Science, USTH, University of Science and Technology of
Hanoi; supervisor: Dr. Son Tran Giang, ICT Lab).

<!-- TODO: add a demo screenshot or GIF of the web app, e.g.
![Demo](docs/demo.gif) -->

**Final model (v4_M, YOLOv8m) on a held-out test set of 1,491 images:**
**mAP50 0.877**, precision 0.854, recall 0.837, and **0.826 recall on the
`without_helmet` (violation) class**.

## Problem

Motorcycles dominate traffic in Vietnam, and helmet use is mandatory. Spotting
violations by hand doesn't scale. A single detector has two weaknesses: it
misses small and partially hidden heads, and it flags pedestrians, who are
allowed to go without a helmet. This project tackles both:

1. **Detection:** a YOLOv8 model with two classes, `with_helmet` and `without_helmet`.
2. **Association:** a `without_helmet` detection counts as a violation only if
   it sits on a motorcycle.

## Approach

```
frame ──► Helmet model (YOLOv8m, trained here) ──► with_helmet / without_helmet boxes ─┐
      └─► Motorcycle model (COCO YOLOv8s, class "motorcycle") ──► expanded zones ──────┴─► violation / suppressed
```

- **Dataset:** 27,241 real images (no augmented copies) merged from 7 public
  Roboflow Universe datasets. Class names from the different sources were mapped
  to 2 classes, polygon labels were converted to boxes, and several thousand
  labels were annotated or corrected by hand. Split: 23,580 train / 2,170 val /
  1,491 test, about 55/45 `with_helmet`/`without_helmet`.
- **Training:** an ablation study of 10 runs over 5 dataset versions
  (YOLOv8n/s/m, COCO-pretrained), on a single RTX 3050 Laptop GPU with 4 GB
  VRAM (batch size 4–8 at 640×640).
- **Association rule:** each motorcycle box is widened ×1.3, extended upward by
  2.0× its height (to reach the rider's head) and downward by 0.1×. A
  `without_helmet` box whose centre falls in that zone is a violation. Otherwise
  it is suppressed.
- **Deployment threshold:** confidence 0.329, the F1-optimal point on the v4_M
  F1-confidence curve.

## Results

Test set (1,491 images), final model **v4_M**:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| all | 0.854 | 0.837 | 0.877 | 0.543 |
| without_helmet | 0.844 | 0.826 | 0.866 | – |

Both project targets were met: mAP50 > 0.80 and violation recall > 0.75.

**What the ablation showed:**

- **Data diversity gave the biggest gain.** Going from 3 to 7 sources (3,357 to
  9,570 images) raised mAP50 from 0.596 to 0.851.
- **Class rebalancing came next.** Raising violations from 25.1% to 45.6% of
  training boxes added +0.135 mAP50 and +0.203 violation recall over v2_M. It
  also cut the recall gap between the two classes from 0.143 to 0.022.
- **There was a label-noise ceiling around 0.85 mAP50.** Training longer
  (0.849), using a 3.7× larger model (0.852) and tripling augmentation
  (0.841 / 0.836) did not beat it.
- **Fine-tuning from weights trained on noisy labels hurt** (0.639 → 0.615).
  Restarting from COCO weights worked better.

## Repository layout

```
configs/config.yaml        all paths, thresholds and hyperparameters
src/helmet_detection/      reusable package
  ├── labels.py            class remapping, polygon → bbox conversion
  ├── association.py       motorcycle zone expansion and violation rule
  ├── pipeline.py          two-model inference + drawing
  └── config.py            config loading, path resolution
scripts/                   numbered pipeline steps (download → train → evaluate → run)
app/web_app.py             Gradio demo (image, webcam, video)
tests/                     unit tests (no GPU or weights needed)
archive/                   early exploratory scripts, kept for reference
```

## Installation

Requires Python 3.10+ (developed with Python 3.11, PyTorch 2.5.1, Windows 11).

```bash
git clone https://github.com/ethica79/Helmet-detection-on-motorcycle-driver-in-vietnam.git
cd Helmet-detection-on-motorcycle-driver-in-vietnam
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# For GPU support, install the CUDA build of torch first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### Model weights

- **Helmet model (v4_M):** download `helmet_v4_M.pt` from the
  [Releases page](https://github.com/ethica79/Helmet-detection-on-motorcycle-driver-in-vietnam/releases)
  and put it in `weights/`.
  <!-- TODO: upload best.pt of run v4_M as a release asset named helmet_v4_M.pt -->
- **Motorcycle model:** stock COCO `yolov8s.pt`. Ultralytics downloads it on first run.

## Usage

**Web demo** (image upload, live webcam, video file):

```bash
python app/web_app.py
```

**Live webcam or video file:**

```bash
python scripts/08_detect_video.py --source 0              # webcam
python scripts/08_detect_video.py --source traffic.mp4 --save out.mp4
```

**Batch run on a folder of images** (saves annotated images and a log):

```bash
python scripts/09_association_batch.py --images path/to/frames
```

**Reproduce training and evaluation:**

```bash
cp .env.example .env                       # add your Roboflow API key
python scripts/01_download.py              # download the merged dataset
python scripts/02_convert_labels.py        # unify classes, polygons → boxes
python scripts/03_eda.py                   # dataset statistics
python scripts/04_train.py                 # train (defaults: v4_M settings)
python scripts/05_analyze_training.py      # training curves
python scripts/06_evaluate_test.py         # metrics on the test split
python scripts/07_analyze_test.py          # test report and plots
```

All settings live in `configs/config.yaml`. Every script accepts
`--config path/to/other.yaml` and `--help`.

**Tests:**

```bash
pytest
```

## Limitations

- The ~0.85 mAP50 ceiling comes from label noise in the public source datasets.
  Better labels, not bigger models, are the next step.
- The association rule is geometric. It can still misfire with heavy occlusion
  or very dense traffic.
- The thesis version of the association step used every class from the COCO
  model (including `person`) as a motorcycle zone. This repository restricts it
  to `motorcycle`. The detection metrics above are unaffected, but the
  association step has not been re-evaluated quantitatively since this fix.
- This is a research prototype. It has not been deployed on real traffic infrastructure.

## Acknowledgements and licenses

- Code: MIT License (see `LICENSE`).
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) is licensed
  under AGPL-3.0. Redistributing or serving this project together with
  Ultralytics must follow that license.
- Training images come from 7 public
  [Roboflow Universe](https://universe.roboflow.com/) datasets, each under its
  own license. They are listed in `scripts/01_download.py`.
- Thanks to Dr. Son Tran Giang (ICT Lab, USTH) for supervising the thesis.
