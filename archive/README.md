# archive/

Exploratory scripts kept for reference only. They are **not** part of the
pipeline and still contain the original hardcoded Windows paths.

- `merge_datasets.py`: first local merge of the 7 Roboflow sources (later replaced by merging on Roboflow).
- `phase2_baseline.py`: baseline using COCO `yolov8m.pt` class 3 (motorcycle) before switching to a dedicated motorcycle detector.
- `_extract_frames.py`: extracts evenly spaced frames from a traffic video for the Phase 2 tests.
