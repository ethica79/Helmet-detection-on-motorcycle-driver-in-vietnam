"""Two-model inference pipeline: helmet detector + motorcycle detector + association."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .association import AssociationResult, Box, ExpansionFactors, associate
from .config import resolve

# BGR colours used for drawing
GREEN = (0, 200, 0)
RED = (0, 0, 255)
CYAN = (200, 200, 0)
ORANGE = (0, 120, 200)


class ViolationPipeline:
    """Runs both models on a frame and applies the motorcycle association rule.

    Models are loaded lazily on first use and shared across threads (Gradio
    may call the pipeline from several workers).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.helmet_weights = resolve(cfg["models"]["helmet_weights"])
        self.moto_weights = resolve(cfg["models"]["moto_weights"])
        self.helmet_conf = float(cfg["inference"]["helmet_conf"])
        self.moto_conf = float(cfg["inference"]["moto_conf"])
        self.factors = ExpansionFactors(**cfg["association"])
        self.with_id = int(cfg["classes"]["with_helmet"])
        self.without_id = int(cfg["classes"]["without_helmet"])
        self._models: tuple[YOLO, YOLO] | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[YOLO, YOLO]:
        with self._lock:
            if self._models is None:
                for p in (self.helmet_weights, self.moto_weights):
                    if not Path(p).exists():
                        raise FileNotFoundError(
                            f"Weights not found: {p}. See README > Model weights."
                        )
                self._models = (YOLO(str(self.helmet_weights)), YOLO(str(self.moto_weights)))
        return self._models

    def detect(self, frame_bgr: np.ndarray) -> tuple[AssociationResult, list[Box]]:
        """Run both models on one BGR frame. Returns the association and moto boxes."""
        helmet_model, moto_model = self._load()
        img_h, img_w = frame_bgr.shape[:2]

        h_res = helmet_model(frame_bgr, conf=self.helmet_conf, verbose=False)[0]
        m_res = moto_model(frame_bgr, conf=self.moto_conf, verbose=False)[0]

        moto_boxes: list[Box] = [tuple(b) for b in m_res.boxes.xyxy.tolist()]
        helmet_dets = [
            (tuple(b), float(c), int(k))
            for b, c, k in zip(h_res.boxes.xyxy.tolist(),
                               h_res.boxes.conf.tolist(),
                               h_res.boxes.cls.tolist())
        ]
        result = associate(helmet_dets, moto_boxes, img_w, img_h, self.factors,
                           self.with_id, self.without_id)
        return result, moto_boxes

    def annotate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect and draw results on a copy of the frame."""
        result, moto_boxes = self.detect(frame_bgr)
        return draw(frame_bgr, result, moto_boxes)


def draw(frame_bgr: np.ndarray, result: AssociationResult, moto_boxes: list[Box]) -> np.ndarray:
    """Draw motorcycles, helmets, violations, suppressed detections and a counter overlay."""
    out = frame_bgr.copy()

    for box in moto_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), ORANGE, 1)

    for dets, colour, label, thickness in (
        (result.helmets, GREEN, "helmet", 2),
        (result.violations, RED, "VIOLATION", 2),
        (result.suppressed, CYAN, None, 1),
    ):
        for box, conf in dets:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
            if label:
                cv2.putText(out, f"{label} {conf:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    for i, text in enumerate([
        f"Helmets:     {len(result.helmets)}",
        f"Violations:  {len(result.violations)}",
        f"Motorcycles: {len(moto_boxes)}",
    ]):
        y = 22 + i * 22
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out
