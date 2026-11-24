"""Rule-based association between helmet detections and motorcycles.

A ``without_helmet`` detection becomes a confirmed violation only if its centre
falls inside an expanded motorcycle box. Otherwise it is suppressed (most
often a pedestrian). The box is expanded much more upward than downward,
because the rider's head sits above the motorcycle body.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]       # x1, y1, x2, y2 in pixels
Detection = tuple[Box, float]                 # (box, confidence)


@dataclass(frozen=True)
class ExpansionFactors:
    """Asymmetric expansion of a motorcycle box (see configs/config.yaml)."""

    x: float = 1.3
    up: float = 2.0
    down: float = 0.1


@dataclass
class AssociationResult:
    """Helmet detections split by outcome."""

    helmets: list[Detection] = field(default_factory=list)
    violations: list[Detection] = field(default_factory=list)
    suppressed: list[Detection] = field(default_factory=list)


def expand_box(box: Box, factors: ExpansionFactors, img_w: float, img_h: float) -> Box:
    """Expand a motorcycle box into an association zone, clipped to the image."""
    x1, y1, x2, y2 = box
    cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
    return (
        max(0.0, cx - w * factors.x / 2),
        max(0.0, y1 - h * factors.up),
        min(float(img_w), cx + w * factors.x / 2),
        min(float(img_h), y2 + h * factors.down),
    )


def center_in_box(box: Box, cx: float, cy: float) -> bool:
    """True if the point (cx, cy) lies inside ``box`` (edges included)."""
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def associate(
    helmet_dets: Sequence[tuple[Box, float, int]],
    moto_boxes: Sequence[Box],
    img_w: float,
    img_h: float,
    factors: ExpansionFactors = ExpansionFactors(),
    with_helmet_id: int = 0,
    without_helmet_id: int = 1,
) -> AssociationResult:
    """Classify helmet-model detections as helmet, violation or suppressed.

    Args:
        helmet_dets: ``(box, confidence, class_id)`` from the helmet model.
        moto_boxes: motorcycle boxes from the motorcycle model.
        img_w, img_h: image size, used to clip the expanded zones.
        factors: expansion factors for the motorcycle zones.
    """
    zones = [expand_box(b, factors, img_w, img_h) for b in moto_boxes]
    result = AssociationResult()

    for box, conf, cls in helmet_dets:
        if cls == with_helmet_id:
            result.helmets.append((box, conf))
        elif cls == without_helmet_id:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if any(center_in_box(z, cx, cy) for z in zones):
                result.violations.append((box, conf))
            else:
                result.suppressed.append((box, conf))

    return result
