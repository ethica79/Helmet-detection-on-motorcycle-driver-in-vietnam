"""Label utilities: class remapping across source datasets and polygon-to-bbox conversion.

All label lines are in YOLO format with normalized coordinates:
- bbox:    ``class cx cy w h``
- polygon: ``class x1 y1 x2 y2 x3 y3 ...`` (at least 3 points)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Each source dataset uses different class names for the same thing.
# Maps every known variant to the project IDs; None means drop the annotation.
NAME_TO_ID: dict[str, int | None] = {
    "helmet":           0,
    "with_helmet":      0,
    "with helmet":      0,
    "no helmet":        1,
    "no-helmet":        1,
    "no_helmet":        1,
    "nohelmet":         1,
    "without_helmet":   1,
    "without helmet":   1,
    "head":             1,
    "null":             None,  # no useful annotation info
    "0":                None,  # unnamed/placeholder class
}

PROJECT_CLASS_NAMES = ["with_helmet", "without_helmet"]


@dataclass
class ConversionStats:
    """Counters reported by :func:`convert_label_lines`."""

    remapped: int = 0
    polygons: int = 0
    dropped: int = 0


def build_remap(names: Sequence[str]) -> dict[int, int | None]:
    """Map a dataset's class IDs (index in ``names``) to project IDs.

    Unknown names map to ``None`` and their annotations are dropped.
    """
    return {old_id: NAME_TO_ID.get(name.strip().lower()) for old_id, name in enumerate(names)}


def polygon_to_bbox(parts: Sequence[str]) -> str:
    """Convert a YOLO polygon line (already split) into a bbox line.

    The bbox is the tightest axis-aligned box around all polygon points.
    """
    cls = parts[0]
    coords = [float(v) for v in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return f"{cls} {cx:.6f} {cy:.6f} {x_max - x_min:.6f} {y_max - y_min:.6f}"


def convert_label_lines(
    lines: Iterable[str], remap: Mapping[int, int | None]
) -> tuple[list[str], ConversionStats]:
    """Remap classes and convert polygons to bboxes for one label file.

    Lines whose class maps to ``None`` and malformed lines are dropped.
    """
    stats = ConversionStats()
    out: list[str] = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        new_cls = remap.get(int(parts[0]))
        if new_cls is None:
            stats.dropped += 1
            continue
        if str(new_cls) != parts[0]:
            parts[0] = str(new_cls)
            stats.remapped += 1

        if len(parts) == 5:
            out.append(" ".join(parts))
        elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            out.append(polygon_to_bbox(parts))
            stats.polygons += 1
        else:
            stats.dropped += 1

    return out, stats
