"""
Convert YOLO segmentation/polygon labels to YOLO bounding box format.
Polygon format: class x1 y1 x2 y2 x3 y3 ... (variable number of points)
BBox format:    class cx cy w h
"""

from pathlib import Path

DATASET_ROOT = Path(r"C:\ProjectHate\datasets\helmet_merged\projecthate-3")
SPLITS       = ["train", "valid", "test"]

def polygon_to_bbox(parts):
    """Convert polygon points to cx cy w h (all normalized 0-1)."""
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]  # every other value starting at 0 = x coords
    ys = coords[1::2]  # every other value starting at 1 = y coords

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

total_converted = 0
total_skipped   = 0
total_already   = 0

for split in SPLITS:
    lbl_dir = DATASET_ROOT / split / "labels"
    if not lbl_dir.exists():
        print(f"Skipping {split} — labels dir not found")
        continue

    converted = 0
    for lbl_path in lbl_dir.glob("*.txt"):
        content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue

        new_lines = []
        changed = False

        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                # Already bbox format
                new_lines.append(line.strip())
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                # Polygon format — convert
                new_lines.append(polygon_to_bbox(parts))
                changed = True
            else:
                # Malformed line — skip
                total_skipped += 1
                continue

        if changed:
            lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            converted += 1
        else:
            total_already += 1

    print(f"[{split}] Converted: {converted}")
    total_converted += converted

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Converted to bbox : {total_converted}")
print(f"  Already bbox      : {total_already}")
print(f"  Malformed lines   : {total_skipped}")
