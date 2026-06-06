"""
1. Remap class IDs to project standard (0=with_helmet, 1=without_helmet)
2. Convert YOLO segmentation/polygon labels to YOLO bounding box format.

Polygon format: class x1 y1 x2 y2 x3 y3 ... (variable number of points)
BBox format:    class cx cy w h
"""

from pathlib import Path
import yaml

DATASET_ROOT = Path(r"C:\ProjectHate\datasets\helmet_merged\projecthate-8")
SPLITS       = ["train", "valid", "test"]

# Maps any known class name variant → our standard class ID
# Maps any known class name variant → our standard class ID (None = drop)
# All keys are lowercase matching is done with .lower() in build_remap()
NAME_TO_ID = {
    # with_helmet variants
    "helmet":           0,
    "with_helmet":      0,
    "with helmet":      0,
    # without_helmet variants
    
    "no helmet":        1,
    "no-helmet":        1,
    "no_helmet":        1,
    "nohelmet":         1,
    "no-helmet":        1,
    "without_helmet":   1,
    "without helmet":   1,
    "head":             1,
    # ambiguous / drop
    "null":             None,  # no annotation info
    "0":                None,  # unnamed class
}


def build_remap(data_yaml_path):
    """Read data.yaml and return {old_id: new_id} mapping. None = drop."""
    with open(data_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    remap = {}
    for old_id, name in enumerate(names):
        new_id = NAME_TO_ID.get(name.strip().lower())
        remap[old_id] = new_id  # None means drop
        if new_id is None:
            print(f"  WARNING: unknown class '{name}' (id={old_id}) — will be dropped")
    return remap


def polygon_to_bbox(parts):
    """Convert polygon points to cx cy w h (all normalized 0-1)."""
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w  = x_max - x_min
    h  = y_max - y_min

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# --- Build class remap from data.yaml ---
data_yaml_path = DATASET_ROOT / "data.yaml"
remap = build_remap(data_yaml_path)
print(f"Class remap: {remap}\n")

total_converted = 0
total_remapped  = 0
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
            if not parts:
                continue

            old_cls = int(parts[0])
            new_cls = remap.get(old_cls)

            if new_cls is None:
                # Drop classes not in our standard
                total_skipped += 1
                changed = True
                continue

            if new_cls != old_cls:
                parts[0] = str(new_cls)
                changed = True
                total_remapped += 1

            if len(parts) == 5:
                # Already bbox format
                new_lines.append(" ".join(parts))
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

# Fix data.yaml names to our standard
with open(data_yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["names"] = ["with_helmet", "without_helmet"]
data["nc"] = 2
with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
print(f"\ndata.yaml names updated to: ['with_helmet', 'without_helmet']")

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Polygon → bbox    : {total_converted}")
print(f"  Class IDs remapped: {total_remapped}")
print(f"  Already correct   : {total_already}")
print(f"  Lines dropped     : {total_skipped}")