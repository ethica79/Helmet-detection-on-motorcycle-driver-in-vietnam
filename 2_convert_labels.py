"""
Remap class IDs to project standard (0=with_helmet, 1=without_helmet)
and convert any polygon/segmentation labels to bounding boxes.
"""

from pathlib import Path
import yaml

DATASET_ROOT = Path(r"C:\ProjectHate\datasets\helmet_merged\projecthate-8")
SPLITS       = ["train", "valid", "test"]

# Each source dataset uses slightly different class names for the same thing.
# This maps all known variants to our two standard IDs. None means drop the annotation.
NAME_TO_ID = {
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


# Reads data.yaml to find out what class IDs the dataset currently uses,
# then builds a mapping from those old IDs to our standard 0/1 IDs.
def build_remap(data_yaml_path):
    with open(data_yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    remap = {}
    for old_id, name in enumerate(data.get("names", [])):
        new_id = NAME_TO_ID.get(name.strip().lower())
        remap[old_id] = new_id
        if new_id is None:
            print(f"  WARNING: unknown class '{name}' (id={old_id}) will be dropped")
    return remap


# Converts a YOLO segmentation polygon line to a bounding box line.
# Polygon format:  class x1 y1 x2 y2 x3 y3 ... (variable points, normalized)
# Bbox format:     class cx cy w h (normalized)
def polygon_to_bbox(parts):
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
        print(f"Skipping {split} - labels dir not found")
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

            # Drop any class that doesn't map to our standard
            if new_cls is None:
                total_skipped += 1
                changed = True
                continue

            if new_cls != old_cls:
                parts[0] = str(new_cls)
                changed = True
                total_remapped += 1

            if len(parts) == 5:
                # Already in bbox format, just keep it
                new_lines.append(" ".join(parts))
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                # Polygon convert to bbox
                new_lines.append(polygon_to_bbox(parts))
                changed = True
            else:
                # Malformed line, skip it
                total_skipped += 1
                continue

        if changed:
            lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            converted += 1
        else:
            total_already += 1

    print(f"[{split}] Converted: {converted}")
    total_converted += converted

# Update data.yaml so YOLO knows the correct class names when training
with open(data_yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["names"] = ["with_helmet", "without_helmet"]
data["nc"] = 2
with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

print(f"\nDone!")
print(f"  Polygon -> bbox:     {total_converted}")
print(f"  Class IDs remapped:  {total_remapped}")
print(f"  Already correct:     {total_already}")
print(f"  Lines dropped:       {total_skipped}")
