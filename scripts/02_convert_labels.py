"""Remap class IDs to the project standard (0=with_helmet, 1=without_helmet)
and convert polygon/segmentation labels to bounding boxes, in place.
"""

from __future__ import annotations

import yaml

from helmet_detection.config import base_parser, load_config, resolve
from helmet_detection.labels import PROJECT_CLASS_NAMES, build_remap, convert_label_lines

SPLITS = ["train", "valid", "test"]


def main() -> None:
    args = base_parser("Remap classes and convert polygons to bboxes.").parse_args()
    cfg = load_config(args.config)
    dataset_root = resolve(cfg["paths"]["dataset"])
    data_yaml = dataset_root / "data.yaml"

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    remap = build_remap(data.get("names", []))
    for old_id, new_id in remap.items():
        if new_id is None:
            print(f"  WARNING: unknown class '{data['names'][old_id]}' (id={old_id}) will be dropped")
    print(f"Class remap: {remap}\n")

    totals = {"files_changed": 0, "remapped": 0, "polygons": 0, "dropped": 0}
    for split in SPLITS:
        lbl_dir = dataset_root / split / "labels"
        if not lbl_dir.exists():
            print(f"Skipping {split}: {lbl_dir} not found")
            continue

        changed_files = 0
        for lbl_path in lbl_dir.glob("*.txt"):
            content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            new_lines, stats = convert_label_lines(content.splitlines(), remap)
            if stats.remapped or stats.polygons or stats.dropped:
                lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                changed_files += 1
            totals["remapped"] += stats.remapped
            totals["polygons"] += stats.polygons
            totals["dropped"] += stats.dropped

        print(f"[{split}] files changed: {changed_files}")
        totals["files_changed"] += changed_files

    # Update data.yaml so YOLO trains with the project class names
    data["names"] = PROJECT_CLASS_NAMES
    data["nc"] = len(PROJECT_CLASS_NAMES)
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print("\nDone!")
    print(f"  Files changed:      {totals['files_changed']}")
    print(f"  Polygon -> bbox:    {totals['polygons']}")
    print(f"  Class IDs remapped: {totals['remapped']}")
    print(f"  Lines dropped:      {totals['dropped']}")


if __name__ == "__main__":
    main()
