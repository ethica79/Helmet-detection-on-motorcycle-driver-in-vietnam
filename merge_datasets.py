import os
import shutil
import yaml
from pathlib import Path

ROOT_DIR = r"C:\ProjectHate"

DATASET_MAPPINGS = {
    "nckh_2023": {
        "path": r"C:\ProjectHate\datasets\helmet\nckh_2023\Helmet-Detection-Project-19",
        "mapping": {0: 0, 1: None, 2: None}
    },
    "khadatkar": {
        "path": r"C:\ProjectHate\datasets\helmet\khadatkar\Helmet-and-no-helmet-rider-detection-6",
        "mapping": {0: 0, 1: 1, 2: None}
    },
    "spresearchwork": {
        "path": r"C:\ProjectHate\datasets\helmet\spresearchwork\Motorcycle-riders-without-helmet-1",
        "mapping": {0: 1, 1: 0}
    },
    "thien_phuoc": {
        "path": r"C:\ProjectHate\datasets\helmet\thien_phuoc\Nón-bảo-hiểm-1",
        "mapping": {0: 1, 1: 0}  # no=without_helmet, yes=with_helmet
    },
    "tuandung": {
        "path": r"C:\ProjectHate\datasets\helmet\tuandung\nón-bảo-hiểm-1",
        "mapping": {0: 0, 1: 1}  # helmet=with_helmet, non_helmet=without_helmet
    },
    "abdullah": {
        "path": r"C:\ProjectHate\datasets\helmet\abdullah\NO-Helmet-NO-Ride-2",
        "mapping": {0: 0, 1: 1, 2: None}  # helmet=with_helmet, nohelmet=without_helmet, riders=drop
    },
    "alex": {
        "path": r"C:\ProjectHate\datasets\helmet\alex\motorbike-helmet-3",
        "mapping": {0: 0}  # helmets=with_helmet only, no without_helmet annotations
    },
}

OUTPUT_DIR = r"C:\ProjectHate\datasets\helmet_merged"
SPLITS = ["train", "valid", "test"]

def process_label_file(src_label, dst_label, mapping):
    new_lines = []
    with open(src_label, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            orig_class = int(parts[0])
            new_class = mapping.get(orig_class)
            if new_class is None:
                continue
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts))
    if new_lines:
        with open(dst_label, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        return True
    return False

def merge():
    for split in SPLITS:
        Path(f"{OUTPUT_DIR}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/{split}/labels").mkdir(parents=True, exist_ok=True)

    counters = {split: 0 for split in SPLITS}
    skipped = 0

    for ds_name, ds_info in DATASET_MAPPINGS.items():
        ds_path = ds_info["path"]
        mapping = ds_info["mapping"]
        print(f"\nProcessing: {ds_name}")

        for split in SPLITS:
            img_dir = Path(f"{ds_path}/{split}/images")
            lbl_dir = Path(f"{ds_path}/{split}/labels")

            if not img_dir.exists():
                print(f"No {split} split found, skipping")
                continue

            images = list(img_dir.glob("*.*"))
            for img_path in images:
                stem = img_path.stem
                lbl_path = lbl_dir / f"{stem}.txt"

                if not lbl_path.exists():
                    skipped += 1
                    continue

                new_name = f"{ds_name}_{stem}"
                dst_img = Path(f"{OUTPUT_DIR}/{split}/images/{new_name}{img_path.suffix}")
                dst_lbl = Path(f"{OUTPUT_DIR}/{split}/labels/{new_name}.txt")

                success = process_label_file(lbl_path, dst_lbl, mapping)
                if success:
                    shutil.copy2(img_path, dst_img)
                    counters[split] += 1
                else:
                    skipped += 1

        print(f"Done")

    yaml_content = {
        "path": OUTPUT_DIR,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["with_helmet", "without_helmet"]
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n" + "=" * 50)
    print("Merge complete!")
    print("=" * 50)
    print(f"  train: {counters['train']} images")
    print(f"  valid: {counters['valid']} images")
    print(f"  test:  {counters['test']} images")
    print(f"  skipped: {skipped} files")

if __name__ == "__main__":
    merge()