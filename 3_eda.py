"""
EDA — Exploratory Data Analysis for helmet_merged dataset
Student-friendly version — no numpy, simple logic only.
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image

DATASET_ROOT = Path(r"C:\ProjectHate\datasets\helmet_merged\projecthate-8")
SPLITS       = ["train", "valid", "test"]
CLASS_NAMES  = {0: "with_helmet", 1: "without_helmet"}
OUTPUT_FILE  = Path(r"C:\ProjectHate\report\eda\eda_results.txt")


def average(lst):
    return sum(lst) / len(lst) if lst else 0


def read_split(split):
    img_dir = DATASET_ROOT / split / "images"
    lbl_dir = DATASET_ROOT / split / "labels"
    records = []

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        try:
            w, h = Image.open(img_path).size
        except Exception:
            continue

        boxes = []
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, encoding="utf-8", errors="ignore") as lbl_file:
                for line in lbl_file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cls = int(parts[0])
                            bw  = float(parts[3])
                            bh  = float(parts[4])
                            boxes.append((cls, bw, bh))
                        except ValueError:
                            continue

        records.append({"w": w, "h": h, "boxes": boxes})

    return records


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        def log(text=""):
            print(text)
            out.write(text + "\n")

        def sep():
            log("-" * 55)

        log("Loading dataset")
        all_records = {s: read_split(s) for s in SPLITS}
        log("Done\n")

        # 1. Dataset overview: count images, boxes, class distribution
        log("1. DATASET OVERVIEW")
        sep()
        log(f"{'Split':<8} {'Images':>8} {'Total Boxes':>12} {'with_helmet':>13} {'without_helmet':>16}")
        sep()

        for split in SPLITS:
            records = all_records[split]
            count_0 = 0
            count_1 = 0

            for r in records:
                for (cls, bw, bh) in r["boxes"]:
                    if cls == 0:
                        count_0 += 1
                    elif cls == 1:
                        count_1 += 1

            total_boxes = count_0 + count_1
            pct_0 = 100 * count_0 / total_boxes if total_boxes > 0 else 0
            pct_1 = 100 * count_1 / total_boxes if total_boxes > 0 else 0

            log(f"{split:<8} {len(records):>8} {total_boxes:>12} "
                f"{count_0:>8} ({pct_0:.1f}%)  {count_1:>8} ({pct_1:.1f}%)")

        # 2. Image size stats
        log("\n2. IMAGE SIZES (pixels)")
        sep()

        all_widths   = []
        all_heights  = []
        unique_sizes = defaultdict(int)

        for split in SPLITS:
            for r in all_records[split]:
                all_widths.append(r["w"])
                all_heights.append(r["h"])
                unique_sizes[(r["w"], r["h"])] += 1

        log(f"{'':12} {'min':>6} {'max':>6} {'average':>9}")
        sep()
        log(f"{'width':<12} {min(all_widths):>6} {max(all_widths):>6} {average(all_widths):>9.0f}")
        log(f"{'height':<12} {min(all_heights):>6} {max(all_heights):>6} {average(all_heights):>9.0f}")

        log(f"\nUnique resolutions: {len(unique_sizes)}")
        log(f"\n{'Resolution':<14} {'Count':>6}")
        sep()
        sorted_sizes = sorted(unique_sizes.items(), key=lambda x: -x[1])
        for size, cnt in sorted_sizes[:10]:
            log(f"{str(size[0])+'x'+str(size[1]):<14} {cnt:>6}")

        # 3. Count bounding box sizes (width, height, area) by class
        log("\n3. BOUNDING BOX SIZES (normalised 0-1)")
        sep()

        bw_by_class   = {0: [], 1: []}
        bh_by_class   = {0: [], 1: []}
        area_by_class = {0: [], 1: []}

        for split in SPLITS:
            for r in all_records[split]:
                for (cls, bw, bh) in r["boxes"]:
                    if cls in bw_by_class:
                        bw_by_class[cls].append(bw)
                        bh_by_class[cls].append(bh)
                        area_by_class[cls].append(bw * bh)

        log(f"{'Class':<20} {'Metric':<8} {'Min':>6} {'Max':>6} {'Average':>9}")
        sep()
        for cls, name in CLASS_NAMES.items():
            bws   = bw_by_class[cls]
            bhs   = bh_by_class[cls]
            areas = area_by_class[cls]

            if not bws:
                continue

            log(f"{name:<20} {'width':<8} {min(bws):>6.3f} {max(bws):>6.3f} {average(bws):>9.3f}")
            log(f"{'':20} {'height':<8} {min(bhs):>6.3f} {max(bhs):>6.3f} {average(bhs):>9.3f}")
            log(f"{'':20} {'area':<8} {min(areas):>6.3f} {max(areas):>6.3f} {average(areas):>9.3f}")
            log()

        # 4. Count how many objects are in each image
        log("4. OBJECTS PER IMAGE")
        sep()

        all_counts = []
        for split in SPLITS:
            for r in all_records[split]:
                all_counts.append(len(r["boxes"]))

        images_with_zero = sum(1 for c in all_counts if c == 0)

        log(f"{'Total images':<22} {len(all_counts):>8}")
        log(f"{'Average objects/image':<22} {average(all_counts):>8.2f}")
        log(f"{'Max objects in one image':<22} {max(all_counts):>8}")
        log(f"{'Images with 0 boxes':<22} {images_with_zero:>8}")

        log("\nDone!")

    print(f"\nResults saved to: {OUTPUT_FILE}")


main()
