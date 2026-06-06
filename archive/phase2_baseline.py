from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

ROOT_DIR   = Path(__file__).parent
FRAMES_DIR = ROOT_DIR / "data" / "test_frames"
OUTPUT_DIR = ROOT_DIR / "runs" / "phase2"
VIS_DIR    = OUTPUT_DIR / "baseline_vis"
LOG_PATH   = OUTPUT_DIR / "baseline_log.txt"

MOTO_CLASS = 3     # COCO class index for motorcycle
MOTO_CONF  = 0.25


def log(msg, lf):
    print(msg)
    lf.write(msg + "\n")
    lf.flush()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as lf:
        log("=" * 60, lf)
        log("PHASE 2 — BASELINE MOTORCYCLE DETECTION", lf)
        log("=" * 60, lf)
        log(f"Model           : yolov8m.pt (COCO pretrained)", lf)
        log(f"Frames dir      : {FRAMES_DIR}", lf)
        log(f"Conf threshold  : {MOTO_CONF}", lf)
        log(f"COCO class      : {MOTO_CLASS} (motorcycle)", lf)
        log("", lf)

        if not FRAMES_DIR.exists():
            log(f"ERROR: Frames directory not found: {FRAMES_DIR}", lf)
            log("Create the folder and add .jpg/.png images before running.", lf)
            return

        image_paths = sorted(
            p for p in FRAMES_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        if not image_paths:
            log(f"ERROR: No images found in {FRAMES_DIR}", lf)
            return

        log(f"Images found: {len(image_paths)}", lf)
        log(f"Device      : {'GPU' if torch.cuda.is_available() else 'CPU'}", lf)
        log("", lf)

        model = YOLO("yolov8m.pt")

        total_images                   = len(image_paths)
        images_with_moto               = 0
        all_confidences                = []
        moto_counts_per_detected_image = []

        for img_path in image_paths:
            results = model(str(img_path), conf=MOTO_CONF, verbose=False)[0]
            boxes   = results.boxes

            moto_idx   = [i for i, c in enumerate(boxes.cls.tolist()) if int(c) == MOTO_CLASS]
            moto_confs = [boxes.conf[i].item() for i in moto_idx]
            moto_xyxy  = [boxes.xyxy[i].tolist() for i in moto_idx]
            n_moto     = len(moto_idx)

            log(f"[{img_path.name}]  motorcycles={n_moto}", lf)
            for j, (conf, box) in enumerate(zip(moto_confs, moto_xyxy)):
                x1, y1, x2, y2 = box
                log(f"  [{j}] conf={conf:.3f}  box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})", lf)

            if n_moto > 0:
                images_with_moto += 1
                all_confidences.extend(moto_confs)
                moto_counts_per_detected_image.append(n_moto)

            # Save annotated image (blue boxes for motorcycles only)
            img  = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for box, conf in zip(moto_xyxy, moto_confs):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                draw.text((x1, max(0, y1 - 14)), f"moto {conf:.2f}", fill="blue")
            img.save(VIS_DIR / img_path.name)

        # Summary
        pct_with_moto = 100.0 * images_with_moto / total_images if total_images else 0.0
        avg_conf      = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        avg_per_img   = (
            sum(moto_counts_per_detected_image) / len(moto_counts_per_detected_image)
            if moto_counts_per_detected_image else 0.0
        )

        log("", lf)
        log("=" * 60, lf)
        log("SUMMARY", lf)
        log("=" * 60, lf)
        log(f"Total images processed         : {total_images}", lf)
        log(f"Images with motorcycle         : {images_with_moto} ({pct_with_moto:.1f}%)", lf)
        log(f"Avg confidence (when detected) : {avg_conf:.3f}", lf)
        log(f"Avg motorcycles/image          : {avg_per_img:.2f}  (detected images only)", lf)
        log("", lf)
        log(f"Visualizations : {VIS_DIR}", lf)
        log(f"Log            : {LOG_PATH}", lf)


if __name__ == "__main__":
    main()
