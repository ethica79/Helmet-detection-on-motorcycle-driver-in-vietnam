from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

ROOT_DIR     = Path(__file__).parent
FRAMES_DIR   = ROOT_DIR / "data" / "test_frames"
OUTPUT_DIR   = ROOT_DIR / "runs" / "phase2"
VIS_DIR      = OUTPUT_DIR / "pipeline_vis"
LOG_PATH     = OUTPUT_DIR / "pipeline_log.txt"

HELMET_MODEL = ROOT_DIR / "runs" / "detect" / "runs" / "helmet" / "v4_M" / "weights" / "best.pt"
MOTO_MODEL   = "yolov8Motorcyle.pt"

HELMET_CONF      = 0.35
MOTO_CONF        = 0.25
MOTO_EXPAND_X    = 1.3   # horizontal expansion (left/right)
MOTO_EXPAND_UP   = 2.0   # upward expansion × motorcycle height (from top of moto box)
MOTO_EXPAND_DOWN = 0.1   # downward expansion × motorcycle height (from bottom of moto box)


CLASS_WITH_HELMET    = 0
CLASS_WITHOUT_HELMET = 1


def log(msg, lf):
    print(msg)
    lf.write(msg + "\n")
    lf.flush()


def expand_box(x1, y1, x2, y2, expand_x, expand_up, expand_down, img_w, img_h):
    cx  = (x1 + x2) / 2
    w   = x2 - x1
    h   = y2 - y1
    ex1 = max(0.0, cx - (w * expand_x) / 2)
    ex2 = min(float(img_w), cx + (w * expand_x) / 2)
    ey1 = max(0.0, y1 - h * expand_up)       # extend upward to reach rider's head
    ey2 = min(float(img_h), y2 + h * expand_down)
    return ex1, ey1, ex2, ey2


def center_in_box(exp_box, cx, cy):
    x1, y1, x2, y2 = exp_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as lf:
        log("=" * 60, lf)
        log("PHASE 2 — TWO-MODEL ASSOCIATION PIPELINE", lf)
        log("=" * 60, lf)
        log(f"Helmet model  : {HELMET_MODEL}", lf)
        log(f"Moto model    : {MOTO_MODEL}", lf)
        log(f"HELMET_CONF      : {HELMET_CONF}", lf)
        log(f"MOTO_CONF        : {MOTO_CONF}", lf)
        log(f"MOTO_EXPAND_X    : {MOTO_EXPAND_X}  (horizontal)", lf)
        log(f"MOTO_EXPAND_UP   : {MOTO_EXPAND_UP}  (upward × moto height)", lf)
        log(f"MOTO_EXPAND_DOWN : {MOTO_EXPAND_DOWN}  (downward × moto height)", lf)
        log("", lf)

        if not FRAMES_DIR.exists():
            log(f"ERROR: Frames directory not found: {FRAMES_DIR}", lf)
            return

        image_paths = sorted(
            p for p in FRAMES_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )

        if not image_paths:
            log(f"ERROR: No images found in {FRAMES_DIR}", lf)
            return

        log(f"Images found : {len(image_paths)}", lf)
        log(f"Device       : {'GPU' if torch.cuda.is_available() else 'CPU'}", lf)
        log("", lf)

        helmet_model = YOLO(str(HELMET_MODEL))
        moto_model   = YOLO(MOTO_MODEL)

        total_with_helmet = 0
        total_without_raw = 0
        total_confirmed   = 0
        total_suppressed  = 0

        for img_path in image_paths:
            img      = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            helmet_results = helmet_model(str(img_path), conf=HELMET_CONF, verbose=False)[0]
            moto_results   = moto_model(str(img_path), conf=MOTO_CONF, verbose=False)[0]

            h_boxes = helmet_results.boxes
            m_boxes = moto_results.boxes

            # Motorcycle detections 
            moto_raw = [
                (m_boxes.xyxy[i].tolist(), m_boxes.conf[i].item())
                for i, c in enumerate(m_boxes.cls.tolist())
                if int(c) == MOTO_CLASS
            ]
            moto_expanded = [
                expand_box(*box, MOTO_EXPAND_X, MOTO_EXPAND_UP, MOTO_EXPAND_DOWN, img_w, img_h)
                for box, _ in moto_raw
            ]

            # Helmet detections — classify each without_helmet as confirmed or suppressed
            with_helmet_dets     = []
            confirmed_violations = []
            suppressed_dets      = []

            for i, cls in enumerate(h_boxes.cls.tolist()):
                box  = h_boxes.xyxy[i].tolist()
                conf = h_boxes.conf[i].item()

                if int(cls) == CLASS_WITH_HELMET:
                    with_helmet_dets.append((box, conf))

                elif int(cls) == CLASS_WITHOUT_HELMET:
                    bx1, by1, bx2, by2 = box
                    bcx = (bx1 + bx2) / 2
                    bcy = (by1 + by2) / 2
                    associated = any(center_in_box(exp, bcx, bcy) for exp in moto_expanded)
                    if associated:
                        confirmed_violations.append((box, conf))
                    else:
                        suppressed_dets.append((box, conf))

            n_with = len(with_helmet_dets)
            n_raw  = len(confirmed_violations) + len(suppressed_dets)
            n_conf = len(confirmed_violations)
            n_supp = len(suppressed_dets)
            n_moto = len(moto_raw)

            total_with_helmet += n_with
            total_without_raw += n_raw
            total_confirmed   += n_conf
            total_suppressed  += n_supp

            log(
                f"[{img_path.name}]  "
                f"with_helmet={n_with}  without_raw={n_raw}  "
                f"confirmed={n_conf}  suppressed={n_supp}  motorcycles={n_moto}",
                lf
            )

            # Annotate and save
            draw = ImageDraw.Draw(img)

            for box, conf in with_helmet_dets:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x1, max(0, y1 - 14)), f"helmet {conf:.2f}", fill="green")

            for box, conf in confirmed_violations:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(0, y1 - 14)), f"VIOLATION {conf:.2f}", fill="red")

            for box, conf in suppressed_dets:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
                draw.text((x1, max(0, y1 - 14)), f"suppressed {conf:.2f}", fill="yellow")

            for box, conf in moto_raw:
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                draw.text((x1, max(0, y1 - 14)), f"moto {conf:.2f}", fill="blue")

            img.save(VIS_DIR / img_path.name)

        # Summary
        supp_rate = 100.0 * total_suppressed / total_without_raw if total_without_raw else 0.0

        log("", lf)
        log("=" * 60, lf)
        log("SUMMARY", lf)
        log("=" * 60, lf)
        log(f"Total with_helmet detections   : {total_with_helmet}", lf)
        log(f"Total without_helmet (raw)     : {total_without_raw}", lf)
        log(f"  Confirmed violations         : {total_confirmed}", lf)
        log(f"  Suppressed (pedestrian)      : {total_suppressed}  ({supp_rate:.1f}% suppression rate)", lf)
        log("", lf)
        log(f"Visualizations : {VIS_DIR}", lf)
        log(f"Log            : {LOG_PATH}", lf)


if __name__ == "__main__":
    main()
