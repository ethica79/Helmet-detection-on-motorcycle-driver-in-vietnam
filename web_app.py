import os
import subprocess
import tempfile
import threading
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

# Paths & thresholds

ROOT_DIR = Path(__file__).parent

HELMET_MODEL_PATH = ROOT_DIR / "runs" / "detect" / "runs" / "helmet" / "v4_M" / "weights" / "best.pt"
MOTO_MODEL_PATH   = ROOT_DIR / "yolov8Motorcyle.pt"

HELMET_CONF      = 0.35
MOTO_CONF        = 0.25
MOTO_EXPAND_X    = 1.3
MOTO_EXPAND_UP   = 2.0   # extends 2x moto height upward to reach rider's head
MOTO_EXPAND_DOWN = 0.1

CLASS_WITH_HELMET    = 0
CLASS_WITHOUT_HELMET = 1

# Model loading (lazy, loaded once on first inference)

_helmet_model: YOLO | None = None
_moto_model:   YOLO | None = None
_lock = threading.Lock()


def load_models():
    global _helmet_model, _moto_model
    with _lock:
        if _helmet_model is None:
            _helmet_model = YOLO(str(HELMET_MODEL_PATH))
            _moto_model   = YOLO(str(MOTO_MODEL_PATH))
    return _helmet_model, _moto_model


# Pipeline core

def _expand_zone(x1, y1, x2, y2, img_w, img_h):
    """Asymmetric expansion: wide upward to reach rider's head above vehicle body."""
    cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
    return (
        max(0.0,          cx - w * MOTO_EXPAND_X / 2),
        max(0.0,          y1 - h * MOTO_EXPAND_UP),
        min(float(img_w), cx + w * MOTO_EXPAND_X / 2),
        min(float(img_h), y2 + h * MOTO_EXPAND_DOWN),
    )


def _inside(zone, cx, cy):
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def run_pipeline(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Two-model association pipeline on one BGR frame.
    - Helmet model  -> with_helmet / without_helmet detections
    - Moto model    -> motorcycle bounding boxes (single-class custom model)
    - without_helmet inside a motorcycle zone -> VIOLATION (red)
    - without_helmet outside any zone         -> suppressed pedestrian (cyan)
    """
    helmet_model, moto_model = load_models()
    img_h, img_w = frame_bgr.shape[:2]

    h_res = helmet_model(frame_bgr, conf=HELMET_CONF, verbose=False)[0]
    m_res = moto_model(frame_bgr,   conf=MOTO_CONF,   verbose=False)[0]

    moto_boxes = [m_res.boxes.xyxy[i].tolist() for i in range(len(m_res.boxes))]
    moto_zones = [_expand_zone(*b, img_w, img_h) for b in moto_boxes]

    helmets    = []
    violations = []
    suppressed = []

    for i, cls in enumerate(h_res.boxes.cls.tolist()):
        box  = h_res.boxes.xyxy[i].tolist()
        conf = h_res.boxes.conf[i].item()
        if int(cls) == CLASS_WITH_HELMET:
            helmets.append((box, conf))
        else:
            bx1, by1, bx2, by2 = box
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if any(_inside(z, cx, cy) for z in moto_zones):
                violations.append((box, conf))
            else:
                suppressed.append((box, conf))

    # draw results
    out = frame_bgr.copy()

    for box in moto_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 120, 0), 1)

    for box, conf in helmets:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"helmet {conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    for box, conf in violations:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"VIOLATION {conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    for box, conf in suppressed:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 200), 1)

    # stats overlay
    for i, text in enumerate([
        f"Helmets:     {len(helmets)}",
        f"Violations:  {len(violations)}",
        f"Motorcycles: {len(moto_boxes)}",
    ]):
        y = 22 + i * 22
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out


# Gradio wrapper functions

def detect_image(img_rgb: np.ndarray) -> np.ndarray:
    """Gradio passes RGB; pipeline expects BGR."""
    if img_rgb is None:
        return None
    frame_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = run_pipeline(frame_bgr)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def reencode_h264(src: str) -> str:
    """Re-encode to H.264 + yuv420p so browsers can play the video inline."""
    dst = tempfile.mktemp(suffix=".mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-i", src,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            dst,
        ],
        check=True,
        capture_output=True,
    )
    return dst


def detect_video(video_path: str) -> str:
    """Process each frame and return a browser-playable H.264 video."""
    if video_path is None:
        return None

    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_path = tempfile.mktemp(suffix=".mp4")
    writer   = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(run_pipeline(frame))

    cap.release()
    writer.release()

    out_path = reencode_h264(raw_path)
    os.unlink(raw_path)
    return out_path


# Gradio UI

with gr.Blocks(title="Helmet Violation Detection", theme=gr.themes.Base()) as demo:

    gr.Markdown(
        "# Helmet Violation Detection\n"
        "Two-model pipeline: **helmet detector** (YOLOv8m, v4_M) "
        "+ **motorcycle detector** with asymmetric zone association.\n\n"
        "Green: with helmet &nbsp;|&nbsp; Red: violation (no helmet on motorcycle) "
        "&nbsp;|&nbsp; Cyan: suppressed (pedestrian) &nbsp;|&nbsp; Orange: motorcycle"
    )

    with gr.Tabs():

        with gr.Tab("Image"):
            with gr.Row():
                img_in  = gr.Image(sources=["upload"], label="Input image", type="numpy")
                img_out = gr.Image(label="Detection result", type="numpy")
            gr.Button("Detect", variant="primary").click(
                fn=detect_image, inputs=img_in, outputs=img_out
            )

        with gr.Tab("Webcam (live)"):
            with gr.Row():
                cam_in  = gr.Image(sources=["webcam"], streaming=True,
                                   label="Webcam feed", type="numpy")
                cam_out = gr.Image(label="Detection result", type="numpy")
            cam_in.stream(fn=detect_image, inputs=cam_in, outputs=cam_out)

        with gr.Tab("Video"):
            with gr.Row():
                vid_in  = gr.Video(label="Upload video")
                vid_out = gr.Video(label="Annotated output")
            gr.Button("Process video", variant="primary").click(
                fn=detect_video, inputs=vid_in, outputs=vid_out
            )

demo.launch()
