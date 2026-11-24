"""Gradio demo: image, webcam and video tabs running the full violation pipeline.

Run from the repository root:  python app/web_app.py
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np

from helmet_detection.config import DEFAULT_CONFIG, load_config
from helmet_detection.pipeline import ViolationPipeline

LEGEND = (
    "Green: with helmet &nbsp;|&nbsp; Red: violation (no helmet on a motorcycle) "
    "&nbsp;|&nbsp; Cyan: suppressed (no motorcycle nearby, e.g. pedestrian) "
    "&nbsp;|&nbsp; Orange: motorcycle"
)


def _temp_mp4() -> str:
    """Create an empty temporary .mp4 path (safe replacement for tempfile.mktemp)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        return f.name


def reencode_h264(src: str) -> str:
    """Re-encode to H.264 + yuv420p so browsers can play the video inline."""
    dst = _temp_mp4()
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", src,
         "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p", dst],
        check=True,
        capture_output=True,
    )
    return dst


def build_demo(pipeline: ViolationPipeline) -> gr.Blocks:
    """Create the Gradio interface around a pipeline instance."""

    def detect_image(img_rgb: np.ndarray | None) -> np.ndarray | None:
        """Gradio passes RGB; the pipeline expects BGR."""
        if img_rgb is None:
            return None
        result_bgr = pipeline.annotate(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    def detect_video(video_path: str | None) -> str | None:
        """Process every frame and return a browser-playable H.264 video."""
        if video_path is None:
            return None
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        raw_path = _temp_mp4()
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(pipeline.annotate(frame))
        finally:
            cap.release()
            writer.release()

        try:
            return reencode_h264(raw_path)
        finally:
            Path(raw_path).unlink(missing_ok=True)

    with gr.Blocks(title="Helmet Violation Detection", theme=gr.themes.Base()) as demo:
        gr.Markdown("# Helmet Violation Detection\n" + LEGEND)

        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    img_in = gr.Image(label="Upload image", type="numpy")
                    img_out = gr.Image(label="Detection result", type="numpy")
                gr.Button("Detect", variant="primary").click(
                    fn=detect_image, inputs=img_in, outputs=img_out
                )

            with gr.Tab("Webcam (live)"):
                with gr.Row():
                    cam_in = gr.Image(sources=["webcam"], streaming=True,
                                      label="Webcam feed", type="numpy")
                    cam_out = gr.Image(label="Detection result", type="numpy")
                cam_in.stream(fn=detect_image, inputs=cam_in, outputs=cam_out)

            with gr.Tab("Video"):
                with gr.Row():
                    vid_in = gr.Video(label="Upload video")
                    vid_out = gr.Video(label="Annotated output")
                gr.Button("Process video", variant="primary").click(
                    fn=detect_video, inputs=vid_in, outputs=vid_out
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Helmet violation detection web demo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    build_demo(ViolationPipeline(load_config(args.config))).launch(share=args.share)


if __name__ == "__main__":
    main()
