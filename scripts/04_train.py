"""Train a YOLOv8 helmet detector. Hyperparameters come from the ``train`` section of the config."""

from __future__ import annotations

import torch
from ultralytics import YOLO

from helmet_detection.config import base_parser, load_config, resolve


def main() -> None:
    parser = base_parser("Train a YOLOv8 helmet detector.")
    parser.add_argument("--run-name", help="override train.run_name")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--batch", type=int, help="override train.batch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr = cfg["train"]
    run_name = args.run_name or tr["run_name"]
    project = resolve(cfg["paths"]["runs_dir"]) / "helmet"
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")

    model = YOLO(tr["model"])
    model.train(
        data=str(resolve(cfg["paths"]["dataset"]) / "data.yaml"),
        epochs=args.epochs or tr["epochs"],
        imgsz=tr["imgsz"],
        batch=args.batch or tr["batch"],
        project=str(project),
        name=run_name,
        patience=tr["patience"],   # early stopping
        save=True,
        plots=True,
        workers=tr["workers"],
        cls=tr["cls"],             # higher = penalise wrong class more
        device=device,
    )

    print("Training complete!")
    print(f"Best weights: {project / run_name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
