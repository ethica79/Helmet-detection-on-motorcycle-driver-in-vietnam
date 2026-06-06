from ultralytics import YOLO
import torch


def main():
    DATA_YAML = r"C:\ProjectHate\datasets\helmet_merged\projecthate-8\data.yaml"
    MODEL     = "yolov8m.pt"
    PROJECT   = "runs/helmet"
    RUN_NAME  = "v4_M"
    EPOCHS    = 100
    IMG_SIZE  = 640
    BATCH     = 4  # keep low if running on a single GPU with limited VRAM

    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    model = YOLO(MODEL)

    model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        project  = PROJECT,
        name     = RUN_NAME,
        patience = 20,   # stop early if no improvement for 20 epochs
        save     = True,
        plots    = True,
        workers  = 0,    # set to 0 on Windows to avoid DataLoader multiprocessing issues
        cls      = 1.0,  # weight for classification loss (higher = penalise wrong class more)
        device   = "0" if torch.cuda.is_available() else "cpu",
    )

    print("Training complete!")
    print(f"Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")


if __name__ == '__main__':
    main()
