from ultralytics import YOLO
import torch

def main():
    DATA_YAML = r"C:\ProjectHate\datasets\helmet_merged\projecthate-8\data.yaml"
    MODEL     = "yolov8m.pt"  # Fresh start
    PROJECT   = "runs/helmet"
    RUN_NAME  = "v4_M"        # Fresh start on Dataset 8 (55/45 class ratio)
    EPOCHS    = 100
    IMG_SIZE  = 640
    BATCH     = 4

    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    model = YOLO(MODEL)

    results = model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        project  = PROJECT,
        name     = RUN_NAME,
        patience = 20,
        save     = True,
        plots    = True,
        workers  = 0,
        cls      = 1.0,
        device   = "0" if torch.cuda.is_available() else "cpu",
    )

    print("Training complete!")
    print(f"Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

if __name__ == '__main__':
    main()