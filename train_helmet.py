from ultralytics import YOLO
import torch

def main():
    DATA_YAML = r"C:\ProjectHate\datasets\helmet_merged\data.yaml"
    MODEL     = r"runs/detect/runs/helmet/v14/weights/best.pt"  # Start from the best weights of the previous run"
    PROJECT   = "runs/helmet"
    RUN_NAME  = "v2"  # Change this for each run to avoid overwriting previous results
    EPOCHS    = 50
    IMG_SIZE  = 640
    BATCH     = 8

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
        device   = "0" if torch.cuda.is_available() else "cpu",
    )

    print("Training complete!")
    print(f"Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")

if __name__ == '__main__':
    main()