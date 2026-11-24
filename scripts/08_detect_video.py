import cv2
from ultralytics import YOLO

# config
MODEL_PATH = r"C:\ProjectHD\runs\detect\runs\helmet\v4_M\weights\best.pt"
SOURCE     = 0  # 0 = webcam, or path to a video file
CONF       = 0.5
SAVE_OUT   = False
OUT_PATH   = r"C:\ProjectHD\output.mp4"

COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}  # green = with_helmet, red = without_helmet
NAMES  = {0: "with helmet", 1: "no helmet"}

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open source: {SOURCE}")

# Set up a video writer if we want to save the annotated output
writer = None
if SAVE_OUT:
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print("Running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF, verbose=False)[0]

    for box in results.boxes:
        cls   = int(box.cls)
        conf  = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLORS.get(cls, (255, 255, 0))
        label = f"{NAMES.get(cls, cls)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Helmet Detection", frame)
    if writer:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer:
    writer.release()
    print(f"Saved to {OUT_PATH}")
cv2.destroyAllWindows()
