from pathlib import Path
import cv2

VIDEO_PATH  = Path(__file__).parent / "video" / "traffic1.mp4"
OUTPUT_DIR  = Path(__file__).parent / "data" / "test_frames"
NUM_FRAMES  = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
dur   = total / fps if fps else 0

print(f"Video: {VIDEO_PATH.name}")
print(f"Total frames: {total}  |  FPS: {fps:.1f}  |  Duration: {dur:.1f}s")
print(f"Extracting {NUM_FRAMES} evenly spaced frames...")

indices = [int(i * total / NUM_FRAMES) for i in range(NUM_FRAMES)]
saved = 0

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        continue
    out_path = OUTPUT_DIR / f"frame_{idx:06d}.jpg"
    cv2.imwrite(str(out_path), frame)
    saved += 1

cap.release()
print(f"Saved {saved} frames to {OUTPUT_DIR}")
