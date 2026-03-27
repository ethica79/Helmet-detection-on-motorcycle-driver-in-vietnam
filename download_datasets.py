import os
import time
from roboflow import Roboflow

API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual Roboflow API key
ROOT_DIR = r"C:\ProjectHate"
BASE_DIR = os.path.join(ROOT_DIR, "datasets")

rf = Roboflow(api_key=API_KEY)

helmet_datasets = [
    {
        "name": "NCKH-2023 (Vietnamese)",
        "workspace": "nckh-2023",
        "project": "helmet-detection-project",
        "version": 19,
        "save_dir": os.path.join(BASE_DIR, "helmet", "nckh_2023")
    },
    {
        "name": "Helmet & No-Helmet Rider Detection",
        "workspace": "gw-khadatkar-and-sv-wasule",
        "project": "helmet-and-no-helmet-rider-detection",
        "version": 6,
        "save_dir": os.path.join(BASE_DIR, "helmet", "khadatkar")
    },
    {
        "name": "Motorcycle Riders Without Helmet",
        "workspace": "spresearchwork",
        "project": "motorcycle-riders-without-helmet",
        "version": 1,
        "save_dir": os.path.join(BASE_DIR, "helmet", "spresearchwork")
    },
    {
        "name": "Non Bao Hiem (thien-phuoc)",
        "workspace": "thien-phuoc",
        "project": "non-bao-hiem",
        "version": 1,
        "save_dir": os.path.join(BASE_DIR, "helmet", "thien_phuoc")
    },
    {
        "name": "Non Bao Hiem (tuandung)",
        "workspace": "tuandung-3ed5z",
        "project": "non-bao-hiem-n5bdk",
        "version": 1,
        "save_dir": os.path.join(BASE_DIR, "helmet", "tuandung")
    },
    {
        "name": "No Helmet No Ride (abdullah)",
        "workspace": "abdullah-kqdi3",
        "project": "no-helmet-no-ride",
        "version": 2,
        "save_dir": os.path.join(BASE_DIR, "helmet", "abdullah")
    },
    {
        "name": "Motorbike Helmet (alex)",
        "workspace": "alex-56cf0",
        "project": "motorbike-helmet",
        "version": 3,
        "save_dir": os.path.join(BASE_DIR, "helmet", "alex")
    },
]

def download_with_retry(workspace, project_name, version_num, save_dir, fmt="yolov8", max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            os.makedirs(save_dir, exist_ok=True)
            os.chdir(save_dir)
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            version.download(fmt)
            os.chdir(ROOT_DIR)  # always return to absolute root
            return True
        except Exception as e:
            os.chdir(ROOT_DIR)
            if attempt < max_retries:
                wait = attempt * 5
                print(f"   ⚠️  Attempt {attempt} failed. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return False

print("=" * 50)
print("Starting helmet dataset downloads...")
print("=" * 50)

for ds in helmet_datasets:
    if os.path.exists(ds["save_dir"]) and os.listdir(ds["save_dir"]):
        print(f"\n✅ Already exists, skipping: {ds['name']}")
        continue
    print(f"\n⬇️  Downloading: {ds['name']}")
    success = download_with_retry(
        ds["workspace"],
        ds["project"],
        ds["version"],
        ds["save_dir"],
        fmt=ds.get("format", "yolov8")
    )
    if success:
        print(f"Saved to: {ds['save_dir']}")
    else:
        print(f"Skipped: {ds['name']}")

print("\n" + "=" * 50)
print("Download complete!")
print("=" * 50)