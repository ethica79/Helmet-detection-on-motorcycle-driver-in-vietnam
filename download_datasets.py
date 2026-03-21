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
]

def download_with_retry(workspace, project_name, version_num, save_dir, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            os.makedirs(save_dir, exist_ok=True)
            os.chdir(save_dir)
            project = rf.workspace(workspace).project(project_name)
            version = project.version(version_num)
            version.download("yolov8")
            os.chdir(ROOT_DIR)  # always return to absolute root
            return True
        except Exception as e:
            os.chdir(ROOT_DIR)
            if attempt < max_retries:
                wait = attempt * 5
                print(f"   ⚠️  Attempt {attempt} failed. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Failed after {max_retries} attempts: {e}")
                return False

print("=" * 50)
print("Starting helmet dataset downloads...")
print("=" * 50)

for ds in helmet_datasets:
    print(f"\n⬇️  Downloading: {ds['name']}")
    success = download_with_retry(
        ds["workspace"],
        ds["project"],
        ds["version"],
        ds["save_dir"]
    )
    if success:
        print(f"✅  Saved to: {ds['save_dir']}")
    else:
        print(f"❌  Skipped: {ds['name']}")

print("\n" + "=" * 50)
print("Download complete!")
print("=" * 50)