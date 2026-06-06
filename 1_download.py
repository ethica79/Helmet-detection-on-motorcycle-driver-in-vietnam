import os
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROOT_DIR = r"C:\ProjectHate"
BASE_DIR = os.path.join(ROOT_DIR, "datasets")

rf = Roboflow(api_key=API_KEY)

# The merged dataset we actually use for training.
# The commented-out entries below are the original sources that were
# combined into "ProjectHate" on Roboflow — kept here for reference.
helmet_datasets = [
    {
        "name": "ProjectHate (merged + augmented)",
        "workspace": "therences-workspace",
        "project": "projecthate",
        "version": 8,
        "save_dir": os.path.join(BASE_DIR, "helmet_merged")
    },
    # Original datasets merged into "ProjectHate" above:
    # - NCKH-2023 (Vietnamese):            workspace=nckh-2023,                  project=helmet-detection-project,            version=19
    # - Helmet & No-Helmet Rider:           workspace=gw-khadatkar-and-sv-wasule, project=helmet-and-no-helmet-rider-detection, version=6
    # - Motorcycle Riders Without Helmet:   workspace=spresearchwork,              project=motorcycle-riders-without-helmet,    version=1
    # - Non Bao Hiem (thien-phuoc):         workspace=thien-phuoc,                project=non-bao-hiem,                        version=1
    # - Non Bao Hiem (tuandung):            workspace=tuandung-3ed5z,             project=non-bao-hiem-n5bdk,                  version=1
    # - No Helmet No Ride (abdullah):       workspace=abdullah-kqdi3,             project=no-helmet-no-ride,                   version=2
    # - Motorbike Helmet (alex):            workspace=alex-56cf0,                 project=motorbike-helmet,                    version=3
]


# Roboflow's SDK downloads into the current working directory,
# so we cd into the target folder before downloading, then always
# return to ROOT_DIR in the finally block regardless of success/failure.
def download(workspace, project_name, version_num, save_dir, fmt="yolov8"):
    os.makedirs(save_dir, exist_ok=True)
    os.chdir(save_dir)
    try:
        rf.workspace(workspace).project(project_name).version(version_num).download(fmt)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False
    finally:
        os.chdir(ROOT_DIR)


print("Starting helmet dataset downloads...")

for ds in helmet_datasets:
    # Skip if the folder already exists and has files in it
    if os.path.exists(ds["save_dir"]) and os.listdir(ds["save_dir"]):
        print(f"Already exists, skipping: {ds['name']}")
        continue
    print(f"Downloading: {ds['name']}")
    success = download(
        ds["workspace"],
        ds["project"],
        ds["version"],
        ds["save_dir"],
        fmt=ds.get("format", "yolov8")
    )
    if success:
        print(f"  Saved to: {ds['save_dir']}")
    else:
        print(f"  Failed: {ds['name']}")

print("Done.")
