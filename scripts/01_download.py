"""Download the merged helmet dataset from Roboflow.

Requires a Roboflow API key in the ROBOFLOW_API_KEY environment variable
(or in a .env file at the repository root, see .env.example).

The merged Roboflow project was built mainly from these public Roboflow Universe sources
(plus several small datasets added later to strengthen the without_helmet class):
- NCKH-2023 (Vietnamese):          nckh-2023/helmet-detection-project v19
- Helmet & No-Helmet Rider:         gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection v6
- Motorcycle Riders Without Helmet: spresearchwork/motorcycle-riders-without-helmet v1
- Non Bao Hiem (thien-phuoc):       thien-phuoc/non-bao-hiem v1
- Non Bao Hiem (tuandung):          tuandung-3ed5z/non-bao-hiem-n5bdk v1
- No Helmet No Ride (abdullah):     abdullah-kqdi3/no-helmet-no-ride v2
- Motorbike Helmet (alex):          alex-56cf0/motorbike-helmet v3
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

from helmet_detection.config import REPO_ROOT, base_parser, load_config, resolve


@contextlib.contextmanager
def working_dir(path: Path) -> Iterator[None]:
    """Temporarily cd into ``path``: the Roboflow SDK downloads into the CWD."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    args = base_parser("Download the merged helmet dataset from Roboflow.").parse_args()
    cfg = load_config(args.config)
    dl = cfg["download"]

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and fill it in.")

    save_dir = resolve(cfg["paths"]["datasets_dir"]) / "helmet_merged"
    if save_dir.exists() and any(save_dir.iterdir()):
        print(f"Already downloaded, skipping: {save_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dl['workspace']}/{dl['project']} v{dl['version']} to {save_dir}")
    project = Roboflow(api_key=api_key).workspace(dl["workspace"]).project(dl["project"])
    with working_dir(save_dir):
        project.version(dl["version"]).download(dl["format"])
    print("Done.")


if __name__ == "__main__":
    main()
