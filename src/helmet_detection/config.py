"""Load the project configuration and resolve paths against the repository root."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Falls back to ``configs/config.yaml``."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Return ``path`` unchanged if absolute, otherwise relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the ``--config`` option shared by every script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="path to the YAML config (default: configs/config.yaml)")
    return parser
