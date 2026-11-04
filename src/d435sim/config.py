"""YAML config loading and derived camera intrinsics (bpy-free)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f) or {}


def intrinsics(d455: dict) -> np.ndarray:
    """3x3 K from width/height/hfov; square pixels, integer pixel-centre convention."""
    w, h = d455["width"], d455["height"]
    fx = (w / 2.0) / np.tan(np.deg2rad(d455["hfov_deg"]) / 2.0)
    return np.array([[fx, 0, (w - 1) / 2.0], [0, fx, (h - 1) / 2.0], [0, 0, 1.0]])
