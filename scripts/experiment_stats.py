"""Shared scene-level bootstrap statistics for experiment reports."""
from __future__ import annotations

import numpy as np
import pandas as pd


def boot_indices(n_scenes: int, n_boot: int, seed: int) -> np.ndarray:
    """Return shared scene resamples so differences between estimates are paired."""
    return np.random.default_rng(seed).integers(0, n_scenes, (n_boot, n_scenes))


def per_scene_counts(
    df: pd.DataFrame, scenes: list[str], cls: str, mask_col: str = "success"
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-scene (successes, instances), aligned with scenes."""
    selected = df if cls == "all" else df[df["class"] == cls]
    grouped = selected.groupby("scene")[mask_col].agg(["sum", "size"]).reindex(scenes).fillna(0)
    return grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)


def boot_rate(successes: np.ndarray, totals: np.ndarray, indices: np.ndarray) -> np.ndarray:
    resampled_totals = totals[indices].sum(1)
    return np.where(
        resampled_totals > 0,
        successes[indices].sum(1) / np.maximum(resampled_totals, 1),
        np.nan,
    )


def ci(values: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2
    return float(np.nanquantile(values, tail)), float(np.nanquantile(values, 1 - tail))


def rate_with_ci(
    successes: np.ndarray, totals: np.ndarray, indices: np.ndarray, level: float
) -> dict:
    lower, upper = ci(boot_rate(successes, totals, indices), level)
    rate = float(successes.sum() / totals.sum()) if totals.sum() else np.nan
    return {"rate": rate, "ci_lo": lower, "ci_hi": upper, "n_inst": int(totals.sum())}
