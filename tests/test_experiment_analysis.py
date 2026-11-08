import json
import sys
from pathlib import Path

import cv2
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_analysis  # noqa: E402
from d435sim import sensor  # noqa: E402
from d435sim.config import load_config  # noqa: E402


def test_experiment_analysis_generates_tables_and_figures(tmp_path, monkeypatch):
    config = load_config("experiment")
    saved_noise_sweep = []
    original_savefig = Figure.savefig

    def check_noise_sweep(fig, filename, *args, **kwargs):
        if Path(filename).name == "noise_sweep.png":
            fig.canvas.draw()
            for ax in fig.axes:
                np.testing.assert_allclose(ax.get_xticks(), config["noise_scales"])
                assert isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
                assert len(ax.xaxis.get_minorticklocs()) == 0
                labels = [label for label in ax.get_xticklabels() if label.get_visible()]
                bounds = [label.get_window_extent(fig.canvas.get_renderer()) for label in labels]
                assert all(not left.overlaps(right) for left, right in zip(bounds, bounds[1:]))
            assert len(fig.axes) == 2
            saved_noise_sweep.append(True)
        return original_savefig(fig, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_noise_sweep)
    arms = list(dict.fromkeys([
        *config["levels"],
        *(f"Lfull_minus_{effect}" for effect in config["loo_effects"]),
        *config["arms"],
    ]))
    scenes = []
    rows = []
    for scene_index, scene_id in enumerate(("00000", "00001")):
        scene = tmp_path / "test" / scene_id
        scene.mkdir(parents=True)
        scenes.append(scene)
        inst = np.zeros((12, 12), np.uint16)
        inst[:, :6] = 1
        inst[:, 6:] = 2
        cv2.imwrite(str(scene / "inst.png"), inst)
        objects = [{"id": 1, "class": "profile"}, {"id": 2, "class": "bracket"}]
        (scene / "meta.json").write_text(json.dumps({"objects": objects}))
        raw = np.full((12, 12), 0.5 + 0.1 * scene_index, np.float32)
        np.save(scene / "depth_gt.npy", raw)
        for level_index, level in enumerate(config["figures"]["hole_levels"]):
            depth = raw.copy()
            if level_index:
                depth[0, :level_index] = 0
            np.save(scene / f"depth_{level}.npy", depth)
        for arm_index, arm in enumerate(arms):
            for class_index, cls in enumerate(("profile", "bracket")):
                success = (arm_index + scene_index + class_index) % 3 != 0
                rows.append({
                    "scene": scene_id,
                    "level": arm,
                    "class": cls,
                    "success": success,
                    "failure_type": "" if success else "axis_flip",
                    "mean_depth_m": float(raw.mean()),
                    "n_valid_points": 72 - scene_index,
                    "lateral_offset_mm": 1.0 + arm_index if cls == "profile" else np.nan,
                    "axis_angle_deg": 1.0 + scene_index if cls == "profile" else np.nan,
                    "centroid_err_mm": 2.0 + arm_index if cls == "bracket" else np.nan,
                    "rot_err_deg": 2.0 + scene_index if cls == "bracket" else np.nan,
                })
    csv_path = tmp_path / "loc.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out = tmp_path / "analysis"

    summary = experiment_analysis.generate_all(
        out, csv_path, config, sensor.make_cfg("ladder"), scenes
    )

    assert summary["scenes"] == 2
    assert saved_noise_sweep == [True]
    for name in ("success_by_arm.csv", "continuous_errors.csv",
                 "isolated_ablation.csv", "noise_sweep.csv",
                 "hole_statistics.csv", "degradation_ladder.png",
                 "analysis_summary.json"):
        assert (out / name).is_file()


def test_range_statistics_reports_the_configured_cutoff_band(tmp_path):
    config = load_config("experiment")
    config["range_analysis"]["working_distance_m"] = [0.2, 0.9]
    d455 = load_config("camera")

    result = experiment_analysis.range_and_noise_statistics(
        [], config, sensor.make_cfg("ladder"), d455, tmp_path
    )

    metric = "fraction_of_working_distance_within_0.1m_above_0.3m_cutoff"
    near_cutoff = result[result.metric == metric]
    assert len(near_cutoff) == 1
    assert near_cutoff.iloc[0].value == pytest.approx(0.1 / 0.7)
