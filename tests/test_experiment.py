import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_stats as rt  # noqa: E402
import run_experiment as experiment_runner  # noqa: E402
import experiment_analysis as analysis  # noqa: E402
from d435sim import sensor  # noqa: E402
from d435sim.config import load_config  # noqa: E402

LEVEL = 0.95


def test_bootstrap_ci_covers_truth_and_shared_resamples():
    rng = np.random.default_rng(0)
    S, per = 60, 10
    n = np.full(S, per, float)
    s_a = rng.binomial(per, 0.5, S).astype(float)
    s_b = np.minimum(s_a + rng.binomial(per, 0.05, S), per)          # b is a per-scene superset of a
    idx = rt.boot_indices(S, 1000, 1)
    r = rt.rate_with_ci(s_a, n, idx, LEVEL)
    assert r["ci_lo"] < s_a.sum() / n.sum() < r["ci_hi"]
    boot = rt.boot_rate(s_b, n, idx)
    assert np.all(boot >= rt.boot_rate(s_a, n, idx))


def test_ci_resamples_scenes_not_instances():
    # all instances of a scene share one outcome: a scene-level CI must be wider than an instance-level one
    S, per = 20, 50
    s = np.array([per, 0] * (S // 2), float)
    n = np.full(S, float(per))
    idx = rt.boot_indices(S, 2000, 0)
    scene_ci = rt.rate_with_ci(s, n, idx, LEVEL)
    width_scene = scene_ci["ci_hi"] - scene_ci["ci_lo"]
    p, N = s.sum() / n.sum(), n.sum()
    assert width_scene > 3 * 2 * 1.96 * np.sqrt(p * (1 - p) / N)  # far above the iid-instance width


def test_per_scene_counts_are_aligned_with_requested_scenes():
    df = pd.DataFrame({"scene": ["b", "a", "a"], "class": ["profile", "bracket", "profile"], "success": [True, False, True]})
    s, n = rt.per_scene_counts(df, ["a", "b", "c"], "all")
    assert s.tolist() == [1, 1, 0] and n.tolist() == [2, 1, 0]


def test_experiment_config_does_not_require_git_tags():
    cfg = load_config("experiment")
    assert "required_tag" not in cfg
    assert "frozen_paths" not in cfg


def test_experiment_arm_plan_has_all_declared_conditions():
    cfg = load_config("experiment")
    arms = experiment_runner._arm_specs(cfg, sensor.make_cfg("ladder"))
    assert len(arms) == 20
    assert cfg["noise_scale_arms"]["1.0"] == "L2"
    assert arms["noise_s8"]["effects"] == ["quant", "noise"]
    assert arms["noise_s8"]["noise_scale"] == 8.0


def test_continuous_bootstrap_uses_shared_scene_resamples():
    estimate, lower, upper = analysis._bootstrap_quantiles(
        np.array([1.0, 3.0]), np.array([0, 1]), np.array([[1, 0], [0, 2]]), 0.5, 1.0
    )
    assert estimate == pytest.approx(2.0)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(3.0)
