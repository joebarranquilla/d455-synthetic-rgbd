import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_tables as rt  # noqa: E402
import run_experiment as rx  # noqa: E402
from d455sim.config import load_config  # noqa: E402

LEVEL = 0.95


def test_bootstrap_ci_covers_truth_and_pairing():
    rng = np.random.default_rng(0)
    S, per = 60, 10
    n = np.full(S, per, float)
    s_a = rng.binomial(per, 0.5, S).astype(float)
    s_b = np.minimum(s_a + rng.binomial(per, 0.05, S), per)          # b is a per-scene superset of a
    idx = rt.boot_indices(S, 1000, 1)
    r = rt.rate_with_ci(s_a, n, idx, LEVEL)
    assert r["ci_lo"] < s_a.sum() / n.sum() < r["ci_hi"]
    d = rt.paired_diff(s_a, n, s_b, n, idx, LEVEL)
    assert d["diff"] > 0 and not d["no_detectable_effect"]           # consistent gain: CI excludes 0
    same = rt.paired_diff(s_a, n, s_a, n, idx, LEVEL)
    assert same["diff"] == 0 and same["no_detectable_effect"] and same["ci_lo"] == same["ci_hi"] == 0


def test_noise_only_difference_flagged_as_no_effect():
    rng = np.random.default_rng(3)
    S = 40
    n = np.full(S, 8.0)
    a, b = rng.binomial(8, 0.6, S).astype(float), rng.binomial(8, 0.6, S).astype(float)
    d = rt.paired_diff(a, n, b, n, rt.boot_indices(S, 1000, 2), LEVEL)
    assert d["no_detectable_effect"]


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


def test_per_scene_counts_alignment_and_severity():
    df = pd.DataFrame({"scene": ["b", "a", "a"], "class": ["profile", "bracket", "profile"], "success": [True, False, True]})
    s, n = rt.per_scene_counts(df, ["a", "b", "c"], "all")
    assert s.tolist() == [1, 1, 0] and n.tolist() == [2, 1, 0]
    cfg = load_config("loc")
    d = pd.DataFrame({"class": ["profile", "bracket"], "lateral_offset_mm": [10.0, np.nan], "axis_angle_deg": [1.0, np.nan],
                      "centroid_err_mm": [np.nan, 2.5], "rot_err_deg": [np.nan, 40.0]})
    sev = rt.severity(d, cfg)
    assert sev[0] == pytest.approx(10 / cfg["profile"]["success"]["lateral_mm"])
    assert sev[1] == pytest.approx(40 / cfg["bracket"]["success"]["rot_deg"])


def test_refuses_without_freeze_tag(monkeypatch):
    cfg = load_config("experiment")
    monkeypatch.setattr(rx, "git", lambda *a: type("R", (), {"stdout": "", "returncode": 0})())
    with pytest.raises(SystemExit, match="v1-freeze is missing"):
        rx.check_freeze(cfg)
    monkeypatch.setattr(rx, "git", lambda *a: type("R", (), {"stdout": "v1-freeze\n", "returncode": 1})())
    with pytest.raises(SystemExit, match="differ from tag"):
        rx.check_freeze(cfg)
