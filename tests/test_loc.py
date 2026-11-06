import numpy as np
import pytest

from d435sim import geometry as geo
from d435sim import loc
from d435sim.config import intrinsics, load_config

CFG = load_config("loc")
K = intrinsics(load_config("camera"))
SEED = 3
NOISE_M = 0.001
N_DENSE = 30000


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return loc._rodrigues(a * np.deg2rad(deg))


def visible_cloud(spec, T, rng):
    """Sample the model densely, keep the nearest point per pixel (hidden-surface removal), add noise."""
    pts = geo.sample_model_points(spec, N_DENSE, None, SEED) @ T[:3, :3].T + T[:3, 3]
    u = np.rint(pts[:, 0] / pts[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.rint(pts[:, 1] / pts[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    order = np.lexsort((pts[:, 2], v, u))
    first = np.r_[True, (u[order][1:] != u[order][:-1]) | (v[order][1:] != v[order][:-1])]
    P = pts[order][first]
    return P + rng.normal(0, NOISE_M, P.shape)


def pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


PROFILE = {"class": "profile", "dims_m": [0.03, 0.03, 0.25]}
BRACKET = {"class": "bracket", "dims_m": [0.03, 0.03, 0.03]}


@pytest.mark.parametrize("R", [rot([1, 0, 0], 100) @ rot([0, 0, 1], 30),
                               rot([0, 1, 0], 70) @ rot([1, 0, 0], 100),
                               rot([1, 1, 0], 160)])
def test_profile_recovered(R):
    T_gt = pose(R, [0.03, -0.02, 0.6])
    rng = np.random.default_rng(0)
    P = visible_cloud(PROFILE, T_gt, rng)
    res = loc.localize(P, PROFILE, T_gt, CFG)
    assert res["success"], res
    assert res["lateral_offset_mm"] < 3 and res["axis_angle_deg"] < 3


def test_profile_axis_sign_and_along_axis_not_in_success():
    T_gt = pose(rot([1, 0, 0], 90), [0, 0, 0.6])
    flipped = T_gt @ np.diag([1.0, -1.0, -1.0, 1.0])  # reversed axis, same shape
    shifted = T_gt.copy()
    shifted[:3, 3] += 0.05 * T_gt[:3, 2]  # 5 cm along the axis
    assert loc.evaluate_pose(flipped, T_gt, PROFILE, CFG)["axis_angle_deg"] == pytest.approx(0, abs=1e-6)
    ev = loc.evaluate_pose(shifted, T_gt, PROFILE, CFG)
    assert ev["along_axis_err_mm"] == pytest.approx(50, rel=1e-6) and ev["success"]


def test_too_few_points():
    res = loc.localize(np.random.default_rng(0).normal(size=(10, 3)) * 0.01 + [0, 0, 0.6], PROFILE, np.eye(4), CFG)
    assert not res["success"] and res["failure_type"] == "too_few_points"
    assert res["n_valid_points"] == 10


@pytest.mark.parametrize("R", [rot([0, 1, 0], 20) @ rot([1, 0, 0], 200),   # legs facing away from the camera
                               rot([1, 0, 0], 150) @ rot([0, 0, 1], 60),
                               rot([0, 0, 1], 100) @ rot([0, 1, 0], 150)])
def test_bracket_flip_hypotheses(R):
    T_gt = pose(R, [0.0, 0.02, 0.55])
    P = visible_cloud(BRACKET, T_gt, np.random.default_rng(1))
    res = loc.localize(P, BRACKET, T_gt, CFG)
    assert res["success"], res
