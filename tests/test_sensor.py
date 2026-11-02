import numpy as np
import pytest

from d455sim import sensor as s
from d455sim.config import intrinsics, load_config

CFG = s.make_cfg()
K = intrinsics(CFG["d455"])
B = CFG["d455"]["baseline_m"]
H, W = 480, 848


def make_ctx(normals=None, inst=None, objects=()):
    meta = {"K": K.tolist(), "baseline_m": B, "seed": 7, "objects": list(objects)}
    n = np.zeros((H, W, 3), np.float32) if normals is None else normals
    if normals is None:
        n[..., 2] = -1.0
    return s.build_ctx(meta, n, np.zeros((H, W), np.uint16) if inst is None else inst, bg_roughness=0.7)


def test_disparity_depth_round_trip_and_quant_bound():
    z = np.linspace(0.4, 6.0, 1000)
    d = s.depth_to_disp(z, K[0, 0], B)
    np.testing.assert_allclose(s.disp_to_depth(d, K[0, 0], B), z, rtol=1e-12)
    zq = s.quant(z.astype(np.float32).reshape(1, -1), make_ctx(), CFG, np.random.default_rng(0))[0]
    dq = s.depth_to_disp(zq, K[0, 0], B)
    ok = zq > 0
    assert ok.all()
    assert np.abs(dq - d).max() <= 2.0 ** -CFG["d455"]["disparity_frac_bits"] / 2 * 1.01
    assert s.quant(np.full((2, 2), 7.0, np.float32), make_ctx(), CFG, np.random.default_rng(0)).max() == 0


@pytest.mark.parametrize("z0", [0.6, 1.5, 3.0])
def test_noise_std_matches_analytic(z0):
    z = np.full((300, 300), z0, np.float32)
    out = s.noise(z, make_ctx(), CFG, np.random.default_rng(1))
    expected = z0 ** 2 * CFG["d455"]["subpixel_rms_px"] * CFG["noise"]["scale"] / (K[0, 0] * B)
    assert np.std(out - z0) == pytest.approx(expected, rel=0.03)


def test_occlusion_two_plane_step():
    z = np.full((H, W), 1.0, np.float32)
    z[:, 400:] = 0.5
    out = s.occlusion_shadow(z, K, B, CFG["edges"]["occlusion"]["tol_rel"])
    d_bg, d_fg = K[0, 0] * B / 1.0, K[0, 0] * B / 0.5
    shadow = (out[:, :400] == 0)[:, int(d_bg) + 2:].sum(axis=1)  # ignore the left-border band
    assert np.all(np.abs(shadow - (d_fg - d_bg)) <= 2)
    assert np.all(out[:, 400:] == 0.5)
    assert np.all(out[:, : int(d_bg)] == 0)  # warped outside the right image


def test_dropout_fraction_matches_probability():
    ctx = make_ctx()
    out = s.dropout(np.full((H, W), 1.0, np.float32), ctx, CFG, np.random.default_rng(2))
    p = s.dropout_prob(s.incidence_angle(ctx["normals"], K), np.full((H, W), 0.7), CFG["dropout"]).mean()
    assert (out == 0).mean() == pytest.approx(p, abs=0.01)


def test_levels_and_leave_one_out():
    assert s.level_effects("L2", CFG) == ["quant", "noise"]
    assert s.level_effects("Lfull_minus_edges", CFG) == ["quant", "noise", "dropout"]
    z = np.full((H, W), 1.0, np.float32)
    ctx = make_ctx()
    np.testing.assert_array_equal(s.apply_level(z, ctx, CFG, "L0"), z)
    a = s.apply_level(z, ctx, CFG, "L4")
    np.testing.assert_array_equal(a, s.apply_level(z, ctx, CFG, "L4"))
