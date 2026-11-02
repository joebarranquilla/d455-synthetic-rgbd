import json

import numpy as np
import pytest

from d455sim import geometry as g
from d455sim.config import intrinsics, load_config

SCENE = load_config("scene")
GEOM = SCENE["geometry"]


@pytest.mark.parametrize("mm", [20, 30, 40])
def test_cross_section_area(mm):
    p = GEOM["profile_params"][str(mm)]
    s = mm / 1000
    segs = GEOM["circle_segments"]
    cut = 4 * (p["opening"] * p["lip"] + p["t_width"] * (p["depth"] - p["lip"]))
    r = p["bore_d"] / 2
    bore = 0.5 * segs * r * r * np.sin(2 * np.pi / segs)
    assert g.profile_area(s, p, segs) == pytest.approx(s * s - cut - bore, rel=1e-9)


@pytest.mark.parametrize("mm", [20, 30, 40])
def test_profile_mesh_volume_and_samples(mm):
    s, L = mm / 1000, 0.2
    V, F = g.profile_mesh(s, L, GEOM)
    vol = np.einsum("ij,ij->i", V[F][:, 0], np.cross(V[F][:, 1], V[F][:, 2])).sum() / 6
    area = g.profile_area(s, GEOM["profile_params"][str(mm)], GEOM["circle_segments"])
    assert vol == pytest.approx(area * L, rel=1e-6)
    pts = g.sample_model_points({"class": "profile", "dims_m": [s, s, L]}, 500, GEOM)
    assert pts.shape == (500, 3) and np.abs(pts[:, 2]).max() <= L / 2 + 1e-9


def test_bracket_mesh_and_symmetry():
    V, F = g.bracket_mesh(GEOM)
    vol = np.einsum("ij,ij->i", V[F][:, 0], np.cross(V[F][:, 1], V[F][:, 2])).sum() / 6
    assert 0 < vol < 0.03 * 0.03 * 0.003 * 2
    assert len(g.symmetries("profile", GEOM)) >= 4
    assert any(np.linalg.det(T[:3, :3]) < 0 for T in g.symmetries("bracket", GEOM))


def test_dist_to_zdepth():
    K = intrinsics(load_config("d455"))
    h, w = 480, 848
    z = np.full((h, w), 1.5)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(z)], -1)
    dist = z * np.linalg.norm(ray, axis=-1)
    np.testing.assert_allclose(g.dist_to_zdepth(dist, K), z, rtol=1e-5)
    assert g.dist_to_zdepth(dist, K).dtype == np.float32
    c = g.dist_to_zdepth(dist, K)[h // 2, w // 2]
    assert c == pytest.approx(1.5, rel=1e-4)


def test_meta_schema():
    from d455sim.meta import build_meta, validate_meta
    K = intrinsics(load_config("d455"))
    obj = {"id": 1, "class": "profile", "dims_m": [0.02, 0.02, 0.2], "T_cam_obj": np.eye(4),
           "material": {"roughness": 0.3, "metallic": 1.0, "anisotropy": 0.1, "type": "silver_anodised"},
           "n_visible_px": 10}
    m = build_meta(K, np.eye(4), 0.095, 7, [obj])
    validate_meta(json.loads(json.dumps(m)))
    with pytest.raises(AssertionError):
        validate_meta({"K": m["K"]})
