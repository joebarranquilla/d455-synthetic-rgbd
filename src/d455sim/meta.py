"""meta.json construction and validation (bpy-free)."""
from __future__ import annotations

import numpy as np

OBJ_KEYS = {"id", "class", "dims_m", "T_cam_obj", "material", "n_visible_px"}
MAT_KEYS = {"roughness", "metallic", "anisotropy", "type"}


def build_meta(K: np.ndarray, T_world_cam: np.ndarray, baseline_m: float, seed: int, objects: list[dict]) -> dict:
    objs = []
    for o in objects:
        d = dict(o)
        d["T_cam_obj"] = np.asarray(o["T_cam_obj"], float).tolist()
        d["dims_m"] = [float(x) for x in o["dims_m"]]
        d["n_visible_px"] = int(o["n_visible_px"])
        objs.append(d)
    return {"K": np.asarray(K, float).tolist(), "T_world_cam": np.asarray(T_world_cam, float).tolist(),
            "baseline_m": float(baseline_m), "seed": int(seed), "objects": objs}


def validate_meta(m: dict) -> None:
    assert {"K", "T_world_cam", "baseline_m", "seed", "objects"} <= set(m), "missing top-level keys"
    assert np.shape(m["K"]) == (3, 3) and np.shape(m["T_world_cam"]) == (4, 4)
    for o in m["objects"]:
        assert OBJ_KEYS <= set(o), "missing object keys"
        assert np.shape(o["T_cam_obj"]) == (4, 4)
        assert MAT_KEYS <= set(o["material"])
        assert o["class"] in ("profile", "bracket")
