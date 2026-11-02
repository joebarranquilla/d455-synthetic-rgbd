"""Cycles rendering: RGB (denoised, AA) + geometry passes (no AA)."""
from __future__ import annotations

import blenderproc as bproc
import bpy
import numpy as np

from . import geometry as geo

_outputs_enabled = False


def init(cfg: dict, d455: dict, K: np.ndarray) -> None:
    r = cfg["render"]
    bproc.init()
    bproc.renderer.set_render_devices(use_only_cpu=r["compute_device"] != "GPU",
                                      desired_gpu_device_type=r["compute_device_type"])
    bproc.camera.set_intrinsics_from_K_matrix(K, d455["width"], d455["height"])


def render_sample(cfg: dict, d455: dict, K: np.ndarray, T_world_cam_cv: np.ndarray, cycles_seed: int) -> dict:
    """Returns rgb uint8 HxWx3, depth float32 (0=invalid), normals float16 (camera frame), inst uint16, cls uint8."""
    from .scene import cv_to_gl
    r = cfg["render"]
    bpy.context.scene.cycles.seed = cycles_seed
    bproc.utility.reset_keyframes()
    bproc.camera.add_camera_pose(cv_to_gl(T_world_cam_cv))

    global _outputs_enabled
    if not _outputs_enabled:
        bproc.renderer.enable_distance_output(activate_antialiasing=False)
        bproc.renderer.enable_normals_output()
        bproc.renderer.enable_segmentation_output(map_by=["inst_id", "class_id"],
                                                  default_values={"inst_id": 0, "class_id": 0})
        _outputs_enabled = True

    # geometry pass first: ~1 spp, near-zero pixel filter => no antialiasing on depth/normals/masks
    bproc.renderer.set_denoiser(None)
    bproc.renderer.set_max_amount_of_samples(r["geom_spp"])
    bpy.context.scene.cycles.filter_width = r["geom_filter_width_px"]
    g = bproc.renderer.render()

    bproc.renderer.set_max_amount_of_samples(r["rgb_spp"])
    bproc.renderer.set_denoiser(r["denoiser"])
    bpy.context.scene.cycles.filter_width = r["rgb_filter_width_px"]
    rgb = np.asarray(bproc.renderer.render()["colors"][0])
    rgb = np.clip(np.nan_to_num(rgb)[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)

    dist = np.asarray(g["distance"][0], np.float64)
    depth = geo.dist_to_zdepth(dist, K)
    ok = np.isfinite(dist) & (dist < r["invalid_distance_m"]) & (depth >= d455["min_range_m"]) \
        & (depth <= d455["max_range_m"])
    depth = np.where(ok, depth, 0.0).astype(np.float32)

    n_w = np.asarray(g["normals"][0], np.float64)[..., :3]
    R_cv = T_world_cam_cv[:3, :3]
    n_c = n_w @ R_cv  # = (R^T n) for row vectors
    n_c /= np.maximum(np.linalg.norm(n_c, axis=-1, keepdims=True), 1e-9)
    n_c[depth == 0] = 0.0

    return {"rgb": rgb, "depth": depth, "normals": n_c.astype(np.float16),
            "inst": np.asarray(g["inst_id_segmaps"][0]).astype(np.uint16),
            "cls": np.asarray(g["class_id_segmaps"][0]).astype(np.uint8)}
