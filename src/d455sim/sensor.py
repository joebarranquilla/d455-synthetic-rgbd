"""D455 depth degradation effects (bpy-free). Each effect: f(depth, ctx, cfg, rng) -> depth, 0 = invalid.

cfg  = ladder.yaml dict plus cfg["d455"] (see make_cfg).
ctx  = {"normals" HxWx3, "inst" HxW, "meta" dict, "K" 3x3, "baseline" m, "bg_roughness"} (see build_ctx).
"""
from __future__ import annotations

import zlib
from typing import Callable

import cv2
import numpy as np

from .config import load_config


def make_cfg(ladder_name: str = "ladder") -> dict:
    cfg = load_config(ladder_name)
    cfg["d455"] = load_config("d455")
    return cfg


def build_ctx(meta: dict, normals: np.ndarray, inst: np.ndarray, bg_roughness: float | None = None) -> dict:
    if bg_roughness is None:  # background = table material
        bg_roughness = load_config("scene")["materials"]["table"]["roughness"]
    return {"normals": normals.astype(np.float32), "inst": inst, "meta": meta, "K": np.asarray(meta["K"], float),
            "baseline": float(meta["baseline_m"]), "bg_roughness": float(bg_roughness)}


def depth_to_disp(z: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, fx * baseline / z, 0.0)


def disp_to_depth(d: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, fx * baseline / d, 0.0)


def _valid_min(z: np.ndarray, k: int) -> np.ndarray:
    """Min over a k x k window ignoring invalid (0) pixels; inf where no valid pixel."""
    zz = np.where(z > 0, z, np.inf).astype(np.float32)
    return cv2.erode(zz, np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)


def _valid_max(z: np.ndarray, k: int) -> np.ndarray:
    zz = np.where(z > 0, z, -np.inf).astype(np.float32)
    return cv2.dilate(zz, np.ones((k, k), np.uint8), borderType=cv2.BORDER_REPLICATE)


def discontinuity_mask(z: np.ndarray, thr_m: float) -> np.ndarray:
    """Pixels whose depth differs from a valid right/bottom neighbour by more than thr_m (both sides marked)."""
    v = z > 0
    m = np.zeros(z.shape, bool)
    dx = np.abs(z[:, 1:] - z[:, :-1]) > thr_m
    dx &= v[:, 1:] & v[:, :-1]
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    dy = np.abs(z[1:] - z[:-1]) > thr_m
    dy &= v[1:] & v[:-1]
    m[1:] |= dy
    m[:-1] |= dy
    return m


def quant(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Round disparity to 2^-frac_bits px and invalidate outside [min_range, max_range]."""
    fx, B, d455 = ctx["K"][0, 0], ctx["baseline"], cfg["d455"]
    step = 2.0 ** -d455["disparity_frac_bits"]
    d = np.round(depth_to_disp(depth, fx, B) / step) * step
    z = disp_to_depth(d, fx, B)
    ok = (z >= d455["min_range_m"]) & (z <= d455["max_range_m"])
    return np.where(ok, z, 0.0).astype(np.float32)


def noise(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise in the disparity domain => sigma_z = z^2 * sigma_d / (fx * B)."""
    fx, B = ctx["K"][0, 0], ctx["baseline"]
    sigma = cfg["d455"]["subpixel_rms_px"] * cfg["noise"]["scale"]
    d = depth_to_disp(depth, fx, B)
    d = np.where(d > 0, d + rng.normal(0.0, sigma, d.shape), 0.0)
    return disp_to_depth(d, fx, B).astype(np.float32)


def incidence_angle(normals: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Angle (rad) between surface normal and the reverse viewing ray, per pixel."""
    h, w = normals.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.abs((normals * ray).sum(-1)), 0.0, 1.0)).astype(np.float32)


def dropout_prob(theta: np.ndarray, rough: np.ndarray, c: dict) -> np.ndarray:
    th0, w, s = np.deg2rad(c["theta0_deg"]), np.deg2rad(c["width_deg"]), np.deg2rad(c["spec_sigma_deg"])
    graze = 1.0 / (1.0 + np.exp(-(theta - th0) / w))
    p = c["p_base"] + c["p_graze"] * graze + c["p_spec"] * (1.0 - rough) * np.exp(-theta ** 2 / (2 * s * s))
    return np.clip(p, 0.0, 1.0)


def _material_roughness(ctx: dict) -> np.ndarray:
    ids = ctx["inst"].astype(np.int64)
    lut = np.full(max(int(ids.max()), max((o["id"] for o in ctx["meta"]["objects"]), default=0)) + 1,
                  ctx["bg_roughness"])
    for o in ctx["meta"]["objects"]:
        lut[o["id"]] = o["material"]["roughness"]
    return lut[ids]


def _coherent_uniform(shape: tuple, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    """Spatially smooth field mapped to uniform [0,1) by rank, so `field < p` drops a fraction p in blobs."""
    f = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), sigma_px)
    ranks = np.empty(f.size, np.float32)
    ranks[np.argsort(f, axis=None)] = (np.arange(f.size) + 0.5) / f.size
    return ranks.reshape(shape)


def dropout(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    c = cfg["dropout"]
    p = dropout_prob(incidence_angle(ctx["normals"], ctx["K"]), _material_roughness(ctx), c)
    holes = _coherent_uniform(depth.shape, c["field_sigma_px"], rng) < p
    return np.where(holes, 0.0, depth).astype(np.float32)


def occlusion_shadow(depth: np.ndarray, K: np.ndarray, baseline: float, tol_rel: float) -> np.ndarray:
    """Forward-warp left depth into the right view (z-buffered); invalidate left pixels not seen there."""
    h, w = depth.shape
    valid = depth > 0
    u = np.broadcast_to(np.arange(w), (h, w))
    ur = np.rint(u - depth_to_disp(depth, K[0, 0], baseline)).astype(np.int64)
    inb = valid & (ur >= 0) & (ur < w)
    flat = np.arange(h)[:, None] * w + np.clip(ur, 0, w - 1)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, flat[inb], depth[inb])
    seen = inb & (depth <= zbuf[flat] * (1.0 + tol_rel))
    return np.where(seen, depth, 0.0).astype(np.float32)


def edges(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """(a) baseline occlusion shadow, (b) flying pixels, (c) foreground fattening."""
    c = cfg["edges"]
    z = depth
    if c["occlusion"]["enabled"]:
        z = occlusion_shadow(z, ctx["K"], ctx["baseline"], c["occlusion"]["tol_rel"])
    disc = discontinuity_mask(z, c["flying"]["grad_thr_m"])  # computed once, used by (b) and (c)
    out = _flying_pixels(z, disc, cfg, rng)
    return _foreground_fattening(z, out, disc, cfg)


def _flying_pixels(z: np.ndarray, disc: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    c = cfg["edges"]["flying"]
    if not c["enabled"]:
        return z.copy()
    k = 2 * c["band_px"] + 1
    band = cv2.dilate(disc.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & (z > 0)
    lo, hi = _valid_min(z, k), _valid_max(z, k)
    pick = band & (rng.random(z.shape) < c["prob"]) & np.isfinite(lo) & np.isfinite(hi)
    out = z.copy()
    out[pick] = lo[pick] + rng.random(z.shape)[pick] * (hi[pick] - lo[pick])
    return out.astype(np.float32)


def _foreground_fattening(z: np.ndarray, out: np.ndarray, disc: np.ndarray, cfg: dict) -> np.ndarray:
    c = cfg["edges"]["fatten"]
    if not c["enabled"]:
        return out
    mn = _valid_min(z, c["window_px"])
    fat = disc & np.isfinite(mn) & (z > 0)
    return np.where(fat, np.minimum(out, mn), out).astype(np.float32)


def edge_occlusion(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    del rng
    c = cfg["edges"]["occlusion"]
    if not c["enabled"]:
        return depth.astype(np.float32, copy=True)
    return occlusion_shadow(depth, ctx["K"], ctx["baseline"], c["tol_rel"])


def edge_flying(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    del ctx
    c = cfg["edges"]["flying"]
    disc = discontinuity_mask(depth, c["grad_thr_m"])
    return _flying_pixels(depth, disc, cfg, rng)


def edge_fatten(depth: np.ndarray, ctx: dict, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    del ctx, rng
    c = cfg["edges"]["flying"]
    disc = discontinuity_mask(depth, c["grad_thr_m"])
    return _foreground_fattening(depth, depth.copy(), disc, cfg)


EFFECTS: dict[str, Callable] = {
    "quant": quant,
    "noise": noise,
    "dropout": dropout,
    "edges": edges,
    "edge_occlusion": edge_occlusion,
    "edge_flying": edge_flying,
    "edge_fatten": edge_fatten,
}


def level_effects(name: str, cfg: dict) -> list[str]:
    """Effects of a level name: L0..L4 or Lfull_minus_<effect>."""
    if name.startswith("Lfull_minus_"):
        drop = name[len("Lfull_minus_"):]
        if drop not in EFFECTS:
            raise ValueError(f"unknown effect {drop}")
        return [e for e in cfg["levels"][cfg["full"]] if e != drop]
    return list(cfg["levels"][name])


def effect_rng(cfg: dict, sample_seed: int, effect: str) -> np.random.Generator:
    """Independent stream per effect, so leave-one-out keeps the other effects' random draws identical."""
    return np.random.default_rng([int(cfg["seed"]), int(sample_seed), zlib.crc32(effect.encode())])


def apply_effects(
    depth: np.ndarray,
    ctx: dict,
    cfg: dict,
    effects: list[str],
    sample_seed: int,
    noise_scale: float | None = None,
) -> np.ndarray:
    unknown = set(effects) - set(EFFECTS)
    if unknown:
        raise ValueError(f"unknown effects: {', '.join(sorted(unknown))}")
    z = depth.astype(np.float32)
    ordered = [e for e in cfg["effect_order"] if e in effects]
    ordered.extend(e for e in effects if e not in cfg["effect_order"])
    for e in ordered:
        effect_cfg = cfg
        if e == "noise" and noise_scale is not None:
            effect_cfg = {**cfg, "noise": {**cfg["noise"], "scale": noise_scale}}
        z = EFFECTS[e](z, ctx, effect_cfg, effect_rng(cfg, sample_seed, e))
    return z


def apply_level(depth: np.ndarray, ctx: dict, cfg: dict, level: str, noise_scale: float | None = None) -> np.ndarray:
    return apply_effects(depth, ctx, cfg, level_effects(level, cfg), ctx["meta"]["seed"], noise_scale)
