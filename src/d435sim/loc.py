"""Estimate and evaluate individual part poses from depth.

Masked depth is back-projected, filtered, and aligned to a sampled model using PCA and trimmed ICP.
Poses are T_cam_obj (model to camera). Profile success uses lateral and axis-angle errors; along-axis
error is reported but excluded because it is unobservable when the ends are occluded.

Bracket rotation error accounts for proper rotational symmetries.
# ASSUMPTION: improper mirror symmetries cannot be matched by a rotation estimate.

Failures are classified as too few points, outlier dominated, wrong scale, axis flip, or ICP diverged.
Clustered mode also reports merged, fragmented, and missed instances.
"""
from __future__ import annotations

import itertools
import time
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import geometry as geo

NAN = float("nan")
_FLIPS = [np.diag(d) for d in ([1.0, 1, 1], [-1.0, -1, 1], [-1.0, 1, -1], [1.0, -1, -1])]  # det +1 sign flips


def _axis_hypotheses(all_axes: bool) -> list[np.ndarray]:
    """4 sign-flip hypotheses, or all 24 proper axis permutations/flips (PCA axes of an L are near-degenerate)."""
    if not all_axes:
        return _FLIPS
    perms = [np.eye(3)[list(p)] for p in itertools.permutations(range(3))]
    return [P @ F for P in perms for F in _FLIPS if np.linalg.det(P @ F) > 0]


# ------------------------------------------------------------------ points
def backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Valid depth pixels inside mask -> camera-frame points (N,3) and their (v, u) pixel indices (N,2)."""
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64)
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return P, np.stack([v, u], 1)


def remove_outliers(P: np.ndarray, cfg: dict) -> np.ndarray:
    """Statistical outlier removal; returns indices of kept points."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, ind = pc.remove_statistical_outlier(cfg["sor"]["nb_neighbors"], cfg["sor"]["std_ratio"])
    return np.asarray(ind, int)


# ------------------------------------------------------------------ model
@lru_cache(maxsize=256)
def _model(cls: str, dims: tuple, cfg_key: tuple):
    n, seed, radius, nn = cfg_key
    pts = geo.sample_model_points({"class": cls, "dims_m": list(dims)}, n, None, seed)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn))
    return pts, np.asarray(pc.normals), cKDTree(pts)


def get_model(spec: dict, cfg: dict):
    m = cfg["model"]
    dims = tuple(round(float(x), 6) for x in spec["dims_m"]) if spec["class"] == "profile" else (0.0,)
    return _model(spec["class"], dims, (m["n_points"], m["seed"], m["normal_radius_m"], m["normal_max_nn"]))


# ------------------------------------------------------------------ ICP
def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def trimmed_icp(P: np.ndarray, model, T_init: np.ndarray, cfg: dict) -> tuple[np.ndarray, float, float]:
    """Trimmed point-to-plane ICP. Returns T_cam_obj, fitness (inlier fraction), rmse (m) of the trimmed set."""
    c = cfg["icp"]
    pts, nrm, tree = model
    T_om = np.linalg.inv(T_init)  # camera -> model
    for _ in range(c["iters"]):
        Pm = P @ T_om[:3, :3].T + T_om[:3, 3]
        d, j = tree.query(Pm)
        ok = d <= min(np.quantile(d, c["trim_frac"]), c["max_corr_m"])
        if ok.sum() < 6:
            break
        p, q, n = Pm[ok], pts[j[ok]], nrm[j[ok]]
        A = np.hstack([np.cross(p, n), n])
        x = np.linalg.lstsq(A, -np.einsum("ij,ij->i", n, p - q), rcond=None)[0]  # min-norm in unobservable dirs
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = _rodrigues(x[:3]), x[3:]
        T_om = D @ T_om
        if np.linalg.norm(x[:3]) < c["tol_rot_rad"] and np.linalg.norm(x[3:]) < c["tol_trans_m"]:
            break
    Pm = P @ T_om[:3, :3].T + T_om[:3, 3]
    d, _ = tree.query(Pm)
    keep = d <= np.quantile(d, c["trim_frac"])
    return np.linalg.inv(T_om), float((d < c["fitness_dist_m"]).mean()), float(np.sqrt((d[keep] ** 2).mean()))


def _pca(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and right-handed axes (columns, descending variance)."""
    c = P.mean(0)
    V = np.linalg.svd(P - c, full_matrices=False)[2].T
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return c, V


def _best(cands: list[tuple]) -> tuple:
    return max(cands, key=lambda r: (round(r[1], 4), -r[2]))


def _init_profile(P: np.ndarray, size: float, roll: float) -> np.ndarray:
    c, V = _pca(P)
    a = V[:, 0]
    t = (P - c) @ a
    mid = c + a * (t.max() + t.min()) / 2  # middle of the visible extent
    ray = mid / np.linalg.norm(mid)
    perp = ray - (ray @ a) * a
    if np.linalg.norm(perp) > 1e-6:
        mid = mid + perp / np.linalg.norm(perp) * size / 2  # only the camera-facing side is seen: move to the axis
    x0 = V[:, 1]
    x = np.cos(roll) * x0 + np.sin(roll) * np.cross(a, x0)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(a, x), a, mid
    return T


def estimate_pose(P: np.ndarray, spec: dict, cfg: dict) -> dict:
    """P: outlier-free camera-frame points. Returns {T, fitness, rmse}."""
    model = get_model(spec, cfg)
    if len(P) > cfg["icp"]["max_points"]:  # keep ICP fast on close, dense instances
        P = P[np.random.default_rng(cfg["seed"]).choice(len(P), cfg["icp"]["max_points"], replace=False)]
    if spec["class"] == "profile":
        n = cfg["profile"]["n_roll_hyp"]
        inits = [_init_profile(P, spec["dims_m"][0], k * (np.pi / 2) / n) for k in range(n)]
    else:
        co, Vo = _pca(P)
        cm, Vm = _pca(model[0])
        inits = []
        for S in _axis_hypotheses(cfg["bracket"]["all_axis_hyp"]):
            T = np.eye(4)
            T[:3, :3] = Vo @ S @ Vm.T
            T[:3, 3] = co - T[:3, :3] @ cm
            inits.append(T)
    T, fit, rmse = _best([trimmed_icp(P, model, T0, cfg) for T0 in inits])
    return {"T": T, "fitness": fit, "rmse": rmse}


# ------------------------------------------------------------------ evaluation
def _angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def evaluate_pose(T_est: np.ndarray, T_gt: np.ndarray, spec: dict, cfg: dict) -> dict:
    """Errors and success flag; fields not defined for the class are NaN."""
    out = dict(axis_angle_deg=NAN, lateral_offset_mm=NAN, along_axis_err_mm=NAN, centroid_err_mm=NAN, rot_err_deg=NAN)
    if spec["class"] == "profile":
        a_e, a_g = T_est[:3, 2], T_gt[:3, 2]
        out["axis_angle_deg"] = float(np.degrees(np.arccos(np.clip(abs(a_e @ a_g), 0, 1))))  # sign-invariant
        d = T_est[:3, 3] - T_gt[:3, 3]
        along = d @ a_g
        out["along_axis_err_mm"] = float(abs(along) * 1e3)
        out["lateral_offset_mm"] = float(np.linalg.norm(d - along * a_g) * 1e3)
        s = cfg["profile"]["success"]
        out["success"] = bool(out["lateral_offset_mm"] < s["lateral_mm"] and out["axis_angle_deg"] < s["angle_deg"])
    else:
        cm = get_model(spec, cfg)[0].mean(0)
        out["centroid_err_mm"] = float(np.linalg.norm(T_est[:3, :3] @ cm + T_est[:3, 3] - T_gt[:3, :3] @ cm - T_gt[:3, 3]) * 1e3)
        syms = [S for S in geo.symmetries("bracket") if np.linalg.det(S[:3, :3]) > 0]
        out["rot_err_deg"] = min(_angle_deg(T_est[:3, :3].T @ T_gt[:3, :3] @ S[:3, :3]) for S in syms)
        s = cfg["bracket"]["success"]
        out["success"] = bool(out["centroid_err_mm"] < s["centroid_mm"] and out["rot_err_deg"] < s["rot_deg"])
    return out


def failure_type(spec: dict, cfg: dict, ev: dict, P: np.ndarray, T_est: np.ndarray | None, removed_frac: float) -> str:
    """'' if successful, else one of the failure types (order documented in the module docstring)."""
    f = cfg["failure"]
    if T_est is None:
        return "too_few_points"
    if ev["success"]:
        return ""
    if removed_frac > f["outlier_removed_frac"]:
        return "outlier_dominated"
    pts = get_model(spec, cfg)[0]
    Pm = (P - T_est[:3, 3]) @ T_est[:3, :3]
    ext_obs = np.quantile(Pm, 1 - f["scale_quantile"], axis=0) - np.quantile(Pm, f["scale_quantile"], axis=0)
    if np.any(ext_obs > (pts.max(0) - pts.min(0)) * (1 + f["scale_tol"])):
        return "wrong_scale"
    if spec["class"] == "profile":
        if ev["axis_angle_deg"] > f["flip_angle_deg"]:
            return "axis_flip"
    elif ev["centroid_err_mm"] < cfg["bracket"]["success"]["centroid_mm"]:
        return "axis_flip"
    return "icp_diverged"


def localize(P_raw: np.ndarray, spec: dict, T_gt: np.ndarray, cfg: dict) -> dict:
    """Full per-instance pipeline on raw back-projected points. Returns one result row (no ids)."""
    t0 = time.perf_counter()
    n_raw = len(P_raw)
    row = dict(n_valid_points=n_raw, n_points=n_raw, removed_frac=NAN, axis_angle_deg=NAN, lateral_offset_mm=NAN, along_axis_err_mm=NAN,
               centroid_err_mm=NAN, rot_err_deg=NAN, success=False, failure_type="too_few_points",
               mean_depth_m=float(P_raw[:, 2].mean()) if n_raw else NAN)
    if n_raw >= cfg["min_points"]:
        P = P_raw[remove_outliers(P_raw, cfg)]
        row["n_points"], row["removed_frac"] = len(P), 1.0 - len(P) / n_raw
        if len(P) >= cfg["min_points"]:
            est = estimate_pose(P, spec, cfg)
            ev = evaluate_pose(est["T"], T_gt, spec, cfg)
            row.update(ev)
            row["failure_type"] = failure_type(spec, cfg, ev, P, est["T"], row["removed_frac"])
    row["runtime_ms"] = (time.perf_counter() - t0) * 1e3
    return row


# ------------------------------------------------------------------ clustered mode
def cluster_pixels(depth: np.ndarray, K: np.ndarray, cfg: dict) -> np.ndarray:
    """RANSAC table-plane removal + DBSCAN. Returns HxW int labels (-1 = table / noise / not sampled)."""
    c = cfg["clustered"]
    o3d.utility.random.seed(int(cfg["seed"]))
    mask = np.zeros(depth.shape, bool)
    mask[:: c["stride_px"], :: c["stride_px"]] = True
    P, pix = backproject(depth, mask, K)
    labels = np.full(depth.shape, -1, int)
    if len(P) < c["dbscan"]["min_points"]:
        return labels
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, inl = pc.segment_plane(c["ransac"]["dist_m"], c["ransac"]["n"], c["ransac"]["iters"])
    rest = np.setdiff1d(np.arange(len(P)), inl)
    if len(rest) < c["dbscan"]["min_points"]:
        return labels
    sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[rest]))
    lab = np.asarray(sub.cluster_dbscan(c["dbscan"]["eps_m"], c["dbscan"]["min_points"]))
    labels[pix[rest, 0], pix[rest, 1]] = lab
    return labels


def match_cluster(inst_mask: np.ndarray, inst: np.ndarray, labels: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, str]:
    """Pick the cluster for one GT instance. Returns (cluster pixel mask or None, '' | merged | fragmented | missed)."""
    c = cfg["clustered"]
    sampled = np.zeros(inst.shape, bool)
    sampled[:: c["stride_px"], :: c["stride_px"]] = True
    m = inst_mask & sampled
    n_inst = max(int(m.sum()), 1)
    ids, cnt = np.unique(labels[m & (labels >= 0)], return_counts=True)
    ok = cnt >= c["min_cluster_px"]
    ids, cnt = ids[ok], cnt[ok]
    if len(ids) == 0 or cnt.max() / n_inst < c["overlap_min_frac"]:
        return None, "missed"
    best = ids[np.argmax(cnt)]
    cl = labels == best
    gt_id = inst[inst_mask].flat[0]
    other_frac = float(((inst[cl] != gt_id) & (inst[cl] > 0)).sum()) / max(int(cl.sum()), 1)
    if other_frac > c["merge_frac"]:
        return cl, "merged"
    if len(ids) > 1 and np.sort(cnt)[-2] / n_inst > c["frag_frac"]:
        return cl, "fragmented"
    return cl, ""
