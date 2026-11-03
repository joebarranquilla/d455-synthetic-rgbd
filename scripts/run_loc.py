"""Localise every instance of each scene and write a CSV (one row per scene, level, instance).

python scripts/run_loc.py --split dev --levels L0 L3 [--mode oracle|clustered] [--n N] [--workers W]

oracle (primary): points come from the GT instance mask. clustered (secondary): table-plane RANSAC + DBSCAN,
clusters matched to GT instances by overlap; adds merged / fragmented / missed failures.
Level L0 falls back to depth_gt.npy if depth_L0.npy was not written.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d455sim import loc  # noqa: E402
from d455sim.config import load_config  # noqa: E402


def load_depth(d: Path, level: str) -> np.ndarray:
    f = d / f"depth_{level}.npy"
    return np.load(f if f.exists() or level != "L0" else d / "depth_gt.npy")


def process_scene(args: tuple) -> list[dict]:
    d, levels, mode = args
    cfg = load_config("loc")
    meta = json.loads((d / "meta.json").read_text())
    inst = cv2.imread(str(d / "inst.png"), cv2.IMREAD_UNCHANGED)
    K = np.asarray(meta["K"])
    rows = []
    for level in levels:
        depth = load_depth(d, level)
        labels = loc.cluster_pixels(depth, K, cfg) if mode == "clustered" else None
        for o in meta["objects"]:
            if o["n_visible_px"] < cfg["min_px"]:
                continue
            m = inst == o["id"]
            seg = ""
            if mode == "clustered":
                cl, seg = loc.match_cluster(m, inst, labels, cfg)
                m = cl if cl is not None else np.zeros_like(m)
            P, _ = loc.backproject(depth, m, K)
            res = loc.localize(P, o, np.asarray(o["T_cam_obj"]), cfg)
            if seg:  # a segmentation failure overrides the pose result
                res["success"], res["failure_type"] = False, seg
            rows.append({"scene": d.name, "level": level, "instance": o["id"], "class": o["class"],
                         "material": o["material"]["type"], "n_visible_px": o["n_visible_px"], "mode": mode, **res})
    return rows


COLUMNS = ["scene", "level", "instance", "class", "material", "mode", "mean_depth_m", "n_visible_px", "n_points",
           "removed_frac", "axis_angle_deg", "lateral_offset_mm", "along_axis_err_mm", "centroid_err_mm",
           "rot_err_deg", "success", "failure_type", "runtime_ms"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--levels", nargs="+", default=["L0"])
    ap.add_argument("--mode", choices=["oracle", "clustered"], default="oracle")
    ap.add_argument("--n", type=int, default=None, help="first N scenes")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="CSV path (default outputs/loc_<split>_<mode>.csv)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    dirs = sorted(p for p in (Path(a.data) / a.split).iterdir() if (p / "meta.json").exists())[: a.n]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(process_scene, [(d, a.levels, a.mode) for d in dirs]) for r in rs]
    df = pd.DataFrame(rows, columns=COLUMNS)
    out = Path(a.out) if a.out else ROOT / "outputs" / f"loc_{a.split}_{a.mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(df.groupby(["level", "class"]).agg(n=("success", "size"), success=("success", "mean"),
                                             ms=("runtime_ms", "mean")).round(3).to_string())
    print(df[~df.success].groupby(["level", "failure_type"]).size().to_string())


if __name__ == "__main__":
    main()
