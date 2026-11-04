"""Save a 3-panel debug PNG for one instance: RGB crop | depth crop | 3D points with GT vs estimated model.

python scripts/debug_loc.py --split dev --scene 0 --instance 3 --level L3 [--data DIR] [--out PNG]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d455sim import loc  # noqa: E402
from d455sim.config import load_config  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_loc import load_depth  # noqa: E402

PAD_PX = 20
MODEL_PTS = 600


def render_panel(d: Path, instance: int, level: str, out: Path, cfg: dict | None = None) -> dict:
    """Write the 3-panel PNG for one instance of scene dir d; returns the localisation result row."""
    cfg = cfg or load_config("loc")
    meta = json.loads((d / "meta.json").read_text())
    o = next(o for o in meta["objects"] if o["id"] == instance)
    K, T_gt = np.asarray(meta["K"]), np.asarray(o["T_cam_obj"])
    inst = cv2.imread(str(d / "inst.png"), cv2.IMREAD_UNCHANGED)
    depth = load_depth(d, level)
    m = inst == instance
    P_raw, _ = loc.backproject(depth, m, K)
    res = loc.localize(P_raw, o, T_gt, cfg)
    P = P_raw[loc.remove_outliers(P_raw, cfg)] if len(P_raw) >= cfg["min_points"] else P_raw
    est = loc.estimate_pose(P, o, cfg) if len(P) >= cfg["min_points"] else None

    vs, us = np.nonzero(m)
    v0, v1, u0, u1 = max(vs.min() - PAD_PX, 0), vs.max() + PAD_PX, max(us.min() - PAD_PX, 0), us.max() + PAD_PX
    rgb = cv2.cvtColor(cv2.imread(str(d / "rgb.png")), cv2.COLOR_BGR2RGB)[v0:v1, u0:u1]
    dc = np.where(m, depth, 0)[v0:v1, u0:u1]

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.contour(m[v0:v1, u0:u1], levels=[0.5], colors="r", linewidths=0.8)
    ax.set_title(f"RGB crop, instance {instance} ({o['class']}, {o['material']['type']})")
    ax = fig.add_subplot(1, 3, 2)
    im = ax.imshow(np.ma.masked_equal(dc, 0), cmap="turbo")
    fig.colorbar(im, ax=ax, label="z [m]")
    ax.set_title(f"depth_{level} (masked)")
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    mp = loc.get_model(o, cfg)[0][:MODEL_PTS]
    Qg = mp @ T_gt[:3, :3].T + T_gt[:3, 3]
    if len(P):
        ax.scatter(*P.T, s=1, c="gray", label="observed")
    ax.scatter(*Qg.T, s=2, c="green", label="GT model")
    if est is not None:
        ax.scatter(*(mp @ est["T"][:3, :3].T + est["T"][:3, 3]).T, s=2, c="red", label="estimated model")
    allp = np.vstack([P, Qg]) if len(P) else Qg
    c, r = allp.mean(0), np.ptp(allp, axis=0).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r), ax.set_ylim(c[1] - r, c[1] + r), ax.set_zlim(c[2] - r, c[2] + r)
    ax.invert_yaxis()  # camera frame: y down
    ax.legend(loc="upper left", fontsize=7)
    txt = ", ".join(f"{k}={res[k]:.1f}" for k in ["axis_angle_deg", "lateral_offset_mm", "along_axis_err_mm",
                                                   "centroid_err_mm", "rot_err_deg"] if not np.isnan(res[k]))
    ax.set_title(f"{'OK' if res['success'] else 'FAIL ' + res['failure_type']}: {txt}", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--scene", type=int, required=True)
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--level", default="L0")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "outputs" / f"loc_debug_{a.scene:05d}_{a.instance}_{a.level}.png"
    render_panel(Path(a.data) / a.split / f"{a.scene:05d}", a.instance, a.level, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
