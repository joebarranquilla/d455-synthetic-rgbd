"""Validate the sensor model on rendered samples.

python scripts/validate_sensor.py --split dev [--n 10] [--out outputs/validate]
Per level: hole fraction per material, depth residual std vs z (with analytic formula), edge profile.
Writes metrics.json and PNG plots.
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

from d455sim import sensor  # noqa: E402

Z_BINS = np.arange(0.4, 2.01, 0.2)
EDGE_MAX_PX = 10


def analytic_std(z: np.ndarray, fx: float, B: float, effects: list[str], cfg: dict) -> np.ndarray:
    """sigma_z = z^2 * sigma_d / (fx*B); sigma_d combines noise and uniform quantisation step."""
    var = 0.0
    if "noise" in effects:
        var += (cfg["d455"]["subpixel_rms_px"] * cfg["noise"]["scale"]) ** 2
    if "quant" in effects:
        var += (2.0 ** -cfg["d455"]["disparity_frac_bits"]) ** 2 / 12.0
    return z ** 2 * np.sqrt(var) / (fx * B)


def load(d: Path, cfg: dict):
    meta = json.loads((d / "meta.json").read_text())
    inst = cv2.imread(str(d / "inst.png"), cv2.IMREAD_UNCHANGED)
    ctx = sensor.build_ctx(meta, np.load(d / "normals.npy"), inst)
    return np.load(d / "depth_gt.npy"), ctx, inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "validate"))
    a = ap.parse_args()
    cfg = sensor.make_cfg()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in (Path(a.data) / a.split).iterdir() if (p / "depth_gt.npy").exists())[: a.n]
    names = list(cfg["levels"]) + [f"Lfull_minus_{e}" for e in cfg["levels"][cfg["full"]]]
    fx = None
    acc = {n: {"holes": {}, "res": [[] for _ in Z_BINS[:-1]], "edge": np.zeros((2, EDGE_MAX_PX + 1))} for n in names}

    for d in dirs:
        gt, ctx, inst = load(d, cfg)
        fx, B = ctx["K"][0, 0], ctx["baseline"]
        mtype = {o["id"]: o["material"]["type"] for o in ctx["meta"]["objects"]}
        valid = gt > 0
        disc = sensor.discontinuity_mask(gt, cfg["edges"]["flying"]["grad_thr_m"])
        dist = cv2.distanceTransform((~disc).astype(np.uint8), cv2.DIST_L2, 3)
        for n in names:
            z = sensor.apply_level(gt, ctx, cfg, n)
            a_n = acc[n]
            for i in np.unique(inst):
                m = valid & (inst == i)
                if m.any():
                    t = mtype.get(int(i), "table_bg")
                    h = a_n["holes"].setdefault(t, [0, 0])
                    h[0] += int((m & (z == 0)).sum())
                    h[1] += int(m.sum())
            both = valid & (z > 0) & ~cv2.dilate(disc.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            for k in range(len(Z_BINS) - 1):
                sel = both & (gt >= Z_BINS[k]) & (gt < Z_BINS[k + 1])
                a_n["res"][k].append((z[sel] - gt[sel]).astype(np.float64))
            di = np.minimum(np.rint(dist), EDGE_MAX_PX).astype(int)
            for r in range(EDGE_MAX_PX + 1):
                sel = valid & (di == r)
                a_n["edge"][0, r] += (sel & (z == 0)).sum()
                a_n["edge"][1, r] += sel.sum()

    centres = 0.5 * (Z_BINS[:-1] + Z_BINS[1:])
    metrics = {}
    for n in names:
        eff = sensor.level_effects(n, cfg)
        std = [float(np.std(np.concatenate(r))) if r and sum(map(len, r)) > 10 else None for r in acc[n]["res"]]
        metrics[n] = {
            "effects": eff,
            "hole_fraction_per_material": {t: h[0] / max(h[1], 1) for t, h in acc[n]["holes"].items()},
            "residual_std_m": dict(zip([f"{c:.2f}" for c in centres], std)),
            "analytic_std_m": dict(zip([f"{c:.2f}" for c in centres], analytic_std(centres, fx, B, eff, cfg).tolist())),
            "edge_invalid_fraction_by_px_to_edge": (acc[n]["edge"][0] / np.maximum(acc[n]["edge"][1], 1)).tolist(),
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for n in names:
        m = metrics[n]
        ax[0].bar(np.arange(len(m["hole_fraction_per_material"])) + 0.0, [0], width=0)  # keep axes tidy
        s = [v if v is not None else np.nan for v in m["residual_std_m"].values()]
        ax[1].plot(centres, s, marker="o", label=n)
        ax[2].plot(m["edge_invalid_fraction_by_px_to_edge"], label=n)
    ax[1].plot(centres, analytic_std(centres, fx, B, cfg["levels"][cfg["full"]], cfg), "k--", label="analytic (full)")
    mats = sorted({t for n in names for t in metrics[n]["hole_fraction_per_material"]})
    ax[0].cla()
    for j, n in enumerate(["L0", "L3", cfg["full"]]):
        ax[0].bar(np.arange(len(mats)) + 0.27 * j, [metrics[n]["hole_fraction_per_material"].get(t, 0) for t in mats],
                  width=0.27, label=n)
    ax[0].set_xticks(np.arange(len(mats)) + 0.27)
    ax[0].set_xticklabels(mats, rotation=30, ha="right")
    ax[0].set_title("hole fraction per material")
    ax[0].legend()
    ax[1].set(title="residual std vs z (flat areas)", xlabel="z [m]", ylabel="std [m]", yscale="log")
    ax[1].legend(fontsize=6)
    ax[2].set(title="invalid fraction vs distance to depth edge", xlabel="px", ylabel="fraction")
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=120)

    gt, ctx, _ = load(dirs[0], cfg)
    tiles = []
    for n in ["L0", "L1", "L2", "L3", "L4"]:
        z = sensor.apply_level(gt, ctx, cfg, n)
        g = (255 * np.clip(z / cfg["d455"]["max_range_m"] * 3, 0, 1)).astype(np.uint8)
        c = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
        c[z == 0] = 0
        cv2.putText(c, n, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        tiles.append(c)
    cv2.imwrite(str(out / "levels.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:] + [tiles[0] * 0])]))
    print(f"wrote {out}/metrics.json, validation.png, levels.png ({len(dirs)} samples)")


if __name__ == "__main__":
    main()
