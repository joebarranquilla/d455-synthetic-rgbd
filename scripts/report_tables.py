"""Tables and plots from outputs/experiment/loc_test.csv (test split).

python scripts/report_tables.py [--csv CSV] [--data DIR] [--out DIR]

Statistics: every CI resamples SCENES (not instances). One shared bootstrap index matrix is used for all
levels, so differences between levels are paired. A paired difference whose CI includes 0 is flagged
`no_detectable_effect`.

Outputs in --out (default outputs/experiment):
1 success_by_level_class.csv/.png   2 error_quantiles.csv   3 success_by_depth|npoints|material.csv/.png
4 paired_vs_previous_level.csv      5 leave_one_out.csv/.png   6 failure_types.csv/.png
7 gallery/<level>/rank_XX_*.png + gallery/index.csv (worst K failures with a pose, Phase 3 debug panel)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d455sim.config import load_config  # noqa: E402

CLASSES = ["profile", "bracket", "all"]
ERR_COLS = {"profile": ["lateral_offset_mm", "axis_angle_deg", "along_axis_err_mm"],
            "bracket": ["centroid_err_mm", "rot_err_deg"]}


# ------------------------------------------------------------------ statistics (pure, tested)
def boot_indices(n_scenes: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_scenes) scene indices, shared by every estimate so differences are paired."""
    return np.random.default_rng(seed).integers(0, n_scenes, (n_boot, n_scenes))


def per_scene_counts(df: pd.DataFrame, scenes: list, cls: str, mask_col: str = "success") -> tuple[np.ndarray, np.ndarray]:
    """Per-scene (successes, instances), aligned with `scenes` (zeros for scenes without instances)."""
    d = df if cls == "all" else df[df["class"] == cls]
    g = d.groupby("scene")[mask_col].agg(["sum", "size"]).reindex(scenes).fillna(0)
    return g["sum"].to_numpy(float), g["size"].to_numpy(float)


def boot_rate(s: np.ndarray, n: np.ndarray, idx: np.ndarray) -> np.ndarray:
    tot = n[idx].sum(1)
    return np.where(tot > 0, s[idx].sum(1) / np.maximum(tot, 1), np.nan)


def ci(x: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))


def rate_with_ci(s: np.ndarray, n: np.ndarray, idx: np.ndarray, level: float) -> dict:
    lo, hi = ci(boot_rate(s, n, idx), level)
    return {"rate": float(s.sum() / n.sum()) if n.sum() else np.nan, "ci_lo": lo, "ci_hi": hi, "n_inst": int(n.sum())}


def paired_diff(s_a, n_a, s_b, n_b, idx: np.ndarray, level: float) -> dict:
    """rate_b - rate_a with a paired bootstrap over scenes; flag when the CI includes 0."""
    d = boot_rate(s_b, n_b, idx) - boot_rate(s_a, n_a, idx)
    lo, hi = ci(d, level)
    point = s_b.sum() / max(n_b.sum(), 1) - s_a.sum() / max(n_a.sum(), 1)
    return {"diff": float(point), "ci_lo": lo, "ci_hi": hi, "no_detectable_effect": bool(lo <= 0 <= hi)}


def severity(df: pd.DataFrame, cfg_loc: dict) -> pd.Series:
    """Worst-first score: max error / success threshold (>1 = failed on that criterion); NaN without a pose."""
    p, b = cfg_loc["profile"]["success"], cfg_loc["bracket"]["success"]
    prof = np.maximum(df["lateral_offset_mm"] / p["lateral_mm"], df["axis_angle_deg"] / p["angle_deg"])
    brk = np.maximum(df["centroid_err_mm"] / b["centroid_mm"], df["rot_err_deg"] / b["rot_deg"])
    return pd.Series(np.where(df["class"] == "profile", prof, brk), index=df.index)


# ------------------------------------------------------------------ tables
def table_success(df, levels, scenes, idx, level, out):
    rows = []
    for lv in levels:
        for c in CLASSES:
            s, n = per_scene_counts(df[df.level == lv], scenes, c)
            rows.append({"level": lv, "class": c, **rate_with_ci(s, n, idx, level)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "success_by_level_class.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, c in enumerate(CLASSES):
        r = t[t["class"] == c]
        x = np.arange(len(r)) + 0.12 * (k - 1)
        ax.errorbar(x, r.rate, yerr=[r.rate - r.ci_lo, r.ci_hi - r.rate], fmt="o-", capsize=3, label=c)
    ax.set_xticks(np.arange(len(levels)), levels)
    ax.set(ylabel="success rate", title="Success per level (95% CI, bootstrap over scenes)", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_by_level_class.png", dpi=120)
    plt.close(fig)
    return t


def table_errors(df, levels, out):
    rows = []
    for lv in levels:
        for c, cols in ERR_COLS.items():
            d = df[(df.level == lv) & (df["class"] == c)]
            for col in cols:
                v = d[col].dropna()
                rows.append({"level": lv, "class": c, "metric": col, "n": len(v),
                             "median": v.median() if len(v) else np.nan, "p95": v.quantile(0.95) if len(v) else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(out / "error_quantiles.csv", index=False)
    return t


def table_by_bin(df, levels, scenes, idx, level, out, name, col, bins=None):
    d = df[df.level.isin(levels)].copy()
    if bins is not None:
        edges = pd.IntervalIndex.from_breaks(bins, closed="left")
        d["bin"] = pd.cut(d[col], edges).astype(str)
        order = [str(b) for b in edges]
    else:
        d["bin"] = d[col].astype(str)
        order = sorted(d["bin"].unique())
    rows = []
    for lv in levels:
        for c in ["profile", "bracket"]:
            for b in order:
                sel = d[(d.level == lv) & (d["class"] == c) & (d.bin == b)]
                if len(sel):
                    s, n = per_scene_counts(sel, scenes, "all")
                    rows.append({"level": lv, "class": c, "bin": b, **rate_with_ci(s, n, idx, level)})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"success_by_{name}.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, c in zip(axs, ["profile", "bracket"]):
        for lv in levels:
            r = t[(t.level == lv) & (t["class"] == c)].set_index("bin").reindex(order).dropna(subset=["rate"])
            ax.plot(r.index, r.rate, "o-", label=lv)
        ax.set(title=f"{c}: success vs {name}", ylim=(0, 1.02))
        ax.tick_params(axis="x", rotation=40)
    axs[0].set_ylabel("success rate")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"success_by_{name}.png", dpi=120)
    plt.close(fig)


def table_paired(df, levels, scenes, idx, level, out):
    rows = []
    for prev, cur in zip(levels[:-1], levels[1:]):
        for c in CLASSES:
            a = per_scene_counts(df[df.level == prev], scenes, c)
            b = per_scene_counts(df[df.level == cur], scenes, c)
            rows.append({"from": prev, "to": cur, "class": c, **paired_diff(*a, *b, idx, level)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "paired_vs_previous_level.csv", index=False)
    return t


def table_loo(df, cfg, scenes, idx, level, out):
    rows = []
    for e in cfg["loo_effects"]:
        for c in CLASSES:
            a = per_scene_counts(df[df.level == cfg["full_level"]], scenes, c)
            b = per_scene_counts(df[df.level == f"Lfull_minus_{e}"], scenes, c)
            r = paired_diff(*a, *b, idx, level)  # positive = success recovered by removing the effect
            rows.append({"removed_effect": e, "class": c, "success_full": a[0].sum() / a[1].sum(),
                         "success_without": b[0].sum() / b[1].sum(), "recovery": r["diff"], "ci_lo": r["ci_lo"],
                         "ci_hi": r["ci_hi"], "no_detectable_effect": r["no_detectable_effect"]})
    t = pd.DataFrame(rows)
    t.to_csv(out / "leave_one_out.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in enumerate(CLASSES):
        r = t[t["class"] == c]
        ax.bar(np.arange(len(r)) + 0.27 * (k - 1), r.recovery, 0.27,
               yerr=[r.recovery - r.ci_lo, r.ci_hi - r.recovery], capsize=2, label=c)
    ax.set_xticks(np.arange(len(cfg["loo_effects"])), cfg["loo_effects"])
    ax.set(ylabel="success gained vs full model", title=f"Leave-one-out (vs {cfg['full_level']})")
    ax.axhline(0, color="k", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "leave_one_out.png", dpi=120)
    plt.close(fig)
    return t


def table_failures(df, names, out):
    f = df[~df.success].groupby(["level", "failure_type"]).size().rename("count").reset_index()
    f["fraction_of_instances"] = f["count"] / f["level"].map(df.groupby("level").size())
    f.to_csv(out / "failure_types.csv", index=False)
    p = f.pivot(index="level", columns="failure_type", values="count").reindex(names).fillna(0)
    ax = p.plot(kind="bar", stacked=True, figsize=(9, 4.5), colormap="tab10")
    ax.set(ylabel="failed instances", title="Failure types per level")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "failure_types.png", dpi=120)
    plt.close(ax.figure)


def gallery(df, levels, k, data_dir, split, loc_cfg, out):
    from debug_loc import render_panel
    d = df[~df.success].copy()
    d["severity"] = severity(d, loc_cfg)
    rows = []
    for lv in levels:
        worst = d[d.level == lv].dropna(subset=["severity"]).nlargest(k, "severity")
        for r, (_, x) in enumerate(worst.iterrows(), 1):
            png = out / "gallery" / lv / f"rank_{r:02d}_scene{x.scene}_inst{int(x.instance)}_{x['class']}.png"
            render_panel(Path(data_dir) / split / x.scene, int(x.instance), lv, png, loc_cfg)
            rows.append({"level": lv, "rank": r, "scene": x.scene, "instance": int(x.instance), "class": x["class"],
                         "material": x.material, "failure_type": x.failure_type, "severity": x.severity, "png": str(png)})
    if rows:
        pd.DataFrame(rows).to_csv(out / "gallery" / "index.csv", index=False)


def main() -> None:
    cfg = load_config("experiment")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "experiment" / "loc_test.csv"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiment"))
    ap.add_argument("--no-gallery", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv, dtype={"scene": str})
    loc_cfg, st = load_config("loc"), cfg["stats"]
    scenes = sorted(df.scene.unique())
    idx = boot_indices(len(scenes), st["n_boot"], st["seed"])
    levels = cfg["levels"]
    all_names = levels + [f"Lfull_minus_{e}" for e in cfg["loo_effects"]]

    table_success(df, levels, scenes, idx, st["ci"], out)
    table_errors(df, all_names, out)
    table_by_bin(df, levels, scenes, idx, st["ci"], out, "depth", "mean_depth_m", cfg["bins"]["depth_m"])
    table_by_bin(df, levels, scenes, idx, st["ci"], out, "npoints", "n_points", cfg["bins"]["n_points"])
    table_by_bin(df, levels, scenes, idx, st["ci"], out, "material", "material")
    table_paired(df, levels, scenes, idx, st["ci"], out)
    table_loo(df, cfg, scenes, idx, st["ci"], out)
    table_failures(df, all_names, out)
    if not a.no_gallery:
        gallery(df, all_names, cfg["gallery_k"], a.data, cfg["split"], loc_cfg, out)
    print(f"wrote tables/plots to {out} ({len(scenes)} scenes, {len(df)} rows)")


if __name__ == "__main__":
    main()
