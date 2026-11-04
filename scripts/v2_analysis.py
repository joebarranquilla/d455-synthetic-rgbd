"""Analysis products for the predeclared v2 synthetic-depth experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d455sim.config import intrinsics, load_config  # noqa: E402
from report_tables import boot_indices, boot_rate, ci, per_scene_counts, rate_with_ci  # noqa: E402

CLASSES = ["all", "profile", "bracket"]
ERROR_METRICS = {
    "lateral_error_mm": ("lateral_offset_mm", "profile"),
    "angular_error_deg": ("axis_angle_deg", "profile"),
    "centroid_error_mm": ("centroid_err_mm", "bracket"),
    "rotation_error_deg": ("rot_err_deg", "bracket"),
}
FAILURE_GROUPS = {
    "orientation/symmetry ambiguity": {"axis_flip"},
    "too few valid points/outlier-dominated": {"too_few_points", "outlier_dominated"},
    "ICP divergence/bad initialization": {"icp_diverged"},
}


def _rate(df: pd.DataFrame, arm: str, cls: str, scenes: list[str], indices: np.ndarray,
          confidence: float) -> dict:
    selected = df[df["level"] == arm]
    successes, totals = per_scene_counts(selected, scenes, cls)
    return rate_with_ci(successes, totals, indices, confidence)


def _bootstrap_quantiles(values: np.ndarray, scene_indices: np.ndarray, indices: np.ndarray,
                         quantile: float, confidence: float) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    weights = indices[:, scene_indices[order]]
    cumulative = np.cumsum(weights, axis=1)
    totals = cumulative[:, -1]
    positions = np.maximum((totals - 1) * quantile, 0)
    lower_rank, upper_rank = np.floor(positions), np.ceil(positions)
    lower_locations = np.argmax(cumulative > lower_rank[:, None], axis=1)
    upper_locations = np.argmax(cumulative > upper_rank[:, None], axis=1)
    weight = positions - lower_rank
    boot = (sorted_values[lower_locations] * (1 - weight) +
            sorted_values[upper_locations] * weight).astype(float)
    boot[totals == 0] = np.nan
    point = float(np.quantile(values, quantile))
    lo, hi = ci(boot, confidence)
    return point, lo, hi


def success_summary(df: pd.DataFrame, arms: list[str], scenes: list[str],
                    indices: np.ndarray, confidence: float, out: Path) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for cls in CLASSES:
            rows.append({"arm": arm, "class": cls, **_rate(df, arm, cls, scenes, indices, confidence)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "success_by_arm_v2.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    for offset, cls in enumerate(CLASSES):
        selected = result[result["class"] == cls]
        x = np.arange(len(arms)) + (offset - 1) * 0.18
        ax.errorbar(x, selected["rate"], yerr=[selected["rate"] - selected["ci_lo"],
                                               selected["ci_hi"] - selected["rate"]],
                    fmt="o", capsize=2, label=cls)
    ax.set_xticks(np.arange(len(arms)), arms, rotation=65, ha="right")
    ax.set(ylabel="success rate", ylim=(0, 1.02), title="Localization success by predeclared arm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "success_by_arm_v2.png", dpi=150)
    plt.close(fig)
    return result


def continuous_errors(df: pd.DataFrame, arms: list[str], scenes: list[str], indices: np.ndarray,
                      quantiles: dict, confidence: float, out: Path) -> pd.DataFrame:
    scene_lookup = {name: index for index, name in enumerate(scenes)}
    rows = []
    for arm in arms:
        arm_rows = df[df["level"] == arm]
        for cls in CLASSES:
            cls_rows = arm_rows if cls == "all" else arm_rows[arm_rows["class"] == cls]
            for label, (column, required_class) in ERROR_METRICS.items():
                values_rows = cls_rows[cls_rows["class"] == required_class].dropna(subset=[column])
                values = values_rows[column].to_numpy(float)
                scene_ids = np.fromiter((scene_lookup[str(value)] for value in values_rows["scene"]),
                                        dtype=np.int64, count=len(values_rows))
                for statistic, quantile in quantiles.items():
                    estimate, lower, upper = _bootstrap_quantiles(
                        values, scene_ids, indices, quantile, confidence
                    )
                    rows.append({"arm": arm, "class": cls, "metric": label, "statistic": statistic,
                                 "estimate": estimate, "ci_lo": lower, "ci_hi": upper, "n": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "continuous_errors_v2.csv", index=False)
    return result


def paired_comparisons(df: pd.DataFrame, levels: list[str], full_level: str, loo_effects: list[str],
                       scenes: list[str], indices: np.ndarray, confidence: float,
                       out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows = []
    for previous, current in zip(levels[:-1], levels[1:]):
        for cls in CLASSES:
            a, b = df[df["level"] == previous], df[df["level"] == current]
            sa, na = per_scene_counts(a, scenes, cls)
            sb, nb = per_scene_counts(b, scenes, cls)
            delta = boot_rate(sb, nb, indices) - boot_rate(sa, na, indices)
            lower, upper = ci(delta, confidence)
            paired_rows.append({"from": previous, "to": current, "class": cls,
                                "success_change": float(sb.sum() / max(nb.sum(), 1) -
                                                        sa.sum() / max(na.sum(), 1)),
                                "ci_lo": lower, "ci_hi": upper,
                                "no_detectable_effect": bool(lower <= 0 <= upper)})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_vs_previous_v2.csv", index=False)

    loo_rows = []
    for effect in loo_effects:
        removed = f"Lfull_minus_{effect}"
        for cls in CLASSES:
            full_s, full_n = per_scene_counts(df[df["level"] == full_level], scenes, cls)
            removed_s, removed_n = per_scene_counts(df[df["level"] == removed], scenes, cls)
            boot = boot_rate(removed_s, removed_n, indices) - boot_rate(full_s, full_n, indices)
            lower, upper = ci(boot, confidence)
            loo_rows.append({"removed_effect": effect, "class": cls, "full_level": full_level,
                             "success_full": float(full_s.sum() / max(full_n.sum(), 1)),
                             "success_without": float(removed_s.sum() / max(removed_n.sum(), 1)),
                             "recovery": float(removed_s.sum() / max(removed_n.sum(), 1) -
                                               full_s.sum() / max(full_n.sum(), 1)),
                             "ci_lo": lower, "ci_hi": upper,
                             "no_detectable_effect": bool(lower <= 0 <= upper)})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(out / "leave_one_out_v2.csv", index=False)
    return paired, loo


def isolated_ablation(df: pd.DataFrame, full_level: str, single_arms: list[str], scenes: list[str],
                      indices: np.ndarray, confidence: float, out: Path) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        rates: dict[str, tuple[float, np.ndarray]] = {}
        for arm in ["L0", full_level, *single_arms]:
            successes, totals = per_scene_counts(df[df["level"] == arm], scenes, cls)
            rates[arm] = (float(successes.sum() / max(totals.sum(), 1)), boot_rate(successes, totals, indices))
        single_deltas = {arm: rates["L0"][0] - rates[arm][0] for arm in single_arms}
        boot_single_deltas = [rates["L0"][1] - rates[arm][1] for arm in single_arms]
        delta_l4 = rates["L0"][0] - rates[full_level][0]
        boot_delta_l4 = rates["L0"][1] - rates[full_level][1]
        residual = delta_l4 - sum(single_deltas.values())
        boot_residual = boot_delta_l4 - np.sum(boot_single_deltas, axis=0)
        lower, upper = ci(boot_residual, confidence)
        l4_lower, l4_upper = ci(boot_delta_l4, confidence)
        single_boot = {arm: rates["L0"][1] - rates[arm][1] for arm in single_arms}
        sum_boot = np.sum(list(single_boot.values()), axis=0)
        sum_lower, sum_upper = ci(sum_boot, confidence)
        row = {"class": cls, "success_L0": rates["L0"][0],
                     "delta_L4": delta_l4, "delta_L4_ci_lo": l4_lower, "delta_L4_ci_hi": l4_upper,
                     "sum_single_effect_deltas": sum(single_deltas.values()),
                     "sum_single_ci_lo": sum_lower, "sum_single_ci_hi": sum_upper,
                     "residual_L4_minus_sum_singles": residual, "ci_lo": lower, "ci_hi": upper,
                     "superadditivity_detectable": bool(lower > 0),
                     "result": "detectable positive superadditivity" if lower > 0 else
                     "detectable subadditivity" if upper < 0 else "no detectable superadditivity"}
        for arm, delta in single_deltas.items():
            single_lower, single_upper = ci(single_boot[arm], confidence)
            row[f"delta_{arm}"] = delta
            row[f"delta_{arm}_ci_lo"] = single_lower
            row[f"delta_{arm}_ci_hi"] = single_upper
        row["success_L4"] = rates[full_level][0]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "isolated_ablation_v2.csv", index=False)
    return result


def noise_sweep(success: pd.DataFrame, errors: pd.DataFrame, scale_arms: dict[str, str],
                out: Path) -> pd.DataFrame:
    rows = []
    for scale, arm in scale_arms.items():
        for cls in CLASSES:
            rate_row = success[(success["arm"] == arm) & (success["class"] == cls)].iloc[0]
            metrics = errors[(errors["arm"] == arm) & (errors["class"] == cls)]
            row = {"noise_scale": float(scale), "arm": arm, "class": cls,
                   "success_rate": rate_row["rate"], "success_ci_lo": rate_row["ci_lo"],
                   "success_ci_hi": rate_row["ci_hi"]}
            for metric in ERROR_METRICS:
                for statistic in ("median", "p90"):
                    selected = metrics[(metrics["metric"] == metric) & (metrics["statistic"] == statistic)]
                    row[f"{metric}_{statistic}"] = selected["estimate"].iloc[0]
                    row[f"{metric}_{statistic}_ci_lo"] = selected["ci_lo"].iloc[0]
                    row[f"{metric}_{statistic}_ci_hi"] = selected["ci_hi"].iloc[0]
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(out / "noise_sweep_v2.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for cls in CLASSES:
        rows_for_class = result[result["class"] == cls].sort_values("noise_scale")
        axs[0].errorbar(rows_for_class["noise_scale"], rows_for_class["success_rate"],
                        yerr=[rows_for_class["success_rate"] - rows_for_class["success_ci_lo"],
                              rows_for_class["success_ci_hi"] - rows_for_class["success_rate"]],
                        marker="o", capsize=2, label=cls)
    axs[0].set(xscale="log", xlabel="disparity-noise scale s", ylabel="success rate", ylim=(0, 1.02))
    for cls in ("profile", "bracket"):
        metric = "lateral_error_mm" if cls == "profile" else "centroid_error_mm"
        selected = result[result["class"] == cls].sort_values("noise_scale")
        axs[1].plot(selected["noise_scale"], selected[f"{metric}_p90"], marker="o", label=cls)
    axs[1].set(xscale="log", xlabel="disparity-noise scale s", ylabel="90th-percentile position error (mm)")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Noise-scale sweep (otherwise L2)")
    fig.tight_layout()
    fig.savefig(out / "noise_sweep_v2.png", dpi=150)
    plt.close(fig)
    return result


def edge_component_summary(success: pd.DataFrame, edge_arms: dict[str, str], out: Path) -> pd.DataFrame:
    names = {"occlusion": "occlusion invalidation", "flying_pixels": "flying pixels",
             "foreground_fattening": "foreground fattening"}
    result = success[success["arm"].isin(edge_arms.values())].copy()
    reverse = {value: key for key, value in edge_arms.items()}
    result["component"] = result["arm"].map(reverse).map(names)
    result.to_csv(out / "edge_components_v2.csv", index=False)
    return result


def _failure_group(code: str) -> str:
    for group, codes in FAILURE_GROUPS.items():
        if code in codes:
            return group
    return "other"


def l0_failure_analysis(df: pd.DataFrame, scenes: list[str], indices: np.ndarray, config: dict,
                        confidence: float, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = df[df["level"] == "L0"].copy()
    baseline["failure_group"] = baseline["failure_type"].map(_failure_group)
    rows = []
    groups = [*FAILURE_GROUPS, "other"]
    for cls in CLASSES:
        selected = baseline if cls == "all" else baseline[baseline["class"] == cls]
        failed = selected[~selected["success"]]
        for group in groups:
            count = int((failed["failure_group"] == group).sum())
            rows.append({"class": cls, "failure_group": group, "count": count,
                         "failed_instances": len(failed), "all_instances": len(selected),
                         "fraction_of_failures": count / len(failed) if len(failed) else 0.0})
    failures = pd.DataFrame(rows)
    failures.to_csv(out / "l0_failure_types_v2.csv", index=False)

    bin_rows = []
    for column, bins in config["baseline_bins"].items():
        values = baseline[column]
        categories = pd.cut(values, bins=bins, right=False, include_lowest=True)
        for interval in categories.cat.categories:
            for cls in CLASSES:
                selected = baseline[(categories == interval) &
                                    ((baseline["class"] == cls) if cls != "all" else np.ones(len(baseline), bool))]
                if selected.empty:
                    continue
                successes, totals = per_scene_counts(selected, scenes, "all")
                estimate = rate_with_ci(successes, totals, indices, confidence)
                bin_rows.append({"bin_variable": column, "bin": str(interval), "class": cls, **estimate})
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(out / "l0_success_by_bins_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = failures[failures["class"].isin(["profile", "bracket"])].pivot(
        index="failure_group", columns="class", values="count").reindex(groups).fillna(0)
    pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    ax.set(ylabel="failed instances", title="L0 failure categories by class")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out / "l0_failure_types_v2.png", dpi=150)
    plt.close(fig)
    return failures, bins, baseline


def hole_statistics(scenes: list[Path], levels: list[str], connectivity: int,
                    invalid_value: float, out: Path) -> pd.DataFrame:
    totals: dict[tuple[str, str], dict[str, object]] = {}
    for level in levels:
        for cls in ("all", "profile", "bracket"):
            totals[(level, cls)] = {"pixels": 0, "invalid": 0, "components": []}
    for scene in scenes:
        inst = cv2.imread(str(scene / "inst.png"), cv2.IMREAD_UNCHANGED)
        meta = json.loads((scene / "meta.json").read_text())
        for level in levels:
            depth = np.load(scene / f"depth_{level}.npy")
            for obj in meta["objects"]:
                cls = obj["class"]
                mask = inst == obj["id"]
                invalid = mask & (depth == invalid_value)
                for group in ("all", cls):
                    bucket = totals[(level, group)]
                    bucket["pixels"] = int(bucket["pixels"]) + int(mask.sum())
                    bucket["invalid"] = int(bucket["invalid"]) + int(invalid.sum())
                    if invalid.any():
                        count, _, stats, _ = cv2.connectedComponentsWithStats(
                            invalid.astype(np.uint8), connectivity=connectivity)
                        components = bucket["components"]
                        components.extend(int(area) for area in stats[1:count, cv2.CC_STAT_AREA])
    rows = []
    for (level, cls), values in totals.items():
        areas = np.asarray(values["components"], dtype=float)
        rows.append({"level": level, "class": cls, "object_pixels": values["pixels"],
                     "invalid_pixels": values["invalid"],
                     "fraction_invalid": values["invalid"] / values["pixels"] if values["pixels"] else np.nan,
                     "hole_count": int(areas.size),
                     "hole_size_median_px": float(np.median(areas)) if areas.size else np.nan,
                     "hole_size_p90_px": float(np.quantile(areas, 0.9)) if areas.size else np.nan,
                     "hole_size_p99_px": float(np.quantile(areas, 0.99)) if areas.size else np.nan,
                     "hole_size_max_px": float(areas.max()) if areas.size else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(out / "hole_statistics_v2.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for cls in ("all", "profile", "bracket"):
        selected = result[result["class"] == cls].set_index("level").reindex(levels)
        axs[0].plot(levels, selected["fraction_invalid"], "o-", label=cls)
    axs[0].set(ylabel="invalid pixels / object-mask pixels", title="Invalid fraction")
    axs[0].legend()
    all_components = {level: totals[(level, "all")]["components"] for level in levels}
    box_data = [all_components[level] for level in levels]
    nonempty = [(level, data) for level, data in zip(levels, box_data) if data]
    if nonempty:
        axs[1].boxplot([data for _, data in nonempty], tick_labels=[level for level, _ in nonempty],
                       showfliers=False)
        axs[1].set_yscale("log")
    else:
        axs[1].text(0.5, 0.5, "No invalid object pixels", ha="center", va="center",
                    transform=axs[1].transAxes)
    axs[1].set(ylabel="connected invalid component area (pixels)", title="Hole-size distribution")
    fig.tight_layout()
    fig.savefig(out / "hole_statistics_v2.png", dpi=150)
    plt.close(fig)
    return result


def ladder_figure(scenes: list[Path], scene_id: str, levels: list[str], figure_config: dict,
                  out: Path) -> None:
    scene = next((path for path in scenes if path.name == scene_id), None)
    if scene is None:
        raise ValueError(f"predeclared ladder scene {scene_id} is missing")
    reference = np.load(scene / f"depth_{levels[0]}.npy")
    cmap = plt.get_cmap("viridis").with_extremes(bad="white")
    error_cmap = plt.get_cmap("magma").with_extremes(bad="white")
    fig, axes = plt.subplots(len(levels), 2, figsize=(10, 2.4 * len(levels)), squeeze=False)
    for row, level in enumerate(levels):
        depth = np.load(scene / f"depth_{level}.npy")
        image = np.ma.masked_where(depth == 0, depth)
        axes[row, 0].imshow(image, cmap=cmap, vmin=figure_config["depth_min_m"],
                            vmax=figure_config["depth_max_m"])
        axes[row, 0].set_ylabel(level)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        common = (depth > 0) & (reference > 0)
        error = np.ma.masked_where(~common, np.abs(depth - reference))
        axes[row, 1].imshow(error, cmap=error_cmap, vmin=0.0,
                            vmax=figure_config["absolute_error_max_m"])
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
    axes[0, 0].set_title("Depth z (m)")
    axes[0, 1].set_title("Absolute error vs L0 (m)")
    depth_bar = fig.colorbar(axes[0, 0].images[0], ax=axes[:, 0], fraction=0.025, pad=0.02)
    depth_bar.set_label("z depth (m)")
    error_bar = fig.colorbar(axes[0, 1].images[0], ax=axes[:, 1], fraction=0.025, pad=0.02)
    error_bar.set_label("absolute depth error (m)")
    fig.suptitle(f"V2 degradation ladder, test scene {scene_id}; shared color limits")
    fig.savefig(out / "degradation_ladder_v2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def range_and_noise_statistics(scenes: list[Path], experiment_config: dict, ladder_config: dict,
                               d455: dict, out: Path) -> pd.DataFrame:
    total_valid = 0
    below_ideal = 0
    for scene in scenes:
        depth = np.load(scene / "depth_gt.npy")
        valid = depth > 0
        total_valid += int(valid.sum())
        below_ideal += int((valid & (depth < experiment_config["range_analysis"]["ideal_min_range_m"])).sum())

    working_min, working_max = experiment_config["range_analysis"]["working_distance_m"]
    ideal_min = experiment_config["range_analysis"]["ideal_min_range_m"]
    cutoff = float(d455["min_range_m"])
    near_margin = experiment_config["range_analysis"]["near_cutoff_margin_m"]
    working_span = working_max - working_min
    below_ideal_span = max(0.0, min(ideal_min, working_max) - working_min)
    near_cutoff_span = max(0.0, min(cutoff + near_margin, working_max) - working_min)
    fx = intrinsics(d455)[0, 0]
    disparity_sigma = float(d455["subpixel_rms_px"] * ladder_config["noise"]["scale"])
    values = [
        {"metric": "fraction_of_working_distance_below_0.6m_ideal_minimum",
         "value": below_ideal_span / working_span, "unit": "fraction"},
        {"metric": "fraction_of_working_distance_within_0.05m_above_0.4m_cutoff",
         "value": near_cutoff_span / working_span, "unit": "fraction"},
        {"metric": "valid_raw_test_pixels_below_0.6m",
         "value": below_ideal / total_valid if total_valid else np.nan, "unit": "fraction"},
        {"metric": "valid_raw_test_pixels", "value": total_valid, "unit": "pixels"},
        {"metric": "profile_lateral_success_threshold",
         "value": experiment_config["range_analysis"]["profile_lateral_success_mm"], "unit": "mm"},
        {"metric": "bracket_centroid_success_threshold",
         "value": load_config("loc")["bracket"]["success"]["centroid_mm"], "unit": "mm"},
        {"metric": "datasheet_error_at_4m",
         "value": experiment_config["range_analysis"]["datasheet_error_at_4m_percent"], "unit": "percent"},
    ]
    for depth_m in experiment_config["range_analysis"]["sigma_depths_m"]:
        sigma_z = depth_m ** 2 * disparity_sigma / (fx * d455["baseline_m"])
        values.append({"metric": f"analytical_sigma_z_at_{depth_m:g}_m",
                       "value": sigma_z, "unit": "m"})
        values.append({"metric": f"analytical_sigma_z_at_{depth_m:g}_m",
                       "value": sigma_z * 1000, "unit": "mm"})
        if depth_m == experiment_config["range_analysis"]["datasheet_reference_depth_m"]:
            values.append({"metric": "analytical_sigma_z_at_4m_percent_of_range",
                           "value": sigma_z / depth_m * 100, "unit": "percent"})
    result = pd.DataFrame(values)
    result.to_csv(out / "range_statistics_v2.csv", index=False)
    return result


def generate_all(out: Path, csv_path: Path, experiment_config: dict,
                 ladder_config: dict, scene_dirs: list[Path]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path, dtype={"scene": str})
    scenes = sorted(df["scene"].unique())
    stats = experiment_config["bootstrap"]
    indices = boot_indices(len(scenes), stats["replicates"], stats["seed"])
    ladder_levels = experiment_config["levels"]
    loo_effects = experiment_config["loo_effects"]
    noise_arms = experiment_config["noise_scale_arms"]
    arms = [*ladder_levels, *(f"Lfull_minus_{effect}" for effect in loo_effects),
            *experiment_config["arms"].keys()]
    arms = list(dict.fromkeys(arms))

    success = success_summary(df, arms, scenes, indices, stats["confidence"], out)
    errors = continuous_errors(df, arms, scenes, indices, experiment_config["continuous_quantiles"],
                              stats["confidence"], out)
    paired, loo = paired_comparisons(df, ladder_levels, experiment_config["full_level"], loo_effects,
                                     scenes, indices,
                                     stats["confidence"], out)
    isolated = isolated_ablation(
        df, experiment_config["full_level"],
        [name for name in experiment_config["isolated_effect_arms"]
         if name != experiment_config["full_level"]],
        scenes, indices, stats["confidence"], out
    )
    edge = edge_component_summary(success, experiment_config["edge_component_arms"], out)
    sweep = noise_sweep(success, errors, noise_arms, out)
    failures, bins, _ = l0_failure_analysis(df, scenes, indices, experiment_config,
                                            stats["confidence"], out)
    holes = hole_statistics(scene_dirs, experiment_config["figures"]["hole_levels"],
                            experiment_config["figures"]["hole_connectivity"],
                            experiment_config["figures"]["invalid_depth_value_m"], out)
    ladder_figure(scene_dirs, experiment_config["figures"]["ladder_scene_ids"][0],
                  ladder_levels, experiment_config["figures"], out)
    range_stats = range_and_noise_statistics(scene_dirs, experiment_config, ladder_config,
                                             load_config("d455"), out)
    summary = {
        "scenes": len(scenes),
        "rows": len(df),
        "success": success.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "leave_one_out": loo.to_dict(orient="records"),
        "isolated_ablation": isolated.to_dict(orient="records"),
        "edge_components": edge.to_dict(orient="records"),
        "noise_sweep": sweep.to_dict(orient="records"),
        "l0_failures": failures.to_dict(orient="records"),
        "l0_bins": bins.to_dict(orient="records"),
        "hole_statistics": holes.to_dict(orient="records"),
        "range_statistics": range_stats.to_dict(orient="records"),
    }
    (out / "analysis_summary_v2.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True,
                   default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n"
    )
    return summary
