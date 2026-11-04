"""Run the test experiment and generate its tables and figures.

Run from the repository root with: python scripts/run_experiment.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d435sim import sensor  # noqa: E402
from d435sim.config import load_config  # noqa: E402
import make_depth  # noqa: E402
import run_loc  # noqa: E402
import experiment_analysis  # noqa: E402


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)


def blender_version() -> str:
    candidates = [shutil.which("blender"), os.environ.get("BLENDER"), str(ROOT / ".venv" / "bin" / "blender")]
    failures = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            path = path / "blender"
        if not path.is_file():
            continue
        try:
            result = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                                    check=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{path}: {exc}")
            continue
        first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
        if first_line:
            return first_line
        failures.append(f"{path}: version command returned no output")
    if failures:
        return "unavailable; " + "; ".join(failures)
    return "unavailable: no Blender executable found in PATH, BLENDER, or .venv/bin"


def _scene_dirs(data_dir: Path, split: str) -> list[Path]:
    root = data_dir / split
    if not root.is_dir():
        raise SystemExit(f"missing split directory: {root}")
    return sorted(path for path in root.iterdir() if (path / "meta.json").is_file())


def _arm_specs(experiment: dict, ladder: dict) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    configured_scales = sorted(float(scale) for scale in experiment["noise_scale_arms"])
    declared_scales = sorted(float(scale) for scale in experiment["noise_scales"])
    if configured_scales != declared_scales:
        raise ValueError("noise_scale_arms must cover exactly the declared noise_scales")
    scale_one_arm = experiment["noise_scale_arms"].get("1.0")
    if scale_one_arm not in experiment["levels"] or sensor.level_effects(scale_one_arm, ladder) != ["quant", "noise"]:
        raise ValueError("the s=1 noise-scale result must reuse the declared L2 arm")
    primary = [*experiment["levels"],
               *(f"Lfull_minus_{effect}" for effect in experiment["loo_effects"])]
    for arm in primary:
        specs[arm] = {"effects": sensor.level_effects(arm, ladder), "noise_scale": None}
    for arm, effects in experiment["arms"].items():
        specs[arm] = {"effects": list(effects), "noise_scale": None}
    for scale_text, arm in experiment["noise_scale_arms"].items():
        if arm == "L2":
            continue
        specs[arm] = {"effects": ["quant", "noise"], "noise_scale": float(scale_text)}
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    experiment = load_config("experiment")
    ladder = sensor.make_cfg("ladder")
    out = args.out or ROOT / experiment["outputs"]["directory"]
    out.mkdir(parents=True, exist_ok=True)
    scenes = _scene_dirs(args.data, experiment["split"])
    expected = experiment["bootstrap"]["expected_scenes"]
    if len(scenes) != expected:
        raise SystemExit(f"REFUSING TO RUN: expected {expected} test scenes, found {len(scenes)}.")
    arms = _arm_specs(experiment, ladder)
    jobs = [(scene, name, "ladder", spec["effects"], spec["noise_scale"])
            for scene in scenes for name, spec in arms.items()]

    started = time.time()
    with Pool(args.workers) as pool:
        pool.map(make_depth.process, jobs)
        print(f"regenerated depth maps for {len(scenes)} test scenes x {len(arms)} arms",
              flush=True)
        rows = [row for scene_rows in pool.map(
            run_loc.process_scene, [(scene, list(arms), "oracle") for scene in scenes], chunksize=1
        ) for row in scene_rows]

    csv_path = out / experiment["outputs"]["localization_csv"]
    pd.DataFrame(rows, columns=run_loc.COLUMNS).to_csv(csv_path, index=False)
    summary = experiment_analysis.generate_all(out, csv_path, experiment, ladder, scenes)
    for row in summary["range_statistics"]:
        if str(row["metric"]).startswith("analytical_sigma_z_at_"):
            print(f'{row["metric"]}: {row["value"]:.6g} {row["unit"]}')
    dev_scenes = _scene_dirs(args.data, "dev") if (args.data / "dev").is_dir() else None
    metadata = {
        "current_commit": git("rev-parse", "HEAD").stdout.strip(),
        "blender_version": blender_version(),
        "split_sizes": {"test": len(scenes), "dev": len(dev_scenes) if dev_scenes is not None else None},
        "dev_split_present": dev_scenes is not None,
        "test_scene_ids": [scene.name for scene in scenes],
        "arm_count": len(arms),
        "localization_rows": len(rows),
        "reproduction_command": "python scripts/run_experiment.py",
        "elapsed_seconds": time.time() - started,
        "analysis_summary": str(out / "analysis_summary.json"),
        "all_arm_summary_rows": len(summary["success"]),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {csv_path} and experiment tables/figures to {out}")


if __name__ == "__main__":
    main()
