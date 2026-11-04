"""Run the full experiment on the TEST split: all levels L0-L4 plus leave-one-out, oracle-mask localisation.

python scripts/run_experiment.py [--data DIR] [--n N] [--workers W] [--out CSV]

Refuses to run unless git tag `v1-freeze` exists and the frozen paths (configs, src) are identical to the tag.
Step 1 writes depth_<LEVEL>.npy per scene (skipped if present), step 2 localises every instance, one CSV row
per (scene, level, instance). Then run scripts/report_tables.py.
"""
import argparse
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from d455sim.config import load_config  # noqa: E402
import make_depth  # noqa: E402
import run_loc  # noqa: E402


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def check_freeze(cfg: dict) -> None:
    tag = cfg["required_tag"]
    if not git("tag", "--list", tag).stdout.strip():
        sys.exit(f"REFUSING TO RUN: git tag {tag} is missing. Create the freeze tag before running the test experiment.")
    if git("diff", "--quiet", tag, "--", *cfg["frozen_paths"]).returncode != 0:
        sys.exit(f"REFUSING TO RUN: configs/src differ from tag {tag} (parameters are frozen).")


def level_names(cfg: dict) -> list[str]:
    return list(cfg["levels"]) + [f"Lfull_minus_{e}" for e in cfg["loo_effects"]]


def _depth_job(args: tuple) -> None:
    d, name = args
    if not (d / f"depth_{name}.npy").exists():
        make_depth.process((d, name))


def main() -> None:
    cfg = load_config("experiment")
    check_freeze(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n", type=int, default=None, help="first N scenes (debugging the script only)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "experiment" / "loc_test.csv"))
    a = ap.parse_args()
    root = Path(a.data) / cfg["split"]
    dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())[: a.n]
    if not dirs:
        sys.exit(f"no scenes in {root}")
    names = level_names(cfg)
    t0 = time.time()
    with Pool(a.workers) as pool:
        pool.map(_depth_job, [(d, n) for d in dirs for n in names])
        print(f"depth maps ready for {len(dirs)} scenes x {len(names)} levels ({time.time() - t0:.0f} s)", flush=True)
        rows = [r for rs in pool.map(run_loc.process_scene, [(d, names, "oracle") for d in dirs], chunksize=1) for r in rs]
    df = pd.DataFrame(rows, columns=run_loc.COLUMNS)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {time.time() - t0:.0f} s); tag {cfg['required_tag']} = "
          f"{git('rev-list', '-n', '1', cfg['required_tag']).stdout.strip()[:10]}")


if __name__ == "__main__":
    main()
