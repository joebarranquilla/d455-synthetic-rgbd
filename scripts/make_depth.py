"""Write degraded depth maps: make_depth.py --split S (--level L3 | --drop edges) [--data DIR] [--workers N]

Output: <sample>/depth_<NAME>.npy (float32, 0 = invalid); NAME is the level or Lfull_minus_<effect>.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d455sim import sensor  # noqa: E402


def process(args: tuple) -> str:
    d, name = args
    cfg = sensor.make_cfg()
    meta = json.loads((d / "meta.json").read_text())
    ctx = sensor.build_ctx(meta, np.load(d / "normals.npy"), cv2.imread(str(d / "inst.png"), cv2.IMREAD_UNCHANGED))
    out = sensor.apply_level(np.load(d / "depth_gt.npy"), ctx, cfg, name)
    np.save(d / f"depth_{name}.npy", out)
    return d.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--level", help="L0..L4")
    g.add_argument("--drop", help="effect to leave out of the full level")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    name = a.level if a.level else f"Lfull_minus_{a.drop}"
    sensor.level_effects(name, sensor.make_cfg())  # fail early on bad names
    dirs = sorted(p for p in (Path(a.data) / a.split).iterdir() if (p / "depth_gt.npy").exists())
    with Pool(a.workers) as pool:
        done = pool.map(process, [(d, name) for d in dirs])
    print(f"wrote depth_{name}.npy for {len(done)} samples in {Path(a.data) / a.split}")


if __name__ == "__main__":
    main()
