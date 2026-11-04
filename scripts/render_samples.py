# Render samples: blenderproc run scripts/render_samples.py --split dev --n 5 --seed 0
import blenderproc as bproc  # noqa: F401  must be first

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d455sim import render, scene  # noqa: E402
from d455sim.config import intrinsics, load_config  # noqa: E402
from d455sim.meta import build_meta, validate_meta  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()

    cfg, d455 = load_config("scene"), load_config("camera")
    K = intrinsics(d455)
    render.init(cfg, d455, K)
    times = []
    for idx in range(a.start, a.start + a.n):
        t0 = time.time()
        seed = a.seed * 1_000_000 + idx
        rng = np.random.default_rng(seed)
        items = scene.build_scene(cfg, rng)
        T = scene.sample_camera_pose(cfg, rng)
        out = render.render_sample(cfg, d455, K, T, seed % (2 ** 31))

        T_inv = np.linalg.inv(T)
        objects = []
        for it in items:
            s = it["spec"]
            objects.append({"id": s["id"], "class": s["class"], "dims_m": s["dims_m"],
                            "T_cam_obj": T_inv @ np.asarray(it["obj"].get_local2world_mat()),
                            "material": s["material"], "n_visible_px": int((out["inst"] == s["id"]).sum())})
        meta = build_meta(K, T, d455["baseline_m"], seed, objects)
        validate_meta(meta)

        d = Path(a.out) / a.split / f"{idx:05d}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "rgb.png"), cv2.cvtColor(out["rgb"], cv2.COLOR_RGB2BGR))
        np.save(d / "depth_gt.npy", out["depth"])
        np.save(d / "normals.npy", out["normals"])
        cv2.imwrite(str(d / "inst.png"), out["inst"])
        cv2.imwrite(str(d / "cls.png"), out["cls"])
        (d / "meta.json").write_text(json.dumps(meta))
        times.append(time.time() - t0)
        print(f"sample {idx:05d}: {times[-1]:.2f} s ({len(objects)} objects)", flush=True)
    print(f"mean {np.mean(times):.2f} s/sample over {len(times)} samples")


main()
