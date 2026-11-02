"""Montage rows of rgb | depth | inst | normals: python scripts/montage.py DIR [N] OUT"""
import sys
from pathlib import Path

import cv2
import numpy as np


def row(d: Path) -> np.ndarray:
    rgb = cv2.imread(str(d / "rgb.png"))
    depth = np.load(d / "depth_gt.npy")
    valid = depth > 0
    dn = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = depth[valid].min(), depth[valid].max()
        dn[valid] = (255 * (depth[valid] - lo) / max(hi - lo, 1e-6)).astype(np.uint8)
    dc = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
    dc[~valid] = 0
    inst = cv2.imread(str(d / "inst.png"), cv2.IMREAD_UNCHANGED).astype(np.uint32)
    ic = np.stack([(inst * 97) % 256, (inst * 57) % 256, (inst * 17) % 256], -1).astype(np.uint8)
    ic[inst == 0] = 0
    nc = ((np.load(d / "normals.npy").astype(np.float32) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)[..., ::-1]
    return np.hstack([rgb, dc, ic, nc])


if __name__ == "__main__":
    src, n, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    dirs = sorted(p for p in src.iterdir() if p.is_dir())[:n]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([row(d) for d in dirs]))
    print(f"wrote {out} ({len(dirs)} rows)")
