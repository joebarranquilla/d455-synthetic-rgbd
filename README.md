# Synthetic RGB-D for an eye-in-hand RealSense D435

Renders aluminium T-slot profiles and L-brackets (Blender/BlenderProc), degrades the ideal depth with a seeded
D435-inspired sensor model, and evaluates oracle-mask localization. Camera values are in `configs/camera.yaml`
(datasheet: 1280x720, 87 deg HFOV, 50 mm baseline, 0.3-3 m). Sensor coefficients are assumptions, not calibrated
to real captures. (The Python package keeps its original name `d455sim`.)

## Setup
```
python3 -m venv .venv && make deps
make test
```

## Run the experiments (needs a GPU for rendering)
```
# 1. Render the 100 test scenes into data/test (about the slowest step)
.venv/bin/blenderproc run scripts/render_samples.py --split test --n 100 --seed 10000

# 2. v1 (tag v1-freeze) -> outputs/experiment/
make experiment

# 3. v2 (tag v2-protocol) -> outputs/v2/
make experiment-v2
```
`make experiments` runs 2 and 3. Both runners refuse to start if the frozen configs/code differ from their tag, so
commit and tag first (see below). Depth levels are cached as `data/test/*/depth_*.npy`; delete them after any
config change.

Optional: `make sample` renders 5 dev scenes and a montage (`outputs/montage.png`);
`scripts/render_samples.py --split dev` is for debugging only. What v1 and v2 contain: [docs/experiment.md](docs/experiment.md).

## Tags
```
git add -A && git commit -m "D435 camera"
git tag -f v1-freeze && git tag -f v2-protocol
```

## Layout
`configs/` all parameters, `src/d455sim/` library (only `parts.py`, `scene.py`, `render.py` use bpy),
`scripts/` entry points, `tests/` pytest (no GPU/Blender). `data/` and `outputs/` are never committed.
