# Synthetic RGB-D for an eye-in-hand RealSense D435

Renders aluminium T-slot profiles and L-brackets (Blender/BlenderProc), degrades the ideal depth with a seeded
D435-inspired sensor model, and evaluates oracle-mask localization. Camera values are in `configs/camera.yaml`
(datasheet: 1280x720, 87 deg HFOV, 50 mm baseline, 0.3-3 m). Sensor coefficients are assumptions, not calibrated

## Setup
```
python3 -m venv .venv && make deps
make test
```

## Run the experiment (needs a GPU for rendering)
```
# 1. Render the 100 test scenes into data/test (about the slowest step, use 10 for quick testing)
.venv/bin/blenderproc run scripts/render_samples.py --split test --n 100 --seed 10000

Alternatively, if data is available, upload it to `data/test` before running the experiment. Example:
```
data/test/
  00000/
    rgb.png
    depth_gt.npy
    normals.npy
    inst.png
    cls.png
    meta.json
  ...
  00099/
    ...
```

# 2. Run the experiment -> outputs/experiment/
make experiment
```
Depth levels are cached as `data/test/*/depth_*.npy`; delete them after any config change.

Optional: `make sample` renders 5 dev scenes and a montage (`outputs/montage.png`);
`scripts/render_samples.py --split dev` is for debugging only. See [docs/experiment.md](docs/experiment.md) for experiment details.

## Layout
`configs/` all parameters, `src/d435sim/` library (only `parts.py`, `scene.py`, `render.py` use bpy),
`scripts/` entry points, `tests/` pytest (no GPU/Blender). `data/` and `outputs/` are never committed.
