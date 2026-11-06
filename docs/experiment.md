# Experiment

The experiment runs oracle-mask per-instance localization (thresholds in `configs/loc.yaml`) on the 100 test scenes.
Dev scenes are for debugging only. Parameters: `configs/camera.yaml`, `configs/scene.yaml`, `configs/ladder.yaml`.

Run it with `make experiment` or `python scripts/run_experiment.py`. It evaluates cumulative levels L0-L4,
leave-one-effect-out and isolated effects, edge components, a noise-scale sweep, L0 failure types, hole statistics,
and range analysis. Results are written to `outputs/experiment/`.
