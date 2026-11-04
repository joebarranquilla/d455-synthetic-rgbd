# Experiments

Both run oracle-mask per-instance localization (thresholds in `configs/loc.yaml`) on the 100 test scenes. Dev scenes are
for debugging only. Parameters: `configs/camera.yaml`, `configs/scene.yaml`, `configs/ladder.yaml`.

- **v1** (`scripts/run_experiment.py`, `configs/experiment.yaml`, tag `v1-freeze`): cumulative levels L0-L4 and
  leave-one-effect-out. Output `outputs/experiment/`.
- **v2** (`scripts/run_v2.py`, `configs/experiment_v2.yaml`, tag `v2-protocol`): v1 arms plus isolated effects, edge
  components, noise-scale sweep, L0 failure types, hole statistics and range analysis. Output `outputs/v2/`.
