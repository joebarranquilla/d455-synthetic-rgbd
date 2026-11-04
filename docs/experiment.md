# Experiment setup

The primary experiment is the declared v2 protocol in `protocol_v2.md`. It evaluates classical per-instance
localization using ground-truth instance masks (oracle mode) on the held-out test scenes. Development scenes are
used for debugging and visual modelling choices, not for estimating test results.

Localization success thresholds are defined in `configs/loc.yaml`: profiles use lateral position and long-axis
angle, while brackets use centroid and symmetry-aware rotation error. The v2 runner computes success and
continuous-error summaries, paired scene-bootstrap intervals, isolated-effect and noise-scale analyses, L0
failure summaries, hole statistics, and the declared figures.

Run `python scripts/run_v2.py` from the configured project environment. It requires the `v2-protocol` tag and
refuses to proceed if the frozen paths differ from that tag. It regenerates the declared test depth maps from raw
renders and produces every v2 table and figure. Do not tune from test results or modify parameters after the
protocol tag.

The older `scripts/run_experiment.py` entry point is retained for the v1-freeze experiment. Sensor effects are
modelled approximations; neither protocol is a calibrated model of real-camera performance.
