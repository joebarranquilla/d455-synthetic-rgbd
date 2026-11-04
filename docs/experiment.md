# Experiment setup

The test-split experiment evaluates classical per-instance localization using the ground-truth instance masks
(oracle mode). It runs the cumulative sensor levels L0-L4 and leave-one-effect-out variants of L4. The test data
is generated separately from the development data; development scenes are used for debugging, not tuning against
test results.

Localization success thresholds are defined in `configs/loc.yaml`: profiles use lateral position and long-axis
angle, while brackets use centroid and symmetry-aware rotation error. The report script computes confidence
intervals by resampling scenes and uses shared bootstrap samples when comparing levels. Its tables and plots
include success rates, error quantiles, level comparisons, leave-one-out results, failure types, and an optional
gallery of difficult cases.

The sensor effects are modelled approximations; the experiment describes the behavior of this configuration and
localization method, not calibrated real-camera performance.

The experiment settings are in `configs/experiment.yaml`. Running `scripts/run_experiment.py` requires the
`v1-freeze` tag and matching frozen configuration/source files. It reuses existing scenes in `data/test/` and
creates sensor-level depth files there only when they are missing.
