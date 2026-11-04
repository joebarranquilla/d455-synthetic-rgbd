# D455 synthetic-depth experiment: v2 protocol

This protocol is committed and tagged before the v2 test run. The test experiment is limited to the specified `p_graze = 0.30`; there is no p_graze sweep or comparison.

## Declared deviation

`v1-freeze` specifies `p_graze = 0.60`. The sole v2 parameter deviation is `p_graze = 0.30`, declared in `configs/ladder_v2.yaml` and listed in `configs/ladder_v2.diff`. The user reports that this visually motivated choice was made by hand after inspecting perturbed **dev** depth maps; no test-set metrics had been seen. The time of the decision is unknown. The value is not calibrated to D455 data.

## Fixed sensor model and order

- Base seed `1234`; scene seeds are stored in scene metadata. Each effect uses an independent stream keyed by base seed, scene seed, and effect name.
- Operation order: disparity noise, disparity quantization, dropout, then edge operations. L2 is noise **before** quantization.
- Camera parameters: `848 x 480`, horizontal FOV `87 deg`, baseline `0.095 m`, validity range `[0.4, 6.0] m`, disparity noise `0.08 px RMS` times scale, disparity fractional bits `6`.
- Quantization step `2^-6 px`.
- Dropout: `p_base=0.01`, `p_graze=0.30`, `theta0_deg=70`, `width_deg=6`, `p_spec=0.20`, `spec_sigma_deg=12`, coherent-field `sigma=4 px`.
- Edge settings: occlusion relative tolerance `0.02`; flying-pixel depth-gradient threshold `0.05 m`, band half-width `1 px`, probability `0.30`; foreground-fattening min-filter window `3 px`.
- The existing localization configuration and algorithm are unchanged: oracle masks; skip objects below `100` visible mask pixels; minimum valid points `50`; SOR `20` neighbors and standard-deviation ratio `2.0`; model cloud `4000` points, seed `0`, normal radius `0.008 m`, max neighbors `30`; ICP `40` iterations, max `3000` observed points, trim fraction `0.8`, max correspondence `0.02 m`, translation tolerance `1e-5 m`, rotation tolerance `1e-4 rad`, fitness distance `0.004 m`. Profiles use four roll hypotheses and succeed at lateral `<5 mm` and axis `<5 deg`; brackets use four sign-flip hypotheses and succeed at centroid `<5 mm` and symmetry-aware rotation `<10 deg`. Existing failure thresholds are outlier-removed fraction `0.5`, scale tolerance `0.25`, scale quantile `0.01`, and profile flip angle `45 deg`. No baseline tuning or algorithm changes.

## Test arms

All arms use the same test scenes and the same scene-level bootstrap resampling. The exact effect definitions and values are fixed by `configs/ladder_v2.yaml` and `configs/experiment_v2.yaml`.

1. **Cumulative:** L0 (ideal), L1 (+ quantization), L2 (L1 + noise, applied before quantization), L3 (+ dropout), L4 (+ bundled edges).
2. **Leave-one-effect-out:** L4 without quantization, without noise, without dropout, and without edges.
3. **Isolated effects:** quantization only, noise only, dropout only, edges only, and cumulative L4.
4. **Edge components:** occlusion invalidation only, flying pixels only, and foreground fattening only.
5. **Noise scale:** `s = {0.5, 1, 2, 4, 8}`, with quantization plus disparity noise and all other effects disabled, i.e. otherwise L2. The `s=1` result is the shared L2 arm, not a second run.

No spatially correlated-noise variant is included. No other p_graze value is evaluated.

## Outcomes and analysis

- Success rate for overall, profile, and bracket instances; report 95% scene-bootstrap confidence intervals.
- For each arm, median and 90th percentile of lateral error, angular error, centroid error, and rotation error, overall and by class, with 95% scene-bootstrap intervals and sample counts. Errors not defined for a class remain missing.
- Cumulative paired comparisons, four leave-one-out comparisons, and isolated-arm deltas. Define degradation as `success(L0) - success(arm)`. Report `degradation(L4) - sum(degradation(single-effect arms))`, with a paired scene-bootstrap 95% CI; superadditivity is detectable only if the interval excludes zero in the positive direction.
- Noise-scale success and continuous-error summaries versus `s`.
- L0 failure counts by class in the categories orientation/symmetry ambiguity (`axis_flip`), too few valid points/outlier-dominated, ICP divergence/bad initialization, and other. These are mappings from the existing failure codes, not new algorithmic diagnoses. Also tabulate L0 success by valid point-count and mean-depth bins from the experiment config.
- Hole statistics at this configuration only: fraction of pixels invalid within object masks and connected-component hole-size distributions for L0-L4, by class and overall.
- One fixed scene (`00000`) ladder figure: L0-L4 depth and absolute error relative to L0, using fixed shared depth/error color limits.
- Fraction of valid raw test-depth pixels below `0.6 m`; analytical `sigma_z` at `0.45`, `0.9`, and `4 m` for the configured disparity noise, compared with the `5 mm` success threshold and the datasheet's `<2% at 4 m` figure.
- Scene bootstrap: all `100` test scenes, `2000` replicates, `95%` intervals, seed `0`, shared resampling indices across arms. Continuous outcomes are median (`q=0.5`) and 90th percentile (`q=0.9`), with intervals from the same scene resamples. L0 bins are mean depth `[0.4,0.6,0.8,1.0,1.3,1.7,6.0] m` and valid point count `[0,200,500,1000,2000,4000,1e9]`. No test-based parameter changes or scene selection.
- Working-distance analysis uses `0.45--0.9 m`, D455 ideal minimum `0.6 m`, the configured `0.4 m` cutoff, and a `0.10 m` near-cutoff band. It reports `sigma_z` at `0.45`, `0.9`, and `4.0 m`, the configured `5 mm` localization position threshold, and the datasheet's `<2%` error figure at `4 m`.

The single reproduction entry point is `python scripts/run_v2.py`; it verifies the v2 tag/config and regenerates all declared tables and figures from raw test renders. Test-derived outputs are not inputs to parameter selection.
