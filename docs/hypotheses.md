# Pre-registered hypotheses

Written and committed **before** any run on the `test` split, then frozen with git tag `v1-freeze`.
After the tag, no parameter in `configs/*.yaml` and no localisation or sensor code may change.
Tuning and debugging happened on `dev` only. Results are reported whatever they turn out to be.

## Setup being tested
- Sensor ladder (`configs/ladder.yaml`): L0 ideal, L1 +quantisation, L2 +noise, L3 +dropout, L4 +edges (cumulative).
  Leave-one-out: `Lfull_minus_<effect>` removes one effect from L4.
- Method: classical per-instance localisation (`src/d455sim/loc.py`), oracle GT masks. No learning.
- Success: profile lateral < 5 mm and axis angle < 5 deg; bracket centroid < 5 mm and rotation < 10 deg
  (symmetry-aware). Along-axis error of profiles is reported but not part of success.
- Statistics: 95% bootstrap CIs that resample **scenes**, not instances. Paired differences use the same
  resampled scenes for both levels. "No detectable effect" = the CI of the difference includes 0.

## Hypotheses
Direction is predicted for each; "negligible" means the paired CI includes 0 or the drop is below 2 points.

| # | Hypothesis | What would refute it |
|---|-----------|----------------------|
| H1 | **Quantisation (L0 -> L1) is negligible** for both classes. The 1/64 px step gives sub-millimetre depth steps in this working range (0.4-1 m). | CI of the L0 -> L1 difference excludes 0 with a drop of 2 points or more. |
| H2 | **Noise (L1 -> L2) lowers success only slightly** (a few points at most), more for brackets (small, few points) than for profiles. | A drop above 10 points, or profiles hurt more than brackets. |
| H3 | **Dropout (L2 -> L3) is the largest single degradation**, and hurts brackets more than profiles. It reduces `n_points`, so success should fall with fewer points. | Dropout is not the largest drop in the leave-one-out table (H6), or brackets are not hurt more. |
| H4 | **Edge effects (L3 -> L4) have a small effect on profiles and a larger one on brackets.** Brackets are small, so a large share of their pixels lies near depth discontinuities (shadow, flying pixels, fattening). | The L3 -> L4 CI includes 0 for brackets. |
| H5 | **Success falls with distance** and with fewer points, for every level above L0. Noise grows with z^2. | No monotone trend in the success vs mean-depth and vs `n_points` bins. |
| H6 | **Leave-one-out ranking:** removing dropout from the full model recovers the most success; removing quantisation recovers about nothing. | A different effect gives the largest recovery, or removing quantisation recovers 2 points or more. |
| H7 | **Dominant failure types:** at L0-L2 mostly `icp_diverged` (partial views, symmetry); at L3-L4 `axis_flip` (bracket) and `too_few_points` / `icp_diverged` increase. `wrong_scale` and `outlier_dominated` stay rare in oracle mode. | `wrong_scale` or `outlier_dominated` exceeds 10% of failures at any level. |
| H8 | **Material:** among profiles, black anodised parts fail more often than silver or brushed aluminium once dropout is on. Dropout in the model depends on roughness only, so the effect is expected to be weak. Exploratory. | Black anodised is not worse, or the gap is within the CI. |
| H9 | **Profile lateral error is small even when along-axis error is large**: the along-axis error median is at least 3x the lateral median for L0-L4. | The ratio is below 3. |

## Reference numbers (dev, 5 scenes, only for orientation, no claim)
Oracle success L0 / L2 / L3: profiles 100% / 100% / 97%, brackets 93% / 87% / 60%.
These come from a smoke test, not a tuning target. The test results may differ.

## Known limitations (stated in advance)
- All sensor coefficients are heuristics and were not calibrated against a real D455. Findings describe
  sensitivity of this method to the modelled effects, not absolute real-world performance.
- Dropout depends on material roughness and incidence angle only, not on albedo or texture.
- Scenes are rendered with a single physics-settled bin setup, so conclusions apply to that distribution.
- Clustered (no-oracle) mode is secondary and not part of the pre-registered claims.
