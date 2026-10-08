# Localization upgrade — measurement notes

Plan: `docs/plan-localization.md` (approved 2026-10-08). Every number here
carries the command, seed list and commit hash that produced it. Negative
results stay in. A stage becomes the default only after its pre-declared gate
(plan, "Verification") passes on the synthetic harness AND the real-data
proxies do not regress.

## Baseline (stage 0, re-measured; replaces output/validation_spacing.csv of 2026-09-01)

## Stage gates

| Stage | Gate | Result | Commit | Shipped as |
|---|---|---|---|---|
| 4 | counted = automatic partition on every auto test case | **PASS** 30/30 synthetic layouts (rng 0–5 × 1–5 tiles, truth seeds), counted (deformable) = auto = truth; `capped` False on all 30 and True on the constructed lattice with a 20-node cap. Printed phantom: counted (deformable) = auto on all 8 tiles; counted chord differs on the 25/31 pair | d81b45d | `fit_tiles(score="deformable")` opt-in; `fit_tiles_prior` (planner) uses it; `fit_tiles` default and `assess_implant` / `reconstruct` stay `"chord"` |
| 6 | margins > 2 on clean cases; constructed shared-seed case < 1 and flagged; normal σ grows with seed noise; centre covariance within 2× of the empirical scatter; planner cost < 0.5 s | **PASS** clean margins all `inf` (no same-count alternative); shared-seed case 4·10⁻⁶, flagged; normal σ 0.1 < 0.3 < 0.6 mm monotone; centre-covariance ratio 1.13–1.32 when every residual row carries the same noise (0.45–0.53 with noise-free axes, 3.8–4.1 with 10° axis noise, see below); margins 2–4 ms + uncertainty 6 ms on the printed phantom. Printed 25/31 pair: margin 1.87 under the bent-tile score → **not** flagged (0.27, flagged, under the chord score) | d81b45d | `margins=False` opt-in on `fit_tiles` / `fit_tiles_auto` / `fit_tiles_prior`; planner suggest calls `margins=True`; `DeformableFit.compute_uncertainty()` lazy, run for reported tiles in `_finish` |

### Stage 4 — one scoring rule (commit d81b45d)

What changed:
- `fit_tiles(..., score="chord" | "deformable")`, default `"chord"`.
- With `"deformable"`, every quad that passes the chord gates is fitted with
  `fit_deformable(seed_pts, seed_axes, kind="full")`, the same call auto mode makes.
- Its score becomes `max(1e-3, deformable_score(fit))`.
  - The floor keeps every item positive, so the selector's `count·1000 + score` still puts "more tiles" first.
  - Auto mode drops such quads instead. With the count given, they stay usable, only ranked last.
- The fit is attached as `pose.deform`. The pose normal and in-plane axis come from it, through `auto._deformed_pose`, so they are oriented the same way as in auto mode.
- `residual_mm` becomes the bent-tile RMS.
- Pairs keep their chord score. Degraded completion is unchanged.
- `fit_tiles_prior` (the planner's count-given path) now uses `"deformable"`.
- `pipeline.assess_implant` and the counted `reconstruct` call stay on `"chord"`, the rule they were calibrated on.
- `TileFitResult.capped` (also on `AutoFitResult`) reports whether the exact search hit its node cap. Before this change, a capped counted search fell back to the greedy answer without saying so.

Synthetic gate:
- Command: `python -m pytest tests/test_localization_scoring.py -k partition_equals_auto` (30 cases).
- Cases: `make_head_phantom(n_tiles=1..5, rng_seed=0..5)`, truth seeds and axes.
- Shortcut: the cases are built on a 4 mm grid. The generator draws the layout before it rasterises, and `test_truth_seeds_do_not_depend_on_spacing` checks that the 4 mm and 0.8 mm truth are identical.
- Result: the counted (deformable) partition equals the auto partition and the truth on all 30 cases. `capped` is False everywhere.
- `test_auto_matches_counted_fit` (chord vs auto, detected seeds) still passes.
- The cap flag is tested on a 6 × 6 lattice of 36 seeds with n_full = 9. It is False with the default cap and True with `_SEARCH_NODE_CAP = 20`, and the capped selection stays disjoint.

Runtime on the printed phantom (32 seeds, 14 quads pass the chord gates, min of 3 runs):

| Path | Time |
|---|---|
| counted chord | 0.01 s |
| counted deformable | 1.08 s (about 77 ms per bent-tile fit) |
| auto | 2.19 s (includes the loose tier) |
| planner, count given (`fit_tiles_prior(n_full=8)`), old: chord + 8 bent-tile fits | 0.65 s |
| planner, count given, new: deformable + margins + uncertainty | 1.29 s (+0.64 s) |

Other agents' test runs shared the machine, so treat these as ±20 %.

### Stage 6 — uncertainty outputs (commit d81b45d)

**Pose covariance.** `DeformableFit.compute_uncertainty()` fills the fields in place.
- It is lazy: the search never pays for it.
- `_finish` calls it for the reported tiles: supported and relaxed-tentative tiles. Triplet-completed tiles are skipped, because their 4th seed sits on the model, so s² and JᵀJ would both be optimistic.
- `cov_x = s²(JᵀJ)⁻¹` uses the solver Jacobian of the winning start. `s² = 2·cost/(m − 9)`, where m = 18 rows: 12 position, 4 axis and 2 curvature-prior.
- JᵀJ is column-scaled, then pseudo-inverted over eigenvalues above 10⁻¹⁰ × max. `ψ` has no effect on a flat or equal-curvature sheet, so its column can vanish.
- `cov_cond` and `cov_rank` are reported.
- `center_cov` propagates `cov_x` through the mean predicted seed position, using a forward-difference Jacobian (9 evaluations).
- `normal_sigma_deg` = √trace of the propagated normal covariance: the RMS tilt over both tangent directions.
- The 2-seed rigid fallback has no least-squares solution, so all fields stay None.
- Cost: about 0.7 ms per tile, 6 ms for the 8 printed tiles.

**Calibration** (`python scripts/validation_loc_scoring.py --calibration`):
- Setup: 200 draws per row, 0.3 mm Gaussian noise per seed coordinate, `hinge_starts=False`.
- Ratio = mean trace(`center_cov`) / trace of the empirical covariance of the fitted centres.

| κ (1/mm) | axis noise | ratio | normal σ predicted / empirical RMS (°) |
|---|---|---|---|
| 0 | none | 0.45 | 1.47 / 2.03 |
| 0 | σ/w_axis = 4.3° (every row at one level) | **1.13** | 2.37 / 2.24 |
| 0 | 5° | 1.36 | 2.60 / 2.30 |
| 0 | 10° | 3.82 | 4.35 / 2.87 |
| 0.05 | none | 0.53 | 1.77 / 2.15 |
| 0.05 | 4.3° | **1.32** | 2.81 / 2.43 |
| 0.05 | 5° | 1.56 | 3.06 / 2.52 |
| 0.05 | 10° | 4.09 | 4.99 / 3.31 |

Reading:
- The textbook estimator is calibrated when its own assumption holds: one noise variance for every residual row. That is the 4.3° row, and the test case `test_center_cov_matches_empirical_scatter`.
- The pooled s² follows whichever residual class dominates:
  - with noise-free axes it under-reports by about 2×;
  - with 10° axis error it over-reports by about 4×.
- Real PCA axes on bloomed capsules are closer to the second case, so on real scans these numbers are conservative.
- The principled fix is stage 5's whitened residuals (per-seed Σᵢ). With them s² should come out near 1, to be checked with NEES there. No constant was tuned here.

Unit tests (`tests/test_localization_scoring.py`):
- Mean `normal_sigma_deg` over 20 draws rises monotonically for 0.1 / 0.3 / 0.6 mm seed noise: 0.57 / 1.65 / 3.28° (rng 11, noise-free axes).
- The centre-covariance ratio is 1.21 for the 4.3° case above (rng 5, κ = 0.05); the test asserts [0.5, 2].

**Partition margins** (`margins=True` on `fit_tiles`, `fit_tiles_auto` and `fit_tiles_prior`):
- For each selected tile, the selector is run again with that item removed. Its seeds stay available to every other grouping.
- Auto mode uses `_PerCountSelector` at the chosen count n_sel. The counted path uses `fit._Selector`; there, a lower achievable count (Δvalue ≥ 1000) means no alternative exists.
- Margin = best total − best total of the same count without the tile, or `inf` when no alternative exists.
- `partition_alternatives[tile_id]` holds the alternative's seed groups.
- `ambiguous_tiles` lists margins below `AMBIGUOUS_MARGIN = 1.0`. That is 0.5 mm of bent-tile RMS on one tile, from the `DEF_W_RMS = 2/mm` term of `deformable_score`. On the chord path the same number means about 0.67 mm of chord RMS.
- Degraded completions and tentative tiles carry no margin.
- The planner's suggest status appends `T<k> ambiguous (margin x.x)` (`planner._suggest_notes`).

Synthetic results:
- Clean cases: every margin is `inf`. With exactly 4n true seeds there is no other grouping of n tiles.
- Constructed shared-seed case: 3 corners of a square plus two candidates for the 4th corner, ±1.5 mm out of plane and mirror-symmetric. Margin 4·10⁻⁶, flagged, alternative = the other quad. Counted gives the same.
- Same case with one candidate on the corner and the other 3.5 mm off: margin 1.29, not flagged. A corner 3 mm off-plane costs only 1.12: the bent-tile model absorbs much of it by twisting.

## Real-data proxies (no truth)

### Printed 8-tile phantom: stages 4 and 6 (commit d81b45d)

Commands:
- Seed cache: `python scripts/validation_loc_scoring.py --extract`. This runs `load_volume` + `reconstruct` in 24 s, giving 32 seeds at 0.59 × 0.59 × 1.0 mm, saved to `output/loc_scoring/cache/phantom8_seeds.pkl`.
- Comparison: `python scripts/validation_loc_scoring.py`, output in `output/loc_scoring/phantom8_measure.txt`.
- Seeds 25 and 31 are 3.93 mm apart.

| Tile (seeds) | counted chord | counted deformable | auto |
|---|---|---|---|
| {0,1,2,5} | inf | inf, rms 0.26 mm | inf |
| {10,11,17,18} | inf | inf, 0.31 | inf |
| {7,13,15,20} | inf | inf, 0.40 | inf |
| {3,4,6,9} | inf | inf, 0.50 | inf |
| {14,22,23,28} | 1.72 | 2.13, 0.56 | inf (the alternative {21,22,23,28} collides with the crumpled tile, which auto selects) |
| **25 / 31 pair** | {25,26,29,30} + {8,16,24,31}, margin **0.27 → ambiguous** | {8,16,24,25} + {26,29,30,31}, margin **1.87** | same as counted deformable, **1.87** |
| {12,19,21,27} crumpled | degraded completion (no margin) | degraded completion (no margin) | loose tier, inf |

Where seeds 25 and 31 go:

| Scoring | Seed 25 | Seed 31 | Margin |
|---|---|---|---|
| chord | {25,26,29,30} | {8,16,24,31} | 0.27 (totals 10.83 vs 10.56) |
| deformable (counted and auto) | {8,16,24,25} | {26,29,30,31} | 1.87 |

- Per tile under the bent-tile score: 6.28 + 6.26 for the chosen reading, 5.78 + 4.89 for the chord reading. Bent-tile RMS 0.78 / 0.87 mm vs 1.11 / 1.47 mm.
- The best alternative swaps both tiles at once: it is exactly the other reading.
- Stage 4 therefore makes the two modes agree on the bent-tile reading.
- Under the pre-declared threshold, stage 6 does not flag the pair: 1.87 > 1.0. The chord score would have flagged it (0.27).

Pose uncertainty (auto):
- Centre σ (√trace) is 0.29–0.41 mm on the four clean tiles and 0.46 mm on the crumpled one. It is 0.69–0.95 mm on {14,22,23,28} and on the 25/31 pair.
- Normal σ is 2.8–3.9° on the clean tiles, 6.4° on the crumpled one, and 5.9–9.3° on the others.
- These track the bent-tile RMS. They inherit the pooled-s² caveat above: axis misfit on bloomed 1 mm capsules inflates them.

Runtime added for the planner suggest call:
- Margins: 2–4 ms (one selector run per tile).
- Uncertainty: about 6 ms.
- Count-given path: +0.64 s from stage 4's deformable scoring of 14 quads instead of fitting 8 selected ones.
- Count-unknown path: about +10 ms.
- All well under the 0.5 s margin budget.

## With-truth track

## Open decisions

- **Margin per configuration vs per swapped tile.** The 25/31 alternative changes TWO tiles, so its 1.87 is about 0.93 per tile, or 0.47 mm of bent-tile RMS each. That is below the 1.0 line if the threshold were read per changed tile. The plan defines the margin as the plain best-vs-second-best total, and that is what shipped. Normalising by the number of swapped tiles is a one-line change. Decide it on a case with truth, not on this phantom.
- **Pooled s².** The Gauss–Newton s² mixes position, axis and prior rows. See the stage 6 calibration table. Revisit once stage 5's whitened residuals land, so that s² carries per-seed Σᵢ.

## Runs log

| Date | Command | Seeds | Commit | Wall time |
|---|---|---|---|---|
| 2026-10-08 | `python scripts/validation_loc_scoring.py --extract` | printed 8-tile phantom (32 seeds) | d81b45d | 25 s |
| 2026-10-08 | `python scripts/validation_loc_scoring.py` | cached printed-phantom seeds | d81b45d | 11 s |
| 2026-10-08 | `python scripts/validation_loc_scoring.py --calibration` | synthetic bent tile, rng 1, 200 draws × 8 rows | d81b45d | 60 s |
| 2026-10-08 | `python -m pytest tests/test_tiles*.py tests/test_tile_model.py tests/test_implant_assessment.py tests/test_localization_plumbing.py tests/test_localization_scoring.py` | phantom rng per test | d81b45d | 157 s, 254 passed (215 before + 39 new) |
