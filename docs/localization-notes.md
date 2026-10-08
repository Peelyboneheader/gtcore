# Localization upgrade — measurement notes

Plan: `docs/plan-localization.md` (approved 2026-10-08). Every number here
carries the command, seed list and commit hash that produced it. Negative
results stay in. A stage becomes the default only after its pre-declared gate
(plan, "Verification") passes on the synthetic harness AND the real-data
proxies do not regress.

## Baseline (stage 0, re-measured; replaces output/validation_spacing.csv of 2026-09-01)

All stage 0 numbers: code commit **fe4a1a8** (branch `loc/harness`), clean
tree, 2026-10-08. Each block's first line is the exact command. Outputs (not
tracked, `output/` is git-ignored) are named after each block; every run is
appended to a `runs.csv` with its commit.

### Seed renderer and its calibration (`gtcore/phantom/seed_render.py`)

Forward model, deliberately sharing no code with any localization or fitting
model: exact capsule (cylinder with hemispherical caps, 4.5 mm end to end,
0.8 mm across, 2.128 mm³) marked in/out on a 0.1 mm grid in voxel-index
space (m = ceil(spacing / 0.1) samples per voxel and axis, over a box of whole
voxels covering the capsule plus 4 PSF σ), Gaussian PSF (σ 0.45 mm) on that
grid, block average to the true voxel footprint (the slab integration), sum
into the volume, noise 10 HU, clip at 3071 HU. Checks
(`tests/test_seed_render.py`): mass Σ(v − bg)·V = contrast × capsule volume
within 2 % (measured 0.7 %); noise-free grey centroid = truth within 0.02 mm
on 0.4 × 0.5 × 0.7 mm (measured ≤ 0.014 mm over 200 random seeds; on 2.0 and
2.8 mm slabs the block-averaged centroid of 10 random seeds is off by up to
0.07 / 0.46 mm in z, the slab information limit); a 0.5 mm rendering block-averaged to 2.0 mm =
the direct 2.0 mm rendering within 1 % of peak; ~8 ms per seed.

Real seed peaks (max HU within ±1 mm of each detected seed, commit fe4a1a8):

- printed 8-tile phantom (Philips 0.59 × 0.59 × 1.0 mm, O-MAR): **31/32 seeds
  at 3071 HU**, the last at 2862;
- PostOp CT: the series is **1 mm slices** (SliceThickness 1, Siemens,
  0.52 mm pixels) with 64 slices present at irregular 1 mm multiples (steps
  1–11 mm), which the loader rebuilds onto a 2.0 mm grid (52 of 89 slices
  interpolated). The 22 tile-assigned seeds peak at a **median 2936 HU on the
  measured slices, 4/22 at 3071**; the 1500–1950 HU of `docs/data-notes.md`
  holds only on interpolated grid slices.

One contrast cannot reproduce both stage 0 targets: for random seed
orientations the 1 mm / 2 mm peak ratio is only ~1.3 (the 0.8 mm capsule
cross-section, not the slab, limits a 1 mm peak), so a contrast giving a
1700 HU median on 0.5 × 0.5 × 2.0 mm gives a 2200 HU median and no
saturation on 0.59 × 0.59 × 1.0 mm. Two contrasts (HU above a 35 HU
background) were therefore calibrated by bisection over 400 seeds at uniform
sub-voxel offsets and axes:

- `METAL_HU_POSTOP = 8380` (= `DEFAULT_METAL_HU`): median peak 1700 HU on
  0.5 × 0.5 × 2.0 mm (the stage 0 target);
- `METAL_HU_PRINTED = 13500`: 31/32 seeds reach 3071 HU on
  0.59 × 0.59 × 1.0 mm, as measured on the printed phantom.

Physical cross-check (not a target, not a default):
`METAL_HU_POSTOP_MEASURED = 10900` reproduces on 0.52 × 0.52 × 1.0 mm BOTH the
PostOp's measured median (2936 HU) and its saturated fraction (18 %, real
4/22) with one parameter; it would put the 0.5 × 0.5 × 2.0 mm median at
2200 HU. The 1700 HU target is therefore pessimistic for that scanner (see
Open decisions).

`python scripts/validation_seed_localization.py --calibrate` (400 seeds per
row, layout rng 0, noise seeds 0–399, unclipped peaks):

| grid | contrast | metal_hu | median peak | IQR | P5–P95 | ≥ 3071 |
|---|---|---|---|---|---|---|
| G1 (0.59x0.59x1.0) | postop | 8380 | 2203 | 2138–2267 | 1972–2382 | 0 % |
| G1 (0.59x0.59x1.0) | printed | 13500 | 3526 | 3423–3629 | 3152–3816 | 96 % |
| G2 (0.5x0.5x2.0) | postop | 8380 | 1696 | 1556–1938 | 1255–2247 | 0 % |
| G2 (0.5x0.5x2.0) | printed | 13500 | 2712 | 2484–3098 | 1994–3600 | 27 % |
| G3 (1 mm slabs, irregular gaps, loader grid 0.5x0.5x2.0) | postop | 8380 | 1776 | 1299–2229 | 730–2366 | 0 % |
| G3 (1 mm slabs, irregular gaps, loader grid 0.5x0.5x2.0) | printed | 13500 | 2839 | 2073–3568 | 1150–3794 | 45 % |
| G3L (0.5x0.5x1.0, odd slices interpolated) | postop | 8380 | 2222 | 2070–2297 | 1328–2439 | 0 % |
| G3L (0.5x0.5x1.0, odd slices interpolated) | printed | 13500 | 3558 | 3309–3680 | 2115–3905 | 82 % |
| G4 (0.7x0.7x2.8) | postop | 8380 | 1202 | 1107–1443 | 874–1812 | 0 % |
| G4 (0.7x0.7x2.8) | printed | 13500 | 1913 | 1762–2299 | 1391–2893 | 4 % |

`make_head_phantom(seed_render="analytic")` uses `METAL_HU_PRINTED` (the
binary phantom's 8000 HU painted capsules peak at ~4000 HU on its 0.7 mm grid,
i.e. the same saturating regime) and lets each capsule displace the local
tissue (contrast = 35 + metal_hu − local HU). Without the displacement the
two seeds that sit in the cavity's air pocket rendered ~270 HU too dark and
were lost at 0.7 mm (partition 3/5); fixed in 63c9ab2. The default
`seed_render="binary"` phantom is bit-identical to 3cf35af (sha256 test).

### Synthetic localization harness (G1–G4)

Grids: **G1** 0.59 × 0.59 × 1.0 mm (printed contrast); **G2**
0.5 × 0.5 × 2.0 mm (PostOp contrast); **G3** PostOp-like gaps (PostOp
contrast): rendered at 0.5 × 0.5 × 1.0 mm, slices kept with irregular steps
of 1/2/3 slices (p = ¼, ½, ¼, median 2), rebuilt by the loader's own rule
onto its grid (median kept spacing = 2.0 mm; `drop_and_interpolate(...,
grid="loader")`), so it is a 0.5 × 0.5 × 2.0 mm volume of 1 mm slabs where
every grid slice without a kept slice at its position is interpolated, like
the PostOp; **G4** 0.7 × 0.7 × 2.8 mm (PostOp contrast). The literal reading
"1 mm grid, every other slice interpolated" is available as `--grids G3L`
but not used: the pipeline then applies its thin-slice 2000 HU threshold and
detects only 5–10 % of the seeds at PostOp contrast (1 rng: recall 0.10
sparse / 0.05 crowded).

Layouts (fixed): **sparse** = 40 seeds per volume, every pair ≥ 8 mm apart
(dart throwing); **crowded** = 40 seeds grown so each has a neighbour
7–8 mm away. Layout rng 0–4 (same truth on every grid), 200 seeds per grid
and layout, centres continuous in a 44 mm box (uniform sub-voxel offsets),
axes uniform on the sphere; noise seed 1000·(rng + 100·crowded) + grid index.
Baseline = `detect_seed_candidates` at `seed_detection_params(spacing)`
(2000 HU at ≤ 1.2 mm, 1200 HU at 2.0 mm, 1000 HU at 2.8 mm) +
`filter_seed_shaped`. Matching: Hungarian ≤ 3 mm. "thr shift" = median
centre shift between detection at 1200 and 2000 HU for seeds matched in both
(n in brackets; at 2.0–2.8 mm few seeds exceed 2000 HU). "axis°" = median
seed-axis error. "peak HU (sat)" = median truth-seed peak and % at 3071.
ms/seed = detection time / candidates (small 60 mm volumes).

`python scripts/validation_seed_localization.py --methods baseline` — commit fe4a1a8, 2026-10-08; layouts rng 0-4 × 40 seeds per volume (200 per grid and layout); match: Hungarian ≤ 3 mm; contrast G1 13500 HU, G2–G4 8380 HU above 35 HU background; PSF σ 0.45 mm, noise 10 HU, clip 3071 HU; wall 17 s

| grid | layout | method | recall | FP | bias i / j / k (mm) | RMS i / j / k (mm) | 3D mean ± SD | P95 | max | axis° med | thr shift (n) | NEES (in 95 %) | ms/seed | peak HU (sat) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | sparse | baseline | 1.00 | 0 | 0.005 / -0.008 / -0.009 | 0.09 / 0.09 / 0.11 | 0.13 ± 0.10 | 0.37 | 0.45 | 2.9 | 0.09 (200) | – | 0.3 | 3071 (96 %) |
| G1 | crowded | baseline | 1.00 | 0 | 0.001 / -0.004 / 0.005 | 0.09 / 0.08 / 0.12 | 0.13 ± 0.10 | 0.35 | 0.51 | 2.6 | 0.09 (200) | – | 0.2 | 3071 (94 %) |
| G2 | sparse | baseline | 0.94 | 14 | -0.004 / -0.011 / 0.043 | 0.33 / 0.30 / 0.40 | 0.45 ± 0.40 | 1.30 | 1.54 | 18.9 | 0.39 (45) | – | 0.2 | 1676 (0 %) |
| G2 | crowded | baseline | 0.96 | 18 | 0.011 / -0.049 / 0.022 | 0.35 / 0.32 / 0.42 | 0.47 ± 0.42 | 1.31 | 1.48 | 23.4 | 0.52 (39) | – | 0.2 | 1678 (0 %) |
| G3 | sparse | baseline | 0.76 | 41 | 0.043 / 0.086 / 0.133 | 0.49 / 0.48 / 0.78 | 0.86 ± 0.57 | 1.90 | 2.20 | 47.2 | 0.21 (77) | – | 0.2 | 1628 (0 %) |
| G3 | crowded | baseline | 0.78 | 39 | 0.075 / 0.037 / 0.033 | 0.45 / 0.42 / 0.68 | 0.78 ± 0.49 | 1.68 | 1.91 | 44.0 | 0.28 (91) | – | 0.2 | 1836 (0 %) |
| G4 | sparse | baseline | 0.91 | 10 | 0.043 / -0.015 / -0.055 | 0.36 / 0.31 / 0.64 | 0.68 ± 0.40 | 1.42 | 1.68 | 55.4 | 0.10 (2) | – | 0.1 | 1193 (0 %) |
| G4 | crowded | baseline | 0.85 | 10 | 0.018 / 0.015 / 0.042 | 0.34 / 0.34 / 0.64 | 0.70 ± 0.41 | 1.53 | 1.76 | 54.4 | 0.14 (3) | – | 0.1 | 1220 (0 %) |

Reading: thin slices (G1) are already at 0.13 mm mean (z RMS 0.11 mm) with
0.09 mm threshold sensitivity; the coarse grids carry the error, dominated
by the slice axis (z RMS 0.40–0.42 mm at G2, 0.64 mm at G4, 0.68–0.78 mm at
G3) with recall 0.85–0.96 and 10–18 false positives per 200 seeds. Every false
positive on G2–G4 (132 in all, incl. G3) lies 1.2–2.6 mm from a true seed:
it is the second fragment of a split seed, the stage 3 target.
G3 (PostOp-like gaps) is the worst grid: recall 0.76–0.78, 39–41
false positives, P95 1.7–1.9 mm, z RMS 0.68–0.78 mm. The seed axis is
meaningless at ≥ 2 mm (median 19–55°). No method yet reports `cov_ras`, so
NEES is empty.

### Head phantom (`scripts/validation_spacing.py`)

Phantom `make_head_phantom(spacing=0.7, n_tiles=3)`, rng 0–4, slices
block-averaged ×1–4; detection adaptive (pipeline parameters) and fixed
(2000 HU); counted tile fit (`fit_tiles(..., 3)`); seeds matched Hungarian
< 2 mm (the historical gate).

Binary (historical) phantom:

`python scripts/validation_spacing.py --refine none --realizations 5` — commit fe4a1a8, 2026-10-08; head phantom 0.7 mm, 3 tiles, rng 0-4, slab-averaged ×1-4; render binary; refine none; fuse off; match Hungarian < 2 mm; wall 7 s

| slices (mm) | recall adaptive / fixed | seeds | 3D mean ± SD (mm) | P95 | max | z bias | z RMS | xy RMS | partition | tile centre err (mm) | normal err (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 1.00 / 1.00 | 60 | 0.20 ± 0.16 | 0.50 | 0.67 | 0.015 | 0.09 | 0.23 | 5/5 | 0.11 | 7.0 |
| 1.4 | 1.00 / 0.88 | 60 | 0.29 ± 0.20 | 0.57 | 1.31 | 0.013 | 0.18 | 0.30 | 4/5 | 0.15 | 7.0 |
| 2.1 | 0.97 / 0.17 | 58 | 0.50 ± 0.26 | 0.97 | 0.99 | 0.062 | 0.39 | 0.40 | 2/5 | 0.19 | 7.6 |
| 2.8 | 0.98 / 0.00 | 59 | 0.70 ± 0.36 | 1.25 | 1.81 | 0.069 | 0.62 | 0.48 | 2/5 | 0.35 | 6.5 |

Seed errors pooled over the matched seeds of all realizations (adaptive detection): mean ± SD, P95, max of the 3-D error; z = slice axis.  Tile errors: mean over the correctly partitioned scans; the truth normal is the radial direction from the cavity centre (generator definition), so ~7° is a floor, not a fit error.

Analytic phantom (exact capsules, printed contrast, clipped at 3071 after
slab averaging):

`python scripts/validation_spacing.py --refine none --realizations 5 --seed-render analytic` — commit fe4a1a8, 2026-10-08; head phantom 0.7 mm, 3 tiles, rng 0-4, slab-averaged ×1-4; render analytic (contrast 13500 HU, clip 3071); refine none; fuse off; match Hungarian < 2 mm; wall 8 s

| slices (mm) | recall adaptive / fixed | seeds | 3D mean ± SD (mm) | P95 | max | z bias | z RMS | xy RMS | partition | tile centre err (mm) | normal err (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 1.00 / 1.00 | 60 | 0.10 ± 0.08 | 0.25 | 0.44 | -0.006 | 0.06 | 0.12 | 5/5 | 0.05 | 7.0 |
| 1.4 | 1.00 / 0.73 | 60 | 0.34 ± 0.25 | 0.67 | 1.26 | -0.013 | 0.26 | 0.33 | 5/5 | 0.20 | 6.9 |
| 2.1 | 0.97 / 0.10 | 58 | 0.54 ± 0.35 | 1.10 | 1.15 | 0.058 | 0.49 | 0.41 | 3/5 | 0.24 | 8.1 |
| 2.8 | 0.92 / 0.00 | 55 | 0.74 ± 0.42 | 1.46 | 1.78 | 0.051 | 0.72 | 0.45 | 0/5 | – | – |

Seed errors pooled over the matched seeds of all realizations (adaptive detection): mean ± SD, P95, max of the 3-D error; z = slice axis.  Tile errors: mean over the correctly partitioned scans; the truth normal is the radial direction from the cavity centre (generator definition), so ~7° is a floor, not a fit error.

Against the stale `output/validation_spacing.csv` (2026-09-01 13:47,
adaptive mode, mean of per-scan means; same 5 rng): recall 1.00 / 1.00 /
1.00 / 1.00 → **1.00 / 1.00 / 0.97 / 0.98**; error 0.19 / 0.27 / 0.46 /
0.67 mm → 0.20 / 0.29 / 0.50 / 0.70 mm; max 0.68 / 0.76 / 1.05 / 1.64 mm →
0.67 / 1.31 / 0.99 / 1.81 mm; **partition 5/5 / 5/5 / 5/5 / 2/5 → 5/5 / 4/5 /
2/5 / 2/5** at 0.7 / 1.4 / 2.1 / 2.8 mm. The stale file was optimistic: the
tile partition already fails 1/5 at 1.4 mm and 3/5 at 2.1 mm (2.1 mm: rng 1
and 3 match only 11 of 12 seeds within 2 mm, rng 2 mis-partitions with all
12 matched; 1.4 mm: rng 1 mis-partitions with all 12 matched). Not bisected between the phantom
changes (tile re-draw on collision) and the detection / fitting changes since
2026-09-01. The plan's "partition fails 3 of 5 times at 2.8 mm" still holds;
its "2.8 mm partition 2/5 → ≥ 3/5" gate is unchanged, and 2.1 mm (2/5) now
needs the same attention.

The analytic phantom halves the thin-slice error (0.7 mm: 0.20 → 0.10 mm,
P95 0.50 → 0.25 mm): about half of the "0.2 mm on thin slices" is the binary
phantom's own voxel-painted capsule, not the localizer. From 1.4 mm on the
two phantoms agree within 0.05 mm in mean error; the analytic one loses more
seeds at 2.8 mm (recall 0.92, partition 0/5).

## Stage gates

| Stage | Gate | Result | Commit | Shipped as |
|---|---|---|---|---|
| 5 — hierarchical WLS + posterior seeds | identical fit without covariance; head phantom 2.1 / 2.8 mm mean 3D error ≥ 15 % below raw; tile centre / normal not worse; 2.8 mm partition ≥ 3/5; posterior NEES in [0.5, 2]; stable across slack 0.1–0.5 mm | **PENDING-COVARIANCE** (stand-in: analytic slab covariance). Identity: bit-identical (max diff 0.0 over 288 fits vs 3cf35af; frozen reference in the tests) ✓. 2.1 mm: −0.3 % ✗. 2.8 mm: −15.2 % (borderline ✓). Centre unchanged; normal 3.77→2.66° (2.1) and 6.44→2.85° (2.8) ✓. Partition 2.8 mm 5/5 (raw also 5/5); 2.1 mm 2/5, same as raw (split-fragment detections). NEES conditional 2.02 / 3.00 ✗, PEV 1.25 / 1.76 ✓ (input covariance itself 1.19 / 1.86). Slack sweep: 2.1 / 2.8 mm within 2.5 % ✓; 0.7 mm at slack 0.1 is 16 % worse than raw ✗ | b09cb39 | opt-in `reconstruct(fuse_tiles=True)`, default off |
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

## Stage 5 — hierarchical weighted least squares (stand-in covariance)

**Status: PENDING-COVARIANCE.** Measured with the analytic slab covariance
as a stand-in for the stage-2 grey-level covariance (not merged yet). The
coordinator re-runs the gate after stage 2 merges; if the 15 % gate fails
then, stage 5 is dropped and that is reported.

### What was built (commit b09cb39, branch `loc/hwls`)

- `deform.fit_deformable(..., seed_cov=None, slack_mm=SLACK_MM)`,
  `SLACK_MM = 0.3` mm. Seed i's 3 position residuals are multiplied by
  W_i = (C_i / λ_min(C_i))^(−1/2), C_i = Σ_i + slack²·I, eigenvalue ratio
  clamped to ≤ 4 (weights in [0.5, 1]). The best-determined direction keeps
  unit weight in mm, so the calibrated gates keep their meaning.
  `DeformableFit` gains `weighted_residuals_mm`, `mahalanobis_sq` and the
  properties `wrms_mm` (= `rms_mm` without covariance) and `chi_rms`
  (√(mean rᵢᵀCᵢ⁻¹rᵢ / 3), dimensionless).
- `auto.fit_tiles_auto(..., seed_cov=None)` and `fit_tiles_prior(...,
  seed_cov=None)` forward the per-candidate covariance to every bent-tile
  fit (main loop, loose tier, cover-pass quads, triplets — the inferred 4th
  seed gets its mates' mean covariance — and the counted prior's attached
  fits). The rms gates and `deformable_score` read `wrms_mm`. The
  chord / similarity prefilters stay unweighted.
- **Axis term on coarse scans (a finding, not in the plan).** With
  `seed_cov`, the bent-tile fits drop the seed-axis term when the slice
  spacing is above `AXES_RELIABLE_DZ_MM` = 1.2 mm. This is the rule the
  cover pass already applies. Without it the stage makes things worse; see
  below. The pre-fit axis-coherence gates are unchanged.
- New `gtcore/tiles/fuse.py::posterior_seed_positions(result, centers,
  cov, slack_mm=0.3, slack_tentative_mm=1.0, cov_mode="conditional")`.
  - The mean is ŝᵢ = xᵢ + Kᵢ(mᵢ(θ̂) − xᵢ) with Kᵢ = Σᵢ(Σᵢ + Σₛ)⁻¹. This is
    algebraically (Σᵢ⁻¹+Σₛ⁻¹)⁻¹(Σᵢ⁻¹xᵢ+Σₛ⁻¹mᵢ), written in a form that
    never inverts Σᵢ.
  - mᵢ comes from the correspondence stored on the fit, checked against the
    fit's own residuals. If that check fails, a minimum-distance
    permutation (≤ 24 candidates) is used.
  - Tentative and degraded tiles use slack 1.0 mm. Seeds that no tile owns
    pass through unchanged.
  - `cov_mode="conditional"` returns (Σᵢ⁻¹+Σₛ⁻¹)⁻¹.
  - `cov_mode="pev"` returns the linearised prediction-error covariance,
    which also carries the uncertainty of θ̂ refitted from the same seeds:
    ŝ − s = (I − K + KGL)ε − K(I − GL)δ, where G is the model Jacobian and L
    the Gauss–Newton sensitivity of the fit. GL·dx agrees with
    finite-difference refits within the solver tolerance (~1e-3 mm on
    0.02 mm perturbations).
  - `slab_covariance_stand_in(affine, n)` = (1/12 + 0.1²)·M Mᵀ, with
    M = affine[:3, :3].
- `pipeline.reconstruct(..., fuse_tiles=False)`. With the flag on, tiles
  and `seeds.cov_ras`, the tiles are fitted with `seed_cov` (counted mode
  goes through `fit_tiles_prior(cover=False)`), then the posterior is
  computed.
  - `vol.meta["seed_posterior"]` and `PipelineResult.meta["seed_posterior"]`
    hold the raw centres and covariances, the posterior (conditional and
    PEV covariance) and the per-tile summary.
  - **The switch:** only with the flag on does `PipelineResult.seeds`, the
    dose / planner / `to_placed_tiles` feed, carry the posterior. The raw
    detections stay in `PipelineResult.meta["seeds_unfused"]`.
  - Without `cov_ras` nothing is fused and the reason is recorded.
  - `PipelineResult` gains a trailing `meta` field; the field order is
    unchanged.

### Commands, seeds, commit

```
# synthetic head phantom, spacing 0.7 mm, 3 tiles, rng_seed 0-4,
# block-averaged along z to 0.7 / 1.4 / 2.1 / 2.8 mm (factors 1-4)
PYTHONPATH=. python scripts/validation_fuse.py --mc --csv <out.csv>
PYTHONPATH=. python -m pytest -q tests/test_tiles_fuse.py -s   # 13 tests, ~17 s
```

Commit b09cb39 (branch `loc/hwls`, base 3cf35af). Detection uses
`seed_detection_params` + `detect_seed_candidates` + `filter_seed_shaped`.
Truth matching is Hungarian, 2 mm gate. Errors are means over all matched
detections. Partition is correct when every truth tile is recovered by
exactly one fitted tile (supported + tentative) with no mixed or
unmatched member. The tile normal is compared with the plane normal of the
truth seeds. NEES = mean eᵀΣ⁻¹e / 3 over fused seeds.

### Results (5 realizations per row, slack 0.3 mm)

| Slice | 3D raw → post | z raw → post | Partition raw / w | Tile centre raw / post | Tile normal raw / w (deg) | NEES input / cond / PEV | Auto fit raw / w (s) | Posterior | Redundant share |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 0.200 → 0.185 (−7.3 %) | 0.066 → 0.069 (+4 %) | 5/5 / 5/5 | 0.106 / 0.106 | 1.59 / 1.59 | 0.47 / 0.58 / 0.43 | 0.15 / 0.15 | 54 µs/seed | 0.26 |
| 1.4 | 0.291 → 0.281 (−3.4 %) | 0.159 → 0.144 (−9.9 %) | 5/5 / 5/5 | 0.158 / 0.158 | 2.08 / 1.40 | 0.73 / 1.04 / 0.73 | 0.23 / 0.06 | 50 µs/seed | 0.30 |
| 2.1 | 0.519 → 0.518 (−0.3 %) | 0.336 → 0.333 (−0.8 %) | 2/5 / 2/5 | 0.246 / 0.246 | 3.77 / 2.66 | 1.19 / 2.02 / 1.25 | 0.26 / 0.11 | 63 µs/seed | 0.22 |
| 2.8 | 0.702 → 0.596 (−15.2 %) | 0.554 → 0.422 (−23.8 %) | 5/5 / 5/5 | 0.311 / 0.311 | 6.44 / 2.85 | 1.86 / 3.00 / 1.76 | 0.24 / 0.08 | 63 µs/seed | 0.36 |

How to read the columns:
- "Redundant share" is the share of the raw error energy that lies outside
  the tangent space of the bent-tile model at the true seeds. That is the
  only part any tile fit can see (3 of 12 dimensions, 0.25 for i.i.d.
  errors).
- The posterior costs 0.5–1 ms per 12-seed scan, or 2–4 ms with the PEV
  covariance.
- The weighted auto fit is faster on coarse scans (no axis term).

Per realization, mean 3D error raw → posterior (mm):

| Slice | rng 0 | rng 1 | rng 2 | rng 3 | rng 4 |
|---|---|---|---|---|---|
| 0.7 | 0.218 → 0.198 | 0.171 → 0.158 | 0.184 → 0.174 | 0.233 → 0.200 | 0.192 → 0.194 |
| 1.4 | 0.302 → 0.279 | 0.306 → 0.308 | 0.239 → 0.248 | 0.309 → 0.283 | 0.301 → 0.288 |
| 2.1 | 0.484 → 0.471 | 0.563 → 0.561 | 0.484 → 0.493 | 0.587 → 0.600 | 0.479 → 0.464 |
| 2.8 | 0.848 → 0.711 | 0.644 → 0.495 | 0.698 → 0.691 | 0.739 → 0.575 | 0.583 → 0.507 |

Reading:
- **Little or no gain below 2.8 mm.**
  - 0.7 mm: −7 % in 3D, but 0.015 mm in absolute terms.
  - 1.4 mm: −3 %.
  - 2.1 mm: none. The 2.1 mm z errors are mostly common to the four seeds of
    a tile (redundant share 0.22), and the tile pose absorbs them.
  - 2.8 mm: −15 % in 3D and −24 % in z, just at the gate.
- **Tile centre:** unchanged by construction. With equal covariances the
  posterior shifts of a tile sum to zero, because the translation is free.
- **Tile normal:** clearly better on coarse scans (6.4° → 2.9° at 2.8 mm).
  That comes from leaving out the degenerate axes, not from the position
  weighting.
- **Partition:** unchanged in every scan. The 3 failures at 2.1 mm are
  split-blob fragments (a truth seed detected twice, one copy outside the
  2 mm gate), which is stage 3's problem. The plan's "2.8 mm partition
  2/5" came from the counted-mode spacing study. The auto mode measured
  here already partitions 5/5 at 2.8 mm.
- **NEES:** the conditional covariance is over-confident on thick slices
  (2.0, 3.0). The PEV covariance tracks the input covariance (1.25 vs 1.19,
  1.76 vs 1.86): the posterior is as calibrated as Σᵢ. The stand-in Σᵢ is
  itself off: too large at 0.7 mm (0.47) and too small at 2.8 mm (1.86).
  The NEES gate is therefore really a gate on the stage-2 covariance plus
  `cov_mode="pev"`.

**Without the axis rule the stage is harmful.** This was measured before
the rule was added, on rng 0–1 only. Mean 3D error:
- 2.8 mm: 0.746 → 1.117 mm (+50 %);
- 2.1 mm: 0.524 → 0.686 mm (+31 %);
- tile normal error at 2.8 mm: 10.3° → 14.9°.

The bent-tile fit on the true grouping shows why. Mean model-point error
mᵢ(θ̂) vs truth at 2.8 mm (rng 0–4, the 14 tiles with all 4 seeds
matched; raw seeds 0.70 mm):

| Fit | Model-point error at 2.8 mm (mm) |
|---|---|
| With axes, unweighted | 0.85 |
| With axes, weighted | 1.16 |
| Without axes, unweighted | 0.61 |
| Without axes, weighted | 0.58 |

A capsule inside one 2.8 mm slab has a PCA axis that is off by 20–40°.
Down-weighting z makes the 4 mm/rad axis term relatively stronger, and it
tilts the tile.

### Slack sensitivity (fit and posterior with the same slack; mean 3D / |z| error, mm; posterior NEES conditional)

| Slice | 0.1 | 0.2 | 0.3 (default) | 0.5 |
|---|---|---|---|---|
| 0.7 | 0.233 / 0.107, 2.86 | 0.197 / 0.077, 0.86 | 0.185 / 0.069, 0.58 | 0.187 / 0.065, 0.49 |
| 1.4 | 0.293 / 0.149, 4.16 | 0.284 / 0.145, 1.49 | 0.281 / 0.144, 1.04 | 0.282 / 0.146, 0.83 |
| 2.1 | 0.521 / 0.334, 8.86 | 0.519 / 0.334, 3.07 | 0.518 / 0.333, 2.02 | 0.518 / 0.333, 1.49 |
| 2.8 | 0.589 / 0.416, 13.35 | 0.590 / 0.418, 4.56 | 0.596 / 0.422, 3.00 | 0.610 / 0.436, 2.25 |

Partition is identical at every slack (5/5, 5/5, 2/5, 5/5).
- At 2.1 and 2.8 mm the error is flat across 0.1–0.5 mm (within 2.5 %).
- At 0.7 mm, slack 0.1 is 16 % *worse* than raw (0.233 vs 0.200). The
  phantom's own seeds deviate from the bent-tile model by 0.07–0.26 mm rms
  per tile (mean 0.15; truth seeds fitted with no noise). A slack below
  that over-trusts the model.
- That is ≈ 0.17 mm per axis (rms·2/√3 for 3 residual DOF), so 0.3 mm
  sits about 1.7× above it. Do not report a slack below 0.2 mm.

### The 3-redundant-DOF caveat, measured

A full tile gives 12 coordinates against 9 parameters, so only 3
error modes are visible to any tile fit:
- the twist out of the tile plane (1);
- two chord combinations in the plane (2).

Common-mode and rigid-like errors (shift, tilt) are absorbed into the
pose. Two measurements put numbers on the limit.

**Model-only Monte-Carlo** (`validation_fuse.py --mc`):
- 400 bent tiles per cell, random orientation;
- true seeds = model + N(0, 0.3² I);
- detections = true + N(0, Σ_slab);
- weighted fit, no axes, then the posterior;
- result is the best case, with no detection effects.

| Slice | κ = 0 (flat) | κ = 0.03 | κ = 0.07 | κ = 0.15 /mm |
|---|---|---|---|---|
| 0.7 | −3.5 % (z −3.0 %) | −2.8 % | −3.8 % | −4.0 % |
| 1.4 | −6.3 % (z −8.3 %) | −5.7 % | −6.4 % | −5.6 % |
| 2.1 | −9.7 % (z −12.2 %) | −9.1 % | −9.3 % | −7.5 % (z −9.1 %) |
| 2.8 | −12.7 % (z −15.1 %) | −12.0 % | −11.4 % | −9.5 % (z −11.1 %) |

Reading:
- Even with an exact model and an exact covariance, the ceiling is about
  10–13 % in 3D at 2.1–2.8 mm.
- Near-flat tiles gain slightly more than strongly curved ones:
  - 2.8 mm: 12.7 % flat vs 9.5 % at κ = 0.15;
  - 2.1 mm: 9.7 % vs 7.5 %.
- The difference is small. The curvature prior (1.5 mm per 1/mm, i.e.
  0.1 mm of residual at κ = 0.07) carries little information.
- Every head-phantom tile is curved (κ 0.075–0.088 /mm), so the phantom
  has no near-flat tiles to compare against.
- The 2.8 mm phantom gain (15 %) is above the Monte-Carlo figure. Two
  likely reasons:
  - its errors put more energy into the redundant modes (share 0.36, vs
    0.25 for isotropic i.i.d. errors);
  - its seeds deviate from the model less (≈ 0.17 mm per axis) than the
    Monte-Carlo's 0.3 mm.
- At 2.1 mm the share is lower (0.22), hence no gain.
- Conclusion for the paper:
  - the stage-5 gain is real but bounded, about 10–15 % on 2.8 mm slices;
  - it is near zero where the slab error is common to all four seeds of a
    tile;
  - the Monte-Carlo figure at 2.1 mm is about 10 %, so a 15 % gate there
    is unlikely to pass unless stage 2 changes the error structure.

## Open decisions

- Stage 5 posterior covariance:
  - Option: make `cov_mode="pev"` the default. The conditional
    (Σᵢ⁻¹+Σₛ⁻¹)⁻¹ is over-confident on thick slices (NEES 2–3) because it
    ignores the uncertainty of θ̂. The posterior mean is the same in both
    modes.
  - The pipeline already stores both (`cov_ras`, `cov_ras_pev`).
- Stage 5 axis rule:
  - With `seed_cov`, fits above 1.2 mm drop the axis term. The unweighted
    supported path still fits on degenerate axes on coarse scans (tile
    normal error 6.4° at 2.8 mm vs 2.9° without).
  - Changing that is outside stage 5: it alters the calibrated selection
    and needs its own gate.
- Stage 5 weight normalisation:
  - The normalisation is per seed (λ_min(Cᵢ)), which drops relative
    precision between seeds. That is immaterial with the stand-in, where
    all seeds share one Σ.
  - Re-check once stage-2 covariances differ between seeds. The alternative
    is one normaliser per tile, which keeps relative precision but shifts
    the mm gates.
- **Margin per configuration vs per swapped tile.** The 25/31 alternative changes TWO tiles, so its 1.87 is about 0.93 per tile, or 0.47 mm of bent-tile RMS each. That is below the 1.0 line if the threshold were read per changed tile. The plan defines the margin as the plain best-vs-second-best total, and that is what shipped. Normalising by the number of swapped tiles is a one-line change. Decide it on a case with truth, not on this phantom.
- **Pooled s².** The Gauss–Newton s² mixes position, axis and prior rows. See the stage 6 calibration table. Revisit once stage 5's whitened residuals land, so that s² carries per-seed Σᵢ.
| 0 | harness + fresh baseline (no gate) | renderer tests pass; baselines above | fe4a1a8 | `seed_render`, scripts, `meta["interpolated_k"]` |

## Real-data proxies (no truth)

`scripts/validation_realdata_proxies.py`, no refinement, no fusion, fresh
pipeline runs (`--no-cache`); `reconstruct` does not yet accept
`refine_seeds` / `fuse_tiles` (checked by signature; the script then applies
them after the pipeline when asked). Near-flat = max |κ| ≤ 0.04 /mm of the
bent-tile fit: four seeds on a cylinder are coplanar, so flatness is not
observable from positions alone, and the fitted κ is inflated by noisy PCA
axes (no tile of this phantom has max |κ| ≤ 0.015); the chord residual
against the fitted bent tile is the curvature-free variant. Split halves are
decimations (even / odd 1 mm slices → 1 mm slabs at 2 mm pitch), so their
errors are independent and disagreement/√2 is the single-scan precision; at
2 mm pitch a seed inside one slab snaps to that slab's centre, hence the
~1 mm median |Δk|.

`python scripts/validation_realdata_proxies.py --no-cache` — commit fe4a1a8, 2026-10-08; refine none, fuse off; wall 101 s

**Printed 8-tile phantom (1 mm).** 32 seeds (32 raw blobs); 8 supported + 0 tentative tiles, 0 unassigned; evidence supports n=8 (no further tile candidate). Real seed peaks: 31/32 at 3071 HU (min 2862). Pipeline 25 s.

| tile | kind / confidence | seeds | bent-tile RMS (mm) | κ1 / κ2 (1/mm) | fold (°) | sides (mm) | axis spread (°) |
|---|---|---|---|---|---|---|---|
| 0 | full / supported | 0,1,2,5 | 0.26 | 0.039 / 0.127 | 145 | 7.13 7.20 9.04 9.18 | 15.5 |
| 1 | full / supported | 10,11,17,18 | 0.31 | 0.007 / 0.050 | 58 | 8.30 9.13 9.15 9.49 | 4.7 |
| 2 | full / supported | 7,13,15,20 | 0.40 | 0.037 / 0.016 | 43 | 8.75 9.22 9.33 9.70 | 5.1 |
| 3 | full / supported | 3,4,6,9 | 0.50 | 0.033 / 0.011 | 37 | 8.50 9.06 9.20 10.17 | 7.8 |
| 4 | full / supported | 14,22,23,28 | 0.56 | 0.085 / 0.073 | 97 | 7.69 7.78 7.88 9.07 | 23.0 |
| 5 | full / supported | 8,16,24,25 | 0.78 | -0.040 / 0.102 | 117 | 7.40 9.66 11.08 11.65 | 30.3 |
| 6 | full / supported | 26,29,30,31 | 0.87 | 0.046 / -0.020 | 52 | 9.10 9.13 9.18 11.20 | 14.9 |
| 7 | full / supported, degraded | 12,19,21,27 | 0.46 | 0.053 / 0.251 | 288 | 4.80 5.91 7.92 8.76 | 14.5 |

- bent-tile RMS: mean 0.52 mm, max 0.87 mm
- side chords, near-flat tiles (max |κ| ≤ 0.040 /mm, n = 2): 9.24 ± 0.52 mm vs nominal 10 mm; all tiles: 8.77 ± 1.40 mm
- side chord − bent-tile model chord (all tiles): 0.070 ± 0.625 mm
- within-tile seed-axis coherence: mean |cos| 0.947, mean spread 14.5°
- close pair: seeds [25, 31] at 3.93 mm (25/31 at 3.93 mm); auto partition {25: [8, 16, 24, 25], 31: [26, 29, 30, 31]} vs counted {25: [25, 26, 29, 30], 31: [8, 16, 24, 31]} -> INCONSISTENT

**Split-half repeatability** (1 mm phantom decimated to two 2.0 mm volumes; reference = the 32 full-resolution seeds; detection threshold of the 2 mm branch).

| half | detected | matched to 1 mm | FP | vs 1 mm: bias i / j / k (mm) | RMS i / j / k (mm) | 3D mean ± SD | P95 | max |
|---|---|---|---|---|---|---|---|---|
| even | 37 | 31 | 6 | 0.117 / -0.047 / 0.053 | 0.74 / 0.27 / 0.53 | 0.75 ± 0.58 | 1.78 | 1.94 |
| odd | 30 | 29 | 1 | -0.010 / 0.059 / -0.003 | 0.48 / 0.27 / 0.41 | 0.49 ± 0.48 | 1.34 | 1.78 |

Odd − even, 28 seeds found in both halves:

| axis | Bland–Altman bias (mm) | SD | 95 % LoA (mm) | disagreement/√2 (mm) | median abs diff (mm) |
|---|---|---|---|---|---|
| i | -0.140 | 1.049 | -2.20 to 1.92 | 0.735 | 1.109 |
| j | 0.094 | 0.467 | -0.82 to 1.01 | 0.331 | 0.261 |
| k | -0.062 | 0.827 | -1.68 to 1.56 | 0.576 | 1.000 |

3-D |odd − even|: mean 1.24 mm, P95 1.91 mm; 3-D disagreement/√2 = 0.99 mm.

**PostOp CT** (0.52x0.52x2.00 grid, 52 of 89 slices interpolated). 731 raw blobs -> 53 in-vault candidates; 4 supported + 2 tentative tiles, 5 unassigned in-implant seeds (NN distances 3.4, 10.0, 7.2, 3.0, 3.4 mm); fit_tiles_prior(ImplantPrior()) agrees: True; implant confirmed. Peak HU (±1 mm) of the 22 tile-assigned seeds: median 2936, 4 at 3071; 1 of them centred on an interpolated slice. Pipeline 15 s.

**Negative controls.**

| scan | loaded | grid (mm) | candidates | verdict | reason |
|---|---|---|---|---|---|
| CT 3D printed | True | 0.68x0.68x1.00 (248 slices) | 0 | absent | only 0 candidates (a tile needs 4 seeds) |
| DOEJOHNPOSTCT | True | 0.52x0.52x1.00 (204 slices) | 31 | confirmed | 6 gate-passing tile quads, 6 grouped within one cavity-sized region |

Reading against the plan's real-data criteria at baseline: printed phantom
32/32 seeds, 8/8 tiles (holds); 25/31 partitioned differently by the auto and
the counted fit (the documented ambiguity; stage 6's margin should flag it);
PostOp 4 supported + 2 tentative, 5 unassigned of which 3 are 3.0–3.4 mm
from a neighbour (split duplicates, the stage 3 target); tile-free printed
scan: 0 candidates, absent. **The pre-implant DOE scan is already
"confirmed" at baseline** (6 tile-like quads from calcifications, the known
limitation in `docs/data-notes.md`), so the plan's "negative controls stay
not confirmed" cannot be a no-regression check for that scan: the criterion
can only be "verdict and quad count do not get worse" (see Open decisions).

## With-truth track

Hook ready, no truth yet: `python scripts/validation_realdata_proxies.py
--truth <csv>` with columns `x,y,z[,tile_id]` (mm, any rigid frame, header
row, one row per physical seed). Registration: rigid ICP with a Hungarian
assignment per iteration from 24 principal-axis + 300 random rotation
starts, trimmed (closest 80 %) Kabsch; reports per-seed error after one
global rigid fit and after per-tile rigid fits (truth `tile_id` groups, else
the detected partition). Checked: a rotated copy of the 32 detected printed
seeds with 0.3 mm/axis noise comes back at 0.48 ± 0.20 mm (3-D) with zero
bias; `tests/test_validation_realdata_truth.py` covers misses and false
positives.

## Open decisions

1. **Contrast target.** Stage 0 calibrated `DEFAULT_METAL_HU` to a 1700 HU
   median at 0.5 × 0.5 × 2.0 mm as instructed, but the PostOp is a 1 mm
   series whose measured seed peaks (median 2936 HU, 18 % saturated) imply
   10 900 HU, i.e. a 2200 HU median for a true 2 mm acquisition. The harness
   at 8380 HU is pessimistic for detection on G2–G4; switching is one
   constant (`METAL_HU_POSTOP_MEASURED`).
2. **G3 definition.** Implemented as PostOp-like (1 mm slabs, irregular
   gaps, loader grid 2.0 mm) instead of the literal 1 mm grid with every
   other slice interpolated (G3L), which the 2000 HU thin-slice branch cannot
   detect at PostOp contrast.
3. **DOE negative control** is "confirmed" before any change; the
   verification criterion for it needs rewording (no new quads / verdict not
   worse), or the assessment fixed separately.
4. **Near-flat threshold** (κ ≤ 0.04 /mm admits 2 of 8 printed tiles); the
   chord residual against the bent-tile model is the better proxy for the
   stage 2/5 comparisons.

## Runs log

| Date | Command | Seeds | Commit | Wall time |
|---|---|---|---|---|
| 2026-10-08 | `python scripts/validation_fuse.py --mc --csv ...` (stage 5, stand-in covariance) | phantom rng 0-4; MC rng 0 | b09cb39 | 166 s |
| 2026-10-08 | `pytest tests/test_tiles_fuse.py` (13 tests) + `tests/test_tiles*.py tests/test_localization_plumbing.py` | fixed | b09cb39 | 17 s; 151 passed in 147 s |

### Coordinator decision on stage 5 (2026-10-08, after loc/hwls merged at 04b4d20)

Gate ("head phantom 2.1/2.8 mm mean 3D error ≥ 15 % better than raw seeds"):
**not met** at 2.1 mm (−0.3 %), **just met** at 2.8 mm (−15.2 %); z error −1 %
and −24 %. The agent's model-only Monte-Carlo ceiling (exact bent-tile model,
exact covariance) is 9–10 % at 2.1 mm and 10–13 % at 2.8 mm, so the plan's
15 % gate sat above what the method can deliver on these cavities: 78 % of
the thick-slice error is a shift common to all four seeds of a tile, which
the tile pose absorbs and no tile-based estimator can see. This is the
3-redundant-DOF caveat from the plan, now measured.

Decision: stage 5 ships **opt-in only** (`reconstruct(fuse_tiles=False)`,
`fit_deformable(seed_cov=None)` default, bit-identical to before). It is
re-evaluated once stage 2 supplies real per-seed covariances; if the 2.1 mm
gain stays below the ceiling-adjusted bar of 8 %, the paper reports it as a
negative result with the ceiling analysis. The conditional posterior
covariance is over-confident (NEES 2–3 on thick slices); the pose-error-
propagated covariance (`cov_mode="pev"`) is calibrated and will be the
default if the stage ever ships.

Separate gated item opened (owner: coordinator, after stage 4/6 merge): on
scans thicker than 1.2 mm the DEFAULT unweighted bent-tile fit still uses the
degenerate PCA seed axes; dropping the axis term there (as the cover pass
already does) cut tile normal error from 6.4° to 2.9° at 2.8 mm in the
weighted path. Changing the default alters the calibrated selection, so it
gets its own before/after gate on `tests/test_tiles_auto.py` cases and the
real scans.
| 2026-10-08 | `python scripts/validation_loc_scoring.py --extract` | printed 8-tile phantom (32 seeds) | d81b45d | 25 s |
| 2026-10-08 | `python scripts/validation_loc_scoring.py` | cached printed-phantom seeds | d81b45d | 11 s |
| 2026-10-08 | `python scripts/validation_loc_scoring.py --calibration` | synthetic bent tile, rng 1, 200 draws × 8 rows | d81b45d | 60 s |
| 2026-10-08 | `python -m pytest tests/test_tiles*.py tests/test_tile_model.py tests/test_implant_assessment.py tests/test_localization_plumbing.py tests/test_localization_scoring.py` | phantom rng per test | d81b45d | 157 s, 254 passed (215 before + 39 new) |
| 2026-10-08 | `python scripts/validation_seed_localization.py --methods baseline` | layout rng 0–4 × 40 × sparse/crowded × G1–G4 | fe4a1a8 | 17 s |
| 2026-10-08 | `python scripts/validation_seed_localization.py --calibrate` | 400 seeds per grid × contrast, rng 0 | fe4a1a8 | ~40 s |
| 2026-10-08 | `python scripts/validation_spacing.py --refine none --realizations 5` | phantom rng 0–4 | fe4a1a8 | 7 s |
| 2026-10-08 | `python scripts/validation_spacing.py --refine none --realizations 5 --seed-render analytic` | phantom rng 0–4 | fe4a1a8 | 8 s |
| 2026-10-08 | `python scripts/validation_realdata_proxies.py --no-cache` | real scans (printed8, PostOp, CT 3D printed, DOE) | fe4a1a8 | 101 s |
