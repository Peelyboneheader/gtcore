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
| 2026-10-08 | `python scripts/validation_seed_localization.py --methods baseline` | layout rng 0–4 × 40 × sparse/crowded × G1–G4 | fe4a1a8 | 17 s |
| 2026-10-08 | `python scripts/validation_seed_localization.py --calibrate` | 400 seeds per grid × contrast, rng 0 | fe4a1a8 | ~40 s |
| 2026-10-08 | `python scripts/validation_spacing.py --refine none --realizations 5` | phantom rng 0–4 | fe4a1a8 | 7 s |
| 2026-10-08 | `python scripts/validation_spacing.py --refine none --realizations 5 --seed-render analytic` | phantom rng 0–4 | fe4a1a8 | 8 s |
| 2026-10-08 | `python scripts/validation_realdata_proxies.py --no-cache` | real scans (printed8, PostOp, CT 3D printed, DOE) | fe4a1a8 | 101 s |
