# Independent audit of the clinical case against the TPS files (2026-10-08)

A second, independent pass over `DOEJOHNPOSTCT` and its MIM structure set
and dose grid, done without `gtcore`'s own readers (plain pydicom +
scikit-image rasterisation, trilinear dose sampling) so that the numbers in
`docs/clinical-validation-notes.md` are checked by code that shares nothing
with them. Where the two disagree it is said so. Scripts live outside the
repo (session scratchpad); every number below is reproducible from the
three DICOM inputs alone.

Inputs (all one frame of reference, `2.16.840.1.114362...742.381`):

| File | What it is |
|---|---|
| `DOEJOHNPOSTCT/` (204 slices) | post-implant head CT, Siemens Force, Hr40, 0.52 x 0.52 x 1.0 mm |
| `..._RTst_..._GESTA` | MIM structure set "GESTALT Trial: Final Saved Session - Resliced contours", 24 ROIs, every contour on a CT slice (max z offset 0.00 mm) |
| `..._RTDOSE_..._Pre-Op.Dose` | MIM plan dose, 1 x 1 x 1.667 mm, Gy, max 813 Gy. **The label is misleading: this is the implant dose** (median 651 Gy at the seed positions) |

## 1. Which D90 is which

Both figures that circulated are real; they belong to different structures.
Trilinear RTDOSE at CT voxel centres, rx 60 Gy:

| Structure | Volume | D90 | D100 | V100 | V150 |
|---|---|---|---|---|---|
| `CTV_GT` (the clinical target) | 19.0 cc | **63.65 Gy** | 39.2 | 93.7 % | 48.2 % |
| `CTV_Post` | 23.4 cc | 50.7 Gy | 29.3 | 81.8 % | 39.3 % |
| full 5 mm rind of `Cav_Post` | 45.0 cc | 45.1 Gy | 20.2 | 66.9 % | 24.3 % |
| full 5 mm rind, clipped to `Brain` | 35.4 cc | 46.5 Gy | | 67.8 % | |
| `Cav_Post` | 35.1 cc | 93.6 Gy | 30.5 | 99.1 % | 91.8 % |
| `Res_Post` | 0.8 cc | 59.0 Gy | 46.6 | 87.6 % | |

- "63.0 Gy" is the same CTV_GT number with nearest-neighbour dose sampling.
- `CTV_GT` is 0 % inside `Cav_Post`, every voxel within 5.2 mm of it, and
  covers only 42 % of the full 5 mm rind (47 % for `CTV_Post`, which reaches
  10.6 mm). `CTV_GT` lies 100 % inside `CTV_Post`. The clinical target is a
  hand-trimmed partial rind, so an automatic cavity + 5 mm HR-CTV has to be
  compared with the full-rind row (45 Gy), with `CTV_GT` as the upper bound.
  This agrees with the hrctv notes.

## 2. Seed count: the plan has 32 sources, the contour has 30

- `Seeds` ROI: 92 contours, 30 connected components in 3D (26-connectivity),
  8-15 mm^3 each, nearest-neighbour spacing 4.7 / 7.7 / 10.4 mm
  (min / median / max). Thirty is the honest count of the contour.
- The RTDOSE has **32 local maxima above 200 Gy** (merged at 3 mm). Thirty
  sit within 1.5 mm of a contoured seed; **two have no contour** (8.3 and
  8.5 mm from the nearest one). Both carry 640-670 Gy, the same as the
  contoured seeds, so the TPS modelled 32 sources = 8 full tiles. "7.5
  tiles" was inferred from the incomplete contour and is wrong.
- The two uncontoured sources are 0.8 and 0.9 mm from gtcore detections 6
  and 14. On the CT both are seed-shaped (6.5 and 4.9 mm^3 above 2000 HU,
  peak 3071 / 2758 HU, elongated like the matched seeds) and both are
  members of fitted tiles. Detection 6 lies outside the physician's `Brain`
  contour, which is presumably why it was not contoured.

gtcore (feature/hrctv at 99724ef, `reconstruct(n_full_tiles="auto")`), seeds
matched to the 30 contour centroids by Hungarian assignment:

| | |
|---|---|
| detections / contoured seeds | 31 / 30 |
| matched within 2 mm | 29 / 30; error mean 0.31, median 0.23, p95 0.88, max 0.89 mm |
| extra detections | 2 = the two uncontoured sources above |
| contoured seed missed | 1 (contour seed 4): found as a raw blob 0.15 mm away (4.6 mm^3, elongation 3.2, passes the shape filter) and **dropped by the vault filter**: it lies 2.1 mm outside `cranial_interior` under the craniotomy, and the filter dilates the interior by 2 voxels (about 1 mm). Contoured seeds sit -2.1 to +2.1 mm (median +2.1) from that mask surface, so the margin is marginal on this case. Fix: dilate by a few mm, or test against the body/skull instead. |

Tile inference with all 32 seeds (31 + the filtered one): auto mode and the
count-constrained `fit_tiles(n_full=8)` both return the same **6 tiles**;
the 8 leftover seeds (3, 5, 9, 10, 13, 15, 23, 31) contain no 4-subset with
square-like chords: one pair is 4.7 mm apart, another four form an 8-9 mm
chain. Crumpled or stacked tiles outside the bent-sheet model: 6 of 8 is the
result to report, with that reason.

## 3. Dose engine: no seed-strength discrepancy

Every GammaTile seed is 3.5 U (Jacob, 2026-10-08). TG-43 (`dose_at_points`,
TG-43U1S2 consensus data) against the RTDOSE at `CTV_GT` voxels more than
5 mm from any seed:

| seed set | ratio TG-43 / RTDOSE, median [p5, p95] | `CTV_GT` D90 | V100 |
|---|---|---|---|
| 30 contoured seeds | 0.932 [0.773, 1.030] | 5789 cGy | 87.3 % |
| 31 gtcore detections (shipped) | 0.969 [0.877, 1.044] | 6080 | 90.9 % |
| **32 seeds (31 + the vault-filtered one)** | **0.989 [0.902, 1.066]** | **6192** | 91.9 % |
| RTDOSE | 1 | 6365 | 93.7 % |

The 7 % deficit seen with 30 seeds is the two missing sources, not seed
strength. With the full source list the engine agrees with the TPS to 1 %
in the field and 2.7 % in `CTV_GT` D90; point by point the residual is
median 0.0 %, IQR -3.3 to +3.5 %, with 69 % of voxels within 5 %. The
"3.76 U" row in the clinical notes should be read as the size of that
artefact, nothing else.

## 4. Cavity volume: where the 44 cc comes from

gtcore cavity (seed-sheet rule, reach 14) on this case: **43.9 cc, Dice
0.63** against `Cav_Post` (35.1 cc); covers 71 % of it; 57 % of gtcore's
volume lies inside it. Two separate failures:

- **19.0 cc leaks into brain.** 99 % of the gtcore-only volume is inside
  the `Brain` ROI, HU p10/p50/p90 = 6 / 27 / 51, and 18.2 cc of it lies more
  than 3 mm beyond the seed hull. The rule grows through voxels below
  `FLUID_HI_HU` = 26 with a 14 mm reach; peri-cavity oedema sits at 15-30
  HU, so the growth walks into oedema.
- **10.2 cc of `Cav_Post` is not covered.** Only 31 % of it is above 26 HU
  (clot or debris); 68 % is fluid or air the rule could have reached but
  clipped: contoured seeds lie 1.5-7.5 mm (median 3.8) *inside* the
  physician's wall, and the sheet clip is 3 mm. 61 % of the uncovered volume
  is more than 3 mm from the seed hull.

For scale, the contents of `Cav_Post` are HU p10/p50/p90 = -355 / -8 / 61:
12 % air, 24 % above 26 HU.

Geometry alone does better on this case. The convex hull of the 30
contoured seeds is 11.3 cc and 97 % inside `Cav_Post`:

| rule (no HU test) | volume | Dice | covers |
|---|---|---|---|
| hull dilated 3 mm | 20.9 cc | 0.69 | 55 % |
| hull dilated 5 mm | 30.0 cc | 0.79 | 74 % |
| hull dilated 7 mm | 41.4 cc | 0.82 | 89 % |
| (hull + 10 mm) & HU < 26 & Brain | 32.2 cc | 0.78 | 74 % |
| shipped sheet rule | 43.9 cc | 0.63 | 71 % |

Section 5 tests a combined rule on this case and on the synthetic phantom.

## 5. Candidate cavity rule, tested on both data sets

Prototype (scratch code, not in `gtcore`): **core** = convex hull of the
seeds dilated by `margin` with no HU test (so clot and debris inside the
implant count as cavity); **growth** = voxels below `thr` HU geodesically
within `reach` of the core; everything clipped to `tol` beyond the seed sheet
(`_sheet_normals`) and to the cranial interior; 3 mm closing; holes filled.
Seeds are given (the 32 clinical positions; truth seeds on the phantom) so
only the cavity rule is tested. Truth: `Cav_Post`; `truth.masks["cavity"]`
on `make_head_phantom` at (1.0 mm, 3 tiles, rng 0), (1.0 mm, 5 tiles, rng 1),
(0.7 mm, 3 tiles, rng 2). 36 parameter sets per case; full tables and the two inspection figures are
in `docs/figures/clinical-audit/` (`cavity_proto.csv`, `cavity_adaptive.csv`,
`cavity_overlay.png`, `seed_inspection.png`).

| rule | clinical Dice / covers / cc | phantom Dice (3 cases) |
|---|---|---|
| shipped sheet rule (reach 14) | 0.66 / 72 % / 40.6 | 0.90, 0.84, 0.94 |
| core 3 mm, tol 5, thr 20, reach 10 | **0.80** / 82 % / 36.5 | 0.83, 0.79, 0.90 |
| core 3 mm, tol 7, thr 20, reach 10 | 0.81 / 84 % / 38.5 | 0.83, 0.79, 0.90 |
| core 5 mm, tol 5, thr 20, reach 10 | 0.77 / 83 % / 40.0 | 0.85, 0.84, 0.89 |
| core 3 mm, tol 5, thr 26, reach 14 | 0.71 / 83 % / 46.7 | 0.88, 0.89, 0.92 |
| core 3 mm, tol 5, no growth | 0.69 / 57 % / 22.1 | 0.49, 0.39, 0.69 |
| core 7 mm, tol 5, no growth | 0.77 / 78 % / 35.9 | 0.66, 0.58, 0.77 |

What the sweep says:

- **The leak is the HU threshold, not the reach.** At thr 26 the clinical
  Dice never exceeds 0.71 for any core / tolerance / reach; at thr 20 it
  reaches 0.80. Peri-cavity oedema on this scan sits at 15-30 HU (the
  gtcore-only volume has HU p10/p50/p90 = 6/27/51); a 26 HU fluid class
  contains it.
- **An adaptive threshold does not help.** Midway between the cavity core
  (median 11 HU inside the seed hull, air excluded) and brain (median 40
  HU) is 22-28 HU, i.e. inside the oedema band: clinical Dice 0.68-0.78.
  Intensity cannot separate oedema from cavity contents on this case
  (`docs/figures/clinical-audit/cavity_adaptive.csv`).
- **The sheet tolerance should be about 5 mm, not 3.** Contoured seeds lie
  1.5-7.5 mm (median 3.8) inside `Cav_Post`; tol 3 caps clinical coverage
  at 71-72 % whatever else is set, tol 5 lifts it to 82-83 %. On the
  phantom tol 5 costs 0.00-0.03 Dice.
- **Geometry carries the cavity; intensity only extends it.** Without any
  growth, a 7 mm core already scores 0.77 clinically, but 0.58-0.77 on the
  phantom, whose implants are one-sided (the far wall is far from every
  seed). So the growth step stays, bounded by a lower threshold and a
  shorter reach.
- **Why the phantom pulls the other way.** The phantom fills the cavity
  with 18 HU fluid (+4 HU noise) against 35 HU brain and has no oedema; a
  20 HU threshold cuts into its own fluid, which is why thr 20 / reach 10
  costs it 0.04-0.07 Dice. Clinical cavity contents are at -8 HU (median
  of `Cav_Post`; 11 HU inside the hull) and the brain next to them is
  oedematous. The phantom is the less realistic of the two here: lowering
  its fluid to ~10 HU and adding a 20-30 HU oedema rim around the cavity
  would make the phantom gate representative and should be done before any
  rule is re-tuned on it.
- Seed-to-wall geometry is consistent between the two data sets: phantom
  seeds are placed 3 mm inside the wall (`SEED_INSET_MM`); their centres
  read -1 mm against `truth.masks["cavity"]` only because the lumen mask
  excludes the painted capsules.

Recommended change to `gtcore.segment.cavity` (seed-sheet branch), in this
order: (1) add the HU-free core (hull + 3 mm); (2) `SHEET_TOL_MM` 3 → 5;
(3) `FLUID_HI_HU` 26 → ~20 with `HULL_REACH_MM` 14 → ~10, re-gated on the
phantom after its fluid / oedema HU are made realistic. Expected: clinical
Dice ≈ 0.80 (from 0.66), phantom ≈ 0.80-0.90. The remaining clinical gap
(about 18 % of `Cav_Post` uncovered, 20 % of ours outside it) is clot above
any threshold and oedema below it, which is where a physician-editable
cavity in the planner is the honest answer.

## 6. Summary of corrections to the clinical notes

- Implant = 8 full tiles / 32 sources, not 7.5 tiles / 30.
- Seed strength is 3.5 U; no discrepancy with the TPS once all 32 sources
  are used. Remove the 3.76 U inference.
- gtcore seed detection on this case: 29/30 contoured seeds at 0.23 mm
  median, plus the 2 uncontoured sources; 1 seed lost to the vault filter
  (fixable); tile inference 6/8.
- Cavity: 0.63-0.66 Dice today; the mechanism and a tested path to ~0.80
  are in section 5.

## 7. The clinical target is derivable from the implant

`CTV_GT` is not an arbitrary hand trim: 90 % of its voxels lie within
10.3 mm of a seed (max 13.5 mm). Taking the full 5 mm rind of `Cav_Post`
and keeping only the part within `d` of any of the 32 seeds, clipped to the
`Brain` contour:

| d | volume | Dice vs `CTV_GT` | covers `CTV_GT` | RTDOSE D90 / V100 (unclipped) |
|---|---|---|---|---|
| 8 mm | 8.5 cc | 0.58 | 42 % | 8112 cGy / 99 % |
| **10 mm** | 16.9 cc | **0.88** | 83 % | 6699 cGy / 96 % |
| 12 mm | 25.2 cc | 0.85 | 98 % | 5787 cGy / 87 % |
| 14 mm | 31.7 cc | 0.74 | 99 % | 5091 cGy / 74 % |
| full rind | 35.4 cc | 0.69 | 99 % | 4528 cGy / 67 % |

(`CTV_GT`: 19.0 cc, D90 6365 cGy.) So "the tiled wall plus 5 mm" is the
clinical convention, and it is a function of the seed positions, which
gtcore measures. An automatic HR-CTV defined this way reproduces the
physician's target to Dice 0.85-0.88 with no hand editing, and its D90
brackets the clinical value. This is the natural target definition for the
intraoperative use case: the wall the tiles were placed to treat.
