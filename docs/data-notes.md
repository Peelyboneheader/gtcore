# Test-data status and findings (2026-09-01 overnight run)

| Dataset | Status | Findings |
|---|---|---|
| Synthetic phantom (`gtcore.phantom`) | ✅ full ground truth | 12/12 seeds @ 0.16 mm mean; brain Dice 0.961, cavity 0.881 |
| `DOE^JOHN...` head CT (204 sl, 0.52×0.52×1.0 mm) | ✅ complete series, **pre-implant** | negative control: 0 true seeds; vault filter removes all dental FPs; ~30 dense-bone candidates remain for tile-stage rejection |
| `CT 3D printed` (tile-less printed phantom) | ⚠️ **18 of ~244 slices present** | negative control once synced; slices present show mostly CT table |
| `3D-Printed Phantom-8tiles (223)` | ⚠️ **34 of ~223 slices present** | THE physical validation case (8 tiles = 32 seeds, count known); waiting on sync |
| `PostOp CT` (real case) | ⚠️ 64 slices, median dz 2.0 mm, gaps to 11 mm → loader rebuilt to 89-slice true grid (52 interpolated) | genuine post-implant: L-frontal cavity + air + seed cluster + streaks visible. **Seeds peak at only 1500–1950 HU** (partial volume at 2 mm + interpolation) → below the 2000 HU detector. Lowering to 1400 floods with 500+ bone/streak candidates. This export cannot support reliable seed localization. |

## Actions taken
- Loader now detects non-uniform slice positions, rebuilds the volume on the
  TRUE z grid (linear interpolation across gaps), reports
  `slices_present / slices_on_grid / slices_interpolated` in `meta`, and warns.
  Previously SimpleITK silently spread slices uniformly — up to ~9 mm z error
  on the PostOp scan. Tests: `tests/test_io_gaps.py`.

## Requests for Jacob
1. Re-copy / fully sync the two printed-phantom folders and, if possible, the
   original thin-cut (≤1.25 mm) PostOp series. OneDrive: right-click →
   "Always keep on this device". The hourly job rechecks the folders.
2. For the PostOp case: how many tiles (full/half) were implanted? The count
   is an algorithm input (challenge vi).

## Backlog identified from real data
- **Coarse-scan seed detection**: at ≥2 mm slices, use the cavity (air/blood
  region) as the search prior and detect seeds as local maxima near its wall
  instead of global thresholding; and/or resample+matched-filter. Also produces
  a paper figure: detection rate & localization error vs slice spacing
  (synthetic phantom resampled to 0.6/1.0/1.5/2.0/3.0 mm).
- Streak-artifact MAR beyond bloom inpainting (reprojection NMAR) — unchanged.

## Update (overnight, ~05:15)
- Adaptive spacing-aware detection landed (`gtcore.pipeline.seed_detection_params`):
  phantom study `output/validation_spacing.png` — recall 1.00 at 1.4/2.1/2.8 mm
  slices vs 0.58/0.00/0.00 fixed; partition 1.00 through 2.1 mm.
- **PostOp CT reanalyzed with adaptive detection**: 731 raw blobs -> 53 in-vault
  candidates -> a 27-seed cluster (35x36x30 mm) at the cavity; `fit_tiles`
  recovers **4 complete tiles (residuals 0.28-0.94 mm)** and saturates at 4 for
  any requested count, rejecting 11 leftovers. Still need from Jacob: the true
  implanted tile count (full/half) and ideally the thin-cut export.

## 8-tile printed phantom — VALIDATED (morning, final)
Series is complete at 157 slices (the "(223)" in the folder name is not a
slice count). Philips 0.59x0.59x1.0 mm, O-MAR on. Results:
- 32/32 seeds detected (exactly 8 tiles x 4; zero false positives after fixes)
- 8/8 tiles recovered with fit residuals 0.32-1.38 mm; one tile physically
  crumpled during placement (sides squeezed to ~5 mm) is recovered by the
  count-constrained degraded-completion pass and flagged `degraded=True`.
Fixes this scan drove: (1) voxel-quantization covariance in seed PCA (a seed
lying flat in one slice had elongation ~1e6); (2) vault filter fails OPEN
when no credible cranial interior exists (printed phantom has no skull);
(3) count-constrained completion for crumpled tiles (opt-in, pipeline on).

## PostOp CT — cavity segmentation verdict
Cavity mask is 0 voxels on this export under every prior (none / all 53 / the
27-seed cluster): the 52 interpolated slices smear the air/fluid boundaries
segmentation depends on, and the brain mask collapses to a degenerate blob.
The planner now says "no cavity surface in this scan" instead of eating
clicks. Expect both to work on the original thin-cut series.

## Automatic implant detection (assess_implant) — capabilities and limits
Tri-state verdict (confirmed / uncertain / absent) from manufactured-geometry
evidence: gate-passing 4-seed quads, non-bone context, grouping within one
cavity-sized region. Correct on every implanted scan tested (synthetic,
8-tile physical, PostOp real) and on synthetic negatives (scatter, chains).
KNOWN LIMITATION, measured on the DOE pre-implant negative control: dense
physiologic calcifications (pineal/choroid/falx) cluster near the third
ventricle, survive the shape filters (which select rod-like blobs by
construction), and form 6 chance quads with genuinely tile-like geometry
(residual 0.2-0.7 mm, axis coherence 0.94-0.98) -> false "confirmed".
No cheap feature separates them (tried: linkage clustering, bone masks,
shell-HU context, peak HU, elongation). Resolution: the verdict is EVIDENCE,
not authority -- surgeon knows whether an implant exists; the principled
discriminator (model-selection with deformable tile physics) is the
feature/tile-autogen work.
