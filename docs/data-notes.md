# Test-data status and findings (2026-09-01 overnight run)

| Dataset | Status | Findings |
|---|---|---|
| Synthetic phantom (`gtcore.phantom`) | ✅ full ground truth | 12/12 seeds @ 0.16 mm mean; brain Dice 0.961, cavity 0.881 |
| `DOEJOHNPOSTCT` (`DOE^JOHN...` head CT, "STEALTH 1.0 Hr40", 204 sl, 0.52×0.52×1.0 mm, 2023-10-26 09:09:16) | ✅ complete series, **post-implant 1 mm thin-cut of the SAME acquisition as `PostOp CT`** (corrected 2026-10-08; previously recorded as "pre-implant negative control") | the reference dataset for the PostOp export: contiguous 1 mm slices, same in-plane origin / 0.5195 mm pixels / orientation, all 64 PostOp slice positions coincide with thin-cut slices (different kernel, mean \|ΔHU\| ≈ 40). 57 raw blobs → 31 in-vault candidates; **6 supported tiles** (bent-tile residuals 0.18–0.67 mm), 7 unassigned; its tiles reproduce the PostOp supported tiles to ≤ 0.2 mm. The "~30 dense-bone candidates" of the 2026-09-01 reading are the implant. |
| `CT 3D printed` (tile-less printed phantom) | ✅ complete (248 slices, 0.68×0.68×1.0 mm) | the only real negative control on this machine: 0 seed candidates, implant "absent" (updated 2026-10-08; was "18 of ~244 slices present") |
| `3D-Printed Phantom-8tiles (223)` | ✅ complete (157 slices, 0.59×0.59×1.0 mm) | THE physical validation case (8 tiles = 32 seeds, count known): 32/32 seeds, 8/8 tiles (see below; was "34 of ~223 slices present") |
| `PostOp CT` (real case, 2023-10-26 09:11:14) | ⚠️ 64 slices of **1.0 mm thickness at 2.0 mm spacing** (1 mm slabs with 1 mm unimaged gaps; DICOM `SliceThickness` 1.0), gaps to 11 mm → loader rebuilt to 89-slice 2.0 mm grid (52 interpolated) | genuine post-implant: L-frontal cavity + air + seed cluster + streaks visible. Seed peaks on the **measured** slices: median 2936 HU, 4/22 tile-assigned seeds saturated at 3071 HU; the weak **1500–1950 HU** peaks recorded on 2026-09-01 occur only on **interpolated** slices (corrected 2026-10-08). With the adaptive 1200 HU floor: 731 raw blobs → 53 in-vault candidates (50 after the stage-3 fragment merge), 4 supported + 2 tentative tiles, 3 unassigned. The thin-cut `DOEJOHNPOSTCT` is the same acquisition and serves as its 1 mm reference. |

## Corrections (2026-10-08, localization stage 3; details in `docs/localization-notes.md`)
- **`DOEJOHNPOSTCT` is not a pre-implant scan.** It is the contiguous 1 mm
  thin-cut ("STEALTH 1.0 Hr40", 204 slices, acquired 2023-10-26 09:09:16) of
  the same post-implant acquisition as the `PostOp CT` export (series time
  09:11:14, two minutes later): same in-plane origin, 0.5195 mm pixels and
  orientation; all 64 PostOp slice positions coincide with thin-cut slices.
  Its 6 supported tiles match the PostOp tiles within 0.2 mm. The "6 chance
  calcification quads → false confirmed" finding below was therefore a
  correct detection of the implant, and **there is currently no real
  pre-implant negative control on this machine** (the tile-free printed
  scan is the only real negative). The thin-cut is now the reference
  dataset for the PostOp export (`scripts/validation_realdata_proxies.py`
  matches PostOp seeds to it without any transform).
- **`PostOp CT` geometry.** The slices are 1.0 mm slabs (DICOM
  `SliceThickness` 1.0) at 2.0 mm spacing with 1 mm unimaged gaps, not
  contiguous 2 mm slabs; the seeds on measured slices peak at a median of
  2936 HU (4/22 saturated) and the 1500–1950 HU peaks of the 2026-09-01
  reading are confined to interpolated slices. The duplicate candidate
  pairs (34/35, 4/37) were slab-boundary fragments of single seeds and are
  rejoined by the stage-3 fragment merge (unassigned 5 → 3).
- The scans moved out of OneDrive on 2026-10-08: they now live under
  `C:\Users\jacob\Documents\` (the scripts fall back to the old OneDrive
  path; `GT_DATA_ROOT` overrides).

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
   *(Resolved 2026-10-08: both phantom folders are complete, and the thin-cut
   is already here — it is `DOEJOHNPOSTCT`, see Corrections above. What is
   still missing is a real pre-implant scan to serve as a negative control.)*
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
  implanted tile count (full/half) and ideally the thin-cut export *(the
  thin-cut is `DOEJOHNPOSTCT`, identified 2026-10-08; the count is still
  unknown — the thin-cut reads 6 supported tiles)*.

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
8-tile physical, PostOp real, and the DOEJOHNPOSTCT thin-cut of the PostOp
acquisition) and on synthetic negatives (scatter, chains).
**Correction (2026-10-08).** Previously recorded as: "KNOWN LIMITATION,
measured on the DOE pre-implant negative control: dense physiologic
calcifications (pineal/choroid/falx) cluster near the third ventricle,
survive the shape filters (which select rod-like blobs by construction),
and form 6 chance quads with genuinely tile-like geometry (residual
0.2-0.7 mm, axis coherence 0.94-0.98) -> false 'confirmed'. No cheap
feature separates them (tried: linkage clustering, bone masks, shell-HU
context, peak HU, elongation)." That reading rested on the mislabel of
`DOEJOHNPOSTCT` as pre-implant: it is the 1 mm thin-cut of the same
post-implant acquisition as `PostOp CT` (Corrections above), so those 6
quads ARE the implant (they reproduce the PostOp supported tiles within
0.2 mm), the "confirmed" verdict on that scan is correct, and no feature
could have separated them because there was nothing to separate. The
calcification false-positive risk is therefore UNMEASURED, not refuted:
there is currently no real pre-implant head CT on this machine, and the
only real negative control is the tile-free printed scan (0 candidates,
"absent"). The comments in `pipeline.assess_implant` and `reconstruct`
that cite the "pre-implant negative control" (~30 dense-bone candidates,
6 quads) rest on the same mislabel.
Unchanged resolution: the verdict is EVIDENCE, not authority -- the surgeon
knows whether an implant exists; the principled discriminator
(model-selection with deformable tile physics) is the feature/tile-autogen
work.

## PostOp CT — "T only gives 3 tiles" (2026-09-02)

Jacob reported 3 suggested tiles for a cluster of many seeds.  Re-run of
the exact planner path on the current folder (64 slices present, 89-slice
2.0 mm grid, 52 interpolated; 1200 HU adaptive floor): 731 blobs -> 53
in-vault candidates -> calibrated auto fit **n = 4** ("no further tile
candidate", not the node cap; marginal gains 7.2 / 7.2 / 6.5 / 6.4 all
far above the 3.5 penalty, no 5th disjoint candidate).  The 3 seen earlier
is consistent with an older sync state of the folder (fewer slices -> a
different grid and threshold branch); the search itself is exact.

Why the 37 leftovers are not tiles: 26 are far from the implant (bone /
streak clutter, now reported as `clutter`); of the 11 inside the cluster,
two pairs are split-blob duplicates 3.0-3.4 mm apart (34/35, 4/37), the
only two 4-candidate groups with tile-like chords have bent-tile residuals
of 2.0-2.5 mm and incoherent axes, and several L-shaped triplets exist
(arms 8-12 mm) — i.e. tiles with one seed lost to partial volume.

With the cover pass (`docs/autogen-notes.md`): **4 supported + 2 tentative
tiles** (both by triplet completion, inferred seeds flagged), **5 seeds
unassigned** inside the implant (the two duplicates and two singles) and
shown magenta in the planner; no half tiles assumed.  `--tiles 6` gives
the same reading with an explicit shortfall message; `--tiles 4` keeps the
4 and lists 11 unassigned.  Every remaining gap on this export is a
detection limit of the 2 mm interpolated series, not a fitting one; the
thin-cut export is still the real fix.

*Update 2026-10-08 (localization stage 3):* the two "split-blob duplicates"
are slab-boundary fragments of one seed each (a capsule crossing the 1 mm
unimaged gap between two 1 mm slabs); `reconstruct` now rejoins them by
default on such exports (`merge_fragments`), leaving 4 supported + 2
tentative tiles and **3 unassigned** real seeds — the same three the thin-cut
`DOEJOHNPOSTCT` (6 supported tiles, 7 unassigned) leaves unassigned, so no
detection repair can place them. The thin-cut is on this machine (see
Corrections); the implanted tile count is still unknown.
