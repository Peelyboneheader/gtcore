# Plan: more accurate seed localization and tile inference (modest compute)

## Context

Jacob asked how seed extraction and tile localization work today, and whether they can be made more accurate without much added complexity or runtime. Decisions taken: prioritize coarse 2 mm exports first, but keep thin-slice behaviour at least as good; up to ~5 s extra per clinical scan is acceptable; new methods ship behind flags and become the default only after pre-declared accuracy criteria pass; give a validation plan both with and without independent ground truth for the printed phantom.

### How it works today

**Seeds** (`gtcore/seeds/detect.py`, called from `gtcore/pipeline.py::reconstruct` on the raw HU volume, native grid, before metal-artifact reduction):
1. Global threshold chosen only from the slice thickness (`pipeline.seed_detection_params`: 2000 HU at ≤ 1.2 mm, then 1500 / 1200 / 1000 HU).
2. 26-connected components, a volume gate, then merged blobs are cut by unweighted k-means using the median volume of all blobs.
3. Centre = HU-weighted mean of the above-threshold voxels only; long axis = principal axis with a voxel-size correction.
4. Shape filter on volume and elongation, vault filter, optional threshold search when the seed count is known.

The known capsule size (4.5 × 0.8 mm), the scanner's 3071 HU clip, and interpolated gap slices are never used. No per-seed uncertainty is produced.

**Tiles** (`gtcore/tiles/`):
- *Count known* (`fit.py::fit_tiles`): enumerate all 4-seed groups that pass chord-length, flatness and axis-alignment gates; score by how square the group is (scale-free, not the 10 mm pitch); exact branch-and-bound picks the disjoint set; pose = seed mean + plane normal.
- *Count unknown* (`auto.py::fit_tiles_auto`): every gated group gets a bent-tile fit (`deform.py`, 9 parameters, uses the 10 mm pitch and 3 mm seed offset); the count is chosen by a per-tile penalty; a "cover pass" then adds tentative tiles, including 3-seed L-shapes completed with an inferred 4th seed.
- *Cross-check* (`surface.py`): fits each tile onto the cavity wall; reported only.

**Measured accuracy:** about 0.2 mm mean seed error on thin slices, growing to about 0.5 mm at 2.1 mm and 0.7 mm (max 1.6) at 2.8 mm slices, where the tile partition fails 3 of 5 times. Real scans have no seed truth.

**Weak points found:**
- Centroids depend on the threshold and on where the seed falls in the voxel grid.
- Thick-slice z error has no uncertainty model.
- Split-blob duplicates appear about 3 mm apart on the PostOp scan.
- Counted and automatic modes score differently and disagree on the printed phantom.
- Two bugs:
  - `fit.py:583` drops `spacing_mm` when `reconstruct` runs auto mode, so coarse scans use the 1 mm tolerance.
  - `auto.py:904-910` keeps the plane-fit normal while the drawn tile comes from the bent-tile fit.
- The baseline CSV (`output/validation_spacing.csv`, 13:47) is older than the phantom code (20:59). It must be re-measured before any gain is claimed.

### Key physical limit

A seed lying inside one 2–3 mm slab carries almost no information about its z position inside that slab (about slab/√12 ≈ 0.6–0.8 mm). No single-seed method removes that. The largest thick-slice gain comes from letting the tile model pin z, using an honest per-seed uncertainty. That is why the plan pairs a cheap better centroid with a tile-based fusion.

## Preliminary evidence (quick read-only experiment, 2026-10-08)

Synthetic phantom `make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=1)`, 12 seeds, slices block-averaged to 1.4 / 2.1 / 2.8 mm. "Proposed" is an untuned stage 2 prototype without neighbour masking.

**Dose sensitivity, one Cs-131 seed (TG-43, transverse axis):** a 0.5 mm seed error changes local dose by 16.6 % at 5 mm, 11.4 % at 8 mm and 9.5 % at 10 mm. A 1.0 mm error changes it by 29.5 % / 20.9 % / 17.8 %.

**Mean 3D seed error (mm), today → proposed:**

| Slices | Threshold | Today | Proposed | z error, today → proposed | Centre shift when the threshold changes, today → proposed |
|---|---|---|---|---|---|
| 0.7 | 1500 | 0.15 | 0.22 (max 1.12) | 0.04 → 0.13 | 0.02 → 0.00 |
| 1.4 | 2000 | 0.44 | 0.21 | 0.20 → 0.10 | 0.18 → 0.02 |
| 2.1 | 1500 | 0.65 | 0.23 | 0.37 → 0.13 | 0.52 → 0.26 |
| 2.8 | 1000 | 0.64 | 0.38 | 0.52 → 0.31 | 0.11 → 0.03 |

**Reading:**
- On thick slices the proposed estimator cuts mean error 25–65 % and makes the answer largely independent of the arbitrary HU threshold.
- On 0.7 mm slices the unmasked prototype is worse (one 1.12 mm outlier, most likely a neighbouring seed inside the ROI). This is exactly what the planned neighbour masking addresses, and why stage 2 stays behind a gate that requires "not worse on thin slices".

## Methodological basis (for the Medical Physics submission)

Every method below is either a textbook estimator or a standard formulation applied to the GammaTile geometry. Heuristic steps are labelled so the paper can disclose them and report their sensitivity.

| Stage | Method | Status | Precedent / justification |
|---|---|---|---|
| Today | Global threshold, connected components, intensity-weighted centroid, volume-based cluster splitting | Standard, but biased | The classic post-implant CT seed pipeline: threshold → connected volumes → split clusters by seed volume → weighted centroid and orientation. Centroids computed only over above-threshold voxels are known to be biased toward the voxel grid (documented in particle-localization literature). |
| 2 | Background-subtracted intensity-weighted centroid over a fixed ROI | Standard | Used for fiducial localization in CT (intensity-weighted 3D centroid of the marker region) and in sub-pixel localization generally. Unbiased for a symmetric blur when the ROI is not truncated by a threshold. |
| 2 | Per-seed covariance = error propagation of the centroid under noise + uniform within-slab position variance s²/12 | Standard (analytic) | First-order error propagation plus uniform-distribution variance. **No tuned constant.** Calibration is checked, not fitted, with the normalized-error χ² (NEES) consistency test. |
| 3 | Intensity-weighted k-means split; fragment merge | Split: standard. Merge: rule-based (disclose) | Weighted clustering is standard. Merge thresholds are tied to physical seed length (4.5 mm), with a sensitivity sweep reported. |
| 4 | One scoring rule (bent-tile fit) for both modes | Our model | Removes an internal inconsistency; the bent-tile model is already the paper's contribution. |
| 5 | **Hierarchical (random-effects) weighted least squares** of each tile with anisotropic seed covariance | Standard formulation | Generalized least squares with per-point anisotropic covariance is the accepted way to handle anisotropic localization error in point-based registration (Maier-Hein, Fitzpatrick et al., IEEE TPAMI 2012; anisotropic Procrustes, Danilchenko & Fitzpatrick). |
| 6 | Pose covariance s²(JᵀJ)⁻¹; partition score margin | Standard | Gauss–Newton covariance of a least-squares fit; the margin is a plain best-vs-second-best score gap. |
| 7 | Maximum-likelihood fit of a blurred line source | Standard (opt-in, sensitivity analysis) | Model-based least-squares sub-voxel localization (Gaussian/line-spread fitting) is standard. **Saturated voxels are excluded**, not modelled with a censored likelihood, to keep it simple and conventional. |
| 8 | Image check at an inferred seed position | Rule-based (disclose) | Peak above background by k·σ_noise, a conventional detection threshold. |

**Correction to the earlier draft (stage 5).** The draft blended each measured seed with the tile model's prediction after the fit. That double-counts the measurement, because the model was fitted to the same seed. The sound version is a single random-effects least-squares problem:
- measurement model: xᵢ = sᵢ + εᵢ with εᵢ ~ N(0, Σᵢ), and sᵢ = mᵢ(θ) + δᵢ with δᵢ ~ N(0, Σₛ);
- eliminating sᵢ gives a weighted bent-tile fit with weights (Σᵢ + Σₛ)⁻¹;
- the posterior seed estimate is then exactly the inverse-covariance blend, now consistent.

Same compute, one equation for the paper.

**Honest caveat to state in the paper.** A full tile gives 12 coordinates against 9 bent-tile parameters, so only 3 redundant degrees of freedom (plus the axis terms and the curvature prior). The z gain from stage 5 therefore comes mostly from near-flat tiles and the curvature prior. It must be measured; it cannot be assumed. Stage 5 has a pre-declared go/no-go gate and is dropped if the gain is not seen.

**Reporting conventions:**
- localization error as mean ± SD, 95th percentile and max, per axis and 3D;
- detection sensitivity and positive predictive value;
- split-half repeatability with Bland–Altman limits of agreement;
- covariance calibration by the NEES/χ² test;
- fiducial-registration vocabulary (localization error vs target error) for the with-truth track.

## Approach (stages, cheapest and highest-yield first; each ships independently)

| # | Stage | Added runtime | Expected gain |
|---|---|---|---|
| 0 | Validation harness + fresh baseline | test time only | Makes every claim measurable |
| 1 | Bug fixes + `SeedCandidates.subset` | 0 | Consistent poses; correct coarse-scan tolerance |
| 2 | Threshold-free grey-level centroid + per-seed covariance | 1–3 ms/seed | 10–25 % lower mean error at ≥ 1.4 mm; threshold-independent |
| 3 | Merge/split repair | ~0 | Removes PostOp split duplicates |
| 4 | Bent-tile score in counted mode | same as auto (~0.1 s/group) | Counted and automatic agree |
| 5 | Hierarchical weighted least-squares tile fit + posterior seed positions | µs/seed | Expected main thick-slice gain (15–30 % lower z error at 2.1–2.8 mm; 2.8 mm partition 2/5 → ≥ 3/5), limited by 3 redundant DOF per tile; dropped if not measured |
| 6 | Partition margin + tile pose uncertainty | one selector run per tile | Flags ambiguous groupings (printed seeds 25/31) |
| 7 | Line-source model fit (saturation, gap slices) | 10–25 ms/seed (≤ 1.5 s) | Further z gain where a seed straddles slices; calibrated covariance |
| 8 | Image check of inferred 4th seeds | ~1 ms/tile | Tentative tiles marked "recovered" or "no image evidence" |

Stages 0–5 are the core (about 3 days). Stages 6–8 are independent stretch goals. Total added runtime stays well under the 5 s budget (worst case about 2 s at 60 candidates with stage 7).

### Stage 0 — harness and baseline
- New `gtcore/phantom/seed_render.py`:
  - `render_seeds(shape, affine, centers, axes, metal_hu, background_hu, psf_sigma_mm, noise_hu, saturate_hu=3071, fine_mm=0.1, rng_seed)`. It supersamples the exact 4.5 × 0.8 mm capsule on a fine local grid, blurs, block-averages to the true (anisotropic) voxel footprint, adds noise and clips. This is deliberately not the stage 7 fit model.
  - `thick_slices` moves here; the script and test helpers become aliases.
  - `drop_and_interpolate(vol, keep_k)`, mirroring the DICOM gap filler.
  - `random_seed_layout`.
- Calibrate `metal_hu` so that 0.5×0.5×2.0 mm peaks sit around 1700 HU (PostOp) and 0.59×0.59×1.0 mm peaks clip at 3071 (printed phantom).
- `make_head_phantom(..., seed_render="binary"|"analytic", saturate_hu=None)`. The default stays bit-identical.
- `gtcore/io/dicom.py`: record `meta["interpolated_k"]` (which slices were filled). Additive.
- New `scripts/validation_seed_localization.py`:
  - grids G1 0.59×0.59×1.0, G2 0.5×0.5×2.0, G3 = G2 with gap slices, G4 0.7×0.7×2.8;
  - 200 seeds each, random sub-voxel offsets and axes;
  - reports bias and RMS per voxel axis, 3D mean / P95 / max, axis error, threshold sensitivity (1200 vs 2000 HU), covariance calibration (mean normalized squared error, NEES), and ms per seed.
- `scripts/validation_spacing.py` gains `--refine none|centroid|model`, `--fuse`, `--seed-render`, writing tagged CSVs. Re-run it to replace the stale baseline.
- New `scripts/validation_realdata_proxies.py` (no-truth track, see Verification).

### Stage 1 — bug fixes and plumbing
- `tiles/fit.py::fit_tiles(..., spacing_mm=None)`, passed to `fit_tiles_auto`. `pipeline.reconstruct` passes `vol.spacing`.
- `tiles/auto.py::fit_tiles_prior`: after attaching the bent-tile fit, set the normal and in-plane axis from it, oriented the way `_deformed_pose` already does.
- `to_placed_tiles`: orient the bent-tile normal into the cavity for near-flat fits.
- `seeds/detect.py`: `SeedCandidates` gains optional `cov_ras (N,3,3)` and `info`, plus `subset(keep)`. It is used at the three hand-rebuild sites in `pipeline.py` (lines 59, 276, 313) so new fields are not dropped.

### Stage 2 — grey-level centroid + covariance (`method="centroid"`)
New `gtcore/seeds/refine.py`:
- `seed_roi`: an ellipsoidal ROI sized from capsule length, blur and voxel size, with voxels closer to another candidate removed.
- `grey_centroid`:
  - local background = median of a surrounding shell;
  - signed weights (value − background) over the whole ROI, not just above-threshold voxels;
  - 3 re-centring passes;
  - axis reuses `detect._measure`.
- Covariance, analytic with no tuned constants: the noise term σₙ²·Σ(x−ĉ)(x−ĉ)ᵀ/(Σw)², plus a slab term s²/12 along any voxel axis the seed does not span (variance of a uniform position within the slab). It is checked with the NEES/χ² test on the harness, not fitted to it.
- `refine_seed_candidates(vol, cands, method, max_shift_mm=1.5)`: falls back to the original centre if the result moves too far.
- `pipeline.reconstruct(..., refine_seeds=None)` refines on the raw volume after the vault filter and threshold search. The default is off until the stage 2 gates pass.

### Stage 3 — merge/split repair (`detect.py`)
- `_split_merged_blobs`: take the median only over blobs inside the per-seed volume window (default-on; synthetic results unchanged). Optional intensity-weighted Lloyd split with 2 deterministic starts.
- New `_merge_fragments`. It merges two candidates only when all of these hold:
  - they are < 4.0 mm apart;
  - the image stays bright along the line between them;
  - either their axes are collinear with the separation, or on coarse scans the separation lies mainly along z.

  It never re-merges parts of one split blob, so the existing 7 mm two-seed split test is unaffected.
- First, dump the 5 unassigned PostOp candidates (separation, axes, bridge HU) to confirm they are z-fragments before tuning.

### Stage 4 — one scoring rule
- `fit_tiles(..., score="chord"|"deformable")`. The deformable option re-scores each gated group with `auto.deformable_score(deform.fit_deformable(...))`, clamped positive so the count still comes first.
- `fit_tiles_prior` uses the deformable score. `assess_implant` keeps the chord score it was calibrated on.

### Stage 5 — hierarchical weighted least squares (random effects)
- `deform.fit_deformable(..., seed_cov=None, slack_mm=0.3)`: position residuals are whitened by (Σᵢ + Σₛ)^(−1/2), with Σₛ = slack²·I as the per-seed deviation from the ideal bent sheet.
  - The whitening is normalized by its smallest eigenvalue (anisotropy clamped to 4×) so that, with no covariance, output is identical and the calibrated thresholds keep their meaning in mm.
  - `slack_mm` is a physical constant: placement and deformation tolerance of a seed within the collagen carrier. It is reported with a sensitivity sweep (0.1–0.5 mm), not tuned per scan.
- `auto.fit_tiles_auto(..., seed_cov=None)` forwards it to every `fit_deformable` call.
- New `gtcore/tiles/fuse.py::posterior_seed_positions(result, centers, cov)`. Given the fitted θ̂ it returns the posterior seeds `ŝᵢ = (Σᵢ⁻¹+Σₛ⁻¹)⁻¹(Σᵢ⁻¹xᵢ+Σₛ⁻¹mᵢ(θ̂))` and their covariance.
  - This is the exact conditional estimate of the random-effects model above, not a post-hoc blend.
  - Tentative and degraded tiles use a larger slack.
- `reconstruct(..., fuse_tiles=False)` keeps raw and posterior positions. The dose and planner feeds switch only when the flag is on.

### Stage 6 — uncertainty outputs (stretch)
- `DeformableFit`: pose covariance from the solver Jacobian, centre covariance, normal sigma in degrees.
- `fit_tiles_auto(..., margins=False)`: per selected tile, re-run `_PerCountSelector` without it and record the score margin and the alternative. Tiles with margin < 1 are flagged ambiguous. The counted equivalent goes through `fit._Selector`.
- Expose the margins in the planner status line.

### Stage 7 — line-source model fit (`method="model"`, stretch, opt-in)
- Closed-form blurred line source (whitened anisotropic Gaussian ⊗ 4.5 mm segment, erf form) plus a linear background.
- Saturated voxels (at the detected clip ceiling) and interpolated gap slices are excluded from the fit, the conventional treatment, rather than modelled with a censored likelihood.
- `scipy.optimize.least_squares` (trf, ≤ 100 evaluations), started from stage 2.
- Covariance from JᵀJ; falls back to stage 2 on bad fits.
- Optional 2-capsule fit for blobs that stage 3 split, chosen by BIC.

### Stage 8 — image check of inferred seeds (stretch)
- New `gtcore/tiles/verify.py::verify_inferred_seeds`. It looks for image evidence around each inferred 4th seed and, if present, refines it and marks it "recovered"; otherwise "no image evidence".
- It does not promote the tile out of tentative.

### Not doing (cost > benefit)
- Full joint image fit of each tile with its 4 seeds (stage 5's hierarchical fit on the seed estimates is its first-order version).
- Censored (Tobit) likelihoods, robust M-estimators, or Bayesian samplers: correct but harder to explain, for marginal gain here.
- Global re-thresholding, resampling, deconvolution, or metal-artifact-reduction changes.
- Learned detectors.
- Re-tuning the calibrated gates and penalties.
- An anisotropic rewrite of the head phantom.
- Letting the wall fit change tile selection.

## Critical files
- `gtcore/seeds/detect.py`, new `gtcore/seeds/refine.py`, `gtcore/seeds/__init__.py`
- `gtcore/tiles/fit.py`, `gtcore/tiles/auto.py`, `gtcore/tiles/deform.py`, new `gtcore/tiles/fuse.py`, new `gtcore/tiles/verify.py`
- `gtcore/pipeline.py` (flags `refine_seeds`, `fuse_tiles`; spacing pass-through; `subset()`)
- `gtcore/io/dicom.py` (`interpolated_k`)
- `gtcore/phantom/generate.py`, new `gtcore/phantom/seed_render.py`
- scripts: `validation_spacing.py`, new `validation_seed_localization.py`, new `validation_realdata_proxies.py`

Reuse: `detect._measure`, `deform.fit_deformable` / `deformable_score`, `model.kabsch` / `fit_rigid`, `auto._PerCountSelector` / `_cover_pass` / `_deformed_pose`, `fit._Selector`, `pipeline.seed_detection_params`, `scripts/validation_spacing.thick_slices`.

## Verification

### Synthetic, true error (both tracks)
Harness grids G1–G4 plus the head phantom via `validation_spacing.py` (5 realizations × 0.7 / 1.4 / 2.1 / 2.8 mm). Pre-declared gates:

| Stage | Gate |
|---|---|
| 2 | G1 mean ≤ 0.8× baseline; G2/G4 z RMS ≤ 0.85×; threshold sensitivity ≤ 0.1 mm and ≥ 2× smaller; NEES in [0.5, 2]; recall unchanged; ≤ 5 ms/seed; head phantom 2.1/2.8 mm mean ≥ 15 % better, 0.7 mm not worse by > 0.02 mm |
| 3 | side-by-side seeds 2.5–3.5 mm apart still split; gap-slice fragment → 1 candidate; 0 false splits among 30 noise blobs |
| 4 | counted = automatic partition on every auto test case |
| 5 | identical fit with no covariance; head phantom 2.1/2.8 mm mean ≥ 15 % better than raw seeds; tile centre/normal not worse; 2.8 mm partition ≥ 3/5; posterior covariance passes NEES in [0.5, 2]; result stable across slack 0.1–0.5 mm. If the 15 % gate fails, stage 5 is not shipped and that is reported |
| 7 | G2/G4 mean ≤ 0.85× stage 2; saturation bias ≤ 0.05 mm/axis; fallback ≤ 5 %; ≤ 30 ms/seed |
| 8 | 2.1 mm missed-seed case "recovered" within 1.0 mm; same case with the seed removed → "no image evidence" |

### Real scans WITHOUT independent truth (`validation_realdata_proxies.py`)
- **Printed phantom, 1 mm:**
  - bent-tile RMS per tile;
  - side chords vs the known 10 mm pitch on near-flat tiles (spread should shrink);
  - seed-axis agreement within a tile;
  - partition margin for seeds 25/31.
- **Split-half repeatability:** build two 2 mm scans from odd and from even slices of the 1 mm printed phantom. Disagreement/√2 estimates 2 mm precision with no truth; the full 1 mm result is the reference. Target ≥ 15 % lower disagreement than baseline.
- **PostOp:** unassigned 5 → ≤ 2; the 4 supported tiles kept; no new supported tiles.
- **Negative controls:** the tile-free printed scan (never analysed so far) and the pre-implant scan stay "not confirmed".
- **Printed phantom result:** stays 32/32 seeds, 8/8 tiles.

### Real scans WITH independent truth (when Jacob supplies it)
- **Design or measured layout available:**
  - match detected seeds to truth seeds with a Hungarian assignment inside a rigid registration (Kabsch + ICP over all 32);
  - report per-seed error after one global rigid fit (absolute accuracy) and after per-tile rigid fits (local accuracy, removing hand-placement error);
  - state the truth's own uncertainty (CAD tolerance or measurement method).
- **Cheap calibration phantom (if a new one can be made):**
  - tape 2–3 flat tiles onto a flat acrylic plate;
  - scan at 1 mm and 2 mm, at 0°, 30° and 45° tilt;
  - truth = coplanar 10 mm square grid per tile; plate plane from the CT;
  - this gives absolute intra-tile error, z-bias versus tilt, and slice-thickness dependence;
  - same scripts with a `--truth <csv|stl>` flag.

### Suite and docs
- Full suite stays green (~7 min, run alone); new tests add < 40 s.
- Opt-in flags become defaults only after the stage gates pass on both tracks. Each switch updates `README.md`, `docs/data-notes.md`, and a new section in `docs/autogen-notes.md` with command, seed and commit hash. Frozen expectations in `tests/test_tiles_cover.py` are updated only with before/after numbers recorded.
