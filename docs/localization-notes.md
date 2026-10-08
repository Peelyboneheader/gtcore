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
| 2 | Head phantom: 2.1/2.8 mm mean >= 15 % better, 0.7 mm not worse by > 0.02 mm. Threshold sensitivity <= 0.1 mm and >= 2x smaller. NEES in [0.5, 2]. Recall unchanged. <= 5 ms/seed. Harness: G1 mean <= 0.8x, G2/G4 z RMS <= 0.85x | Head phantom: PASS (2.1 mm -56 %, 2.8 mm -42 %, 0.7 mm -26 %; worst seed +0.08 mm). Threshold: PASS (median 0.000-0.002 mm vs 0.035-0.373). NEES: PASS on supersampled seeds (0.75-1.34, 5 grids); FAIL in 3D on the binary head phantom (10^2-10^3, caused by its voxelized capsules, not the estimator); z NEES 0.59 / 0.69 at 2.1 / 2.8 mm. Recall: unchanged by construction. Runtime: 1.7-3.4 ms/seed. Harness G1-G4: PENDING (stand-in grids: G1 mean 0.11x, G2 z RMS 0.24x, G4 z RMS 0.34x) | 234f3e5, 7afa8c2 | `reconstruct(refine_seeds="centroid")`; default stays `None` until the harness gate runs |

## Stage 2 — grey-level centroid + analytic covariance (branch loc/refine)

Code at 7afa8c2 (stage 2 is 234f3e5; 7afa8c2 checks the shift trust region inside the
iteration and adds `scripts/sweep_seed_refine.py`; every number below is from 7afa8c2).
`gtcore/seeds/refine.py`: `estimate_saturation`, `seed_roi`, `grey_centroid`,
`refine_seed_candidates(vol, cands, method="centroid", max_shift_mm=1.5,
psf_sigma_mm=0.45)`; `pipeline.reconstruct(..., refine_seeds=None)` refines
on the RAW volume after the vault filter and the threshold search, before the
implant assessment, and logs per-seed status in `vol.meta["seed_refine"]`.
Tests: `tests/test_seeds_refine.py` (19 tests, ~9 s).

Commands (every table below):

    python scripts/sweep_seed_refine.py --part head        # rng 0-4, ablations
    python scripts/sweep_seed_refine.py --part threshold   # rng 0-4
    python scripts/sweep_seed_refine.py --part analytic    # supersampled grids
    python -m pytest tests/test_seeds_refine.py -s         # rng 0-2 subset

### Estimator (as protocolled) and the measured deviations

Background-subtracted intensity-weighted centroid: median background of a
neighbour-masked shell (1.5 mm), SIGNED weights `v - b` over the whole
window (clipping at 0 keeps only the positive half of the noise, a pedestal
that pulls the centroid back toward the threshold-based start), 3
re-centred passes. Covariance = noise propagation `sigma_n^2 S / (sum w)^2`
(`sigma_n` = 1.4826 MAD of the shell) + a sampling term + a
background-gradient term; nothing is fitted. Deviations from the protocol
text, each forced by a measurement:

1. **Window shape.** The protocol's orientation-free ellipsoid
   (`L/2 + 3 sigma_eff,a`) is kept as `roi="ellipsoid"` and is used for the
   FIRST pass; later passes use a capsule window matched to the seed
   (`D/2 + 3 sigma_eff,a` around the axis segment). Reason: background
   structure inside the window becomes centroid bias roughly in proportion
   to N r (N voxels, radius r), and the phantom's cavity air-fluid level
   (a 1000 HU step) lies within 5 mm of 2-3 seeds per realization. One such
   seed moved 0.96 mm at 2.1 mm with the ellipsoid. Ellipsoid vs capsule,
   rng 0-4: mean 0.155 / 0.202 / 0.220 / 0.432 vs 0.147 / 0.187 / 0.219 /
   0.408 mm at 0.7 / 1.4 / 2.1 / 2.8 mm; fallbacks 12 / 13 / 11 / 16 vs
   8 / 10 / 10 / 13.
2. **Axis.** Principal axis of the Sheppard-corrected weighted covariance
   (data covariance minus `s_a^2/12`), not `detect._measure`, which takes
   the axis from the uncorrected covariance. A seed lying flat across a
   2.8 mm slab boundary has a larger through-slab spread (two slabs 2.8 mm
   apart: 1.96 mm^2) than along its length (~1.9 mm^2). Its axis flipped to
   z, the capsule window cut its ends, and the pass locked onto the wrong
   axis. Pairs of seeds 7 mm apart at 0.7x0.7x2.8 mm with slab phase ~0.5
   all fell back before the fix and refined to 0.01-0.13 mm after it.
3. **Neighbour masking: the Voronoi cut is the one needed** (default
   `mask="voronoi"`).
   - Head phantom, rng 0-4:
     - Voronoi alone gives the same errors and fallbacks as both masks.
     - The capsule exclusion alone adds 2 / 3 fallbacks at 2.1 / 2.8 mm.
     - No masking: mean 0.152 / 0.197 / 0.249 / 0.424 mm and 3-6 more
       fallbacks.
   - Supersampled pairs 7 mm apart, 6 trials each:
     - Side by side, every mode is fine (<= 0.02 mm at 0.7 mm iso).
     - END TO END without masking, every seed falls back ("extended") at
       0.7 / 2.0 / 2.8 mm.
     - The capsule exclusion alone leaves a 0.74 mm error at 0.5x0.5x2.0 mm.
     - Voronoi gives 0.02 / 0.19 / 0.42 mm; the last two are the slab
       sampling error of a flat seed.
   - Adding the capsule exclusion to Voronoi changed no result and doubles
     the runtime (4.0-6.4 vs 1.9-3.4 ms/seed).
   - The coordinator's prototype outlier at 0.7 mm (1.12 mm, rng 1) is most
     likely the air level, not a neighbour. Masking changes nothing at 0.7 mm
     here, while switching off the background check (item 5) re-creates a
     +1.06 mm outlier at 0.7 mm (rng 3, a seed 4.0 mm above the level).
4. **Sampling term.** The protocol's switch (`s^2/12` where
   `L|u_a| + 2.5 sigma_eff,a < 1.5 s_a`, else 0; kept as `slab="rule"`) is
   badly over-confident: z NEES 12-45 on the supersampled grids and 87-925
   on the head phantom.
   - Replaced by the closed-form variance of a box-sampled centroid over a
     uniform sub-voxel phase (Poisson summation):
     `var_a = s_a^2/(2 pi^2) sum_k |F_a(k/s_a)|^2 / k^2`, where `F_a` is the
     Fourier transform of the seed's blurred profile along axis a. It equals
     `s^2/12` for a thin seed, vanishes for a wide one, and has no threshold.
   - Default `slab="bound"` evaluates it for a seed lying in the slab plane,
     the maximum over orientations. The through-slab axis component is
     exactly what thick slices cannot resolve.
   - With the estimated axis (`slab="exact"`), z NEES is 1.7-5.6
     (supersampled) and 9-208 (head phantom). The bound gives 0.23-0.40
     (supersampled, conservative) and 0.59 / 0.69 (head phantom,
     2.1 / 2.8 mm).
5. **Background structure** (new).
   - `delta = S g / sum(w)` is the shift that a planar background gradient
     `g` (least squares over the shell) would cause.
   - It is added to the covariance as `delta delta^T` and triggers
     `fallback:background` when `|delta| > 0.1 mm` (rule, disclosed).
   - Seeds more than 5 mm below the air level have a median `|delta|` of
     0.002-0.019 mm (max 0.021 mm). The exception is at 2.8 mm with the
     level 6-7 mm above, where it reaches 0.59 mm. Seeds near the level that
     get this far reach up to 0.99 mm.
   - Sensitivity at 0.05 / 0.1 / 0.2 mm: means within 0.01 mm, fallbacks
     +-1. Switched off: the 0.7 mm outlier of item 3.

Other fallbacks return the detected centre with covariance `s_a^2/12`:
- `shift`: the refined centre leaves the 1.5 mm trust region;
- `no_signal`: no positive signal;
- `extended`: the shell holds more than 0.5x the ROI's peak contrast
  (a plate, bone or an unmasked neighbour);
- the window runs off the volume.

Saturation:
- The ceiling (>= 5 voxels at exactly the maximum) is recorded in
  `info["saturation_hu"]`, and per seed in `refine_n_saturated`.
- Saturated voxels are kept, because a symmetric clip keeps the profile
  symmetric.
- Test: supersampled seeds clipped at 1500 HU on 0.59x0.59x1.0 mm keep bias
  <= 0.05 mm per axis and max error <= 0.1 mm.

### Head phantom (binary capsules), rng 0-4, detection as `reconstruct`

| Slices | n | Mean (mm), detection -> refined | Max | z RMS | Worst seed vs detection | Fallbacks | NEES 3D / z | ms/seed |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 60 | 0.200 -> 0.147 (-26 %) | 0.670 -> 0.495 | 0.095 -> 0.068 | +0.080 | 8 | 3358 / 1650 | 3.1-3.4 |
| 1.4 | 60 | 0.291 -> 0.187 (-36 %) | 1.312 -> 1.312 | 0.182 -> 0.116 | +0.099 | 10 | 1729 / 7.0 | 2.1-2.2 |
| 2.1 | 58 | 0.495 -> 0.219 (-56 %) | 0.994 -> 0.855 | 0.395 -> 0.189 | +0.038 | 10 | 861 / 0.59 | 1.8-2.0 |
| 2.8 | 59 | 0.703 -> 0.408 (-42 %) | 1.805 -> 1.805 | 0.624 -> 0.418 | +0.084 | 13 | 525 / 0.69 | 1.7-1.9 |

Notes on this table:
- Means include the fallback seeds at their detection error. The 1.4 and
  2.8 mm maxima are fallback seeds.
- rng 0-2 subset printed by the test: 0.191 -> 0.134, 0.282 -> 0.165,
  0.472 -> 0.197, 0.733 -> 0.380 mm.
- ms/seed includes the once-per-volume saturation scan.

**Where the fallbacks are.** 41 of 237 seed instances fell back.
- 38 are seeds less than 5 mm below, or above, the cavity's air-fluid
  level. There are 49 such instances; 11 refined. By status: `no_signal`
  22, `shift` 10, `background` 5, `extended` 1.
- The other 3 are at 2.8 mm with the level 6.0-6.9 mm above
  (`background`): the window plus shell reaches it through a 2.8 mm slab.
- None of the remaining 185 seeds fell back.
- Real post-op cavities often hold air against the tile's lumen face.
  The fallback rate on clinical scans must be measured on the real-data
  proxies before any default switch.

**On the head phantom, 3D NEES fails.**
- In-plane, the binary phantom paints each capsule on the 0.7 mm grid before
  blurring. A Gaussian blur preserves the centroid of the painted voxels, so
  the in-plane "error" is the painted capsule's voxelization (0.05-0.2 mm),
  while image noise predicts ~0.002-0.01 mm.
- That is a renderer artefact (a real CT samples a band-limited image), so
  calibration is judged on supersampled seeds, and on the head phantom
  along z only, where the sampling term dominates: 0.59 / 0.69 at
  2.1 / 2.8 mm.
- At 0.7 / 1.4 mm the 0.7 mm painting error also dominates z.

### Threshold sensitivity (1200 vs 2000 HU, same seed, truth-matched), rng 0-4

| Slices | Seeds (both refined ok) | Detection median / P90 (mm) | Refined median / P90 | Refined <= 0.1 mm |
|---|---|---|---|---|
| 0.7 | 60 (52) | 0.035 / 0.094 | 0.000 / 0.046 | 97 % |
| 1.4 | 58 (50) | 0.244 / 0.577 | 0.000 / 0.207 | 86 % |
| 2.1 | 43 (33) | 0.335 / 0.988 | 0.001 / 1.006 | 63 % |
| 2.8 | 28 (20) | 0.373 / 0.903 | 0.002 / 0.890 | 68 % |

Pairs where both thresholds refined OK agree to <= 0.001 mm, with one
exception. At 2000 HU a coarse-slice seed can split into fragments
(0.7-1.0 mm^3). The Voronoi cut then halves the seed, and each fragment
refines toward its own half (up to 1.6 mm). That is the stage-3 duplicate
problem, and refinement does not repair it.

### Supersampled seeds (calibration), 72 random seeds per grid, 20 HU noise

Rendering, in order:
- capsules rendered on a fine grid nested in the voxels (3 sub-samples
  in-plane, 3-10 through-slab);
- Gaussian blur (sigma 0.45 mm), then box-averaging to the voxel;
- white noise;
- random sub-voxel phase and axis per seed.

The renderer is `_render_capsules` in the test file, a stand-in for the
stage-0 harness renderer.

| Grid (mm) | Mean (mm) | RMS x / y / z after | NEES 3D bound / exact / rule | z NEES bound / exact / rule |
|---|---|---|---|---|
| 0.59x0.59x1.0 (G1-like) | 0.081 -> 0.009 | 0.006 / 0.006 / 0.007 | 0.90 / 0.94 / 0.94 | 1.02 / 1.11 / 1.12 |
| 0.5x0.5x2.0 (G2-like) | 0.281 -> 0.048 | 0.007 / 0.008 / 0.070 | 0.75 / 1.34 / 5.97 | 0.26 / 2.04 / 12.5 |
| 0.5x0.5x2.1 | 0.351 -> 0.055 | 0.007 / 0.008 / 0.077 | 0.78 / 1.29 / 7.72 | 0.23 / 1.69 / 16.9 |
| 0.7x0.7x2.8 (G4-like) | 0.598 -> 0.169 | 0.013 / 0.013 / 0.218 | 0.75 / 3.13 / 22.9 | 0.40 / 5.62 / 44.6 |
| 0.7 iso | 0.093 -> 0.010 | 0.006 / 0.007 / 0.007 | 1.34 / 1.34 / 1.34 | 1.57 |

- In-plane NEES is 0.83-1.37 on every grid, so the noise term is
  calibrated.
- z RMS, detection -> refined: 0.295 -> 0.070 (G2-like), 0.632 -> 0.218
  (G4-like).
- Residual z bias: -0.012 mm at 2.0-2.1 mm and -0.022 mm at 2.8 mm.

### Sensitivity of the rule-based choices (head phantom, rng 0-4, mean mm at 0.7 / 1.4 / 2.1 / 2.8)

| Variant | Mean | Fallbacks |
|---|---|---|
| default | 0.147 / 0.187 / 0.219 / 0.408 | 8 / 10 / 10 / 13 |
| shell 1.0 mm | 0.147 / 0.187 / 0.219 / 0.417 | 8 / 10 / 10 / 14 |
| shell 2.5 mm | 0.149 / 0.196 / 0.219 / 0.407 | 10 / 12 / 10 / 13 |
| bg check 0.05 mm | 0.147 / 0.187 / 0.219 / 0.418 | 9 / 10 / 10 / 14 |
| bg check 0.2 mm | 0.147 / 0.187 / 0.219 / 0.400 | 8 / 10 / 10 / 12 |
| bg check off | 0.164 / 0.187 / 0.205 / 0.372 (0.7 mm max 1.29) | 7 / 10 / 7 / 9 |
| PSF sigma 0.35 mm | 0.142 / 0.186 / 0.221 / 0.405 (z NEES 2.1 / 2.8: 0.30 / 0.47) | 6 / 10 / 10 / 12 |
| PSF sigma 0.60 mm | 0.149 / 0.196 / 0.219 / 0.413 (z NEES 2.28 / 1.36) | 10 / 12 / 10 / 14 |

The phantom's true blur is 0.45 mm. The estimate is insensitive to the
assumed PSF; the covariance's z term follows it, as it should.

### Limitations to disclose

- Seeds at an air-fluid level or a bone edge fall back to the detected
  centre (no gain, no loss); the fallback rate on real scans is unknown.
- Split-blob duplicates (stage 3) are refined as separate seeds.
- The covariance assumes white noise. Real CT noise is spatially
  correlated, which makes the noise term optimistic; the sampling term is
  unaffected.
- The PSF sigma (0.45 mm) is a scanner parameter, not fitted per scan.

## Real-data proxies (no truth)

## With-truth track

## Open decisions

## Runs log

| Date | Command | Seeds | Commit | Wall time |
|---|---|---|---|---|
| 2026-10-08 | `python scripts/sweep_seed_refine.py --part head` | rng 0-4 | 7afa8c2 | ~20 s |
| 2026-10-08 | `python scripts/sweep_seed_refine.py --part threshold` | rng 0-4 | 7afa8c2 | ~15 s |
| 2026-10-08 | `python scripts/sweep_seed_refine.py --part analytic` | layout rng 1, noise rng 2 | 7afa8c2 | ~15 s |
| 2026-10-08 | `python -m pytest tests/test_seeds_refine.py -s` | rng 0-2 | 7afa8c2 | 9 s |
