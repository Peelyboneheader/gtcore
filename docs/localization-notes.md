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
| 5 — hierarchical WLS + posterior seeds | identical fit without covariance; head phantom 2.1 / 2.8 mm mean 3D error ≥ 15 % below raw; tile centre / normal not worse; 2.8 mm partition ≥ 3/5; posterior NEES in [0.5, 2]; stable across slack 0.1–0.5 mm | **PENDING-COVARIANCE** (stand-in: analytic slab covariance). Identity: bit-identical (max diff 0.0 over 288 fits vs 3cf35af; frozen reference in the tests) ✓. 2.1 mm: −0.3 % ✗. 2.8 mm: −15.2 % (borderline ✓). Centre unchanged; normal 3.77→2.66° (2.1) and 6.44→2.85° (2.8) ✓. Partition 2.8 mm 5/5 (raw also 5/5); 2.1 mm 2/5, same as raw (split-fragment detections). NEES conditional 2.02 / 3.00 ✗, PEV 1.25 / 1.76 ✓ (input covariance itself 1.19 / 1.86). Slack sweep: 2.1 / 2.8 mm within 2.5 % ✓; 0.7 mm at slack 0.1 is 16 % worse than raw ✗ | b09cb39 | opt-in `reconstruct(fuse_tiles=True)`, default off |

## Real-data proxies (no truth)

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
