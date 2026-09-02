# TG-43 engine port — issues found during verification (2026-09-01)

`gtcore/dose/tg43.py` is a **verbatim** port of `DoseInterpolator` from the
AAPM GammaView module (publication 1). Numerical behaviour was preserved
exactly; verification found the following, to be addressed in step v
(dose engine) of publication 2 — none were changed during the port:

1. **Near-axis geometry factor goes negative.** For theta < 0.01 the line-source
   G uses `1/(r^2 - (L/2)^2)`, negative for r < 0.20 cm and singular at 0.20 cm.
   `dose_at_point` clamps with `max(0, rate)`, so dose reads exactly 0 on the
   seed's long axis out to ~2 mm from the centre — inside the treated volume.
   Fix: proper line-source G_L on-axis limit `1/(r^2 - L^2/4)` only valid
   off-segment; use the correct beta/(L*r*sin(theta)) formulation with the
   on-axis analytic limit.
   **Status: FIXED in `gtcore/dose/engine.py` (step v).** `TG43Engine`
   uses `beta/(L r sin(theta))` everywhere, the analytic on-axis limit
   `1/(r^2 - L^2/4)` only for r > L/2 (beyond the tip), and clamps points
   inside the physical capsule cylinder (perpendicular distance < 0.04 cm,
   |axial| <= L/2) to the geometry value at the capsule surface. On-axis
   dose is now positive, finite and monotone; regression tests in
   `tests/test_dose_engine.py` pin both the v2 behaviour and the fact that
   v1 still reads 0 there.
   **Refined (2026-09-01, dose-engine pass).** The side-only clamp left a
   jump at the capsule *end face* (interior clamp value vs the line-source
   value just outside it, ~65 % on axis). Interior points now project to
   the NEAREST capsule-surface point (side or end face) and the whole
   TG-43 product is evaluated there, so the field is continuous across the
   entire can (`test_capsule_boundary_is_continuous`) and nothing inside
   the can exceeds the surface value. Outside the capsule nothing changed.
2. **`_tg43_dose_rate` does not fold theta about 90 deg** (F silently returns 1.0
   for theta > 90); `dose_at_point` folds first and is symmetric. Any direct
   caller of the private method must fold theta itself. Fold inside the core
   method when building the vectorized grid engine.
   **Status: FIXED in `gtcore/dose/engine.py` (step v).** `TG43Engine.dose_rate`
   folds theta (mod 360, mirrored about 180 and 90) as its first step, so the
   core is exactly symmetric for every caller; `rate(30, r) == rate(150, r)`
   bitwise, tested.
3. **`DOSE_CONVERSION_FACTOR = 335.23 * 3.5`** has no documented derivation.
   It should be re-derived explicitly as S_K [U] x integral-to-total-decay
   tau = T_half / ln2 (Cs-131 T_half = 9.689 d -> tau = 335.5 h) — the 335.23
   is plausibly that integral in hours; the 3.5 is presumably S_K per seed in U,
   which must instead come from the implant's assay certificate, not a constant.
   Re-establish provenance before quoting absolute doses.
   **Status: FIXED in `gtcore/dose/engine.py` (step v).**
   `TG43Engine.dose_to_total_decay(rate, sk_per_seed_u)` computes
   `rate * S_K * tau` with `tau = T_half / ln 2` derived in code from
   `T_half = 9.689 d = 232.536 h` (tau = 335.48 h, ~335.5 h). S_K is an
   explicit parameter (default 3.5 U, the historical value) that should be
   fed from the assay certificate. v1's `335.23 * 3.5` is confirmed as
   `tau_v1 * S_K`: 335.23 h is the same tau 0.08% stale, which is exactly the
   residual seen in the v1-vs-v2 regression (+0.074% in v2).
4. **Anisotropy table NaN holes** (r = 0.10, 0.15 cm at theta <= 15 deg) are
   filled by scanning outward to the first non-NaN radius — values near the
   source borrow from much larger radii. Replace with published CS-1 Rev2
   consensus data or explicit extrapolation.
   **Status: PARTIALLY FIXED in `gtcore/dose/engine.py` (step v).** The fill
   is now precomputed once at init (nearest valid value along r at the same
   theta) and documented, instead of a dynamic scan on every lookup; the
   filled values are identical to v1's, preserving regression agreement.
   The *data* limitation stands: r = 0.10/0.15 cm entries at theta <= 20 deg
   still borrow from r = 0.25 cm (0.15 cm for the theta = 20 row) and the
   table should eventually be replaced with published CS-1 Rev2 consensus
   values — flagged for physics review.
   **Scoped (2026-09-01).** All 19 holes lie *inside the titanium can*
   (perpendicular distance < 0.04 cm and |axial| <= 0.225 cm;
   `test_anisotropy_holes_are_inside_the_capsule`). Because interior points
   are projected to the capsule surface before F is looked up, no field
   point in tissue ever reads a borrowed value. The replacement is still
   desirable for completeness but has no dosimetric consequence.
5. **Inconsistent radius floors**: `_radial_dose` clips r to [0.05, 10] while
   `dose_at_point` clips to [0.041, 10]; for r in [0.041, 0.05) the radial term
   freezes while geometry varies. Unify.
   **Status: FIXED in `gtcore/dose/engine.py` (step v).** One clip,
   r in [0.05, 10.0] cm (`TG43Engine.R_CLIP_CM`), applied exactly once in
   `dose_rate`; the same clipped radius feeds G_L, g_L and F. (The F lookup
   still clamps to its own table domain [0.10, 10.0] cm, as v1 did — that is
   interpolation-domain clamping, not a second dose clip.)
   **Refined (2026-09-01).** The 10 cm *ceiling* froze the entire rate
   beyond 10 cm (a 1.6e-3 cGy h^-1 U^-1 plateau, i.e. ~2 cGy per seed
   everywhere out to the grid edge — wrong for whole-head grids). Now only
   the floor is a clip (`R_FLOOR_CM = 0.05`); g_L and F are held at their
   `R_DATA_MAX_CM = 10 cm` values while G_L keeps falling, giving a
   monotone, conservative (attenuation-free) far field
   (`test_far_field_falls_off_beyond_data_domain`). Extrapolating the CLRP
   g_L fit instead would attenuate further but is unvalidated outside its
   domain; the difference is < 1e-3 of the prescription at 20 cm.

Verified normalization at reference point (r0 = 1 cm, theta0 = 90 deg):
rate = 1.054448 cGy h^-1 U^-1 vs Lambda = 1.056 (-0.15%, from g_L(1) = 0.99853
polynomial fit). Transverse falloff monotonic: 4.088 / 1.054 / 0.242 / 0.092 /
0.022 at r = 0.5 / 1 / 2 / 3 / 5 cm.

## Dose-engine refinement pass (2026-09-01)

Beyond the five port defects above, the following were changed in
`gtcore/dose/engine.py` / added in `gtcore/dose/metrics.py` (all pinned in
`tests/test_dose_engine.py` and `tests/test_dose_metrics.py`):

- **Tabulated kernel.** `TG43Engine.dose_rate_tabulated` bilinearly
  interpolates the exact rate on a (ln r, theta) lattice (0.5 % radial
  steps, 0.25 deg; 0.05–50 cm) built lazily once per engine. Max relative
  error vs the analytic rate: 5e-4 for r >= 0.25 cm at any angle, < 1 %
  for r < 0.25 cm outside the capsule; on a real 12-seed grid the two agree
  to 8e-5. `compute_dose_grid` defaults to it (`exact=False`):
  12 seeds / 2 mm / 100 mm cube 0.33 s → 0.17 s, 1 mm 2.6 s → 1.1 s.
- **Sub-voxel isodoses.** `isodose_surfaces` runs marching cubes on the
  log-dose scalar field at ln(level) instead of on a thresholded mask. A
  single-seed isodose at ~5 mm on a 2 mm grid lands 0.04 mm rms / 0.09 mm
  max from the analytic isodose radius (the mask approach could only place
  it at half-voxel positions, up to 1 mm off). Normals are outward; all
  disconnected shells are kept; a level outside the grid range yields an
  empty mesh. Default Taubin smoothing is now 0 (no longer needed).
- **Decay helpers.** `decay_factor`, `sk_decayed(sk, hours_since_assay)`
  and `delivered_fraction(hours)` (= 1 − exp(−t/τ)); `compute_dose_grid`
  and `dose_at_points` take `elapsed_hours` to report dose delivered by a
  given time (meta records `delivered_fraction`). Total-to-decay remains
  the default.
- **Point evaluation.** `dose_at_points(centers, axes, points, ...)` — the
  same seed sum as the grid without a grid (exact by default).
- **Clinical metrics** (`gtcore.dose.metrics`): `dvh` / `DVH.D(pct)` /
  `DVH.V(dose)` as exact order statistics, `dose_metrics` (D90, D100, D50,
  Dmean, Dmax, V100/150/200, volume), `rind_mask` (tissue shell within a
  depth of the cavity on the dose grid, physical-mm EDT, optional exclusion
  mask), `wall_dose` (dose at a depth along each cavity-mesh vertex normal)
  and `surface_coverage` (area-weighted fraction of the wall at >= rx). The
  planner's `u` status line now reports wall coverage at 5 mm.
- **Per-seed S_K sanity.** Negative S_K raises; zero-S_K seeds are skipped.

Phantom readouts with 3 truth tiles (12 seeds, 3.5 U each): 5 mm rind
25.5 cc, D90 13.6 Gy, V100 0.24, V150 0.09; wall coverage at 5 mm for
60 Gy: 20.6 % (three tiles cover part of a ~25 cc cavity's wall, as
expected — the number is a planner readout, not a pass/fail).
