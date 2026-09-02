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

## Verification against the TG-43 source documents (2026-09-01)

Sources read in full or in the relevant sections: CLRP TG-43 database v2
page and workbook for the Proxcelan CS-1 Rev2 (`LDR_Cs131_Proxcelan-CS-1
Rev2.xlsx`); AAPM TG-43U1 erratum (Med Phys 31:3532, 2004); TG-43U1S1
(Med Phys 34:2187, 2007) Sec. III interpolation/extrapolation; TG-43U1S2
(Med Phys 44:e297, 2017) Sec. 3, Appendix A11, Tables AI/AII/AXIII/AXIV/XXIV.
The 2018 erratum to TG-43U1S2 (Med Phys 45:971) is paywalled (HTTP 403 on
every route tried); by its abstract it corrects the report's QA dose-rate
tables, not consensus parameters.

### What matched

| Item | Engine | Source | Result |
|---|---|---|---|
| Formalism | 2D line-source, Eq. (1) TG-43U1 | TG-43U1 | identical |
| G_L(r,θ) | β/(L r sinθ), 1/(r²−L²/4) on axis | TG-43U1 | identical; numerically vs quadrature ≤1e-9 |
| Λ | 1.056 | TG-43U1S2 Table AI (CON Λ = 1.056 ± 0.013) | identical |
| L | 0.40 cm | TG-43U1S2 Table AI (CON L) and CLRP (4.0 mm active) | identical |
| g_L fit a0…a6 | 7.38e-4, −1.198e-2, 9.991e-1, 4.979e-1, 1.07e-2, 1.39e-3, 4.055e-1 | CLRP workbook fitting row | identical, digit for digit; fit range 0.05–10 cm |
| F(r,θ) 32×12 | engine table | CLRP workbook Anisotropy sheet | every entry = CLRP value rounded to 3 dp; hole pattern identical |
| T½ | 9.689 d | TG-43U1S2 Table XXIV | identical |
| Capsule | 0.824 mm OD, 4.50 mm | TG-43U1S2 A11 | now 0.0412 cm radius (was 0.040), 0.225 cm half-length |

### What did not match, and what changed

1. **Dataset choice.** v1 (and v2 until now) paired the consensus Λ with
   the CLRP v2 g_L fit and F table. The AAPM+GEC-ESTRO *consensus* g_L(r)
   and F(r,θ) for this seed (TG-43U1S2 Tables AII/AXIII) come from Rivard
   2007, not CLRP. The two datasets agree within 1.5 % for θ ≥ 15° but
   differ by up to 8 % within ~10° of the seed axis (CLRP higher). The
   engine now ships both as `SeedDataset`s; **`TG43Engine()` defaults to
   `"tg43u1s2"`** (the consensus, i.e. what a TPS is commissioned against —
   the GammaTile commissioning literature validates against TG-43U1S2),
   and `"clrp_v2"` remains available for cross-checks and for the v1
   regression tests.
2. **g_L interpolation/extrapolation.** TG-43U1S1 Sec. III.C requires
   log-linear interpolation of tabulated g_L between adjacent nodes
   (Eq. 3), nearest-neighbour below r_min, and — explicitly *not*
   zeroth-order — a single exponential through the two outermost nodes
   beyond r_max (Eq. 4). The engine's earlier far-field policy (hold g_L
   at 10 cm) violated the last point; it now follows Eq. 4 for both
   datasets (consensus: nodes 9 and 10 cm; CLRP: fit values at 9.5 and
   10 cm). Both worked examples printed in the supplement are reproduced
   (g(1.5) = 0.894 from 1.000/0.800; g(6) = 0.300 from 0.510/0.391).
3. **F extrapolation** (nearest-neighbour in r both below r_min and beyond
   r_max, linear-linear interpolation inside) already matched TG-43U1S1;
   the NaN back-fill is that rule applied once at init. All holes in both
   tables are inside the capsule (test-pinned), as are none of the
   tabulated entries.
4. **CLRP workbook defect (reported here for the record).** The
   `radial_dose` sheet's r labels are shifted by one row beyond 1 cm: the
   row labelled 1.5 cm holds exactly 1.000 (= g_L at r0), the row labelled
   10 cm holds g_L(9.5 cm), and the last, unlabelled row holds g_L(10 cm) =
   0.1467. Re-aligned, the CLRP fit reproduces the column to 0.3 %, and it
   agrees with CON g_L to 0.6 % for r ≤ 6 cm and 1.2 % at 10 cm.

### Not verified (stated plainly)

- The 2018 erratum's corrected QA tables. The 2017 printed Table XII for
  the CS-1 Rev2 is inconsistent with its own consensus parameters (its
  0.75 cm column is 13 % below Eq. (1) at every angle, its 0.1 cm column
  7 % above), which is consistent with the erratum's stated scope, so it
  is not used as a regression target. The engine is instead pinned to
  hand evaluations of Eq. (1) from the consensus tables at tabulated
  nodes (exact to 1e-9), and to the TG-43U1S1 worked examples.
- Anything inside the titanium capsule: TG-43 does not define dose there;
  the engine's nearest-surface projection is an engineering choice and is
  documented as such.
