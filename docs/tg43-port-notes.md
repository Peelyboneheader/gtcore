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
5. **Inconsistent radius floors**: `_radial_dose` clips r to [0.05, 10] while
   `dose_at_point` clips to [0.041, 10]; for r in [0.041, 0.05) the radial term
   freezes while geometry varies. Unify.
   **Status: FIXED in `gtcore/dose/engine.py` (step v).** One clip,
   r in [0.05, 10.0] cm (`TG43Engine.R_CLIP_CM`), applied exactly once in
   `dose_rate`; the same clipped radius feeds G_L, g_L and F. (The F lookup
   still clamps to its own table domain [0.10, 10.0] cm, as v1 did — that is
   interpolation-domain clamping, not a second dose clip.)

Verified normalization at reference point (r0 = 1 cm, theta0 = 90 deg):
rate = 1.054448 cGy h^-1 U^-1 vs Lambda = 1.056 (-0.15%, from g_L(1) = 0.99853
polynomial fit). Transverse falloff monotonic: 4.088 / 1.054 / 0.242 / 0.092 /
0.022 at r = 0.5 / 1 / 2 / 3 / 5 cm.
