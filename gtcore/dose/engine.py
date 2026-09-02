"""Corrected, vectorized TG-43U1 dose engine (step v, refined).

This is the successor to :mod:`gtcore.dose.tg43` (the verbatim GammaView
port, kept unchanged as the regression reference). Seed data — the dose-rate
constant ``Lambda``, active length ``L``, the CLRP v2 ``g_L(r)`` polynomial
and the 32 x 12 anisotropy table ``F(r, theta)`` for the IsoRay Proxcelan
CS-1 Rev2 Cs-131 seed — are copied EXACTLY from ``tg43.py``. What changes is
the *implementation*: the five defects documented in
``docs/tg43-port-notes.md`` are fixed here, and everything is vectorized so a
whole dose grid is a handful of numpy broadcasts.

Formalism (AAPM TG-43U1, 2D line-source):

    Drate(r, theta) = Lambda * [G_L(r, theta) / G_L(r0, theta0)]
                             * g_L(r) * F(r, theta)          [cGy h^-1 U^-1]

with ``r0 = 1 cm``, ``theta0 = 90 deg``; ``r`` in centimetres, ``theta`` in
degrees from the seed long axis.

Fixes relative to ``tg43.py`` (numbers match the port-notes items):

1. **Geometry function.** ``G_L = beta / (L r sin(theta))`` with ``beta`` the
   angle subtended by the active line at the field point. Near the long axis
   the analytic on-axis limit ``1 / (r^2 - L^2/4)`` is used ONLY for
   ``r > L/2`` (beyond the tip of the active line, where it is positive and
   finite). Points *inside the source* — within the physical capsule
   cylinder, i.e. perpendicular distance ``< RHO_SURFACE`` (0.04 cm, the
   0.8 mm-diameter CS-1 capsule radius) at axial position ``<= 0.225 cm``
   (the 4.5 mm titanium can) — are projected onto the NEAREST point of the
   capsule surface (cylindrical side or end face) and the whole TG-43
   product is evaluated there. Dose is therefore positive, finite and
   continuous across the entire capsule boundary, and nothing inside the
   can exceeds the surface value, instead of the v1 behaviour (negative G
   clamped to exactly 0 on the long axis out to ~2 mm). Outside the
   capsule the dose is exactly the TG-43U1 line-source value.
2. **Theta folding** (mod 360, mirror about 180 and 90) happens inside the
   core rate function, so every caller — including the vectorized grid path —
   is symmetric by construction.
3. **Anisotropy NaN holes** (r = 0.10, 0.15 cm at theta <= 15 deg) are filled
   ONCE at init by the nearest valid value along r at the same theta (the
   holes are the leading columns, so "nearest valid" is the first tabulated
   radius with data). This reproduces v1's dynamic scan-right fill exactly
   but is explicit, precomputed and documented. Every hole lies *inside the
   physical capsule* (perpendicular distance < 0.04 cm and axial position
   < 0.225 cm at all of them — see ``test_anisotropy_holes_are_inside_the
   _capsule``), so the borrowed values can only ever be sampled by field
   points inside the titanium can; no point in tissue reads a filled entry
   without also being subject to the capsule clamp of fix 1. The table still
   deserves replacement with published consensus data — see port notes.
4. **One radial floor, one data domain.** ``r`` is floored at
   ``R_FLOOR_CM`` (0.05 cm) exactly once in the core rate function, and the
   floored value feeds G_L, g_L and F alike. The tabulated / fitted data —
   g_L(r) and F(r, theta) — are only defined out to ``R_DATA_MAX_CM``
   (10 cm); beyond it they are held at their 10 cm values while the
   geometry factor keeps falling as ~1/r^2. The original v2 froze the whole
   rate at its 10 cm value (a 1.5e-3 cGy h^-1 U^-1 plateau everywhere beyond
   10 cm), which is unphysical for whole-head grids; holding only the data
   terms is the same domain-clamp policy the F lookup already applied and
   gives a monotone, conservative (attenuation-free) far-field falloff.
5. **Explicit dose conversion.** ``dose_to_total_decay(rate, sk_per_seed_u)``
   = ``rate * S_K * tau`` with ``tau = T_half / ln 2``,
   ``T_half(Cs-131) = 9.689 d = 232.536 h`` so ``tau = 335.48 h`` (~335.5 h).
   v1's opaque ``DOSE_CONVERSION_FACTOR = 335.23 * 3.5`` is exactly
   ``tau_v1 * S_K`` with ``S_K = 3.5 U`` baked in: 335.23 h is a slightly
   stale value of the same integral-to-total-decay tau (0.08 % below
   232.536 / ln 2), and 3.5 U is a *per-implant assay quantity*, not a
   physical constant — here it is an explicit parameter (default 3.5 U) that
   should come from the implant's assay certificate. ``sk_decayed`` and
   ``delivered_fraction`` expose the same exponential for decaying an assay
   S_K to the implant date and for dose delivered by a given elapsed time.

Evaluation paths
----------------
``dose_rate`` is the exact analytic evaluation. ``dose_rate_tabulated``
bilinearly interpolates a kernel table sampled once per engine on a fine
(ln r, theta) lattice (``KERNEL_LN_R_STEP``, ``KERNEL_THETA_STEP_DEG``); it
agrees with the exact path to better than 0.1 % everywhere outside the
capsule and beyond r = 0.25 cm (the interior projection is piecewise and
is not resolved by the lattice, which is immaterial: there is no tissue
there), and it is ~2.5x faster, which is what makes 1 mm whole-cavity
grids interactive.
``compute_dose_grid`` and ``dose_at_points`` take an ``exact`` flag; the
grid defaults to the table, point evaluation to exact.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import trimesh
from skimage import measure

from ..volume import Volume, apply_affine

__all__ = ["TG43Engine", "compute_dose_grid", "dose_at_points",
           "isodose_surfaces"]


class TG43Engine:
    """Vectorized TG-43U1 2D line-source engine for IsoRay Proxcelan CS-1 Rev2.

    All public evaluation methods accept scalars or numpy arrays (broadcast
    together) and return the matching scalar/array.
    """

    # ---------------------------------------------------------------- seed data
    # Copied EXACTLY from gtcore.dose.tg43.DoseInterpolator — do not retune.
    _LAMBDA = 1.056                # Dose-rate constant (cGy h^-1 U^-1)
    _L = 0.40                      # Active source length (cm)
    # g_L(r) = (a0*r^-2 + a1*r^-1 + a2 + a3*r + a4*r^2 + a5*r^3) * exp(-a6*r)
    _GR_COEFFS = np.array([7.38e-04, -1.198e-02, 9.991e-01, 4.979e-01,
                           1.07e-02, 1.39e-03, 4.055e-01])
    # Anisotropy table F(r, theta) — V2 (2019) Proxcelan CS-1 Rev2
    _F_THETAS = np.array([
        0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 73, 75, 78, 80, 82, 84, 85, 86, 87, 88, 89, 90
    ], dtype=np.float64)

    _F_RADII = np.array([
        0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00, 7.50, 10.00
    ], dtype=np.float64)
    _F_TABLE = np.array([
        #  0.10   0.15   0.25   0.50   0.75   1.00   2.00   3.00   4.00   5.00   7.50  10.00
        [  np.nan, np.nan, 0.617, 0.883, 0.903, 0.900, 0.889, 0.884, 0.882, 0.882, 0.880, 0.874],  # 0
        [  np.nan, np.nan, 0.648, 0.864, 0.890, 0.893, 0.891, 0.889, 0.887, 0.885, 0.879, 0.875],  # 1
        [  np.nan, np.nan, 0.679, 0.847, 0.868, 0.878, 0.877, 0.882, 0.878, 0.875, 0.869, 0.866],  # 2
        [  np.nan, np.nan, 0.709, 0.837, 0.860, 0.867, 0.850, 0.847, 0.848, 0.849, 0.850, 0.852],  # 3
        [  np.nan, np.nan, 0.762, 0.822, 0.793, 0.779, 0.787, 0.798, 0.807, 0.814, 0.826, 0.832],  # 5
        [  np.nan, np.nan, 0.786, 0.742, 0.726, 0.730, 0.761, 0.781, 0.794, 0.803, 0.818, 0.827],  # 7
        [  np.nan, np.nan, 0.731, 0.688, 0.702, 0.718, 0.760, 0.783, 0.797, 0.807, 0.822, 0.830],  # 10
        [  np.nan, np.nan, 0.696, 0.694, 0.714, 0.730, 0.772, 0.794, 0.807, 0.817, 0.830, 0.838],  # 12
        [  np.nan, np.nan, 0.761, 0.724, 0.743, 0.758, 0.795, 0.814, 0.826, 0.834, 0.845, 0.852],  # 15
        [  np.nan, 1.141, 0.845, 0.780, 0.794, 0.806, 0.835, 0.849, 0.858, 0.863, 0.871, 0.875],  # 20
        [  1.143, 1.051, 0.889, 0.829, 0.838, 0.847, 0.868, 0.879, 0.885, 0.888, 0.894, 0.897],  # 25
        [  1.131, 1.016, 0.916, 0.868, 0.873, 0.880, 0.895, 0.903, 0.907, 0.910, 0.914, 0.915],  # 30
        [  1.074, 1.001, 0.935, 0.899, 0.902, 0.906, 0.918, 0.923, 0.926, 0.928, 0.930, 0.932],  # 35
        [  1.040, 0.994, 0.950, 0.923, 0.924, 0.927, 0.936, 0.940, 0.942, 0.943, 0.944, 0.946],  # 40
        [  1.021, 0.993, 0.962, 0.942, 0.942, 0.944, 0.951, 0.953, 0.955, 0.956, 0.956, 0.957],  # 45
        [  1.013, 0.993, 0.972, 0.958, 0.957, 0.959, 0.963, 0.965, 0.966, 0.966, 0.967, 0.968],  # 50
        [  1.007, 0.994, 0.979, 0.969, 0.970, 0.970, 0.973, 0.974, 0.975, 0.975, 0.975, 0.976],  # 55
        [  1.004, 0.995, 0.985, 0.978, 0.979, 0.980, 0.981, 0.982, 0.982, 0.982, 0.982, 0.983],  # 60
        [  1.002, 0.996, 0.990, 0.985, 0.986, 0.987, 0.988, 0.988, 0.988, 0.988, 0.988, 0.989],  # 65
        [  1.001, 0.998, 0.994, 0.990, 0.991, 0.992, 0.993, 0.993, 0.993, 0.993, 0.993, 0.993],  # 70
        [  1.001, 0.998, 0.996, 0.993, 0.993, 0.994, 0.995, 0.995, 0.995, 0.995, 0.995, 0.995],  # 73
        [  1.001, 0.999, 0.997, 0.995, 0.995, 0.995, 0.996, 0.996, 0.996, 0.996, 0.996, 0.997],  # 75
        [  1.000, 0.999, 0.998, 0.997, 0.997, 0.997, 0.998, 0.998, 0.998, 0.998, 0.998, 0.998],  # 78
        [  1.000, 0.999, 0.999, 0.998, 0.998, 0.998, 0.998, 0.999, 0.999, 0.999, 0.999, 0.998],  # 80
        [  1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999],  # 82
        [  1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 1.000, 1.000, 0.999, 1.000, 1.000],  # 84
        [  1.000, 1.000, 1.000, 0.999, 1.000, 1.000, 0.999, 1.000, 1.000, 1.000, 1.000, 1.000],  # 85
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 86
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 87
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 88
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 89
        [  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # 90
    ], dtype=np.float64)

    # ------------------------------------------------------------ decay physics
    #: Cs-131 half-life: 9.689 days = 232.536 hours (NNDC).
    T_HALF_HOURS = 9.689 * 24.0                       # 232.536 h
    #: Mean life tau = T_half / ln 2 = 335.48 h (~335.5 h). Integrating an
    #: exponentially decaying dose rate D0*exp(-t/tau) from 0 to infinity gives
    #: total dose = D0 * tau, so total-decay dose = rate * S_K * tau.
    TAU_HOURS = T_HALF_HOURS / math.log(2.0)
    #: Default air-kerma strength per seed [U]. v1 hard-coded 3.5 U inside its
    #: DOSE_CONVERSION_FACTOR (= 335.23 * 3.5 = tau_v1 * S_K); the real number
    #: must come from the implant's assay certificate, hence a parameter here.
    DEFAULT_SK_U = 3.5

    # --------------------------------------------------------- geometry limits
    #: Radial floor [cm]: r below this is evaluated AT the floor (one clip,
    #: applied once in the core rate function, feeding G_L, g_L and F alike).
    R_FLOOR_CM = 0.05
    #: Outer edge of the tabulated / fitted data [cm]. g_L and F are held at
    #: their value here for larger r; the geometry factor is NOT clamped.
    R_DATA_MAX_CM = 10.0
    #: Kept for callers that read the old pair; (floor, data-domain edge).
    R_CLIP_CM = (R_FLOOR_CM, R_DATA_MAX_CM)
    #: Physical capsule radius [cm] (CS-1 capsule is 0.8 mm diameter x 4.5 mm).
    #: Field points inside the capsule cylinder clamp G_L to its surface value.
    _RHO_SURFACE = 0.04
    # PHYSICAL capsule half-length (4.5 mm titanium can), cm. The clamp for
    # "inside the source" must use this, not the ACTIVE half-length L/2 =
    # 0.20 cm: with the active length, on-axis points at z in (0.20, 0.225]
    # cm -- still inside the titanium -- hit the analytic 1/(z^2 - L^2/4)
    # limit near its pole and spiked ~50x (review finding).
    _CAP_HALF_CM = 0.225
    #: Perpendicular distances below this [cm] count as "on axis" and take the
    #: analytic on-axis limit (only reachable beyond the tip, see fix 1).
    _Y_EPS = 1.0e-6

    # ----------------------------------------------------------- kernel table
    #: Lattice step in ln(r) for the tabulated kernel (0.5 % radial steps).
    KERNEL_LN_R_STEP = 0.005
    #: Lattice step in theta [deg]; every integer table angle is a node, so
    #: the piecewise-linear F(theta) is reproduced exactly along theta.
    KERNEL_THETA_STEP_DEG = 0.25
    #: Outer radius [cm] of the tabulated kernel; beyond it the table path
    #: clamps r (the exact path does not). 50 cm exceeds any head-CT extent.
    KERNEL_R_MAX_CM = 50.0

    def __init__(self):
        # Fix 3: precompute the NaN-filled anisotropy table ONCE. The NaN
        # holes sit in the leading (small-r) columns of rows theta <= 20 deg,
        # so "nearest valid value along r at the same theta" is the first
        # non-NaN entry of the row; every hole is back-filled with it. This
        # matches v1's dynamic scan-right fill value-for-value, but the borrow
        # from larger radii is now explicit and happens exactly once.
        filled = self._F_TABLE.copy()
        for row in filled:
            valid = np.flatnonzero(~np.isnan(row))
            if valid.size == 0:      # pragma: no cover - table is never empty
                row[:] = 1.0
                continue
            first = valid[0]
            row[:first] = row[first]
            # Interior/trailing holes (none in this table) would take the
            # nearest valid neighbour along r; assert the assumption instead.
            if np.isnan(row).any():  # pragma: no cover - guarded by data
                idx = np.flatnonzero(np.isnan(row))
                for i in idx:
                    j = valid[np.argmin(np.abs(self._F_RADII[valid]
                                               - self._F_RADII[i]))]
                    row[i] = row[j]
        self._F_FILLED = filled
        # Reference geometry factor G_L(r0=1 cm, theta0=90 deg).
        self._GL_ref = float(self._geometry_factor(np.array(1.0),
                                                   np.array(90.0)))
        # Kernel table is built lazily on first tabulated evaluation.
        self._kernel = None

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _fold_theta(theta_deg):
        """Fold any angle into [0, 90] deg (mod 360, mirror at 180 and 90)."""
        t = np.mod(theta_deg, 360.0)
        t = np.where(t > 180.0, 360.0 - t, t)
        t = np.where(t > 90.0, 180.0 - t, t)
        return t

    def _project_to_capsule(self, y, z):
        """Map field points inside the physical capsule onto its surface.

        Returns ``(y_eff, z_eff, side)``: points with perpendicular distance
        ``y < rho`` and ``|axial| z <= cap half-length`` move to the NEAREST
        capsule-surface point -- the cylindrical side (y -> rho, same z) when
        that is closer, else the end face (same y, z -> cap half-length).
        Points outside pass through unchanged. Evaluating the whole TG-43
        product at the projected point makes the dose rate continuous across
        the entire capsule boundary and constant along each projection ray,
        so nothing inside the titanium can (where there is no tissue) can
        ever exceed the value on the can's surface.
        """
        inside = (y < self._RHO_SURFACE) & (z <= self._CAP_HALF_CM)
        side = inside & ((self._RHO_SURFACE - y) <= (self._CAP_HALF_CM - z))
        end = inside & ~side
        y_eff = np.where(side, self._RHO_SURFACE, y)
        z_eff = np.where(end, self._CAP_HALF_CM, z)
        return y_eff, z_eff, side

    def _geometry_factor_yz(self, y, z):
        """Line-source G_L from perpendicular ``y`` and |axial| ``z`` [cm].

        ``y`` / ``z`` must already be outside the capsule (see
        :meth:`_project_to_capsule`).
        General case: G_L = beta / (L * y) with beta the angle the active
        line subtends at the field point,
        beta = atan2(y, z - L/2) - atan2(y, z + L/2).
        On the long axis (y ~ 0, only reachable beyond the tip, z > L/2)
        the analytic limit 1 / (z^2 - L^2/4) is positive and finite.
        """
        L = self._L
        half = L / 2.0
        near_axis = y < self._Y_EPS
        y_safe = np.maximum(y, self._Y_EPS)       # keep atan2/div well-posed
        beta = np.arctan2(y_safe, z - half) - np.arctan2(y_safe, z + half)
        G = beta / (L * y_safe)
        denom = np.maximum(z * z - half * half, 1.0e-12)
        return np.where(near_axis, 1.0 / denom, G)

    def _geometry_factor(self, r_cm, theta_deg):
        """Line-source G_L(r, theta), vectorized. theta must be in [0, 90].

        Convenience wrapper: converts to (y, z), projects capsule-interior
        points to the surface (fix 1) and evaluates
        :meth:`_geometry_factor_yz`.
        """
        r = np.asarray(r_cm, dtype=float)
        th = np.radians(np.asarray(theta_deg, dtype=float))
        y = r * np.sin(th)                      # perpendicular distance
        z = r * np.abs(np.cos(th))              # |axial| position
        y_eff, z_eff, _side = self._project_to_capsule(y, z)
        return self._geometry_factor_yz(y_eff, z_eff)

    def _radial_dose(self, r_cm):
        """g_L(r) — CLRP v2 polynomial fit, vectorized.

        Expects r >= R_FLOOR_CM; r beyond R_DATA_MAX_CM is held at the edge
        value (the fit is not validated outside its data domain).
        """
        r = np.minimum(np.asarray(r_cm, dtype=float), self.R_DATA_MAX_CM)
        a = self._GR_COEFFS
        return (a[0] * r**-2 + a[1] * r**-1 + a[2] + a[3] * r
                + a[4] * r**2 + a[5] * r**3) * np.exp(-a[6] * r)

    def _anisotropy(self, r_cm, theta_deg):
        """F(r, theta) — bilinear interpolation on the NaN-filled table.

        Lookups are clamped to the table domain (r in [0.10, 10.0] cm,
        theta in [0, 90] deg); this is domain clamping of the interpolation,
        identical to v1, not an extra radial dose clip.
        """
        radii = self._F_RADII
        thetas = self._F_THETAS
        table = self._F_FILLED

        r_c = np.clip(r_cm, radii[0], radii[-1])
        t_c = np.clip(theta_deg, thetas[0], thetas[-1])

        ir = np.clip(np.searchsorted(radii, r_c, side="right") - 1,
                     0, len(radii) - 2)
        it = np.clip(np.searchsorted(thetas, t_c, side="right") - 1,
                     0, len(thetas) - 2)

        t_r = (r_c - radii[ir]) / (radii[ir + 1] - radii[ir])
        t_t = (t_c - thetas[it]) / (thetas[it + 1] - thetas[it])

        v00 = table[it, ir]
        v01 = table[it, ir + 1]
        v10 = table[it + 1, ir]
        v11 = table[it + 1, ir + 1]

        top = v00 * (1.0 - t_r) + v01 * t_r
        bot = v10 * (1.0 - t_r) + v11 * t_r
        return top * (1.0 - t_t) + bot * t_t

    # -------------------------------------------------------------------- core
    def _rate_folded(self, theta_folded_deg, r_cm):
        """Exact rate for theta already in [0, 90] and r already floored.

        Field points inside the physical capsule are projected onto the
        nearest capsule-surface point first, and ALL THREE factors (G_L,
        g_L, F) are evaluated there -- see :meth:`_project_to_capsule`.
        """
        r = np.asarray(r_cm, dtype=float)
        th = np.radians(np.asarray(theta_folded_deg, dtype=float))
        y = r * np.sin(th)
        z = r * np.abs(np.cos(th))
        y_eff, z_eff, side = self._project_to_capsule(y, z)
        moved = side | (z_eff != z)
        if np.any(moved):
            r_eval = np.where(moved, np.hypot(y_eff, z_eff), r)
            th_eval = np.where(moved,
                               np.degrees(np.arctan2(y_eff, z_eff)),
                               theta_folded_deg)
        else:
            r_eval, th_eval = r, theta_folded_deg
        GL = self._geometry_factor_yz(y_eff, z_eff)
        gL = self._radial_dose(r_eval)
        F = self._anisotropy(r_eval, th_eval)
        return self._LAMBDA * (GL / self._GL_ref) * gL * F

    def dose_rate(self, theta_deg, r_cm):
        """TG-43U1 2D dose rate [cGy h^-1 U^-1], vectorized and exact.

        Fix 2: theta is folded into [0, 90] HERE, so the core is exactly
        symmetric for every caller. Fix 4: r is floored at R_FLOOR_CM once,
        and the same floored radius feeds G_L, g_L and F.
        """
        th_in = np.asarray(theta_deg, dtype=float)
        r_in = np.asarray(r_cm, dtype=float)
        scalar = th_in.ndim == 0 and r_in.ndim == 0
        th, r = np.broadcast_arrays(np.atleast_1d(th_in), np.atleast_1d(r_in))

        th_f = self._fold_theta(th)
        r_c = np.maximum(r, self.R_FLOOR_CM)
        rate = self._rate_folded(th_f, r_c)
        return float(rate[0]) if scalar else rate

    # ------------------------------------------------------------ kernel table
    def _build_kernel(self):
        """Sample the exact rate on a (ln r, theta) lattice, once."""
        u0 = math.log(self.R_FLOOR_CM)
        u1 = math.log(self.KERNEL_R_MAX_CM)
        n_u = int(math.ceil((u1 - u0) / self.KERNEL_LN_R_STEP)) + 1
        du = (u1 - u0) / (n_u - 1)
        n_t = int(round(90.0 / self.KERNEL_THETA_STEP_DEG)) + 1
        dt = 90.0 / (n_t - 1)
        u = u0 + du * np.arange(n_u)
        t = dt * np.arange(n_t)
        tt, uu = np.meshgrid(t, u, indexing="ij")          # (n_t, n_u)
        table = self._rate_folded(tt, np.exp(uu))
        self._kernel = {"u0": u0, "du": du, "n_u": n_u, "dt": dt,
                        "n_t": n_t, "table": table}
        return self._kernel

    def dose_rate_tabulated(self, theta_deg, r_cm):
        """Rate via bilinear interpolation of the kernel table [cGy h^-1 U^-1].

        Same folding/flooring semantics as :meth:`dose_rate`; additionally
        clamps r to ``KERNEL_R_MAX_CM``. Interpolation is linear in ln(r)
        and theta, which tracks the ~1/r^2 falloff to ~1e-5 relative.
        """
        th_in = np.asarray(theta_deg, dtype=float)
        r_in = np.asarray(r_cm, dtype=float)
        scalar = th_in.ndim == 0 and r_in.ndim == 0
        th, r = np.broadcast_arrays(np.atleast_1d(th_in), np.atleast_1d(r_in))
        rate = self._rate_tabulated_folded(self._fold_theta(th), r)
        return float(rate[0]) if scalar else rate

    def _rate_tabulated_folded(self, theta_folded_deg, r_cm):
        k = self._kernel if self._kernel is not None else self._build_kernel()
        table = k["table"]
        r_c = np.clip(r_cm, self.R_FLOOR_CM, self.KERNEL_R_MAX_CM)
        fu = (np.log(r_c) - k["u0"]) / k["du"]
        ft = np.asarray(theta_folded_deg, dtype=float) / k["dt"]
        iu = np.clip(fu.astype(np.intp), 0, k["n_u"] - 2)
        it = np.clip(ft.astype(np.intp), 0, k["n_t"] - 2)
        wu = np.clip(fu - iu, 0.0, 1.0)
        wt = np.clip(ft - it, 0.0, 1.0)
        v00 = table[it, iu]
        v01 = table[it, iu + 1]
        v10 = table[it + 1, iu]
        v11 = table[it + 1, iu + 1]
        top = v00 + (v01 - v00) * wu
        bot = v10 + (v11 - v10) * wu
        return top + (bot - top) * wt

    # -------------------------------------------------------------- conversion
    def decay_factor(self, hours):
        """exp(-t / tau): fraction of activity (or S_K) remaining after t."""
        return np.exp(-np.asarray(hours, dtype=float) / self.TAU_HOURS)

    def sk_decayed(self, sk_u, hours_since_assay):
        """Air-kerma strength [U] decayed from the assay time by ``hours``.

        The certificate quotes S_K at an assay date; the implant usually
        happens days later, and it is the *implant-day* S_K that sets dose.
        """
        return np.asarray(sk_u, dtype=float) \
            * self.decay_factor(hours_since_assay)

    def delivered_fraction(self, hours):
        """Fraction of the total-to-decay dose delivered after ``hours``.

        int_0^t exp(-s/tau) ds / tau = 1 - exp(-t/tau). ``None`` or +inf
        means the full total-to-decay dose (fraction 1).
        """
        if hours is None:
            return 1.0
        h = np.asarray(hours, dtype=float)
        if np.any(h < 0.0):
            raise ValueError("elapsed hours must be non-negative")
        return 1.0 - np.exp(-h / self.TAU_HOURS)

    def dose_to_total_decay(self, rate, sk_per_seed_u=DEFAULT_SK_U):
        """Convert a dose rate [cGy h^-1 U^-1] to total-to-decay dose [cGy].

        total = rate * S_K * tau, with tau = T_half / ln 2 = 335.48 h for
        Cs-131 (T_half = 9.689 d = 232.536 h). See module docstring, fix 5:
        v1's 335.23 * 3.5 is the same product with a slightly stale tau and a
        hard-coded 3.5 U air-kerma strength.
        """
        return np.asarray(rate, dtype=float) * float(sk_per_seed_u) \
            * self.TAU_HOURS

    def total_dose_at_point(self, theta_deg, r_cm, sk_per_seed_u=DEFAULT_SK_U):
        """Total-to-decay dose [cGy] at (theta, r) for one seed of S_K [U]."""
        out = self.dose_to_total_decay(self.dose_rate(theta_deg, r_cm),
                                       sk_per_seed_u)
        return float(out) if out.ndim == 0 else out


# ------------------------------------------------------------ shared plumbing
def _prepare_seeds(seed_centers, seed_axes, sk_per_seed_u):
    centers = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
    axes = np.asarray(seed_axes, dtype=float).reshape(-1, 3)
    if centers.shape != axes.shape:
        raise ValueError("seed_centers and seed_axes must both be (N, 3)")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("seed_axes contains a zero vector")
    axes = axes / norms[:, None]
    sk = np.broadcast_to(np.asarray(sk_per_seed_u, dtype=float),
                         (centers.shape[0],)).astype(float)
    if np.any(sk < 0.0):
        raise ValueError("sk_per_seed_u must be non-negative")
    return centers, axes, sk


def _sum_seed_rates(points_mm, centers, axes, sk, eng, exact,
                    interference=None):
    """S_K-weighted sum of per-seed rates at (M, 3) RAS points [cGy/h].

    With an ``interference`` model, each seed's rate is multiplied by the
    line-of-sight transmission from that seed to the point before it is
    summed -- the one place superposition can express occluders at all.  The
    ray geometry is already in hand here, so it is handed over rather than
    recomputed inside the model.
    """
    acc = np.zeros(points_mm.shape[0], dtype=np.float64)
    rate_fn = eng._rate_folded if exact else eng._rate_tabulated_folded
    for s in range(centers.shape[0]):
        if sk[s] == 0.0:
            continue
        d = points_mm - centers[s]
        r2 = np.einsum("ij,ij->i", d, d)
        z = d @ axes[s]                                  # signed axial, mm
        y = np.sqrt(np.maximum(r2 - z * z, 0.0))        # perpendicular, mm
        # theta measured from the long axis, already folded into [0, 90]:
        # a point exactly on the centre (y = z = 0) lands on 90 deg, the
        # transverse plane, and r floors anyway.
        theta = np.degrees(np.arctan2(y, np.abs(z)))
        r_mm = np.sqrt(r2)
        r_cm = np.maximum(r_mm / 10.0, eng.R_FLOOR_CM)
        rate = rate_fn(theta, r_cm)
        if interference is not None:
            rate = rate * interference.transmission_cached(
                s, points_mm, d, r_mm)
        acc += rate * sk[s]
    return acc


# ---------------------------------------------------------------------- grids
def compute_dose_grid(seed_centers, seed_axes, bounds_ras, spacing_mm=2.0,
                      sk_per_seed_u=TG43Engine.DEFAULT_SK_U,
                      engine=None, max_chunk_points=262144,
                      elapsed_hours=None, exact=False, interference=None):
    """Dose [cGy] on an axis-aligned RAS grid.

    Parameters
    ----------
    seed_centers : (N, 3) array
        Seed centres, RAS mm.
    seed_axes : (N, 3) array
        Seed long-axis directions, RAS (normalized internally).
    bounds_ras : (2, 3) array
        Min/max grid corner, RAS mm; grid voxel centres start at the min
        corner and step by ``spacing_mm`` up to (at most) the max corner.
    spacing_mm : float
        Isotropic voxel size in mm.
    sk_per_seed_u : float or (N,) array
        Air-kerma strength per seed [U] on the implant day — from the assay
        certificate, decayed with :meth:`TG43Engine.sk_decayed` if needed.
    engine : TG43Engine, optional
        Reuse an existing engine (tables precomputed).
    max_chunk_points : int
        Grid points per k-slab chunk; bounds peak memory to a few float64
        arrays of this length per seed evaluation.
    elapsed_hours : float, optional
        Report the dose delivered by this many hours after implant instead
        of the total-to-decay dose (``None``).
    exact : bool
        Use the analytic rate instead of the tabulated kernel.
    interference : InterferenceModel, optional
        Inter-seed / tile-carrier attenuation
        (:mod:`gtcore.dose.interference`).  ``None`` (the default) is plain
        TG-43 superposition -- every seed alone in water -- which is what the
        formalism is defined for and what the regression tests pin.  Given a
        model, each seed's dose rate is multiplied by the line-of-sight
        transmission to the field point before summing.  The model's capsule
        order must match ``seed_centers``; that is checked here, not assumed.

    Returns
    -------
    Volume
        ``array[k, j, i]`` of dose in cGy, affine mapping (i, j, k) -> RAS.
    """
    centers, axes, sk = _prepare_seeds(seed_centers, seed_axes, sk_per_seed_u)

    bounds = np.asarray(bounds_ras, dtype=float).reshape(2, 3)
    lo, hi = bounds[0], bounds[1]
    if np.any(hi < lo):
        raise ValueError("bounds_ras must be [min_corner, max_corner]")
    spacing = float(spacing_mm)
    if spacing <= 0.0:
        raise ValueError("spacing_mm must be positive")

    n_xyz = np.floor((hi - lo) / spacing + 1.0e-9).astype(int) + 1
    nx, ny, nz = int(n_xyz[0]), int(n_xyz[1]), int(n_xyz[2])
    xs = lo[0] + spacing * np.arange(nx)
    ys = lo[1] + spacing * np.arange(ny)
    zs = lo[2] + spacing * np.arange(nz)

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = spacing
    affine[:3, 3] = lo

    eng = engine if engine is not None else TG43Engine()
    if interference is not None:
        interference.validate_against(centers)
    fraction = float(eng.delivered_fraction(elapsed_hours))
    scale = eng.TAU_HOURS * fraction

    dose = np.zeros((nz, ny, nx), dtype=np.float64)
    # Chunk over k-slabs so peak memory stays bounded regardless of grid size.
    slab_k = max(1, int(max_chunk_points // max(1, nx * ny)))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")          # (ny, nx)

    for k0 in range(0, nz, slab_k):
        k1 = min(nz, k0 + slab_k)
        zz = zs[k0:k1]                                    # (kz,)
        pts = np.empty((k1 - k0, ny, nx, 3), dtype=np.float64)
        pts[..., 0] = xx[None, :, :]
        pts[..., 1] = yy[None, :, :]
        pts[..., 2] = zz[:, None, None]
        flat = pts.reshape(-1, 3)
        acc = _sum_seed_rates(flat, centers, axes, sk, eng, exact,
                              interference=interference)
        dose[k0:k1] = (acc * scale).reshape(k1 - k0, ny, nx)

    meta = {
        "units": "cGy",
        "kind": ("tg43_total_decay_dose" if elapsed_hours is None
                 else "tg43_dose_at_time"),
        "spacing_mm": spacing,
        "n_seeds": int(centers.shape[0]),
        "sk_per_seed_u": np.asarray(sk).tolist(),
        "tau_hours": eng.TAU_HOURS,
        "elapsed_hours": None if elapsed_hours is None else float(elapsed_hours),
        "delivered_fraction": fraction,
        "kernel": "exact" if exact else "table",
        "interference": (None if interference is None
                         else interference.describe()),
    }
    return Volume(dose, affine, meta)


def dose_at_points(seed_centers, seed_axes, points_ras,
                   sk_per_seed_u=TG43Engine.DEFAULT_SK_U, engine=None,
                   elapsed_hours=None, exact=True, max_chunk_points=262144,
                   interference=None):
    """Dose [cGy] at arbitrary RAS points, no grid.

    Same seed / S_K / elapsed-time / ``interference`` semantics as
    :func:`compute_dose_grid`; defaults to the exact analytic rate. Returns
    ``(M,)`` for ``(M, 3)`` input or a float for a single point.
    """
    centers, axes, sk = _prepare_seeds(seed_centers, seed_axes, sk_per_seed_u)
    pts = np.asarray(points_ras, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts).reshape(-1, 3)
    eng = engine if engine is not None else TG43Engine()
    if interference is not None:
        interference.validate_against(centers)
    scale = eng.TAU_HOURS * float(eng.delivered_fraction(elapsed_hours))
    out = np.empty(pts.shape[0], dtype=np.float64)
    step = max(1, int(max_chunk_points))
    for i0 in range(0, pts.shape[0], step):
        chunk = pts[i0:i0 + step]
        out[i0:i0 + step] = _sum_seed_rates(chunk, centers, axes, sk, eng,
                                            exact,
                                            interference=interference) * scale
    return float(out[0]) if single else out


# -------------------------------------------------------------------- isodose
def isodose_surfaces(dose_volume, levels_cgy, smooth_iterations=0):
    """Extract one triangle mesh per isodose level from a dose Volume.

    Marching cubes runs on the *log-dose scalar field* at ``ln(level)``, not
    on a thresholded mask: the isosurface is then positioned by
    interpolation along each grid edge (sub-voxel accurate) and, because
    dose falls as ~1/r^2, interpolating in log space makes that placement
    nearly exact even on a 2 mm grid. A thresholded mask, by contrast, can
    only put the surface at half-voxel positions (up to 1 mm off at 2 mm).

    Parameters
    ----------
    dose_volume : Volume
        Dose grid (cGy), e.g. from :func:`compute_dose_grid`.
    levels_cgy : sequence of float
        Isodose levels in cGy; each surface encloses ``dose >= level``.
    smooth_iterations : int
        Optional Taubin passes (shrink-free). The scalar isosurface is
        already smooth, so this defaults to 0.

    Returns
    -------
    dict {level: trimesh.Trimesh}
        Normals point outward, away from the high-dose region. A level above
        the grid maximum (or below its minimum) yields an empty mesh. All
        connected shells are kept deliberately: a prescription isodose
        around spatially separated tiles is legitimately multiple closed
        shells, and dropping all but the largest would hide coverage gaps.
    """
    if not isinstance(dose_volume, Volume):
        raise TypeError("dose_volume must be a gtcore Volume")
    arr = np.asarray(dose_volume.array, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    log_arr = np.log(np.maximum(arr, tiny)).astype(np.float32)
    lo, hi = float(log_arr.min()), float(log_arr.max())
    affine = dose_volume.affine

    surfaces: Dict[float, trimesh.Trimesh] = {}
    for level in levels_cgy:
        lv = float(level)
        if not (lv > 0.0):
            raise ValueError("isodose levels must be positive, got %r" % (level,))
        log_lv = math.log(lv)
        if not (lo < log_lv < hi):
            surfaces[lv] = trimesh.Trimesh()
            continue
        verts_kji, faces, _n, _v = measure.marching_cubes(log_arr, level=log_lv)
        verts_ras = apply_affine(affine, verts_kji[:, ::-1])   # [k,j,i]->(i,j,k)
        mesh = trimesh.Trimesh(vertices=verts_ras, faces=faces, process=True)
        if smooth_iterations and len(mesh.faces):
            trimesh.smoothing.filter_taubin(mesh,
                                            iterations=int(smooth_iterations))
        if len(mesh.faces):
            mesh.fix_normals(multibody=True)
        surfaces[lv] = mesh
    return surfaces
