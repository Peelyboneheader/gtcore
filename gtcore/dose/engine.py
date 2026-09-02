"""Corrected, vectorized TG-43U1 dose engine (step v).

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
   0.8 mm-diameter CS-1 capsule radius) at axial position ``<= L/2`` — clamp
   to the geometry value at the capsule surface at the same axial position,
   so dose on the long axis is positive, finite and monotone instead of the
   v1 behaviour (negative G clamped to exactly 0 out to ~2 mm).
2. **Theta folding** (mod 360, mirror about 180 and 90) happens inside the
   core rate function, so every caller — including the vectorized grid path —
   is symmetric by construction.
3. **Anisotropy NaN holes** (r = 0.10, 0.15 cm at theta <= 15 deg) are filled
   ONCE at init by the nearest valid value along r at the same theta (the
   holes are the leading columns, so "nearest valid" is the first tabulated
   radius with data). This reproduces v1's dynamic scan-right fill exactly
   but is explicit, precomputed and documented. The underlying table still
   deserves replacement with published consensus data — see port notes.
4. **One radial clip**: ``r`` is clipped to [0.05, 10.0] cm exactly once, in
   the core rate function, and the clipped value feeds G_L, g_L and F alike.
   (F interpolation additionally clamps to its own table domain
   [0.10, 10.0] cm — that is domain clamping of the lookup, not a second dose
   clip, and v1 did the same.)
5. **Explicit dose conversion.** ``dose_to_total_decay(rate, sk_per_seed_u)``
   = ``rate * S_K * tau`` with ``tau = T_half / ln 2``,
   ``T_half(Cs-131) = 9.689 d = 232.536 h`` so ``tau = 335.48 h`` (~335.5 h).
   v1's opaque ``DOSE_CONVERSION_FACTOR = 335.23 * 3.5`` is exactly
   ``tau_v1 * S_K`` with ``S_K = 3.5 U`` baked in: 335.23 h is a slightly
   stale value of the same integral-to-total-decay tau (0.08 % below
   232.536 / ln 2), and 3.5 U is a *per-implant assay quantity*, not a
   physical constant — here it is an explicit parameter (default 3.5 U) that
   should come from the implant's assay certificate.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import trimesh

from ..segment.surface import mask_to_mesh
from ..volume import Volume

__all__ = ["TG43Engine", "compute_dose_grid", "isodose_surfaces"]


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
    #: One consistent radial clip [cm], applied once in the core rate function.
    R_CLIP_CM = (0.05, 10.0)
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

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _fold_theta(theta_deg):
        """Fold any angle into [0, 90] deg (mod 360, mirror at 180 and 90)."""
        t = np.mod(theta_deg, 360.0)
        t = np.where(t > 180.0, 360.0 - t, t)
        t = np.where(t > 90.0, 180.0 - t, t)
        return t

    def _geometry_factor(self, r_cm, theta_deg):
        """Line-source G_L(r, theta), vectorized. theta must be in [0, 90].

        General case: G_L = beta / (L * r * sin(theta)) with beta the angle
        the active line subtends at the field point,
        beta = atan2(y, z - L/2) - atan2(y, z + L/2),
        where y = r sin(theta) (perpendicular) and z = r cos(theta) (axial).
        Special cases per fix 1 of the module docstring.
        """
        r = np.asarray(r_cm, dtype=float)
        th = np.radians(np.asarray(theta_deg, dtype=float))
        L = self._L
        half = L / 2.0

        sin_t = np.sin(th)
        cos_t = np.abs(np.cos(th))          # theta in [0, 90] -> cos >= 0
        y = r * sin_t                       # perpendicular distance to axis
        z = r * cos_t                       # |axial| position

        # Inside the physical capsule: clamp to the surface value at same z.
        inside = (y < self._RHO_SURFACE) & (z <= self._CAP_HALF_CM)
        y_eff = np.where(inside, self._RHO_SURFACE, y)
        # On/near the long axis beyond the physical tip.
        near_axis = (~inside) & (y < self._Y_EPS)

        y_safe = np.maximum(y_eff, self._Y_EPS)   # keep atan2/div well-posed
        beta = np.arctan2(y_safe, z - half) - np.arctan2(y_safe, z + half)
        G = beta / (L * y_safe)

        # Analytic on-axis limit, valid ONLY for r > L/2 (positive, finite).
        denom = np.maximum(z * z - half * half, 1.0e-12)
        return np.where(near_axis, 1.0 / denom, G)

    def _radial_dose(self, r_cm):
        """g_L(r) — CLRP v2 polynomial fit, vectorized. Expects clipped r."""
        r = np.asarray(r_cm, dtype=float)
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
    def dose_rate(self, theta_deg, r_cm):
        """TG-43U1 2D dose rate [cGy h^-1 U^-1], vectorized.

        Fix 2: theta is folded into [0, 90] HERE, so the core is exactly
        symmetric for every caller. Fix 4: r is clipped to R_CLIP_CM once,
        and the same clipped radius feeds G_L, g_L and F.
        """
        th_in = np.asarray(theta_deg, dtype=float)
        r_in = np.asarray(r_cm, dtype=float)
        scalar = th_in.ndim == 0 and r_in.ndim == 0
        th, r = np.broadcast_arrays(np.atleast_1d(th_in), np.atleast_1d(r_in))

        th_f = self._fold_theta(th)
        r_c = np.clip(r, self.R_CLIP_CM[0], self.R_CLIP_CM[1])

        GL = self._geometry_factor(r_c, th_f)
        gL = self._radial_dose(r_c)
        F = self._anisotropy(r_c, th_f)
        rate = self._LAMBDA * (GL / self._GL_ref) * gL * F
        return float(rate[0]) if scalar else rate

    # -------------------------------------------------------------- conversion
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


# ---------------------------------------------------------------------- grids
def compute_dose_grid(seed_centers, seed_axes, bounds_ras, spacing_mm=2.0,
                      sk_per_seed_u=TG43Engine.DEFAULT_SK_U,
                      engine=None, max_chunk_points=262144,
                      interference=None):
    """Total-to-decay dose [cGy] on an axis-aligned RAS grid.

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
        Air-kerma strength per seed [U] — from the assay certificate.
    engine : TG43Engine, optional
        Reuse an existing engine (table fill / reference factor precomputed).
    max_chunk_points : int
        Grid points per k-slab chunk; bounds peak memory to a few float64
        arrays of this length per seed evaluation.
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
    centers = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
    axes = np.asarray(seed_axes, dtype=float).reshape(-1, 3)
    if centers.shape != axes.shape:
        raise ValueError("seed_centers and seed_axes must both be (N, 3)")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("seed_axes contains a zero vector")
    axes = axes / norms[:, None]

    sk = np.broadcast_to(np.asarray(sk_per_seed_u, dtype=float),
                         (centers.shape[0],))

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

    dose = np.zeros((nz, ny, nx), dtype=np.float64)
    # Chunk over k-slabs so peak memory stays bounded regardless of grid size.
    slab_k = max(1, int(max_chunk_points // max(1, nx * ny)))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")          # (ny, nx)

    for k0 in range(0, nz, slab_k):
        k1 = min(nz, k0 + slab_k)
        zz = zs[k0:k1]                                    # (kz,)
        # Points of this slab: (kz, ny, nx, 3)
        pts = np.empty((k1 - k0, ny, nx, 3), dtype=np.float64)
        pts[..., 0] = xx[None, :, :]
        pts[..., 1] = yy[None, :, :]
        pts[..., 2] = zz[:, None, None]
        flat = pts.reshape(-1, 3)

        acc = np.zeros(flat.shape[0], dtype=np.float64)
        for s in range(centers.shape[0]):
            d = flat - centers[s]
            r_mm = np.sqrt(np.einsum("ij,ij->i", d, d))
            safe = np.maximum(r_mm, 1.0e-12)
            cos_t = np.clip(d @ axes[s] / safe, -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_t))
            # A point exactly on a seed centre has no defined angle; use the
            # transverse plane (r clips to the floor anyway).
            theta = np.where(r_mm < 1.0e-12, 90.0, theta)
            rate = eng.dose_rate(theta, r_mm / 10.0)      # mm -> cm
            if interference is not None:
                # Ray geometry is already in hand; hand it over rather than
                # recomputing (P, 3) differences and norms per seed.
                rate = rate * interference.transmission_cached(
                    s, flat, d, r_mm)
            acc += rate * sk[s]

        dose[k0:k1] = (acc * eng.TAU_HOURS).reshape(k1 - k0, ny, nx)

    meta = {
        "units": "cGy",
        "kind": "tg43_total_decay_dose",
        "spacing_mm": spacing,
        "n_seeds": int(centers.shape[0]),
        "sk_per_seed_u": np.asarray(sk).tolist(),
        "tau_hours": eng.TAU_HOURS,
        "interference": (None if interference is None
                         else interference.describe()),
    }
    return Volume(dose, affine, meta)


# -------------------------------------------------------------------- isodose
def isodose_surfaces(dose_volume, levels_cgy, smooth_iterations=5):
    """Extract one triangle mesh per isodose level from a dose Volume.

    Parameters
    ----------
    dose_volume : Volume
        Total-decay dose grid (cGy), e.g. from :func:`compute_dose_grid`.
    levels_cgy : sequence of float
        Isodose levels in cGy; each level thresholds ``dose >= level``.
    smooth_iterations : int
        Light Taubin smoothing (shrink-free) applied by ``mask_to_mesh``.

    Returns
    -------
    dict {level: trimesh.Trimesh}
        ``largest_only=False`` deliberately: a prescription isodose around
        spatially separated tiles is legitimately multiple closed shells,
        and dropping all but the largest would hide coverage gaps.
    """
    if not isinstance(dose_volume, Volume):
        raise TypeError("dose_volume must be a gtcore Volume")
    surfaces: Dict[float, trimesh.Trimesh] = {}
    arr = dose_volume.array
    for level in levels_cgy:
        mask = arr >= float(level)
        surfaces[float(level)] = mask_to_mesh(
            mask, dose_volume.affine,
            smooth_iterations=int(smooth_iterations),
            largest_only=False,
        )
    return surfaces
