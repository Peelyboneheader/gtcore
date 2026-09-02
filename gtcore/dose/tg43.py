"""TG-43U1 dosimetry for the IsoRay Proxcelan CS-1 Rev2 Cs-131 seed.

Ported verbatim from the ``DoseInterpolator`` class of the **GammaView
Dosimetry** 3D Slicer module (``GammaTile.py``), which is the origin of this
implementation and of every constant and table below. The port strips the
Slicer/VTK/Qt host dependency -- this module needs only numpy -- but the
numerical behaviour is unchanged: the dose-rate constant, the active length,
the CLRP v2 polynomial fit for ``g_L(r)``, and the 32 x 12 anisotropy table
``F(r, theta)`` are byte-for-byte the same values, and the interpolation and
clipping logic is reproduced exactly.

Formalism (AAPM TG-43U1, 2D line-source):

    D(r, theta) = Lambda * [G_L(r, theta) / G_L(r0, theta0)]
                         * g_L(r) * F(r, theta)

with ``r0 = 1 cm`` and ``theta0 = 90 deg``. Distances are in centimetres and
angles in degrees, measured from the seed's long axis.
"""
from __future__ import annotations

import numpy as np

__all__ = ["DoseInterpolator"]


class DoseInterpolator:
    """TG-43U1 2D formalism for IsoRay Proxcelan CS-1 Rev2 (L=0.40 cm)."""

    DOSE_CONVERSION_FACTOR = 335.23 * 3.5  # cGy per air kerma strength; math was derived
    _LAMBDA = 1.056 #Dose rate constant (cGy h^-1 U^-1)
    _L = 0.40 # Active source length (cm)
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

    def __init__(self):
        # Precompute reference geometry factor G_L(r0=1.0, theta0=90°)
        self._GL_ref = self._geometry_factor(1.0, 90.0)

    def _geometry_factor(self, r, theta_deg):
        """Line-source geometry function G_L(r, theta)."""
        L = self._L
        theta_rad = np.radians(theta_deg)
        if abs(theta_deg) < 0.01:
            # On-axis approximation
            return 1.0 / (r * r - (L / 2.0) ** 2)
        sin_t = np.sin(theta_rad)
        cos_t = np.cos(theta_rad)
        r_sin = r * sin_t
        r_cos = r * cos_t
        beta = np.arctan2(r_sin, r_cos - L / 2.0) - np.arctan2(r_sin, r_cos + L / 2.0)
        return beta / (L * r * sin_t)

    def _radial_dose(self, r):
        # g_L(r) — Eq. 3 from TG-43U1, polynomial fit from CLRP v2
        r = float(np.clip(r, 0.05, 10.0))
        a = self._GR_COEFFS
        return (a[0] * r**-2 + a[1] * r**-1 + a[2] + a[3] * r
                + a[4] * r**2 + a[5] * r**3) * np.exp(-a[6] * r)

    def _anisotropy(self, r, theta_deg):
        """Anisotropy function F(r, theta) via bilinear interpolation on hardcoded table."""
        thetas = self._F_THETAS
        radii = self._F_RADII
        table = self._F_TABLE

        theta_c = float(np.clip(theta_deg, thetas[0], thetas[-1]))
        r_c = float(np.clip(r, radii[0], radii[-1]))
        it = int(np.searchsorted(thetas, theta_c, side='right')) - 1
        it = max(0, min(it, len(thetas) - 2))
        it2 = it + 1
        ir = int(np.searchsorted(radii, r_c, side='right')) - 1
        ir = max(0, min(ir, len(radii) - 2))
        ir2 = ir + 1

        def _val(ti, ri):
            v = table[ti, ri]
            if np.isnan(v):
                for k in range(ri + 1, len(radii)):
                    if not np.isnan(table[ti, k]):
                        return table[ti, k]
                for k in range(ri - 1, -1, -1):
                    if not np.isnan(table[ti, k]):
                        return table[ti, k]
                return 1.0
            return v

        v00 = _val(it, ir)
        v01 = _val(it, ir2)
        v10 = _val(it2, ir)
        v11 = _val(it2, ir2)

        # Interpolation fractions
        dr = radii[ir2] - radii[ir]
        t_r = (r_c - radii[ir]) / dr if dr > 1e-12 else 0.0

        dt = thetas[it2] - thetas[it]
        t_t = (theta_c - thetas[it]) / dt if dt > 1e-12 else 0.0

        # Bilinear
        top = v00 * (1.0 - t_r) + v01 * t_r
        bot = v10 * (1.0 - t_r) + v11 * t_r
        return top * (1.0 - t_t) + bot * t_t


    def dose_at_point(self, angle_deg, radius_cm):
        """TG-43U1 dose at a point, returns cGy."""
        angle_deg = angle_deg % 360
        if angle_deg > 180:
            angle_deg = 360 - angle_deg
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        angle_deg = float(np.clip(angle_deg, 0, 90))
        radius_cm_eff = float(np.clip(radius_cm, 0.041, 10.0))

        rate = self._tg43_dose_rate(angle_deg, radius_cm_eff)
        return self.DOSE_CONVERSION_FACTOR * max(0.0, rate)

    def _tg43_dose_rate(self, theta_deg, r):
        """Core TG-43U1 2D calculation: Lambda * [G_L(r,theta)/G_L(r0,theta0)] * g_L(r) * F(r,theta)."""
        GL = self._geometry_factor(r, theta_deg)
        gL = self._radial_dose(r)
        F = self._anisotropy(r, theta_deg)
        return self._LAMBDA * (GL / self._GL_ref) * gL * F
