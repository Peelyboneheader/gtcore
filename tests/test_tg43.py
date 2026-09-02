"""TG-43U1 sanity checks for the ported DoseInterpolator.

These are normalization identities, not a dosimetric validation: at the
reference point (r0 = 1 cm, theta0 = 90 deg) the geometry ratio is 1 by
construction, g_L(1) ~ 1 by the fit, and F(1, 90) = 1 from the table, so the
dose rate must come back at Lambda.
"""
from __future__ import annotations

import numpy as np
import pytest

from gtcore.dose import DoseInterpolator


@pytest.fixture(scope="module")
def calc():
    return DoseInterpolator()


def test_reference_point_returns_lambda(calc):
    """D(r0, theta0) == Lambda to within 2%."""
    rate = calc._tg43_dose_rate(90.0, 1.0)
    lam = DoseInterpolator._LAMBDA
    assert rate == pytest.approx(lam, rel=0.02), (
        "rate=%.6f vs Lambda=%.6f (%.2f%%)" % (rate, lam, 100 * (rate / lam - 1))
    )


def test_reference_point_factors_are_unity(calc):
    """Each of the three factors is individually ~1 at the reference point."""
    assert calc._geometry_factor(1.0, 90.0) / calc._GL_ref == pytest.approx(1.0)
    assert calc._anisotropy(1.0, 90.0) == pytest.approx(1.0, abs=1e-12)
    assert calc._radial_dose(1.0) == pytest.approx(1.0, rel=0.02)


def test_transverse_dose_falls_monotonically_with_radius(calc):
    radii = [0.5, 1.0, 2.0, 3.0, 5.0]
    rates = [calc._tg43_dose_rate(90.0, r) for r in radii]
    assert all(a > b for a, b in zip(rates, rates[1:])), rates
    assert all(r > 0 for r in rates)


def test_dose_at_point_falls_monotonically_with_radius(calc):
    radii = [0.5, 1.0, 2.0, 3.0, 5.0]
    doses = [calc.dose_at_point(90.0, r) for r in radii]
    assert all(a > b for a, b in zip(doses, doses[1:])), doses


@pytest.mark.parametrize("theta", [30.0, 60.0])
def test_polar_angle_folding_is_symmetric(calc, theta):
    """dose_at_point folds theta about 90 deg, so it is exactly symmetric.

    Note the private ``_tg43_dose_rate`` does NOT fold: it clips theta into
    [0, 90] only inside ``_anisotropy``, so calling it with theta > 90
    silently uses F = 1. The folding lives in ``dose_at_point``.
    """
    lo = calc.dose_at_point(theta, 1.0)
    hi = calc.dose_at_point(180.0 - theta, 1.0)
    assert lo == pytest.approx(hi, rel=0.01)
    assert lo > 0.0


@pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0])
def test_full_revolution_folding(calc, theta):
    """Angles are folded modulo 360 and mirrored about 0/90/180."""
    ref = calc.dose_at_point(theta, 1.0)
    for equivalent in (theta + 360.0, -theta, 360.0 - theta, 180.0 - theta):
        assert calc.dose_at_point(equivalent, 1.0) == pytest.approx(ref, rel=1e-12)


def test_anisotropy_reduces_dose_off_transverse(calc):
    """F(r, theta) < 1 near the long axis at the reference radius."""
    f_axis = calc._anisotropy(1.0, 0.0)
    f_transverse = calc._anisotropy(1.0, 90.0)
    assert f_axis < f_transverse
    assert 0.5 < f_axis < 1.0


def test_dose_at_point_applies_conversion_factor(calc):
    rate = calc._tg43_dose_rate(90.0, 1.0)
    dose = calc.dose_at_point(90.0, 1.0)
    assert dose == pytest.approx(DoseInterpolator.DOSE_CONVERSION_FACTOR * rate)


def test_radius_is_clipped_not_singular(calc):
    """Tiny and huge radii stay finite (the source clips to [0.041, 10] cm)."""
    for r in (0.0, 1e-6, 0.001, 50.0, 1e6):
        d = calc.dose_at_point(90.0, r)
        assert np.isfinite(d)
        assert d >= 0.0
    # Anything at or below the clip floor collapses to the same value.
    assert calc.dose_at_point(90.0, 0.0) == pytest.approx(
        calc.dose_at_point(90.0, 0.041)
    )
