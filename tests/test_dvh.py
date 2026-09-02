"""Shell DVH maths behind the planner's dose panel (pure numpy/trimesh).

Scored on an analytic 1/r^2 field around a sphere so every number has a
closed-form expectation: the wall shell sits at r=10 mm, the +5 mm shell at
r=15 mm, and dose falls as (r0/r)^2.
"""
from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from gtcore.dose.dvh import (
    dvh_curve,
    dvh_stats,
    format_report,
    outward_normals,
    sample_doses,
    shell_points,
    shell_report,
)
from gtcore.volume import Volume

R_SPHERE = 10.0
D0 = 4000.0  # dose at r = R_SPHERE


@pytest.fixture(scope="module")
def sphere():
    return trimesh.creation.icosphere(subdivisions=3, radius=R_SPHERE)


@pytest.fixture(scope="module")
def radial_dose():
    """dose(r) = D0 * (R/r)^2 on a 1 mm grid spanning +-30 mm."""
    ax = np.arange(-30.0, 30.0 + 1e-9, 1.0)
    kk, jj, ii = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(ii ** 2 + jj ** 2 + kk ** 2)
    dose = D0 * (R_SPHERE / np.maximum(r, 1.0)) ** 2
    affine = np.eye(4)
    affine[:3, 3] = [-30.0, -30.0, -30.0]
    return Volume(dose, affine)


def test_outward_normals_point_away_from_centroid(sphere):
    n = outward_normals(sphere)
    v = np.asarray(sphere.vertices)
    assert np.all(np.einsum("ij,ij->i", n, v) > 0.0)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    # a flipped mesh (inward normals) must give the same answer
    flipped = sphere.copy()
    flipped.invert()
    assert np.allclose(outward_normals(flipped), n, atol=1e-6)


def test_shell_points_offset_radially(sphere):
    r0 = np.linalg.norm(shell_points(sphere, 0.0), axis=1)
    r5 = np.linalg.norm(shell_points(sphere, 5.0), axis=1)
    assert np.allclose(r0, R_SPHERE, atol=1e-6)
    assert np.allclose(r5, R_SPHERE + 5.0, atol=0.05)


def test_sample_doses_matches_field_and_zero_outside(radial_dose):
    pts = np.array([[R_SPHERE, 0, 0], [0, 20.0, 0], [200.0, 0, 0]])
    d = sample_doses(radial_dose, pts)
    assert d[0] == pytest.approx(D0, rel=0.02)
    assert d[1] == pytest.approx(D0 / 4.0, rel=0.02)
    assert d[2] == 0.0, "points outside the grid read 0 cGy"
    assert sample_doses(radial_dose, np.zeros((0, 3))).shape == (0,)


def test_dvh_curve_and_stats_closed_form():
    doses = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(dvh_curve(doses, [0.5, 1.0, 2.5, 4.0, 4.1]),
                       [1.0, 1.0, 0.5, 0.25, 0.0])
    assert np.allclose(dvh_curve([], [1.0, 2.0]), [0.0, 0.0])

    s = dvh_stats(doses, rx_cgy=2.0)
    assert s["Dmin"] == 1.0 and s["Dmax"] == 4.0 and s["Dmean"] == 2.5
    assert s["D50"] == pytest.approx(np.percentile(doses, 50))
    assert s["D90"] == pytest.approx(np.percentile(doses, 10))
    assert s["V100"] == 0.75   # >= 2
    assert s["V150"] == 0.5    # >= 3
    assert s["V200"] == 0.25   # >= 4
    assert s["n"] == 4

    empty = dvh_stats([], rx_cgy=2.0)
    assert all(v == 0.0 for v in empty.values())


def test_shell_report_orders_shells_and_tracks_falloff(sphere, radial_dose):
    rx = 3000.0
    rep = shell_report(radial_dose, sphere, rx, offsets_mm=(0.0, 5.0, 10.0))
    assert list(rep) == [0.0, 5.0, 10.0]
    wall, s5, s10 = (rep[k]["stats"] for k in (0.0, 5.0, 10.0))
    # analytic: 4000 on the wall, 1778 at +5 mm, 1000 at +10 mm
    assert wall["D50"] == pytest.approx(D0, rel=0.03)
    assert s5["D50"] == pytest.approx(D0 * (10 / 15) ** 2, rel=0.03)
    assert s10["D50"] == pytest.approx(D0 * (10 / 20) ** 2, rel=0.03)
    assert wall["V100"] == pytest.approx(1.0)
    assert s5["V100"] == 0.0 and s10["V100"] == 0.0
    # curves are cumulative: non-increasing, start at 100 %
    for entry in rep.values():
        y = entry["curve_y"]
        assert y[0] == 1.0
        assert np.all(np.diff(y) <= 1e-12)
        assert entry["curve_x"].shape == y.shape


def test_format_report_is_fixed_width_table(sphere, radial_dose):
    rep = shell_report(radial_dose, sphere, 3000.0, offsets_mm=(0.0, 5.0))
    text = format_report(rep, 3000.0)
    lines = text.splitlines()
    assert lines[0].startswith("shell")
    assert lines[1].startswith("wall") and "100%" in lines[1]
    assert lines[2].startswith("+5 mm")
    assert len(lines) == 3
