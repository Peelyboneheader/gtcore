"""Thick-slice axis term of the DEFAULT bent-tile fit (docs/localization-notes,
stage 5 follow-up): above ``AXES_RELIABLE_DZ_MM`` the PCA seed axis of a
capsule inside one slab is degenerate, so the unweighted bent-tile fit now
drops the axis residual term there too (``auto.DROP_AXIS_TERM_ON_COARSE``),
exactly as the weighted path and the cover pass already did.

Acceptance: thin slices are bit-identical with the switch on or off;
``fit_deformable(use_axes=False)`` is the ``seed_axes=None`` fit; on a
coarse scan every bent-tile fit of ``fit_tiles_auto`` / ``fit_tiles_prior``
/ ``fit_tiles(score="deformable")`` reports ``axes_fitted=False`` with the
switch on and ``True`` with it off.  The behaviour tests set the switch
explicitly, so only ``test_shipped_default`` changes if it is reverted.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from gtcore.phantom import make_head_phantom
from gtcore.tiles import ImplantPrior, fit_tiles, fit_tiles_auto, fit_tiles_prior
from gtcore.tiles import auto as auto_mod
from gtcore.tiles.deform import fit_deformable

THIN = (0.7, 0.7, 0.7)
COARSE = (0.7, 0.7, 2.1)


@pytest.fixture(scope="module")
def truth_tiles():
    _vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=3)
    rng = np.random.default_rng(0)
    c = np.array([s.center_ras for s in truth.seeds]) + rng.normal(scale=0.3, size=(12, 3))
    a = np.array([s.axis_ras for s in truth.seeds]) + rng.normal(scale=0.2, size=(12, 3))
    return c, a, truth


def _same_fit(f, g):
    return (np.array_equal(f.pose.R, g.pose.R) and np.array_equal(f.pose.t, g.pose.t)
            and f.params == g.params and f.assignment == g.assignment
            and np.array_equal(f.residuals_mm, g.residuals_mm)
            and f.axis_err_deg == g.axis_err_deg and f.n_evals == g.n_evals)


def test_use_axes_false_is_the_axis_free_fit(truth_tiles):
    c, a, _ = truth_tiles
    with_axes = fit_deformable(c[:4], a[:4])
    off = fit_deformable(c[:4], a[:4], use_axes=False)
    none = fit_deformable(c[:4], None)
    assert _same_fit(off, none) and not _same_fit(off, with_axes)
    assert with_axes.axes_fitted and not off.axes_fitted and not none.axes_fitted
    assert off.axis_err_deg == 0.0


def test_shipped_default():
    # gate: docs/localization-notes.md, "axis term"; flip here if reverted
    assert auto_mod.DROP_AXIS_TERM_ON_COARSE is True


def _poses(res):
    return [(tuple(p.seed_indices), p.deform.pose.R.tobytes(),
             p.deform.pose.t.tobytes(), p.deform.params, p.confidence)
            for p in res.all_tiles]


def test_thin_slices_are_bit_identical_either_way(truth_tiles):
    c, a, truth = truth_tiles
    kw = dict(cavity_center_ras=truth.cavity_center_ras, spacing_mm=THIN)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", True):
        on_auto = fit_tiles_auto(c, a, **kw)
        on_prior = fit_tiles_prior(c, a, ImplantPrior(n_full=3), **kw)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", False):
        off_auto = fit_tiles_auto(c, a, **kw)
        off_prior = fit_tiles_prior(c, a, ImplantPrior(n_full=3), **kw)
    assert _poses(on_auto) == _poses(off_auto)
    assert _poses(on_prior) == _poses(off_prior)
    assert all(p.deform.axes_fitted for p in on_auto.all_tiles + on_prior.all_tiles)
    # no spacing given = thin-cut calibration: axes kept
    assert all(p.deform.axes_fitted for p in fit_tiles_auto(c, a).all_tiles)


def test_coarse_slices_drop_the_axis_term_on_every_path(truth_tiles):
    c, a, truth = truth_tiles
    kw = dict(cavity_center_ras=truth.cavity_center_ras, spacing_mm=COARSE)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", True):
        results = [fit_tiles_auto(c, a, **kw),
                   fit_tiles_prior(c, a, ImplantPrior(n_full=3), **kw),
                   fit_tiles(c, a, 3, cavity_center_ras=truth.cavity_center_ras,
                             spacing_mm=COARSE, score="deformable")]
    for res in results:
        assert res.tiles
        assert all(p.deform is not None and not p.deform.axes_fitted
                   for p in res.tiles)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", False):
        res_off = fit_tiles_auto(c, a, **kw)
        counted_off = fit_tiles(c, a, 3, cavity_center_ras=truth.cavity_center_ras,
                                spacing_mm=COARSE, score="deformable")
    assert all(p.deform.axes_fitted for p in res_off.all_tiles)
    assert all(p.deform.axes_fitted for p in counted_off.tiles)
    # the weighted path drops them regardless of the switch
    cov = np.repeat(np.diag([0.05, 0.05, 0.4])[None], 12, axis=0)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", False):
        res_w = fit_tiles_auto(c, a, seed_cov=cov, **kw)
    assert all(not p.deform.axes_fitted for p in res_w.all_tiles)
    # the counted chord score never fits the bent tile inside the search
    chord = fit_tiles(c, a, 3, cavity_center_ras=truth.cavity_center_ras,
                      spacing_mm=COARSE)
    assert all(p.deform is None for p in chord.tiles)
