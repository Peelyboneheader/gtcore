"""Stages 4 and 6 of docs/plan-localization.md (2026-10-08).

Stage 4 -- one scoring rule: ``fit_tiles(..., score="deformable")`` ranks
the gate-passing quads of the counted path with the bent-tile score auto
mode uses, so the counted and automatic partitions agree; the node cap of
the exact search is reported (``capped``) instead of degrading silently.

Stage 6 -- uncertainty outputs: the Gauss-Newton pose covariance of a
bent-tile fit (``DeformableFit.compute_uncertainty``) and the partition
margin of every selected tile (``margins=True``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gtcore.tiles.fit as fit_mod
from gtcore.phantom import make_head_phantom
from gtcore.tiles import ImplantPrior, fit_tiles, fit_tiles_auto, fit_tiles_prior
from gtcore.tiles.auto import AMBIGUOUS_MARGIN
from gtcore.tiles.deform import (DeformParams, deformed_seed_axes,
                                 deformed_seed_points, fit_deformable)
from gtcore.tiles.model import TilePose6

# The generator draws the tile layout (and so the truth seeds) before it
# rasterises anything: truth seeds do not depend on the voxel spacing.  The
# sweep therefore builds its "0.8 mm" cases on a 4 mm grid (13 s -> 1 s);
# test_truth_seeds_do_not_depend_on_spacing guards that shortcut.
_GRID_MM = 4.0
_TRUTH = {}


def _truth(rng_seed, n_tiles, spacing=_GRID_MM):
    key = (rng_seed, n_tiles, spacing)
    if key not in _TRUTH:
        _vol, truth = make_head_phantom(spacing=spacing, n_tiles=n_tiles,
                                        rng_seed=rng_seed)
        c = np.array([s.center_ras for s in truth.seeds])
        a = np.array([s.axis_ras for s in truth.seeds])
        _TRUTH[key] = (c, a, truth)
    return _TRUTH[key]


def _partition(result):
    return {frozenset(p.seed_indices) for p in result.tiles}


def test_truth_seeds_do_not_depend_on_spacing():
    c08, a08, t08 = _truth(4, 5, spacing=0.8)
    c40, a40, t40 = _truth(4, 5)
    assert np.allclose(c08, c40) and np.allclose(a08, a40)
    assert [t.seed_ids for t in t08.tiles] == [t.seed_ids for t in t40.tiles]


# ------------------------------------------------------------ stage 4
@pytest.mark.parametrize("rng_seed", range(6))
@pytest.mark.parametrize("n_tiles", range(1, 6))
def test_counted_deformable_partition_equals_auto(rng_seed, n_tiles):
    """Gate: counted = automatic partition on every auto synthetic case
    (rng 0-5 x 1-5 tiles, truth seeds).  Clean cases are never ambiguous."""
    c, a, truth = _truth(rng_seed, n_tiles)
    cav = truth.cavity_center_ras
    auto = fit_tiles(c, a, "auto", cavity_center_ras=cav, margins=True)
    counted = fit_tiles(c, a, n_tiles, 0, cavity_center_ras=cav,
                        score="deformable", margins=True)
    want = {frozenset(t.seed_ids) for t in truth.tiles}
    assert _partition(counted) == _partition(auto) == want
    assert counted.all_assigned and counted.score_rule == "deformable"
    assert not counted.capped and not auto.capped
    for res in (auto, counted):
        assert set(res.partition_margins) == {p.tile_id for p in res.tiles}
        assert all(m > 2.0 for m in res.partition_margins.values())
        assert res.ambiguous_tiles == []
    for pose in counted.tiles:
        # the pose comes from the attached bent-tile fit, oriented like auto
        assert pose.deform is not None
        assert abs(float(pose.normal_ras @ pose.deform.pose.normal)) > 1 - 1e-9
        assert float(pose.normal_ras @ (pose.center_ras - cav)) > 0.0
        assert abs(float(pose.axis_ras @ pose.normal_ras)) < 1e-9
        assert pose.residual_mm == pytest.approx(pose.deform.rms_mm)


def test_counted_default_stays_chord():
    c, a, truth = _truth(1, 3)
    res = fit_tiles(c, a, 3, 0, cavity_center_ras=truth.cavity_center_ras)
    assert res.score_rule == "chord" and not res.capped
    assert res.partition_margins is None and res.ambiguous_tiles == []
    assert all(p.deform is None for p in res.tiles)
    with pytest.raises(ValueError):
        fit_tiles(c, a, 3, 0, score="bent")


def test_prior_count_uses_the_deformable_score():
    c, a, truth = _truth(2, 3)
    res = fit_tiles_prior(c, a, ImplantPrior(n_full=3),
                          cavity_center_ras=truth.cavity_center_ras)
    assert res.score_rule == "deformable" and res.partition_margins is None
    assert _partition(res) == {frozenset(t.seed_ids) for t in truth.tiles}


def _lattice(n_side=6, pitch=10.0):
    g = np.arange(n_side) * pitch
    xx, yy = np.meshgrid(g, g)
    c = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    a = np.tile([1.0, 0.0, 0.0], (len(c), 1))
    return c, a


def test_capped_is_reported(monkeypatch):
    c, a = _lattice()                     # 36 seeds, 25 gate-passing quads
    res = fit_tiles(c, a, 9, 0)
    assert not res.capped and len(res.tiles) == 9
    monkeypatch.setattr(fit_mod, "_SEARCH_NODE_CAP", 20)
    capped = fit_tiles(c, a, 9, 0)
    assert capped.capped
    # a capped search still returns a valid disjoint selection
    used = [i for p in capped.tiles for i in p.seed_indices]
    assert len(used) == len(set(used))


# ------------------------------------------------------------ stage 6
_R = Rotation.from_rotvec([0.4, 0.1, -0.3]).as_matrix()


def _five_seeds(z_a, z_b):
    """Three corners of a 10 mm square plus TWO candidates for the 4th
    corner, out of plane by ``z_a`` / ``z_b`` mm: either completes a quad
    with the shared L-triplet (a split / duplicate detection)."""
    loc = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],
                    [10, 10, z_a], [10, 10, z_b]], dtype=float)
    return loc @ _R.T + np.array([20.0, 5.0, -10.0]), np.tile(_R[:, 0], (5, 1))


def test_shared_seed_case_is_ambiguous():
    # mirror-symmetric candidates: both readings explain the seeds equally
    c, a = _five_seeds(1.5, -1.5)
    auto = fit_tiles_auto(c, a, margins=True)
    counted = fit_tiles(c, a, 1, 0, score="deformable", margins=True)
    for res in (auto, counted):
        assert len(res.tiles) == 1
        tid = res.tiles[0].tile_id
        assert res.partition_margins[tid] < AMBIGUOUS_MARGIN
        assert res.ambiguous_tiles == [tid]
        alt = res.partition_alternatives[tid]
        assert len(alt) == 1 and set(alt[0]) != set(res.tiles[0].seed_indices)
        assert set(alt[0]) | set(res.tiles[0].seed_indices) == set(range(5))
    from gtcore.planner import _suggest_notes

    assert _suggest_notes(auto) == ["T1 ambiguous (margin 0.0)"]
    # one candidate ON the corner, the other 3.5 mm off: decided, not flagged
    c, a = _five_seeds(0.0, -3.5)
    clear = fit_tiles_auto(c, a, margins=True)
    assert clear.tiles[0].seed_indices == [0, 1, 2, 3]
    assert clear.partition_margins[0] > AMBIGUOUS_MARGIN
    assert clear.ambiguous_tiles == [] and _suggest_notes(clear) == []


def test_margins_are_opt_in_and_forwarded():
    c, a, truth = _truth(0, 2)
    cav = truth.cavity_center_ras
    off = fit_tiles_auto(c, a, cavity_center_ras=cav)
    assert off.partition_margins is None and off.partition_alternatives is None
    for prior in (ImplantPrior(), ImplantPrior(n_full=2)):
        res = fit_tiles_prior(c, a, prior, cavity_center_ras=cav,
                              margins=True)
        assert set(res.partition_margins) == {0, 1}
        assert np.isinf(list(res.partition_margins.values())).all()


def _bent_tile(kappa=0.05):
    pose = TilePose6(Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix(),
                     np.array([10.0, -20.0, 35.0]), "full", 1.0)
    params = DeformParams(kappa, 0.6 * kappa, 0.3)
    return deformed_seed_points(pose, params), deformed_seed_axes(pose, params)


def test_uncertainty_is_lazy_and_well_formed():
    P, A = _bent_tile()
    fit = fit_deformable(P + 0.2, A, kind="full")
    assert fit.cov_x is None and fit.center_cov is None
    assert fit.normal_sigma_deg is None            # selection pays nothing
    assert fit.compute_uncertainty() is fit
    assert fit.cov_x.shape == (9, 9) and fit.center_cov.shape == (3, 3)
    assert np.allclose(fit.cov_x, fit.cov_x.T)
    assert np.linalg.eigvalsh(fit.center_cov).min() >= -1e-12
    assert fit.cov_rank <= 9 and fit.cov_cond >= 1.0
    # 2-seed rigid fallback: no least-squares solution to linearise
    half = fit_deformable(P[:2], A[:2]).compute_uncertainty()
    assert half.cov_x is None and half.center_cov is None
    # computed for the reported tiles (in _finish), not inside the search
    c, a, truth = _truth(3, 2)
    res = fit_tiles_auto(c, a, cavity_center_ras=truth.cavity_center_ras)
    assert all(p.deform.center_cov is not None for p in res.tiles)


def test_normal_sigma_grows_with_seed_noise():
    P0, A0 = _bent_tile()
    rng = np.random.default_rng(11)
    means = []
    for sigma in (0.1, 0.3, 0.6):
        sig = [fit_deformable(P0 + rng.normal(0.0, sigma, P0.shape), A0,
                              kind="full", hinge_starts=False)
               .compute_uncertainty().normal_sigma_deg for _ in range(20)]
        means.append(float(np.mean(sig)))
    assert means[0] < means[1] < means[2], means


def test_center_cov_matches_empirical_scatter():
    """Calibration under the model's own assumption (one noise level for
    every residual row): 0.3 mm on each seed coordinate and 0.3 / w_axis
    rad on each seed axis.  docs/localization-notes.md (stage 6) records
    how the pooled s^2 drifts when the axis noise is NOT at that level."""
    P0, A0 = _bent_tile()
    rng = np.random.default_rng(5)
    sigma = 0.3
    sig_axis = sigma / 4.0                   # deform._W_AXIS_MM_PER_RAD
    centres, traces = [], []
    for _ in range(200):
        P = P0 + rng.normal(0.0, sigma, P0.shape)
        A = []
        for ax in A0:
            v = rng.normal(0.0, sig_axis, 3)
            v -= (v @ ax) * ax
            A.append((ax + v) / np.linalg.norm(ax + v))
        fit = fit_deformable(P, np.array(A), kind="full",
                             hinge_starts=False).compute_uncertainty()
        centres.append(fit.seed_points().mean(axis=0))
        traces.append(float(np.trace(fit.center_cov)))
    empirical = float(np.trace(np.cov(np.array(centres).T)))
    ratio = float(np.mean(traces)) / empirical
    assert 0.5 <= ratio <= 2.0, ratio
