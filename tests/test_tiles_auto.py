"""Count-free tile configuration search (gtcore.tiles.auto).

Acceptance (plan Step 2): with NO count input, auto mode recovers the true
number of full tiles on the synthetic phantom sweep and refuses to inflate
``n`` on decoys; the per-count score curve saturates at the truth.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gtcore.phantom import BRAIN_RADII, make_head_phantom
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import (
    AutoFitResult,
    ScorePoint,
    TileFitResult,
    fit_tiles,
    fit_tiles_auto,
)
from gtcore.tiles.auto import LAMBDA_FULL

SPACING = 0.8


class _Case:
    def __init__(self, rng_seed, n_tiles, n_half=0):
        vol, truth = make_head_phantom(spacing=SPACING, n_tiles=n_tiles,
                                       n_half_tiles=n_half, rng_seed=rng_seed)
        cands = detect_seed_candidates(vol)
        self.truth_seeds = truth.seeds
        self.truth_tiles = truth.tiles
        self.cavity_center = truth.cavity_center_ras
        self.centers = np.array(cands.centers_ras, copy=True)
        self.axes = np.array(cands.axes_ras, copy=True)

    def det_to_truth(self, max_mm=2.0):
        tc = np.array([s.center_ras for s in self.truth_seeds])
        D = cdist(self.centers, tc)
        nn, nnd = D.argmin(axis=1), D.min(axis=1)
        return [int(nn[i]) if nnd[i] <= max_mm else None
                for i in range(len(self.centers))]

    def partition(self, result):
        d2t = self.det_to_truth()
        d2t += [None] * (10 ** 4)
        return {frozenset(d2t[i] for i in p.seed_indices) for p in result.tiles}

    @property
    def truth_partition(self):
        return {frozenset(t.seed_ids) for t in self.truth_tiles}


_CACHE = {}


def get_case(rng_seed, n_tiles, n_half=0):
    key = (rng_seed, n_tiles, n_half)
    if key not in _CACHE:
        _CACHE[key] = _Case(rng_seed, n_tiles, n_half)
    return _CACHE[key]


def _check_curve(result, n_true):
    curve = result.score_curve
    assert isinstance(result, AutoFitResult)
    assert all(isinstance(p, ScorePoint) for p in curve)
    assert curve[0].n == 0 and curve[0].score == 0.0
    feas = [p for p in curve if p.feasible]
    # unpenalised score is non-decreasing with n (tiles have positive score)
    scores = [p.score for p in feas]
    assert all(b >= a - 1e-9 for a, b in zip(scores, scores[1:]))
    # the true tiles each add more than the penalty; anything beyond adds less
    for p in feas:
        if 1 <= p.n <= n_true:
            assert p.marginal > LAMBDA_FULL, (p.n, p.marginal)
        elif p.n > n_true:
            assert p.marginal < LAMBDA_FULL, (p.n, p.marginal)
    assert result.n_selected == n_true
    assert result.n_expected == n_true
    assert result.all_assigned
    assert "n=%d" % n_true in result.summary()


# --------------------------------------------------------------- full sweep
@pytest.mark.parametrize("rng_seed", (0, 3, 6))
@pytest.mark.parametrize("n_tiles", (1, 2, 4, 5))
def test_auto_recovers_true_count(rng_seed, n_tiles):
    case = get_case(rng_seed, n_tiles)
    result = fit_tiles(case.centers, case.axes, "auto",
                       cavity_center_ras=case.cavity_center)
    assert case.partition(result) == case.truth_partition
    _check_curve(result, n_tiles)
    assigned = [i for p in result.tiles for i in p.seed_indices]
    assert sorted(assigned + result.rejected_indices) == \
        list(range(len(case.centers)))
    for pose in result.tiles:
        assert pose.kind == "full"
        assert float(pose.normal_ras @ (pose.center_ras - case.cavity_center)) > 0.0


def test_auto_matches_counted_fit():
    """Auto mode lands on the same partition the OR count would give."""
    case = get_case(1, 3)
    auto = fit_tiles(case.centers, case.axes, "auto",
                     cavity_center_ras=case.cavity_center)
    counted = fit_tiles(case.centers, case.axes, 3, 0,
                        cavity_center_ras=case.cavity_center)
    assert {frozenset(p.seed_indices) for p in auto.tiles} == \
        {frozenset(p.seed_indices) for p in counted.tiles}
    assert isinstance(counted, TileFitResult)
    assert not isinstance(counted, AutoFitResult)


# ------------------------------------------------------------------- decoys
def _add_decoys(case, n_decoys, rng_seed):
    rng = np.random.default_rng(9000 + rng_seed)
    tc = np.array([s.center_ras for s in case.truth_seeds])
    radii = np.asarray(BRAIN_RADII)
    fakes = []
    while len(fakes) < n_decoys:
        p = rng.uniform(-radii, radii)
        if ((p / radii) ** 2).sum() < 0.8 and cdist([p], tc).min() > 15.0:
            fakes.append(p)
    axes = rng.standard_normal((n_decoys, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return np.vstack([case.centers, fakes]), np.vstack([case.axes, axes])


@pytest.mark.parametrize("rng_seed", (0, 2))
def test_auto_refuses_to_inflate_n_with_decoys(rng_seed):
    case = get_case(rng_seed, 2)
    n_real = len(case.centers)
    centers, axes = _add_decoys(case, 8, rng_seed)
    result = fit_tiles(centers, axes, "auto",
                       cavity_center_ras=case.cavity_center)
    assert result.n_selected == 2
    assert case.partition(result) == case.truth_partition
    for i in range(n_real, n_real + 8):
        assert i in result.rejected_indices


def test_auto_refuses_a_near_square_junk_quad():
    """A 4-point group that passes the chord gates but is far from a
    manufactured tile (rhombus, fanned axes) must not become a 4th tile."""
    case = get_case(3, 3)
    rng = np.random.default_rng(4)
    c0 = case.cavity_center + np.array([70.0, 0.0, 0.0])
    junk = np.array([[0, 0, 0], [7.5, 5.5, 0.0], [1.0, 9.0, 1.5],
                     [8.0, 13.0, 2.0]]) + c0
    junk_axes = rng.standard_normal((4, 3))
    junk_axes /= np.linalg.norm(junk_axes, axis=1, keepdims=True)
    centers = np.vstack([case.centers, junk])
    axes = np.vstack([case.axes, junk_axes])
    result = fit_tiles(centers, axes, "auto",
                       cavity_center_ras=case.cavity_center)
    assert result.n_selected == 3
    assert case.partition(result) == case.truth_partition


def test_pure_clutter_gives_zero_tiles():
    rng = np.random.default_rng(8)
    centers = rng.uniform(-40.0, 40.0, (12, 3))
    axes = rng.standard_normal((12, 3))
    result = fit_tiles(centers, axes, "auto")
    assert result.n_selected == 0
    assert result.tiles == []
    assert result.rejected_indices == list(range(12))
    assert result.all_assigned
    assert "n=0" in result.summary()


# --------------------------------------------------------------- half tiles
def test_half_tiles_opt_in():
    case = get_case(1, 2, n_half=2)
    default = fit_tiles(case.centers, case.axes, "auto",
                        cavity_center_ras=case.cavity_center)
    # halves are never *selected* without permission, but are reported
    assert [p.kind for p in default.tiles] == ["full", "full"]
    assert default.n_selected == 2
    truth_halves = {frozenset(t.seed_ids) for t in case.truth_tiles
                    if t.kind == "half"}
    d2t = case.det_to_truth()
    reported = {frozenset(d2t[i] for i in idx)
                for _s, idx in default.half_candidates}
    assert truth_halves <= reported

    allowed = fit_tiles(case.centers, case.axes, "auto", n_half=1,
                        cavity_center_ras=case.cavity_center)
    assert case.partition(allowed) == case.truth_partition
    assert sorted(p.kind for p in allowed.tiles) == \
        ["full", "full", "half", "half"]
    assert allowed.n_selected == 4
    assert allowed.half_candidates == []


def test_half_penalty_never_splits_a_full_tile():
    """With halves allowed, a genuine (deformed) quad must still be read as
    one full tile, not as two near-perfect pairs."""
    case = get_case(0, 4)
    result = fit_tiles_auto(case.centers, case.axes, allow_half=True,
                            cavity_center_ras=case.cavity_center)
    assert [p.kind for p in result.tiles] == ["full"] * 4
    assert case.partition(result) == case.truth_partition


# ------------------------------------------------------------- degenerate
def test_degenerate_inputs():
    empty = fit_tiles(np.zeros((0, 3)), np.zeros((0, 3)), "auto")
    assert empty.n_selected == 0 and empty.tiles == []
    one = fit_tiles(np.zeros((1, 3)), np.ones((1, 3)), "auto")
    assert one.n_selected == 0 and one.rejected_indices == [0]
    with pytest.raises(ValueError):
        fit_tiles(np.zeros((4, 3)), np.zeros((4, 3)), "sometimes")
    with pytest.raises(ValueError):
        fit_tiles_auto(np.zeros((4, 3)), np.zeros((3, 3)))


def test_determinism():
    case = get_case(2, 4)
    a = fit_tiles(case.centers, case.axes, "auto",
                  cavity_center_ras=case.cavity_center)
    b = fit_tiles(case.centers.copy(), case.axes.copy(), "auto",
                  cavity_center_ras=case.cavity_center)
    assert [p.seed_indices for p in a.tiles] == [p.seed_indices for p in b.tiles]
    assert [(p.n, p.score) for p in a.score_curve if p.feasible] == \
        [(p.n, p.score) for p in b.score_curve if p.feasible]
    assert [p.feasible for p in a.score_curve] == \
        [p.feasible for p in b.score_curve]


# ---------------------------------------------------------- pipeline / CLI
def test_pipeline_auto_mode():
    from gtcore.pipeline import reconstruct

    vol, truth = make_head_phantom(spacing=1.2, n_tiles=2, rng_seed=1)
    result = reconstruct(vol, verbose=False, n_full_tiles="auto")
    assert isinstance(result.tiles, AutoFitResult)
    assert result.tiles.n_selected == 2
    assert len(result.tiles.tiles) == 2
    centers = np.array([p.center_ras for p in result.tiles.tiles])
    t_centers = np.array([t.center_ras for t in truth.tiles])
    D = cdist(centers, t_centers)
    assert sorted(D.argmin(axis=1).tolist()) == [0, 1]
    assert D.min(axis=1).max() < 1.5


def test_cli_accepts_auto():
    from gtcore.cli import _int_or_auto

    assert _int_or_auto("auto") == "auto"
    assert _int_or_auto("AUTO") == "auto"
    assert _int_or_auto("3") == 3
    with pytest.raises(ValueError):
        _int_or_auto("three")
