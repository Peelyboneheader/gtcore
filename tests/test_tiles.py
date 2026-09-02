"""Verification campaign for tile-configuration inference (gtcore.tiles).

Randomized sweeps over phantom seeds and tile counts: every case must recover
the truth tile partition EXACTLY from detected seed candidates, with tight
pose bounds, and reject injected decoys.  Phantoms are generated once per
configuration and cached module-wide (volumes and masks are dropped
immediately; only truth poses and detected candidates are kept).

Normal-error reference: the fitter's normal is a plane fit through the
member seeds, but the phantom's ``TileTruth.normal_ras`` is the *radial*
direction from the cavity centre.  On the exact truth seed positions those
two already disagree by up to ~13.6 deg (the wall-conformed quad is tilted
against the local radius), so the tight 12-deg bound is checked against the
plane fit of the truth seed centres -- what a plane-fit algorithm can
actually be held to -- plus a looser sanity bound and an orientation-sign
check against the radial truth normal.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gtcore.phantom import BRAIN_RADII, make_head_phantom
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import TileFitResult, TilePose, fit_tiles
from gtcore.tiles.fit import _orient_normal, _plane_fit

SPACING = 0.8
SWEEP_SEEDS = range(8)
SWEEP_TILES = (1, 2, 3, 4, 5)
HALF_SEEDS = range(4)

MAX_CENTER_ERR_MM = 1.5
MAX_NORMAL_ERR_DEG = 12.0
MAX_RADIAL_ERR_DEG = 18.0       # vs the radial truth normal (see docstring)


# ------------------------------------------------------------- cached cases
class _Case:
    """Ground truth + detected candidates for one phantom configuration."""

    def __init__(self, rng_seed, n_tiles, n_half):
        vol, truth = make_head_phantom(
            spacing=SPACING, n_tiles=n_tiles, n_half_tiles=n_half,
            rng_seed=rng_seed,
        )
        cands = detect_seed_candidates(vol)
        # Keep only what the tests need; the volume and the boolean masks
        # would otherwise pin ~60 MB per case.
        self.truth_seeds = truth.seeds
        self.truth_tiles = truth.tiles
        self.cavity_center = truth.cavity_center_ras
        self.centers = np.array(cands.centers_ras, copy=True)
        self.axes = np.array(cands.axes_ras, copy=True)

    def det_to_truth(self, max_mm=2.0):
        """Candidate index -> truth seed_id (nearest neighbour, or None)."""
        tc = np.array([s.center_ras for s in self.truth_seeds])
        D = cdist(self.centers, tc)
        nn = D.argmin(axis=1)
        nnd = D.min(axis=1)
        return [int(nn[i]) if nnd[i] <= max_mm else None
                for i in range(len(self.centers))]


_CACHE = {}


def get_case(rng_seed, n_tiles, n_half=0):
    key = (rng_seed, n_tiles, n_half)
    if key not in _CACHE:
        _CACHE[key] = _Case(rng_seed, n_tiles, n_half)
    return _CACHE[key]


# ----------------------------------------------------------------- checkers
def _angle_deg(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def check_result(case, result, n_full, n_half, n_candidates=None):
    """Assert the fitted configuration matches truth exactly and precisely."""
    if n_candidates is None:
        n_candidates = len(case.centers)
    assert isinstance(result, TileFitResult)
    assert result.n_expected == n_full + n_half
    assert result.all_assigned, "expected %d+%d tiles, recovered %d" % (
        n_full, n_half, len(result.tiles))
    assert len(result.tiles) == n_full + n_half

    d2t = case.det_to_truth()
    d2t += [None] * (n_candidates - len(d2t))          # appended decoys
    truth_by_key = {frozenset(t.seed_ids): t for t in case.truth_tiles}
    got = {}
    for pose in result.tiles:
        assert isinstance(pose, TilePose)
        ids = [d2t[i] for i in pose.seed_indices]
        assert None not in ids, "tile used a spurious candidate"
        key = frozenset(ids)
        assert len(key) == len(ids), "tile reused one truth seed twice"
        assert key not in got, "two tiles recovered the same truth tile"
        got[key] = pose
    # EXACT partition match, compared as sets of frozensets of truth seed ids.
    assert set(got) == set(truth_by_key)

    # Assigned + rejected must cover every candidate exactly once.
    assigned = [i for p in result.tiles for i in p.seed_indices]
    assert len(assigned) == len(set(assigned))
    assert sorted(assigned + result.rejected_indices) == \
        list(range(n_candidates))

    for key, pose in got.items():
        t = truth_by_key[key]
        assert pose.kind == t.kind
        assert len(pose.seed_indices) == (4 if t.kind == "full" else 2)

        cerr = float(np.linalg.norm(pose.center_ras - t.center_ras))
        assert cerr < MAX_CENTER_ERR_MM, \
            "tile center error %.2f mm" % cerr

        if t.kind == "full":
            pts = np.array([case.truth_seeds[i].center_ras
                            for i in t.seed_ids])
            ref, _ = _plane_fit(pts)
            ref = _orient_normal(ref, pts.mean(axis=0), case.cavity_center)
        else:
            ref = t.normal_ras
        nerr = _angle_deg(pose.normal_ras, ref)
        assert nerr < MAX_NORMAL_ERR_DEG, \
            "tile normal error %.1f deg (%s)" % (nerr, t.kind)
        # Orientation and rough agreement with the radial truth normal.
        assert float(pose.normal_ras @ t.normal_ras) > 0.0
        assert _angle_deg(pose.normal_ras, t.normal_ras) < MAX_RADIAL_ERR_DEG

        # Fitted t1 axis lies in the tile plane and is a unit vector.
        assert abs(float(pose.axis_ras @ pose.normal_ras)) < 1e-6
        assert abs(np.linalg.norm(pose.axis_ras) - 1.0) < 1e-9
        assert pose.residual_mm < 2.0
    return got


# --------------------------------------------------------------- full sweep
@pytest.mark.parametrize("rng_seed", SWEEP_SEEDS)
@pytest.mark.parametrize("n_tiles", SWEEP_TILES)
def test_full_tile_sweep(rng_seed, n_tiles):
    case = get_case(rng_seed, n_tiles)
    result = fit_tiles(case.centers, case.axes, n_tiles, 0,
                       cavity_center_ras=case.cavity_center)
    check_result(case, result, n_tiles, 0)


# --------------------------------------------------------------- half tiles
@pytest.mark.parametrize("rng_seed", HALF_SEEDS)
def test_half_tile_sweep(rng_seed):
    case = get_case(rng_seed, 2, n_half=2)
    result = fit_tiles(case.centers, case.axes, 2, 2,
                       cavity_center_ras=case.cavity_center)
    got = check_result(case, result, 2, 2)
    kinds = sorted(p.kind for p in got.values())
    assert kinds == ["full", "full", "half", "half"]


# ------------------------------------------------------------------- decoys
def _add_decoys(case, n_decoys, rng_seed):
    """Append fake candidates inside the brain, > 15 mm from every seed."""
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
    return (np.vstack([case.centers, fakes]),
            np.vstack([case.axes, axes]))


@pytest.mark.parametrize("rng_seed", HALF_SEEDS)
def test_decoys_are_rejected(rng_seed):
    case = get_case(rng_seed, 3)
    n_real = len(case.centers)
    centers, axes = _add_decoys(case, 4, rng_seed)
    result = fit_tiles(centers, axes, 3, 0,
                       cavity_center_ras=case.cavity_center)
    for i in range(n_real, n_real + 4):
        assert i in result.rejected_indices, "decoy %d was assigned" % i
    check_result(case, result, 3, 0, n_candidates=len(centers))


def test_decoys_with_half_tiles():
    case = get_case(1, 2, n_half=2)
    n_real = len(case.centers)
    centers, axes = _add_decoys(case, 4, 77)
    result = fit_tiles(centers, axes, 2, 2,
                       cavity_center_ras=case.cavity_center)
    for i in range(n_real, n_real + 4):
        assert i in result.rejected_indices
    check_result(case, result, 2, 2, n_candidates=len(centers))


# -------------------------------------------------------- degenerate inputs
def test_too_few_candidates_no_crash():
    case = get_case(0, 3)
    result = fit_tiles(case.centers[:6], case.axes[:6], 3, 0,
                       cavity_center_ras=case.cavity_center)
    assert not result.all_assigned
    assert len(result.tiles) <= 1                      # at most 6 // 4 quads
    assert result.n_expected == 3
    assigned = [i for p in result.tiles for i in p.seed_indices]
    assert sorted(assigned + result.rejected_indices) == list(range(6))


def test_zero_candidates_no_crash():
    result = fit_tiles(np.zeros((0, 3)), np.zeros((0, 3)), 2, 1)
    assert result.tiles == []
    assert result.rejected_indices == []
    assert not result.all_assigned


def test_nothing_expected_is_empty_result():
    case = get_case(0, 1)
    result = fit_tiles(case.centers, case.axes, 0, 0,
                       cavity_center_ras=case.cavity_center)
    assert result.tiles == []
    assert result.n_expected == 0
    assert result.all_assigned
    assert result.rejected_indices == list(range(len(case.centers)))


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        fit_tiles(np.zeros((3, 3)), np.zeros((2, 3)), 1)
    with pytest.raises(ValueError):
        fit_tiles(np.zeros((4, 3)), np.zeros((4, 3)), -1)


# -------------------------------------------------------------- determinism
def test_determinism():
    case = get_case(2, 4)
    kwargs = dict(cavity_center_ras=case.cavity_center)
    a = fit_tiles(case.centers, case.axes, 4, 0, **kwargs)
    b = fit_tiles(case.centers.copy(), case.axes.copy(), 4, 0, **kwargs)
    assert [p.seed_indices for p in a.tiles] == \
        [p.seed_indices for p in b.tiles]
    assert [p.kind for p in a.tiles] == [p.kind for p in b.tiles]
    assert a.rejected_indices == b.rejected_indices
    for pa, pb in zip(a.tiles, b.tiles):
        assert np.array_equal(pa.center_ras, pb.center_ras)
        assert np.array_equal(pa.normal_ras, pb.normal_ras)
        assert np.array_equal(pa.axis_ras, pb.axis_ras)
        assert pa.residual_mm == pb.residual_mm


# ------------------------------------------------- phantom back-compat guard
def test_half_tile_kwarg_keeps_default_phantom_identical():
    a_vol, a_truth = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=5)
    b_vol, b_truth = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=5,
                                       n_half_tiles=0)
    assert np.array_equal(a_vol.array, b_vol.array)
    assert a_vol.meta == b_vol.meta
    assert len(a_truth.seeds) == len(b_truth.seeds)
    for sa, sb in zip(a_truth.seeds, b_truth.seeds):
        assert np.array_equal(sa.center_ras, sb.center_ras)
    assert all(t.kind == "full" for t in a_truth.tiles)


def test_half_tile_truth_geometry():
    _, truth = make_head_phantom(spacing=2.0, n_tiles=1, n_half_tiles=2,
                                 rng_seed=3)
    assert len(truth.seeds) == 4 + 2 * 2
    assert [t.kind for t in truth.tiles] == ["full", "half", "half"]
    for t in truth.tiles:
        if t.kind != "half":
            continue
        assert len(t.seed_ids) == 2
        a, b = (truth.seeds[i] for i in t.seed_ids)
        d = float(np.linalg.norm(a.center_ras - b.center_ras))
        assert 7.5 < d < 11.0            # wall-conformed 10 mm pair chord
        ang = _angle_deg(a.axis_ras, b.axis_ras)
        assert ang < 35.0                # same-tile axes stay near-parallel


# --------------------------------------------------------- pipeline wiring
def test_pipeline_optional_tile_fitting():
    from gtcore.pipeline import reconstruct

    vol, truth = make_head_phantom(spacing=1.2, n_tiles=2, rng_seed=1)
    plain = reconstruct(vol, verbose=False)
    assert plain.tiles is None                      # omitted -> unchanged

    result = reconstruct(vol, verbose=False, n_full_tiles=2)
    assert result.tiles is not None
    assert result.tiles.n_expected == 2
    assert result.tiles.all_assigned
    assert len(result.tiles.tiles) == 2
    centers = np.array([p.center_ras for p in result.tiles.tiles])
    t_centers = np.array([t.center_ras for t in truth.tiles])
    D = cdist(centers, t_centers)
    # each fitted tile sits on a distinct truth tile
    assert sorted(D.argmin(axis=1).tolist()) == [0, 1]
    assert D.min(axis=1).max() < MAX_CENTER_ERR_MM
