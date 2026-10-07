"""Cover pass of auto mode (gtcore.tiles.auto): explain every seed of the
implant, never a 1- or 3-seed tile, no half tiles without the OR's word.

Acceptance:
- a tile with one seed missed by detection is recovered by triplet
  completion as a FULL tile with the 4th seed *inferred* and flagged;
- on a coarse (2 mm) scan the supported selection is unchanged and the cover
  pass explains the tiles it drops, marking them tentative;
- an OR count larger than the geometry supports is reported as a shortfall
  (tentative tiles capped at the shortfall), never invented from clutter;
- clutter and decoys still never inflate the supported count, and a scan
  with no implant gets no tentative tile at all.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import (
    AutoFitResult,
    ImplantPrior,
    fit_tiles,
    fit_tiles_auto,
    fit_tiles_prior,
    spacing_tolerance,
    to_placed_tiles,
)
from gtcore.volume import Volume


def _detect(vol):
    p = seed_detection_params(vol.spacing)
    cands = filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])
    return np.array(cands.centers_ras), np.array(cands.axes_ras)


def _thick_slices(vol, factor):
    """Block-average along k (partial-volume simulation of thick slices);
    mirrors scripts/validation_spacing.py."""
    nk = (vol.array.shape[0] // factor) * factor
    arr = vol.array[:nk].reshape(-1, factor, *vol.array.shape[1:]).mean(axis=1)
    affine = vol.affine.copy()
    affine[:3, 2] *= factor
    affine[:3, 3] += vol.affine[:3, 2] * (factor - 1) / 2.0
    return Volume(arr.astype(np.float32), affine, dict(vol.meta))


@pytest.fixture(scope="module")
def phantom3():
    vol, truth = make_head_phantom(spacing=0.8, n_tiles=3, rng_seed=2)
    centers, axes = _detect(vol)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = cdist(centers, tc).argmin(axis=1)
    assert len(centers) == 12 and len(set(d2t.tolist())) == 12
    return vol, truth, centers, axes, d2t


def test_spacing_tolerance():
    assert spacing_tolerance(None) == 1.0
    assert spacing_tolerance((0.5, 0.5, 1.0)) == 1.0
    assert spacing_tolerance((0.52, 0.52, 2.0)) == 2.0


def test_implant_prior_defaults_and_seed_count():
    unknown = ImplantPrior()
    assert not unknown.count_known and unknown.n_half == 0
    assert "assuming 0 half" in unknown.describe()
    known = ImplantPrior(n_full=3, n_half=1)
    assert known.n_seeds == 14 and "given" in known.describe()
    with pytest.raises(ValueError):
        ImplantPrior(n_full=-1)


# ------------------------------------------------------- triplet completion
def test_missed_seed_is_completed_to_a_full_tile(phantom3):
    vol, truth, centers, axes, d2t = phantom3
    # drop one seed of the first tile: detection "missed" it
    t0 = truth.tiles[0]
    victim = int(np.where(d2t == t0.seed_ids[0])[0][0])
    keep = [i for i in range(len(centers)) if i != victim]
    c, a = centers[keep], axes[keep]
    res = fit_tiles(c, a, "auto", cavity_center_ras=truth.cavity_center_ras)
    assert isinstance(res, AutoFitResult)
    assert res.n_selected == 2                     # calibrated count unchanged
    assert len(res.tentative_tiles) == 1
    pose = res.tentative_tiles[0]
    assert pose.tentative and pose.kind == "full"
    assert len(pose.seed_indices) == 3 and pose.inferred_seed_ras is not None
    # the inferred seed lands where the missed seed physically is (the
    # bent-tile model predicts a wall-conformed corner to ~1-2 mm; it is a
    # flagged proposal, not a measurement)
    missed = np.array([s.center_ras for s in truth.seeds])[t0.seed_ids[0]]
    assert np.linalg.norm(pose.inferred_seed_ras - missed) < 2.5
    assert np.linalg.norm(pose.center_ras - t0.center_ras) < 1.5
    # every seed of the implant is now explained
    assert res.unassigned_indices == [] and res.all_assigned
    assert "1 tentative" in res.summary() and "1 inferred seed" in res.summary()
    # seed_points hands 4 seeds (3 detected + inferred) to the planner feed
    sc, sa = pose.seed_points(c, a)
    assert sc.shape == (4, 3) and sa.shape == (4, 3)
    placed = to_placed_tiles(res, c, a)
    assert len(placed) == 3 and placed[-1].seed_centers.shape == (4, 3)


def test_triplet_never_overrides_a_judged_quad(phantom3):
    """If a detected candidate already sits where the 4th seed would be, the
    quad was judged upstream: the triplet must not resurrect it."""
    vol, truth, centers, axes, d2t = phantom3
    t0 = truth.tiles[0]
    victim = int(np.where(d2t == t0.seed_ids[0])[0][0])
    # displace the victim by 2.5 mm: the quad now fails the supported gates
    # (rms), a candidate still occupies the corner -> no triplet completion
    c = centers.copy()
    c[victim] += np.array([2.5, 0.0, 0.0])
    res = fit_tiles(c, axes, "auto", cavity_center_ras=truth.cavity_center_ras)
    for pose in res.tentative_tiles:
        assert pose.inferred_seed_ras is None or \
            cdist([pose.inferred_seed_ras], c).min() >= 3.5


# ------------------------------------------------------------ coarse scans
def test_coarse_scan_cover_explains_the_dropped_tiles():
    vol, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=1)
    thick = _thick_slices(vol, 2)                  # 1.4 mm slices
    assert spacing_tolerance(thick.spacing) > 1.2
    centers, axes = _detect(thick)
    tc = np.array([s.center_ras for s in truth.seeds])
    assert cdist(centers, tc).min(axis=0).max() < 2.5   # all seeds found
    # the calibrated selection drops a tile here (degenerate axes at this
    # spacing); without the cover pass its 4 seeds would be lost silently
    bare = fit_tiles_auto(centers, axes, cavity_center_ras=truth.cavity_center_ras,
                          spacing_mm=thick.spacing, cover=False)
    assert bare.n_selected == 2 and len(bare.unassigned_indices) == 4
    res = fit_tiles_auto(centers, axes, cavity_center_ras=truth.cavity_center_ras,
                         spacing_mm=thick.spacing)
    assert res.spacing_tol == pytest.approx(float(np.max(thick.spacing)))
    assert res.n_selected == 2 and len(res.tentative_tiles) == 1
    assert res.unassigned_indices == [] and res.all_assigned
    for pose in res.all_tiles:
        assert pose.kind == "full"
        d = np.linalg.norm(np.array([t.center_ras for t in truth.tiles])
                           - pose.center_ras, axis=1).min()
        assert d < 2.5


def test_coarse_scan_missed_seed_is_completed():
    vol, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=1)
    thick = _thick_slices(vol, 3)                  # 2.1 mm: one seed lost
    centers, axes = _detect(thick)
    tc = np.array([s.center_ras for s in truth.seeds])
    assert int((cdist(centers, tc).min(axis=0) < 2.5).sum()) == 11
    res = fit_tiles_auto(centers, axes, cavity_center_ras=truth.cavity_center_ras,
                         spacing_mm=thick.spacing)
    assert len(res.tiles) + len(res.tentative_tiles) == 3, res.summary()
    assert res.n_inferred_seeds == 1


# ---------------------------------------------------------------- OR count
def test_prior_count_reports_shortfall_and_never_invents(phantom3):
    vol, truth, centers, axes, d2t = phantom3
    res = fit_tiles_prior(centers, axes, ImplantPrior(n_full=3),
                          cavity_center_ras=truth.cavity_center_ras)
    assert res.n_requested == 3 and res.n_selected == 3
    assert res.all_assigned and res.tentative_tiles == []
    assert "asked for 3" in res.summary()

    more = fit_tiles_prior(centers, axes, ImplantPrior(n_full=5),
                           cavity_center_ras=truth.cavity_center_ras)
    assert more.n_selected == 3 and not more.all_assigned
    assert "geometry supports only 3" in more.summary()
    assert more.tentative_tiles == []          # no leftover seeds to use
    assert more.unassigned_indices == []

    # count known but one seed missed: the shortfall is filled by a
    # tentative triplet tile, capped at the shortfall
    t0 = truth.tiles[0]
    victim = int(np.where(d2t == t0.seed_ids[0])[0][0])
    keep = [i for i in range(len(centers)) if i != victim]
    short = fit_tiles_prior(centers[keep], axes[keep], ImplantPrior(n_full=3),
                            cavity_center_ras=truth.cavity_center_ras)
    assert short.n_selected == 2 and len(short.tentative_tiles) == 1
    assert short.tentative_tiles[0].inferred_seed_ras is not None
    assert short.unassigned_indices == []


def test_prior_unknown_is_auto_without_halves():
    vol, truth = make_head_phantom(spacing=0.8, n_tiles=2, n_half_tiles=2,
                                   rng_seed=1)
    centers, axes = _detect(vol)
    res = fit_tiles_prior(centers, axes, ImplantPrior(),
                          cavity_center_ras=truth.cavity_center_ras)
    assert res.prior is not None and not res.prior.count_known
    assert all(p.kind == "full" for p in res.all_tiles)
    assert len(res.tiles) == 2
    assert not any(p.kind == "half" for p in res.tentative_tiles)


# ----------------------------------------------------------------- clutter
def test_no_implant_means_no_tentative_tiles():
    rng = np.random.default_rng(8)
    centers = rng.uniform(-40.0, 40.0, (12, 3))
    axes = rng.standard_normal((12, 3))
    res = fit_tiles(centers, axes, "auto")
    assert res.n_selected == 0 and res.tentative_tiles == []
    assert res.unassigned_indices == [] and res.all_assigned
    assert res.clutter_indices == list(range(12))


def test_far_clutter_is_not_unassigned(phantom3):
    vol, truth, centers, axes, d2t = phantom3
    rng = np.random.default_rng(3)
    far = truth.cavity_center_ras + np.array([80.0, 0.0, 0.0]) \
        + rng.uniform(-5, 5, (5, 3))
    c = np.vstack([centers, far])
    a = np.vstack([axes, rng.standard_normal((5, 3))])
    res = fit_tiles(c, a, "auto", cavity_center_ras=truth.cavity_center_ras)
    assert res.n_selected == 3 and res.all_assigned
    assert res.clutter_indices == list(range(12, 17))
    assert res.unassigned_indices == []
