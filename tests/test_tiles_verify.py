"""Stage 8 of docs/plan-localization.md: image check of inferred seeds
(gtcore.tiles.verify).

* the 2.1 mm missed-seed case of test_tiles_cover (rng 1): the inferred
  4th seed is "recovered" and the grey-centroid position lands within
  1.0 mm of the truth seed;
* the same volume with that seed's voxels painted to background gives
  "no image evidence";
* a result without tentative tiles gets an empty verification;
* the tile stays tentative in every case; reconstruct() and the planner
  status line carry the outcome.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gtcore.phantom import make_head_phantom
from gtcore.phantom.seed_render import thick_slices
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import fit_tiles, fit_tiles_auto, verify_inferred_seeds
from gtcore.tiles.fit import TilePose
from gtcore.tiles.verify import verification_summary


def _detect(vol):
    p = seed_detection_params(vol.spacing)
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


@pytest.fixture(scope="module")
def missed():
    """2.1 mm slabs of the rng-1 3-tile phantom: 11 of 12 seeds detected,
    the 12th inferred by triplet completion (test_tiles_cover)."""
    vol, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=1)
    thick = thick_slices(vol, 3)
    cands = _detect(thick)
    tc = np.array([s.center_ras for s in truth.seeds])
    d = cdist(cands.centers_ras, tc).min(axis=0)
    assert int((d < 2.5).sum()) == 11
    lost = int(np.argmax(d))
    res = fit_tiles_auto(cands.centers_ras, cands.axes_ras,
                         cavity_center_ras=truth.cavity_center_ras,
                         spacing_mm=thick.spacing)
    assert res.n_inferred_seeds == 1
    return SimpleNamespace(vol=thick, truth=truth, cands=cands, res=res,
                           lost_ras=tc[lost])


def _inferred_pose(res):
    return [p for p in res.all_tiles if p.inferred_seed_ras is not None][0]


def test_missed_seed_is_recovered_within_1mm(missed):
    res = copy.deepcopy(missed.res)
    pose = _inferred_pose(res)
    before = pose.inferred_seed_ras.copy()
    ver = verify_inferred_seeds(res, missed.vol, missed.cands)
    assert res.verification is ver and set(ver) == {pose.tile_id}
    r = ver[pose.tile_id]
    assert r["status"] == "recovered"
    assert r["peak_hu"] > r["threshold_hu"] > r["background_hu"]
    # this seed sits in the cavity's air pocket (verify.py docstring): the
    # stage-2 grey centroid falls back and the ROI centroid takes over
    assert (r["refine_status"] == "ok"
            or r["refine_status"].startswith("roi_centroid:"))
    assert np.linalg.norm(r["refined_ras"] - missed.lost_ras) < 1.0
    assert r["cov_ras"].shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(r["cov_ras"]) > 0.0)
    # the image refinement improves on the geometric inference
    assert (np.linalg.norm(r["refined_ras"] - missed.lost_ras)
            < np.linalg.norm(before - missed.lost_ras))
    # never promoted, never moved unless asked
    assert pose.tentative and np.allclose(pose.inferred_seed_ras, before)
    verify_inferred_seeds(res, missed.vol, missed.cands, update_poses=True)
    assert np.allclose(pose.inferred_seed_ras, r["refined_ras"])
    assert pose.tentative
    summary = verification_summary(ver)
    assert summary["n_checked"] == 1 and summary["n_recovered"] == 1
    assert isinstance(summary["tiles"][pose.tile_id]["refined_ras"], list)
    assert "image check: 1 recovered" in res.summary()


def test_seed_painted_out_gives_no_image_evidence(missed):
    vol = missed.vol
    arr = np.array(vol.array, copy=True)
    nk, nj, ni = arr.shape
    K, J, I = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni),
                          indexing="ij", sparse=True)
    # RAS distance of every voxel to the lost seed; paint a 5 mm ball (the
    # 4.5 mm capsule, its bloom and the inferred-position error) with the
    # local background
    ijk = np.stack(np.broadcast_arrays(I, J, K), axis=-1).reshape(-1, 3)
    ras = vol.index_to_ras(ijk.astype(float))
    d = np.linalg.norm(ras - missed.lost_ras[None, :], axis=1).reshape(arr.shape)
    shell = arr[(d > 5.0) & (d <= 7.0)]
    arr[d <= 5.0] = np.median(shell)
    blank = copy.copy(vol)
    blank.array = arr
    res = copy.deepcopy(missed.res)
    pose = _inferred_pose(res)
    ver = verify_inferred_seeds(res, blank, missed.cands)
    r = ver[pose.tile_id]
    assert r["status"] == "no image evidence"
    assert r["refined_ras"] is None and r["cov_ras"] is None
    assert r["peak_hu"] <= r["threshold_hu"]
    assert pose.tentative and pose.inferred_seed_ras is not None
    assert "0 recovered, 1 without evidence" in res.summary()


def test_no_tentative_tile_gives_empty_verification(missed):
    counted = fit_tiles(missed.cands.centers_ras, missed.cands.axes_ras, 2, 0)
    assert all(p.inferred_seed_ras is None for p in counted.tiles)
    assert verify_inferred_seeds(counted, missed.vol, missed.cands) == {}
    assert counted.verification == {}
    bare = fit_tiles_auto(missed.cands.centers_ras, missed.cands.axes_ras,
                          spacing_mm=missed.vol.spacing, cover=False)
    assert bare.verification is None
    assert verify_inferred_seeds(bare, missed.vol, missed.cands) == {}
    assert verification_summary(bare.verification)["n_checked"] == 0


def test_reconstruct_records_the_check():
    from gtcore.pipeline import reconstruct

    # off by default: no key, no attribute (a coarse volume keeps this cheap)
    vol, _truth = make_head_phantom(spacing=2.0, n_tiles=2, rng_seed=1)
    plain = reconstruct(vol, verbose=False, n_full_tiles="auto")
    assert "seed_verify" not in vol.meta
    assert plain.tiles.verification is None
    vol, _truth = make_head_phantom(spacing=1.4, n_tiles=2, rng_seed=1)
    res = reconstruct(vol, verbose=False, n_full_tiles="auto",
                      refine_seeds="centroid")
    sv = vol.meta["seed_verify"]
    assert sv is res.meta["seed_verify"]
    assert set(sv) == {"n_checked", "n_recovered", "n_no_evidence", "tiles"}
    assert sv["n_checked"] == res.tiles.n_inferred_seeds
    assert res.tiles.verification is not None


def test_planner_notes_carry_the_outcome():
    from gtcore.planner import _suggest_notes

    def pose(tid, inferred=True):
        p = TilePose(tile_id=tid, kind="full", seed_indices=[0, 1, 2],
                     center_ras=np.zeros(3), normal_ras=[0, 0, 1],
                     axis_ras=[1, 0, 0], residual_mm=0.3,
                     inferred_seed_ras=np.ones(3) if inferred else None)
        p.confidence = "tentative"
        return p

    fit = SimpleNamespace(all_tiles=[pose(0), pose(1), pose(2)],
                          verification={0: {"status": "recovered"},
                                        1: {"status": "no image evidence"}})
    assert _suggest_notes(fit) == [
        "T1 tentative (1 seed inferred: 4th seed recovered)",
        "T2 tentative (1 seed inferred: no image evidence)",
        "T3 tentative (1 seed inferred)"]
    fit.verification = None
    assert _suggest_notes(fit)[0] == "T1 tentative (1 seed inferred)"
