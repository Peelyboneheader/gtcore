"""Merge/split repair of seed candidates (localization plan, stage 3).

Scenes come from ``tests/merge_split_scenes.py`` (supersampled blurred line
sources; see its docstring for the grids).  Gate (docs/plan-localization.md):
side-by-side seeds 2.5-3.5 mm apart still split; a slab-gap fragment pair
becomes one candidate; no false splits among 30 noise specks.
"""
import numpy as np
import pytest

import merge_split_scenes as M
from gtcore.pipeline import merge_fragments_default
from gtcore.seeds import detect_seed_candidates
from gtcore.seeds.detect import MERGE_MAX_SEP_MM, SEED_LENGTH_MM
from gtcore.volume import Volume

THIN = dict(hu_threshold=2000.0, min_mm3=1.0, max_mm3=15.0)   # pipeline, <= 1.2 mm
G1 = (0.59, 0.59, 1.0)
G1_SHAPE = (32, 52, 52)
LONE_C = np.array([[6.0, 6.0, 6.0], [24.0, 6.0, 8.0], [6.0, 24.0, 10.0],
                   [24.0, 24.0, 12.0]])
LONE_U = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0], [1.0, 0, 1.0]])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _side_by_side(axis, offset_dir, d, rng=11):
    """Four lone seeds (so the lone-seed median exists) plus two parallel
    seeds ``d`` mm apart along ``offset_dir`` (made perpendicular to the
    axis)."""
    u = _unit(axis)
    v = np.asarray(offset_dir, float)
    v = _unit(v - (v @ u) * u)
    mid = np.array([15.2, 15.1, 15.3])
    pair = np.array([mid - d / 2 * v, mid + d / 2 * v])
    vol = M.render(G1_SHAPE, G1, np.vstack([LONE_C, pair]),
                   np.vstack([LONE_U, [u, u]]), 5000.0, rng=rng)
    return vol, pair


def _pair_error(cands, pair):
    d = np.linalg.norm(cands.centers_ras[:, None, :] - pair[None, :, :], axis=2)
    near = cands.centers_ras[d.min(axis=1) < 2.5]
    if len(near) != 2:
        return np.inf, len(near)
    e = np.linalg.norm(near[:, None, :] - pair[None, :, :], axis=2)
    return min(max(e[0, 0], e[1, 1]), max(e[0, 1], e[1, 0])), 2


# ---------------------------------------------------------------- split

@pytest.mark.parametrize("d", [2.5, 3.0, 3.5])
@pytest.mark.parametrize("cfg", [
    ("oblique, stacked", [1, 0, 1], [0, 0, 1]),
    ("in-plane, diagonal offset", [1, 0, 0], [0, 1, 1]),
])
def test_side_by_side_pair_splits_into_two(cfg, d):
    name, axis, off = cfg
    vol, pair = _side_by_side(axis, off, d)
    if d == 2.5:
        # the blooms really touch: without the split they are ONE blob
        whole = detect_seed_candidates(vol, hu_threshold=2000.0, min_mm3=1.0,
                                       max_mm3=60.0, split_merged=False)
        assert len(whole) == 5, "%s at %.1f mm is not a merged blob" % (name, d)
    cands = detect_seed_candidates(vol, merge_fragments=True, **THIN)
    err, n_near = _pair_error(cands, pair)
    assert len(cands) == 6 and n_near == 2, (name, d, len(cands))
    assert err <= 0.5, "%s %.1f mm: worst pair error %.2f mm" % (name, d, err)
    if d == 2.5:
        # split siblings carry one source blob and are never re-merged
        near = np.linalg.norm(cands.centers_ras - pair.mean(axis=0), axis=1) < 2.5
        assert len(set(cands.info["source_blob"][near])) == 1
        assert set(cands.info["split_k"][near]) == {2}


def test_weighted_split_beats_kmeans_on_touching_pairs():
    """The two 2.5 mm one-blob pairs: k-means++ on voxel positions with a
    volume-ratio k (the historical method) cuts one across both seeds and
    the other into three; the intensity-weighted split gets both."""
    bad = 0
    for axis, off in (([1, 0, 1], [0, 0, 1]), ([1, 0, 0], [0, 1, 1])):
        vol, pair = _side_by_side(axis, off, 2.5)
        new = detect_seed_candidates(vol, **THIN)
        old = detect_seed_candidates(vol, split_weighted=False, **THIN)
        assert _pair_error(new, pair)[0] <= 0.5
        e_old, _n = _pair_error(old, pair)
        bad += int(len(old) != 6 or e_old > 0.5)
    assert bad == 2


def test_noise_specks_cause_no_false_split():
    """10 seeds + 30 one- and two-voxel specks at 2200 HU: the specks are
    admitted as candidates (default 0.2 mm^3 floor) but must not drag the
    lone-seed reference down far enough to cut real seeds."""
    rng = np.random.default_rng(3)
    centers = np.array([[x, y, z] for x in (6.0, 15.0, 24.0) for y in (6.0, 24.0)
                        for z in (8.0,)] + [[15.0, 15.0, 12.0], [6.0, 15.0, 14.0],
                                            [24.0, 15.0, 6.0], [15.0, 6.0, 15.0]])
    axes = np.array([M.random_axis(rng) for _ in centers])
    specks = []
    taken = set()
    while len(specks) < 30:
        kji = (int(rng.integers(2, 30)), int(rng.integers(2, 50)), int(rng.integers(2, 50)))
        ras = np.array([kji[2] * G1[0], kji[1] * G1[1], kji[0] * G1[2]])
        if np.min(np.linalg.norm(centers - ras, axis=1)) < 5.0 or kji in taken:
            continue
        taken.add(kji)
        specks.append((kji, 2200.0))
        if len(specks) % 2:   # every other speck is two voxels
            specks.append(((kji[0], kji[1], kji[2] + 1), 2200.0))
            taken.add((kji[0], kji[1], kji[2] + 1))
    n_specks = len({(k, j) for (k, j, _i), _h in specks})  # blobs, not voxels
    vol = M.render(G1_SHAPE, G1, centers, axes, 5000.0, specks=specks, rng=5)
    cands = detect_seed_candidates(vol)            # default 0.2 mm^3 floor
    assert np.all(cands.info["split_k"] == 1), "a blob was split"
    assert len(cands) == len(centers) + n_specks
    d = np.linalg.norm(cands.centers_ras[:, None, :] - centers[None], axis=2)
    assert np.all(d.min(axis=0) < 0.5)


def test_reference_median_ignores_oversize_blobs():
    """Dense-bone-sized blobs above the per-seed window outnumber the seeds:
    with the all-blob median the touching pair is not recognised and stays
    one candidate between the two seeds; with the windowed median it is
    split.  (The oversize boxes themselves are then cut into seed-sized
    pieces too -- clutter left to the shape, vault and tile stages.)"""
    vol, pair = _side_by_side([1, 0, 1], [0, 0, 1], 2.5)
    arr = np.array(vol.array)
    for n, (k, j, i) in enumerate([(3, 4, 40), (3, 14, 40), (3, 30, 40),
                                   (26, 4, 4), (26, 40, 20), (26, 44, 44)]):
        arr[k:k + 3, j:j + 4 + n % 3, i:i + 6] = 2500.0   # 25-38 mm^3 each
    v2 = Volume(arr, vol.affine)
    legacy = detect_seed_candidates(v2, split_window=False, **THIN)
    err, n_near = _pair_error(legacy, pair)
    assert n_near == 1 and not np.isfinite(err)
    cands = detect_seed_candidates(v2, **THIN)
    err, n_near = _pair_error(cands, pair)
    assert n_near == 2 and err <= 0.5


def _rod_population(rng=4):
    """Four dim seeds (small thresholded blobs) and one brighter seed whose
    blob is ~1.8x their volume: the volume rule flags the single bright
    seed as a merged pair."""
    dim = M.render(G1_SHAPE, G1, LONE_C, LONE_U, 3300.0, rng=rng, clip=None)
    bright_c = np.array([[15.2, 15.1, 15.3]])
    bright_u = np.array([_unit([1, 0.3, 0.2])])
    bright = M.render(G1_SHAPE, G1, bright_c, bright_u, 4600.0, noise=0.0, clip=None)
    arr = np.minimum(np.asarray(dim.array) + np.asarray(bright.array) - 35.0, 3071.0)
    return Volume(arr.astype(np.float32), dim.affine), bright_c[0]


def test_halves_guard_keeps_one_capsule_whole():
    vol, c = _rod_population()
    unguarded = detect_seed_candidates(vol, split_guard=False, **THIN)
    near = np.linalg.norm(unguarded.centers_ras - c, axis=1) < SEED_LENGTH_MM
    assert near.sum() == 2, "scene does not provoke a halves split"
    halves = unguarded.centers_ras[near]
    assert np.linalg.norm(halves[0] - halves[1]) < MERGE_MAX_SEP_MM
    cands = detect_seed_candidates(vol, **THIN)
    near = np.linalg.norm(cands.centers_ras - c, axis=1) < SEED_LENGTH_MM
    assert near.sum() == 1
    assert np.linalg.norm(cands.centers_ras[near][0] - c) < 0.3
    # split siblings are never re-merged, even though collinear and close
    both = detect_seed_candidates(vol, split_guard=False, merge_fragments=True,
                                  **THIN)
    assert (np.linalg.norm(both.centers_ras - c, axis=1) < SEED_LENGTH_MM).sum() == 2


# ---------------------------------------------------------------- merge

@pytest.mark.parametrize("elev,zc,tol", [(40, 13.0, 0.3), (45, 13.5, 1.0)])
def test_slab_gap_fragments_merge_to_one(elev, zc, tol):
    """1 mm slices every 2 mm (the PostOp export): an oblique capsule
    crossing the unimaged gap leaves two non-touching traces one slice
    apart; the merge returns one candidate near the true centre."""
    shape, mid = M.scene_box("G2s")
    e = np.radians(elev)
    u = np.array([np.cos(e) * np.cos(0.5), np.cos(e) * np.sin(0.5), np.sin(e)])
    c = np.array([mid[0] + 0.13, mid[1] + 0.21, zc])
    vol = M.render_grid("G2s", shape, [c], [u], rng=3)
    plain = M.detect(vol)
    assert len(plain) == 2
    sep = np.linalg.norm(plain.centers_ras[0] - plain.centers_ras[1])
    assert sep < MERGE_MAX_SEP_MM
    merged = M.detect(vol, merge_fragments=True)
    assert len(merged) == 1
    assert int(merged.info["n_fragments"][0]) == 2
    assert len(merged.info["merge_log"]) == 1
    assert np.linalg.norm(merged.centers_ras[0] - c) < tol
    assert np.linalg.norm(merged.centers_ras[0] - c) < \
        np.min(np.linalg.norm(plain.centers_ras - c, axis=1))


def test_merge_refused_in_crowded_neighbourhood():
    """Same fragment pair with a second (vertical) seed 3.5 mm away whose
    candidate lies within the merge distance of a fragment: the group is
    not isolated, so nothing is merged rather than risk fusing two seeds."""
    shape, mid = M.scene_box("G2s")
    e = np.radians(40)
    u = np.array([np.cos(e) * np.cos(0.5), np.cos(e) * np.sin(0.5), np.sin(e)])
    c = np.array([mid[0] + 0.13, mid[1] + 0.21, 13.0])
    other = c + np.array([0.0, 3.5, 0.0])
    lone = M.render_grid("G2s", shape, [c], [u], rng=3)
    assert len(M.detect(lone, merge_fragments=True)) == 1
    vol = M.render_grid("G2s", shape, [c, other], [u, [0.0, 0.0, 1.0]], rng=3)
    plain = M.detect(vol)
    assert len(plain) == 3
    d = np.linalg.norm(plain.centers_ras[:, None] - plain.centers_ras[None], axis=2)
    assert (d[np.triu_indices(3, 1)] < MERGE_MAX_SEP_MM).sum() >= 2
    merged = M.detect(vol, merge_fragments=True)
    assert len(merged) == 3
    assert not merged.info["merge_log"]


def test_merge_is_a_noop_on_thin_and_unfragmented_scans():
    vol, pair = _side_by_side([1, 0, 1], [0, 0, 1], 3.0)
    a = detect_seed_candidates(vol, **THIN)
    b = detect_seed_candidates(vol, merge_fragments=True, **THIN)
    assert np.array_equal(a.centers_ras, b.centers_ras)
    assert b.info["merge_log"] == ()


def test_info_fields_follow_subset():
    shape, mid = M.scene_box("G2s")
    e = np.radians(40)
    u = np.array([np.cos(e) * np.cos(0.5), np.cos(e) * np.sin(0.5), np.sin(e)])
    c = np.array([mid[0] + 0.13, mid[1] + 0.21, 13.0])
    far = c + np.array([0.0, 0.0, 8.0])
    vol = M.render_grid("G2s", shape, [c, far], [u, [1.0, 0.0, 0.0]], rng=3)
    cands = M.detect(vol, merge_fragments=True)
    n = len(cands)
    for key in ("source_blob", "split_k", "n_fragments"):
        assert cands.info[key].shape == (n,)
    assert isinstance(cands.info["merge_log"], tuple)
    sub = cands.subset(cands.info["n_fragments"] > 1)
    assert len(sub) == 1 and sub.info["n_fragments"].tolist() == [2]
    assert sub.info["merge_log"] == cands.info["merge_log"]


def test_pipeline_merge_default_policy():
    arr = np.zeros((4, 4, 4), np.float32)

    def vol(spacing, thickness=None):
        v = Volume(arr, np.diag(list(spacing) + [1.0]))
        if thickness is not None:
            v.meta["slice_thickness"] = thickness
        return v

    assert merge_fragments_default(vol((0.5, 0.5, 2.0), 1.0))[0]        # PostOp
    assert not merge_fragments_default(vol((0.5, 0.5, 2.0), 2.0))[0]    # contiguous
    assert not merge_fragments_default(vol((0.5, 0.5, 2.0)))[0]         # unknown
    assert not merge_fragments_default(vol((0.59, 0.59, 1.0), 0.5))[0]  # thin


def test_merge_distance_sweep_printed():
    """Sensitivity of the merge to MERGE_MAX_SEP_MM on the PostOp geometry
    (small sample; the full table is scripts/localization_mergesplit_sweep.py
    and docs/localization-notes.md, stage 3)."""
    caps = (3.0, 3.5, 4.0, 4.5, 5.0)
    rows = M.sweep(caps=caps, grids=("G2s",), n_single=40, n_pair=6, seed=11)
    print("\n" + M.format_sweep(rows))
    still = [r["still_fragmented"] for r in rows]
    assert rows[0]["fragmented"] > 0
    assert all(a >= b for a, b in zip(still, still[1:]))
    assert rows[caps.index(MERGE_MAX_SEP_MM)]["still_fragmented"] == 0
