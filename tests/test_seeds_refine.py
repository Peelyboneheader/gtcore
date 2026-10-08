"""Stage 2 of docs/plan-localization.md: threshold-free grey-level centroid
with an analytic per-seed covariance (gtcore.seeds.refine).

Acceptance (plan, "Verification"; numbers recorded in
docs/localization-notes.md):

* head phantom (binary capsules, 0.7 mm, rng 0-2), slices block-averaged
  to 0.7 / 1.4 / 2.1 / 2.8 mm:
  - 0.7 mm mean error not worse than detection by > 0.02 mm and no seed
    worse by > 0.3 mm;
  - 2.1 and 2.8 mm mean error >= 15 % better than detection;
  - centres from 1200 vs 2000 HU detections agree (median <= 0.1 mm);
* two seeds 7 mm apart (side by side and end to end) both within 0.15 mm;
* a non-seed candidate (a plate) falls back without crashing;
* ``subset`` after refinement keeps ``cov_ras`` and the per-seed status;
* covariance calibration: NEES mean in [0.5, 2] at 2.1 mm slices (analytic
  supersampled seeds -- the binary head phantom's in-plane error is its
  own voxelized rendering, not estimator error; see the notes);
* <= 5 ms per seed (asserted <= 20 ms for CI slack).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import cdist

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import (
    SeedCandidates,
    detect_seed_candidates,
    estimate_saturation,
    grey_centroid,
    refine_seed_candidates,
    seed_roi,
)
from gtcore.seeds.refine import sampling_variance
from gtcore.volume import Volume

RNG_SEEDS = (0, 1, 2)
FACTORS = (1, 2, 3, 4)          # 0.7, 1.4, 2.1, 2.8 mm slices


# ------------------------------------------------------------------ helpers
def _thick_slices(vol, factor):
    """Block-average along k; mirrors scripts/validation_spacing.py."""
    if factor == 1:
        return vol
    nk = (vol.array.shape[0] // factor) * factor
    arr = vol.array[:nk].reshape(-1, factor, *vol.array.shape[1:]).mean(axis=1)
    affine = vol.affine.copy()
    affine[:3, 2] *= factor
    affine[:3, 3] += vol.affine[:3, 2] * (factor - 1) / 2.0
    return Volume(arr.astype(np.float32), affine, dict(vol.meta))


def _detect(vol, hu_threshold=None):
    p = seed_detection_params(vol.spacing)
    thr = p["hu_threshold"] if hu_threshold is None else hu_threshold
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=thr, min_mm3=p["min_mm3"],
                               max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


def _match(truth_pts, est_pts, gate=2.0):
    """Greedy truth -> candidate matching as scripts/validation_spacing.py:
    ``{truth index: candidate index}``."""
    if not len(est_pts):
        return {}
    D = cdist(truth_pts, est_pts)
    used, out = set(), {}
    for ti in range(len(truth_pts)):
        for j in np.argsort(D[ti]):
            if int(j) not in used:
                if D[ti, int(j)] < gate:
                    used.add(int(j))
                    out[ti] = int(j)
                break
    return out


def _render_capsules(spacing, centers, axes, *, n_fine=(3, 3, 7),
                     metal_hu=12000.0, bg_hu=20.0, psf=0.45, noise=20.0,
                     rng=0, clip=None, origin=None, shape_ijk=None):
    """Supersampled 4.5 x 0.8 mm capsules: anti-aliased indicator on a fine
    grid nested in the voxels, Gaussian blur, box-average to the voxel
    footprint, white noise, optional clip.  (A test stand-in for the stage-0
    harness renderer; deliberately not the stage-7 fit model.)"""
    s = np.asarray(spacing, float)
    centers = np.atleast_2d(np.asarray(centers, float))
    axes = np.atleast_2d(np.asarray(axes, float))
    lo = (np.floor((centers.min(0) - 10.0) / 8.4) * 8.4 if origin is None
          else np.asarray(origin, float))
    if shape_ijk is None:
        shape_ijk = np.ceil((centers.max(0) + 10.0 - lo) / s).astype(int) + 1
    affine = np.diag([s[0], s[1], s[2], 1.0])
    affine[:3, 3] = lo
    arr = np.full(tuple(np.asarray(shape_ijk)[::-1]), bg_hu, dtype=np.float64)
    nf = np.asarray(n_fine)
    fs = s / nf
    R, L = 0.4, 4.5
    for c, u in zip(centers, axes):
        u = u / np.linalg.norm(u)
        p = np.round((c - lo) / s).astype(int)
        half = np.ceil((L / 2 + R + 3.0) / s).astype(int)
        i0, i1 = p - half, p + half + 1
        ax_ = [lo[a] + (np.arange(i0[a] * nf[a], i1[a] * nf[a]) + 0.5) * fs[a]
               - s[a] / 2 for a in range(3)]
        X, Y, Z = np.meshgrid(*ax_, indexing="ij")
        e = np.stack([X - c[0], Y - c[1], Z - c[2]], -1)
        t = np.clip(e @ u, -(L / 2 - R), L / 2 - R)
        d = np.linalg.norm(e - t[..., None] * u, axis=-1)
        ind = np.clip((R - d) / fs.max() + 0.5, 0.0, 1.0)
        fine = ndimage.gaussian_filter(ind, sigma=psf / fs, mode="constant",
                                       truncate=4.0)
        n = i1 - i0
        coarse = fine.reshape(n[0], nf[0], n[1], nf[1], n[2], nf[2]).mean(
            axis=(1, 3, 5))
        patch = metal_hu * coarse.T                       # [k, j, i]
        lo_ = np.maximum(i0, 0)
        hi_ = np.minimum(i1, np.asarray(arr.shape[::-1]))
        arr[lo_[2]:hi_[2], lo_[1]:hi_[1], lo_[0]:hi_[0]] += patch[
            lo_[2] - i0[2]:hi_[2] - i0[2], lo_[1] - i0[1]:hi_[1] - i0[1],
            lo_[0] - i0[0]:hi_[0] - i0[0]]
    arr += np.random.default_rng(rng).normal(0.0, noise, arr.shape)
    if clip is not None:
        arr = np.minimum(arr, clip)
    return Volume(arr.astype(np.float32), affine)


def _random_layout(n_side, rng, z_layers=2, pitch=16.0):
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side),
                             np.arange(z_layers), indexing="ij"),
                 -1).reshape(-1, 3) * pitch
    centers = g + rng.uniform(-2.0, 2.0, g.shape)
    axes = rng.normal(size=g.shape)
    return centers, axes / np.linalg.norm(axes, axis=1, keepdims=True)


# ------------------------------------------------------------- head phantom
@pytest.fixture(scope="module")
def head():
    """Per slice factor: baseline / refined errors over rng 0-2."""
    out = {f: dict(base=[], ref=[], base_z=[], ref_z=[], status=[],
                   nees_z=[], ms=[], n=0) for f in FACTORS}
    thr = {}
    for r in RNG_SEEDS:
        vol0, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=r)
        t = np.array([s.center_ras for s in truth.seeds])
        for f in FACTORS:
            vol = _thick_slices(vol0, f)
            cands = _detect(vol)
            t0 = time.perf_counter()
            ref = refine_seed_candidates(vol, cands)
            dt = time.perf_counter() - t0
            row = out[f]
            row["ms"].append(1e3 * dt / max(1, len(cands)))
            for ti, j in _match(t, cands.centers_ras).items():
                e0 = cands.centers_ras[j] - t[ti]
                e1 = ref.centers_ras[j] - t[ti]
                row["base"].append(np.linalg.norm(e0))
                row["ref"].append(np.linalg.norm(e1))
                row["base_z"].append(abs(e0[2]))
                row["ref_z"].append(abs(e1[2]))
                row["status"].append(ref.info["refine_status"][j])
                if ref.info["refine_status"][j] == "ok":
                    row["nees_z"].append(e1[2] ** 2 / ref.cov_ras[j][2, 2])
            if f in (2, 3):
                # threshold sensitivity: 1200 vs 2000 HU detections
                lo, hi = _detect(vol, 1200.0), _detect(vol, 2000.0)
                rlo = refine_seed_candidates(vol, lo)
                rhi = refine_seed_candidates(vol, hi)
                mlo, mhi = _match(t, lo.centers_ras), _match(t, hi.centers_ras)
                for ti in set(mlo) & set(mhi):
                    a, b = mlo[ti], mhi[ti]
                    thr.setdefault(f, []).append((
                        np.linalg.norm(lo.centers_ras[a] - hi.centers_ras[b]),
                        np.linalg.norm(rlo.centers_ras[a] - rhi.centers_ras[b])))
    for f, row in out.items():
        for k in ("base", "ref", "base_z", "ref_z", "nees_z", "ms"):
            row[k] = np.asarray(row[k], dtype=float)
        nfb = sum(1 for s in row["status"] if s != "ok")
        print("dz %.1f mm: n %2d  mean %.3f -> %.3f  max %.3f -> %.3f  "
              "|z| %.3f -> %.3f  fallbacks %d  NEES_z %.2f  %.2f ms/seed"
              % (0.7 * f, len(row["base"]), row["base"].mean(),
                 row["ref"].mean(), row["base"].max(), row["ref"].max(),
                 row["base_z"].mean(), row["ref_z"].mean(), nfb,
                 row["nees_z"].mean(), row["ms"].mean()))
    return out, {f: np.asarray(v) for f, v in thr.items()}


def test_thin_slices_not_worse(head):
    row = head[0][1]
    assert row["ref"].mean() <= row["base"].mean() + 0.02, (
        row["ref"].mean(), row["base"].mean())
    assert (row["ref"] - row["base"]).max() <= 0.3


@pytest.mark.parametrize("factor", [3, 4])
def test_thick_slices_gain(head, factor):
    row = head[0][factor]
    assert row["ref"].mean() <= 0.85 * row["base"].mean(), (
        row["ref"].mean(), row["base"].mean())


@pytest.mark.parametrize("factor", [2, 3])
def test_threshold_independence(head, factor):
    pairs = head[1][factor]
    assert len(pairs) >= 20
    base, ref = np.median(pairs[:, 0]), np.median(pairs[:, 1])
    print("dz %.1f: centre shift 1200 vs 2000 HU, median %.3f -> %.3f mm"
          % (0.7 * factor, base, ref))
    assert ref <= 0.1 and ref <= 0.5 * base


def test_runtime_per_seed(head):
    ms = np.concatenate([head[0][f]["ms"] for f in FACTORS])
    print("refinement: %.2f ms/seed mean, %.2f max" % (ms.mean(), ms.max()))
    assert ms.mean() <= 20.0


def test_head_phantom_z_calibration(head):
    """Through-slab NEES of the refined seeds on the head phantom: the
    sampling term dominates z at 2.1-2.8 mm and is checked there (in-plane
    the binary phantom's voxelized capsule, not the estimator, dominates)."""
    for f in (3, 4):
        nz = head[0][f]["nees_z"]
        assert 0.25 <= nz.mean() <= 2.0, (f, nz.mean())


# ------------------------------------------------------- analytic seeds
@pytest.fixture(scope="module")
def coarse_2p1():
    rng = np.random.default_rng(7)
    centers, axes = _random_layout(4, rng)
    vol = _render_capsules((0.5, 0.5, 2.1), centers, axes, noise=20.0, rng=3)
    cands = detect_seed_candidates(vol, hu_threshold=1000.0, min_mm3=0.2,
                                   max_mm3=200.0)
    return vol, centers, cands, refine_seed_candidates(vol, cands)


def test_nees_2p1mm(coarse_2p1):
    vol, centers, cands, ref = coarse_2p1
    D = cdist(ref.centers_ras, centers)
    assert len(ref) == len(centers) and (D.min(axis=1) < 1.0).all()
    ok = ref.info["refine_status"] == "ok"
    assert ok.mean() >= 0.9
    e = ref.centers_ras - centers[D.argmin(axis=1)]
    nees = np.array([v @ np.linalg.solve(C, v) / 3.0
                     for v, C in zip(e[ok], ref.cov_ras[ok])])
    e0 = cands.centers_ras - centers[D.argmin(axis=1)]
    print("analytic 0.5x0.5x2.1 mm: mean error %.3f -> %.3f mm, NEES %.2f"
          % (np.linalg.norm(e0, axis=1).mean(), np.linalg.norm(e, axis=1).mean(),
             nees.mean()))
    assert 0.5 <= nees.mean() <= 2.0
    assert np.linalg.norm(e, axis=1).mean() < 0.5 * np.linalg.norm(e0, axis=1).mean()


def test_subset_keeps_cov_and_status(coarse_2p1):
    _, _, _, ref = coarse_2p1
    sub = ref.subset([0, 2])
    assert sub.cov_ras.shape == (2, 3, 3)
    assert np.allclose(sub.cov_ras, ref.cov_ras[[0, 2]])
    assert list(sub.info["refine_status"]) == list(ref.info["refine_status"][[0, 2]])
    assert sub.info["refine_method"] == "centroid"


@pytest.mark.parametrize("sep, label", [((0.0, 7.0, 0.0), "side by side"),
                                        ((7.0, 0.0, 0.0), "end to end")])
def test_neighbour_7mm(sep, label):
    rng = np.random.default_rng(11)
    worst = 0.0
    for trial in range(3):
        c0 = rng.uniform(-0.35, 0.35, 3)
        centers = np.array([c0, c0 + np.asarray(sep)])
        axes = np.array([[1.0, 0.0, 0.0]] * 2)
        vol = _render_capsules((0.7, 0.7, 0.7), centers, axes,
                               n_fine=(3, 3, 3), rng=trial)
        cands = detect_seed_candidates(vol, hu_threshold=2000.0)
        assert len(cands) == 2
        ref = refine_seed_candidates(vol, cands)
        assert list(ref.info["refine_status"]) == ["ok", "ok"]
        worst = max(worst, cdist(ref.centers_ras, centers).min(axis=1).max())
    print("%s, 7 mm: worst error %.3f mm" % (label, worst))
    assert worst <= 0.15


def test_unmasked_end_to_end_neighbour_is_caught():
    """Without masking the neighbour's end sits in the shell: the
    'extended' check refuses the centroid instead of silently biasing it."""
    centers = np.array([[0.1, 0.2, -0.1], [7.1, 0.2, -0.1]])
    vol = _render_capsules((0.7, 0.7, 0.7), centers, [[1.0, 0, 0]] * 2,
                           n_fine=(3, 3, 3))
    cands = detect_seed_candidates(vol, hu_threshold=2000.0)
    ref = refine_seed_candidates(vol, cands, mask="none")
    assert all(s.startswith("fallback") for s in ref.info["refine_status"])
    assert np.allclose(ref.centers_ras, cands.centers_ras)


def test_non_seed_candidate_falls_back():
    """A plate-like candidate (bone plate / clip) next to a real seed."""
    rng = np.random.default_rng(5)
    seed_c = np.array([[0.0, 0.0, 0.0]])
    vol = _render_capsules((0.7, 0.7, 0.7), seed_c, [[0.0, 1.0, 0.0]],
                           n_fine=(3, 3, 3), origin=(-20.0, -20.0, -20.0),
                           shape_ijk=(58, 58, 58))
    arr = vol.array.astype(float)
    plate = np.zeros(arr.shape)
    ijk = np.round((np.array([12.0, 0.0, 0.0]) + 20.0) / 0.7).astype(int)
    plate[ijk[2] - 1:ijk[2] + 1, ijk[1] - 9:ijk[1] + 9, ijk[0] - 9:ijk[0] + 9] = 1500.0
    arr += ndimage.gaussian_filter(plate, 0.45 / 0.7)
    vol = Volume(arr.astype(np.float32), vol.affine)
    cands = SeedCandidates(
        mask=np.zeros(arr.shape, bool),
        centers_ras=np.array([[12.0, 0.0, 0.0], seed_c[0] + rng.normal(0, 0.2, 3)]),
        axes_ras=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        volumes_mm3=np.array([40.0, 3.0]), elongations=np.array([1.2, 3.0]))
    ref = refine_seed_candidates(vol, cands)
    st = list(ref.info["refine_status"])
    assert st[0].startswith("fallback:"), st
    assert np.allclose(ref.centers_ras[0], cands.centers_ras[0])
    assert st[1] == "ok"
    assert np.linalg.norm(ref.centers_ras[1] - seed_c[0]) < 0.1
    # a fallback reports the voxel-quantization covariance s^2/12
    assert np.allclose(np.diag(ref.cov_ras[0]), 0.7 ** 2 / 12.0)


def test_saturated_seeds_stay_unbiased():
    rng = np.random.default_rng(3)
    centers, axes = _random_layout(3, rng, z_layers=1)
    vol = _render_capsules((0.59, 0.59, 1.0), centers, axes, n_fine=(3, 3, 5),
                           clip=1500.0, rng=4)
    assert estimate_saturation(vol.array) == pytest.approx(1500.0)
    cands = detect_seed_candidates(vol, hu_threshold=1000.0)
    ref = refine_seed_candidates(vol, cands)
    assert ref.info["saturation_hu"] == pytest.approx(1500.0)
    assert (ref.info["refine_n_saturated"] > 0).all()
    e = ref.centers_ras - centers[cdist(ref.centers_ras, centers).argmin(axis=1)]
    assert (ref.info["refine_status"] == "ok").all()
    assert np.abs(e.mean(axis=0)).max() <= 0.05
    assert np.linalg.norm(e, axis=1).max() <= 0.1


# ------------------------------------------------------------------- units
def test_estimate_saturation():
    a = np.random.default_rng(0).normal(0, 50, (10, 10, 10))
    assert estimate_saturation(a) is None
    a[0, 0, :6] = 3071.0
    assert estimate_saturation(a) == 3071.0
    assert estimate_saturation(np.zeros((0, 0, 0))) is None


def test_sampling_variance_limits():
    s = np.array([0.5, 0.5, 2.0])
    thin = sampling_variance(s, np.zeros(3), psf_sigma_mm=1e-6, length_mm=0.0,
                             diameter_mm=0.0)
    assert np.allclose(thin, s ** 2 / 12.0, rtol=2e-2)
    wide = sampling_variance(s, np.zeros(3), psf_sigma_mm=0.45)
    assert wide[0] < 1e-6 and 1e-3 < wide[2] < s[2] ** 2 / 12.0
    # a seed spanning exactly one slab along z has no sampling error
    along = sampling_variance(s, np.array([0.0, 0.0, 1.0]), length_mm=2.0)
    assert along[2] < 1e-4


def test_seed_roi_and_grey_centroid_api():
    c = np.array([[0.13, -0.21, 0.05]])
    vol = _render_capsules((0.7, 0.7, 0.7), c, [[0.0, 0.0, 1.0]],
                           n_fine=(3, 3, 3))
    pts, vals, kji = seed_roi(vol, c[0] + 0.3, axis=[0.0, 0.0, 1.0])
    assert len(pts) == len(vals) == len(kji) > 50
    assert np.allclose(vol.array[kji[:, 0], kji[:, 1], kji[:, 2]], vals)
    pts_e, _, _ = seed_roi(vol, c[0], axis=None)
    assert len(pts_e) > len(pts)          # the orientation-free ellipsoid
    # a neighbour 3 mm away removes the far half-space from the window
    pts_n, _, _ = seed_roi(vol, c[0], others=c + [3.0, 0, 0], axis=[0, 0, 1.0])
    assert (pts_n[:, 0] <= c[0, 0] + 1.5 + 1e-9).all()
    r = grey_centroid(vol, c[0] + [0.3, -0.2, 0.4], axis=[1.0, 0, 0])
    assert r.status == "ok"
    assert np.linalg.norm(r.center_ras - c[0]) < 0.05
    assert abs(abs(r.axis_ras[2]) - 1.0) < 0.05
    assert r.cov_ras.shape == (3, 3) and np.all(np.linalg.eigvalsh(r.cov_ras) > 0)


def test_unknown_options_raise():
    vol = Volume(np.zeros((8, 8, 8), np.float32), np.eye(4))
    cands = SeedCandidates(np.zeros((8, 8, 8), bool), np.zeros((0, 3)),
                           np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError):
        refine_seed_candidates(vol, cands, method="magic")
    with pytest.raises(ValueError):
        refine_seed_candidates(vol, cands, slab="nope")
    empty = refine_seed_candidates(vol, cands)
    assert len(empty) == 0 and empty.cov_ras.shape == (0, 3, 3)


# ---------------------------------------------------------------- pipeline
def test_pipeline_flag(monkeypatch):
    """``reconstruct(refine_seeds="centroid")`` refines on the raw volume
    and logs the per-seed status; the default leaves seeds untouched.
    Segmentation is stubbed (its cost is irrelevant here)."""
    import gtcore.pipeline as pl

    def fake_head(vol, metal_mask=None):
        shape = vol.array.shape
        return dict(body=np.ones(shape, bool), skull=np.zeros(shape, bool),
                    cranial_interior=np.ones(shape, bool),
                    brain=np.zeros(shape, bool))

    monkeypatch.setattr(pl, "segment_head", fake_head)
    monkeypatch.setattr(pl, "segment_cavity",
                        lambda clean, interior, brain, seeds: np.zeros(
                            clean.array.shape, bool))
    monkeypatch.setattr(pl, "mask_to_mesh", lambda m, a, step_size=1: None)

    vol0, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=1)
    vol = _thick_slices(vol0, 3)
    t = np.array([s.center_ras for s in truth.seeds])
    plain = pl.reconstruct(vol.copy_with(), verbose=False)
    assert plain.seeds.cov_ras is None
    v2 = vol.copy_with()
    res = pl.reconstruct(v2, verbose=False, refine_seeds="centroid")
    log = v2.meta["seed_refine"]
    assert log["method"] == "centroid" and log["n"] == len(res.seeds)
    assert len(log["status"]) == len(res.seeds) and log["n_ok"] >= 1
    assert res.seeds.cov_ras.shape == (len(res.seeds), 3, 3)
    assert len(res.seeds) == len(plain.seeds)
    e_plain = np.mean([np.linalg.norm(plain.seeds.centers_ras[j] - t[i])
                       for i, j in _match(t, plain.seeds.centers_ras).items()])
    e_ref = np.mean([np.linalg.norm(res.seeds.centers_ras[j] - t[i])
                     for i, j in _match(t, res.seeds.centers_ras).items()])
    assert e_ref < e_plain
    with pytest.raises(ValueError):
        pl.reconstruct(vol.copy_with(), verbose=False, refine_seeds="bogus")


# ------------------------------------------------- merged-harness follow-ups
def test_interpolated_slices_are_kept_and_inflate_z():
    """PostOp-like gap volume: 1 mm slabs kept at irregular 1-3 mm steps and
    re-gridded by the DICOM loader's rule onto 2 mm with the off-grid slices
    interpolated (``meta["interpolated_k"]``).  Interpolated voxels stay in
    the centroid (linear filling preserves the measured slices' first
    moment) and every seed whose window touches one reports the uniform-slab
    z variance s_k^2/12."""
    from gtcore.phantom.seed_render import drop_and_interpolate

    rng = np.random.default_rng(21)
    centers, axes = _random_layout(3, rng, z_layers=2)
    fine = _render_capsules((0.5, 0.5, 1.0), centers, axes, n_fine=(3, 3, 5),
                            metal_hu=8000.0, rng=5)
    steps = np.random.default_rng(3).choice([1, 2, 3], size=fine.array.shape[0],
                                            p=[0.25, 0.5, 0.25])
    keep = np.concatenate([[0], np.cumsum(steps)])
    keep = keep[keep < fine.array.shape[0]]
    vol = drop_and_interpolate(fine, keep, grid="loader")
    assert vol.meta["interpolated_k"] and vol.spacing[2] == pytest.approx(2.0)
    cands = detect_seed_candidates(vol, hu_threshold=1000.0, min_mm3=0.5,
                                   max_mm3=60.0)
    ref = refine_seed_candidates(vol, cands)
    D = cdist(cands.centers_ras, centers)
    ti = D.argmin(axis=1)
    ok = (D.min(axis=1) < 2.0) & (ref.info["refine_status"] == "ok")
    assert ok.sum() >= 0.6 * len(centers)
    e0 = np.linalg.norm(cands.centers_ras[ok] - centers[ti[ok]], axis=1)
    e1 = np.linalg.norm(ref.centers_ras[ok] - centers[ti[ok]], axis=1)
    print("gap volume: %d seeds refined, mean error %.3f -> %.3f mm"
          % (ok.sum(), e0.mean(), e1.mean()))
    assert e1.mean() < e0.mean()
    interp = set(vol.meta["interpolated_k"])
    touched = 0
    for j in np.flatnonzero(ok):
        k = vol.ras_to_index(ref.centers_ras[j])[2]
        if any(int(round(k)) + d in interp for d in (-1, 0, 1)):
            touched += 1
            assert ref.cov_ras[j][2, 2] >= 2.0 ** 2 / 12.0 - 1e-9
    assert touched >= 1


def test_close_candidates_fall_back_to_detection():
    """Two candidates 2.6 mm apart on ONE seed (a threshold split): the
    Voronoi cut would halve the seed, so both keep their detections."""
    c = np.array([[0.11, -0.07, 0.03]])
    vol = _render_capsules((0.7, 0.7, 0.7), c, [[1.0, 0.0, 0.0]],
                           n_fine=(3, 3, 3))
    frag = np.array([c[0] - [1.3, 0, 0], c[0] + [1.3, 0, 0]])
    cands = SeedCandidates(np.zeros(vol.array.shape, bool), frag,
                           np.array([[1.0, 0, 0]] * 2), np.ones(2), np.ones(2))
    ref = refine_seed_candidates(vol, cands)
    assert list(ref.info["refine_status"]) == ["fallback:close_neighbour"] * 2
    assert np.allclose(ref.centers_ras, frag)


def test_axes_kept_unless_requested(coarse_2p1):
    vol, _, cands, ref = coarse_2p1
    assert np.allclose(ref.axes_ras, cands.axes_ras)
    ok = ref.info["refine_status"] == "ok"
    upd = refine_seed_candidates(vol, cands, update_axes=True)
    assert np.allclose(upd.axes_ras[ok], ref.info["refine_axis_ras"][ok])
    assert np.allclose(upd.centers_ras, ref.centers_ras)


def test_only_the_roi_leaving_the_volume_is_truncation():
    """A seed whose background shell, but not its ROI, crosses the last
    slice is refined; one whose ROI crosses it falls back."""
    s = (0.5, 0.5, 2.0)
    origin = (-12.0, -12.0, -12.0)
    shape = (49, 49, 13)                    # k = 0..12 -> z = -12..12 mm
    for z, expect in ((12.0 - 7.1, "ok"), (12.0 - 1.0, "fallback:roi_truncated")):
        c = np.array([[0.13, -0.1, z]])
        vol = _render_capsules(s, c, [[1.0, 0.0, 0.0]], origin=origin,
                               shape_ijk=shape, n_fine=(3, 3, 10))
        cands = SeedCandidates(np.zeros(vol.array.shape, bool), c + 0.1,
                               np.array([[1.0, 0, 0]]), np.ones(1), np.ones(1))
        ref = refine_seed_candidates(vol, cands)
        assert ref.info["refine_status"][0] == expect
