"""Stage 5 of docs/plan-localization: hierarchical (random-effects) weighted
least squares of the bent-tile fit and the posterior seed positions
(gtcore.tiles.deform seed_cov, gtcore.tiles.fuse, reconstruct(fuse_tiles)).

Acceptance:
- without a covariance the bent-tile fit is bit-identical to the historical
  one (frozen reference from commit 3cf35af, and the residual vector);
- the posterior is the exact conditional estimate: precision-weighted
  midpoint for equal isotropic covariances, moves only along a seed's
  poorly determined axis, shrinks the covariance (Loewner order), trusts
  tentative tiles less, passes unassigned seeds through;
- fit_tiles_auto / fit_tiles_prior forward the per-candidate covariance to
  every bent-tile fit;
- head phantom at 2.1 / 2.8 mm slices (5 realizations, analytic slab
  covariance as the stand-in): the numbers are PRINTED; the asserts are
  only "posterior not worse than raw by > 0.02 mm" and "tile centre /
  normal not worse" (the plan's 15 % gate is evaluated by the coordinator).
"""
from __future__ import annotations

import dataclasses
from functools import partial
from unittest import mock

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import ImplantPrior, fit_tiles_auto, fit_tiles_prior
from gtcore.tiles import auto as auto_mod
from gtcore.tiles import deform as deform_mod
from gtcore.tiles.deform import SLACK_MM, fit_deformable, seed_whitening
from gtcore.tiles.fit import TilePose
from gtcore.tiles.fuse import (
    posterior_seed_positions,
    slab_covariance_stand_in,
)
from gtcore.volume import Volume

# frozen with gtcore/tiles/deform.py at commit 3cf35af (before stage 5):
# phantom rng_seed=3 spacing 1.0, tile 0 truth seeds + N(0, 0.3) noise
REF_P = np.array([
    [27.344503665571974, -0.5692381110632276, 32.61547595724601],
    [21.749858657384895, 3.1282039152588017, 28.005791871549786],
    [21.012720920003343, -2.877101857404508, 36.836991082971565],
    [15.972709727530853, 0.32045807843816687, 32.006071425850536]])
REF_A = np.array([
    [-0.6822012218332999, -0.6050227702739772, 0.41054712321383396],
    [-0.7747886064674712, -0.5504958559896612, 0.31089697300939456],
    [-0.8072744261286867, -0.11647736554207705, 0.578568080898849],
    [-0.7146870414945155, -0.17100427982639288, 0.6782182311032884]])
REF_R = np.array([
    [-0.7572327425476194, 0.6530327201176559, 0.01211775843739056],
    [-0.36618283177228833, -0.4398288654803422, 0.820037013070449],
    [0.5408407411882353, 0.6165215612985577, 0.5721821887519303]])
REF_T = np.array([21.505188629055915, -0.9982369991429972, 31.669156009960798])
REF_KAPPA = (0.08314184086463797, 0.06539768308838653, 0.863768034690651)
REF_RMS = 0.21132809132062882
REF_ASSIGNMENT = (1, 0, 3, 2)


# ------------------------------------------------------------ helpers
def _thick_slices(vol, factor):
    """Block-average along k (partial-volume thick slices); mirrors
    scripts/validation_spacing.py."""
    if factor == 1:
        return vol
    nk = (vol.array.shape[0] // factor) * factor
    arr = vol.array[:nk].reshape(-1, factor, *vol.array.shape[1:]).mean(axis=1)
    affine = vol.affine.copy()
    affine[:3, 2] *= factor
    affine[:3, 3] += vol.affine[:3, 2] * (factor - 1) / 2.0
    return Volume(arr.astype(np.float32), affine, dict(vol.meta))


def _detect(vol):
    p = seed_detection_params(vol.spacing)
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


def _match(truth_c, det_c, gate=2.0):
    D = np.linalg.norm(det_c[:, None, :] - truth_c[None, :, :], axis=2)
    out = np.full(len(det_c), -1, dtype=int)
    for i, j in zip(*linear_sum_assignment(D)):
        if D[i, j] < gate:
            out[i] = j
    return out


def _pose(idx, fit, confidence="supported", degraded=False, tile_id=0):
    return TilePose(tile_id=tile_id, kind="full", seed_indices=list(idx),
                    center_ras=np.zeros(3), normal_ras=fit.pose.normal,
                    axis_ras=fit.pose.t1, residual_mm=fit.rms_mm,
                    degraded=degraded, deform=fit, confidence=confidence)


@pytest.fixture(scope="module")
def truth_tiles():
    _vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=3)
    c = np.array([s.center_ras for s in truth.seeds])
    a = np.array([s.axis_ras for s in truth.seeds])
    return c, a, truth


# ------------------------------------------- 1. identity without covariance
def test_no_covariance_is_bit_identical_to_the_historical_fit():
    f = fit_deformable(REF_P, REF_A)
    g = fit_deformable(REF_P, REF_A, seed_cov=None)
    for fit in (f, g):
        assert np.abs(fit.pose.R - REF_R).max() <= 1e-12
        assert np.abs(fit.pose.t - REF_T).max() <= 1e-12
        assert np.abs(np.array([fit.params.kappa1, fit.params.kappa2,
                                fit.params.psi]) - REF_KAPPA).max() <= 1e-12
        assert abs(fit.rms_mm - REF_RMS) <= 1e-12
        assert fit.assignment == REF_ASSIGNMENT
        assert fit.wrms_mm == fit.rms_mm
        assert fit.weighted_residuals_mm is None and fit.chi_rms is None


def test_residual_vector_unchanged_without_weights():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(scale=0.1, size=3), REF_T,
                        [0.05, 0.02, 0.3]])
    uv = np.array([[-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0]])
    R0 = np.eye(3)
    r_none = deform_mod._residuals(x, R0, REF_P, REF_A, uv, 4.0, 1.5)
    r_eye = deform_mod._residuals(x, R0, REF_P, REF_A, uv, 4.0, 1.5,
                                  np.repeat(np.eye(3)[None], 4, axis=0))
    # historical formula, written out
    R = R0 @ deform_mod._rodrigues(x[:3])
    pts, tu = deform_mod._sheet_and_tangent(uv, x[6], x[7], x[8])
    world = pts @ R.T + x[3:6]
    hist = np.concatenate([(world - REF_P).ravel(),
                           4.0 * np.linalg.norm(np.cross(tu @ R.T, REF_A), axis=1),
                           1.5 * x[6:8]])
    assert np.abs(r_none - hist).max() <= 1e-12
    assert np.abs(r_eye - hist).max() <= 1e-12


def test_isotropic_covariance_keeps_the_unweighted_fit():
    """Per-seed normalisation: an isotropic C_i whitens to the identity, so
    a thin isotropic scan's weighted fit IS the calibrated one."""
    cov = np.repeat((0.2 ** 2) * np.eye(3)[None], 4, axis=0)
    f0 = fit_deformable(REF_P, REF_A)
    f1 = fit_deformable(REF_P, REF_A, seed_cov=cov)
    assert np.abs(f1.pose.R - f0.pose.R).max() < 1e-9
    assert abs(f1.rms_mm - f0.rms_mm) < 1e-9
    assert f1.wrms_mm == pytest.approx(f1.rms_mm, abs=1e-9)
    # chi_rms is properly scaled by C = Sigma + slack^2 I
    c = 0.2 ** 2 + SLACK_MM ** 2
    assert f1.chi_rms == pytest.approx(f1.rms_mm / np.sqrt(3.0 * c), rel=1e-9)


def test_whitening_weights_are_clamped_and_normalised():
    cov = np.array([np.diag([0.01, 0.01, 5.0]), np.diag([0.05, 0.1, 0.2]),
                    np.zeros((3, 3))])
    W, Cinv = seed_whitening(cov, slack_mm=0.3)
    for i in range(3):
        w = np.linalg.eigvalsh(W[i])
        assert w.max() == pytest.approx(1.0, abs=1e-12)  # best direction: mm
        assert w.min() >= 0.5 - 1e-12                     # clamp: ratio <= 4
        C = cov[i] + 0.09 * np.eye(3)
        assert np.allclose(Cinv[i] @ C, np.eye(3), atol=1e-9)
    assert np.linalg.eigvalsh(W[0]).min() == pytest.approx(0.5, abs=1e-12)
    assert np.allclose(W[2], np.eye(3))
    with pytest.raises(ValueError):
        fit_deformable(REF_P, REF_A, seed_cov=np.zeros((3, 3, 3)))


# ------------------------------------------------------- 2. the posterior
def test_equal_isotropic_covariances_give_the_midpoint(truth_tiles):
    c, a, _ = truth_tiles
    rng = np.random.default_rng(1)
    x = c[:4] + rng.normal(scale=0.3, size=(4, 3))
    s = SLACK_MM
    cov = np.repeat(s ** 2 * np.eye(3)[None], 4, axis=0)
    fit = fit_deformable(x, a[:4], seed_cov=cov)
    post, pcov, info = posterior_seed_positions([_pose(range(4), fit)], x, cov,
                                                slack_mm=s)
    m = info["model_ras"]
    assert np.allclose(post, 0.5 * (x + m), atol=1e-12)
    assert np.allclose(pcov, 0.5 * cov, atol=1e-15)
    # the matched model points are the fit's own (its residuals reproduce)
    assert np.allclose(np.linalg.norm(x - m, axis=1), fit.residuals_mm,
                       atol=1e-9)
    assert info["tiles"][0]["match"] == "assignment"


def test_large_z_variance_moves_toward_the_model_only_along_z(truth_tiles):
    c, a, _ = truth_tiles
    rng = np.random.default_rng(2)
    x = c[4:8] + rng.normal(scale=0.4, size=(4, 3))
    cov = np.repeat(np.diag([1e-6, 1e-6, 1.0])[None], 4, axis=0)
    fit = fit_deformable(x, None, seed_cov=cov)
    post, pcov, info = posterior_seed_positions([_pose(range(4), fit)], x, cov)
    d = post - x
    m = info["model_ras"]
    assert np.abs(d[:, :2]).max() < 1e-4
    k = 1.0 / (1.0 + SLACK_MM ** 2)
    assert np.allclose(d[:, 2], k * (m - x)[:, 2], atol=1e-9)
    assert np.abs(d[:, 2]).max() > 0.01


def test_posterior_covariance_shrinks_in_the_loewner_order(truth_tiles):
    c, a, _ = truth_tiles
    rng = np.random.default_rng(3)
    for t in range(3):
        x = c[4 * t:4 * t + 4] + rng.normal(scale=0.4, size=(4, 3))
        A = rng.normal(size=(4, 3, 3)) * 0.3
        cov = np.einsum("kij,klj->kil", A, A) + np.diag([0.02, 0.02, 0.6])
        fit = fit_deformable(x, None, seed_cov=cov)
        pose = _pose(range(4), fit)
        _p, pc, _i = posterior_seed_positions([pose], x, cov)
        _p2, pp, _i2 = posterior_seed_positions([pose], x, cov,
                                                cov_mode="pev")
        assert np.allclose(_p, _p2)                 # same mean in both modes
        for i in range(4):
            assert np.allclose(pc[i], pc[i].T)
            assert np.linalg.eigvalsh(cov[i] - pc[i]).min() >= -1e-12
            assert np.linalg.eigvalsh(SLACK_MM ** 2 * np.eye(3)
                                      - pc[i]).min() >= -1e-12
            # carrying the pose uncertainty only adds variance; with the
            # (normalised, clamped) fit weights it is not the exact GLS, so
            # it is below the prior in total, not in every direction
            assert np.linalg.eigvalsh(pp[i] - pc[i]).min() >= -1e-9
            assert np.trace(pp[i]) <= np.trace(cov[i]) + 1e-9


def test_tentative_and_degraded_tiles_move_less(truth_tiles):
    c, a, _ = truth_tiles
    rng = np.random.default_rng(4)
    x = c[8:12] + rng.normal(scale=0.4, size=(4, 3))
    cov = np.repeat(np.diag([0.05, 0.05, 0.6])[None], 4, axis=0)
    fit = fit_deformable(x, None, seed_cov=cov)
    sup = posterior_seed_positions([_pose(range(4), fit)], x, cov)[2]
    for pose in (_pose(range(4), fit, confidence="tentative"),
                 _pose(range(4), fit, degraded=True)):
        ten = posterior_seed_positions([pose], x, cov)[2]
        assert np.all(ten["shift_mm"] < sup["shift_mm"] + 1e-12)
        assert ten["tiles"][0]["slack_mm"] == 1.0
    assert sup["tiles"][0]["slack_mm"] == SLACK_MM


def test_unassigned_pass_through_and_order_fallback(truth_tiles):
    c, a, _ = truth_tiles
    rng = np.random.default_rng(5)
    x = c[:6] + rng.normal(scale=0.3, size=(6, 3))     # seeds 4, 5 unowned
    cov = np.repeat(np.diag([0.05, 0.05, 0.6])[None], 6, axis=0)
    fit = fit_deformable(x[:4], None, seed_cov=cov[:4])
    post, pcov, info = posterior_seed_positions(
        [_pose(range(4), fit)], x, cov)
    assert np.array_equal(post[4:], x[4:]) and np.array_equal(pcov[4:], cov[4:])
    assert list(info["tile_of"]) == [0, 0, 0, 0, -1, -1]
    assert info["n_fused"] == 4 and info["n_passthrough"] == 2
    # the same tile listed with its seeds in another order: the stored
    # assignment no longer reproduces the residuals -> permutation match
    perm = [2, 0, 3, 1]
    post2, _c2, info2 = posterior_seed_positions(
        [_pose(perm, fit)], x, cov)
    assert info2["tiles"][0]["match"] == "hungarian"
    assert np.allclose(post2[:4], post[:4], atol=1e-12)


# -------------------------------------------------- 3. forwarding in auto
def _recording_fit(calls):
    real = deform_mod.fit_deformable

    def rec(pts, axes=None, *args, **kw):
        calls.append((np.array(pts, dtype=float), kw.get("seed_cov")))
        return real(pts, axes, *args, **kw)
    return rec


def test_auto_and_prior_forward_seed_cov_to_every_fit(truth_tiles):
    c, a, truth = truth_tiles
    keep = [i for i in range(12) if i != 9]     # tile 2 loses a seed
    x, ax = c[keep], a[keep]
    n = len(x)
    cov = np.stack([np.diag([0.01, 0.02, 0.1]) * (i + 1) for i in range(n)])
    for kw in (dict(), dict(spacing_mm=(0.7, 0.7, 2.1))):
        calls = []
        with mock.patch.object(auto_mod, "fit_deformable",
                               _recording_fit(calls)):
            res = fit_tiles_auto(x, ax, cavity_center_ras=truth.cavity_center_ras,
                                 seed_cov=cov, **kw)
        assert res.tentative_tiles and res.tentative_tiles[0].inferred_seed_ras \
            is not None                          # triplet completion ran
        assert calls
        for pts, sc in calls:
            assert sc is not None and sc.shape == (len(pts), 3, 3)
            for j in range(len(pts)):
                hit = np.flatnonzero(np.abs(x - pts[j]).max(axis=1) < 1e-12)
                if len(hit):                     # detected seed: its own cov
                    assert np.array_equal(sc[j], cov[hit[0]])
                else:                            # inferred: mean of mates
                    assert np.allclose(sc[j], sc[:j].mean(axis=0))
        for p in res.all_tiles:
            assert p.deform.weighted_residuals_mm is not None
    calls = []
    with mock.patch.object(auto_mod, "fit_deformable", _recording_fit(calls)):
        res = fit_tiles_prior(x, ax, ImplantPrior(n_full=3),
                              cavity_center_ras=truth.cavity_center_ras,
                              seed_cov=cov)
    assert calls and all(sc is not None for _p, sc in calls)
    with pytest.raises(ValueError):
        fit_tiles_auto(x, ax, seed_cov=cov[:-1])


def test_auto_without_cov_is_unchanged(truth_tiles):
    c, a, truth = truth_tiles
    r0 = fit_tiles_auto(c, a, cavity_center_ras=truth.cavity_center_ras,
                        spacing_mm=(0.7, 0.7, 2.1))
    r1 = fit_tiles_auto(c, a, cavity_center_ras=truth.cavity_center_ras,
                        spacing_mm=(0.7, 0.7, 2.1), seed_cov=None)
    assert [p.seed_indices for p in r0.all_tiles] == \
        [p.seed_indices for p in r1.all_tiles]
    for p, q in zip(r0.all_tiles, r1.all_tiles):
        assert np.array_equal(p.deform.pose.R, q.deform.pose.R)
        assert p.deform.weighted_residuals_mm is None


# --------------------------------------------- 4. pipeline reconstruct()
@pytest.fixture(scope="module")
def coarse_phantom():
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=3, rng_seed=2)
    return _thick_slices(vol, 2), truth


def _light_pipeline(with_cov):
    """Patch the heavy segmentation stages away (empty masks: vault filter
    fails open, no cavity, no meshes) and optionally give the detected
    candidates the analytic slab covariance (stand-in for stage 2)."""
    import gtcore.pipeline as pl

    real_detect = pl.detect_seed_candidates

    def det(v, **kw):
        cands = real_detect(v, **kw)
        if with_cov:
            cands = dataclasses.replace(
                cands, cov_ras=slab_covariance_stand_in(v.affine, len(cands)))
        return cands

    def head(v, metal_mask=None):
        z = np.zeros(v.array.shape, dtype=bool)
        return dict(body=z, skull=z, cranial_interior=z, brain=z)

    return [mock.patch.object(pl, "detect_seed_candidates", det),
            mock.patch.object(pl, "segment_head", head),
            mock.patch.object(pl, "segment_cavity",
                              lambda v, *a, **k: np.zeros(v.array.shape, bool))]


def _run(vol, with_cov, **kw):
    import contextlib
    import warnings

    from gtcore.pipeline import reconstruct

    with contextlib.ExitStack() as st:
        for p in _light_pipeline(with_cov):
            st.enter_context(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return reconstruct(Volume(vol.array, vol.affine, dict(vol.meta)),
                               verbose=False, **kw)


def test_reconstruct_fuse_switches_the_seed_feed(coarse_phantom):
    vol, truth = coarse_phantom
    base = _run(vol, True, n_full_tiles="auto")
    assert "seed_posterior" not in base.meta
    assert "seed_posterior" not in base.volume.meta
    fused = _run(vol, True, n_full_tiles="auto", fuse_tiles=True)
    sp = fused.meta["seed_posterior"]
    assert sp["applied"] and sp is fused.volume.meta["seed_posterior"]
    raw = fused.meta["seeds_unfused"]
    assert np.array_equal(raw.centers_ras, base.seeds.centers_ras)
    assert np.array_equal(sp["raw_centers_ras"], base.seeds.centers_ras)
    assert np.array_equal(fused.seeds.centers_ras, sp["centers_ras"])
    assert np.array_equal(fused.seeds.cov_ras, sp["cov_ras"])
    assert sp["cov_ras_pev"].shape == sp["cov_ras"].shape
    assert fused.seeds.info["fused"] and sp["info"]["n_fused"] >= 8
    assert not np.array_equal(fused.seeds.centers_ras, raw.centers_ras)
    for p in fused.tiles.all_tiles:
        assert p.deform.weighted_residuals_mm is not None
    assert "tile fusion" in fused.timings
    # no covariance -> nothing fused, reason recorded, feed unchanged
    nocov = _run(vol, False, n_full_tiles="auto", fuse_tiles=True)
    assert not nocov.meta["seed_posterior"]["applied"]
    assert "cov_ras" in nocov.meta["seed_posterior"]["reason"]
    assert np.array_equal(nocov.seeds.centers_ras, base.seeds.centers_ras)
    # counted mode goes through fit_tiles_prior (every tile has a bent fit)
    counted = _run(vol, True, n_full_tiles=3, fuse_tiles=True)
    assert counted.meta["seed_posterior"]["applied"]
    assert len(counted.tiles.tiles) == 3
    assert all(p.deform is not None for p in counted.tiles.tiles)


# ---------------------------------------- 5. synthetic gain (head phantom)
def _tile_errors(res, d2t, truth, xyz):
    tc = np.array([s.center_ras for s in truth.seeds])
    tid = np.array([s.tile_id for s in truth.seeds])
    ce, ne = {}, {}
    for p in res.all_tiles:
        g = {int(tid[d2t[i]]) if d2t[i] >= 0 else -1 for i in p.seed_indices}
        if len(g) != 1 or -1 in g or len(p.seed_indices) != 4:
            continue
        t = g.pop()
        T = tc[tid == t]
        ce[t] = float(np.linalg.norm(xyz[p.seed_indices].mean(axis=0)
                                     - T.mean(axis=0)))
        nt = np.linalg.svd(T - T.mean(axis=0))[2][-1]
        cosang = min(1.0, abs(float(p.deform.pose.normal @ nt)))
        ne[t] = float(np.degrees(np.arccos(cosang)))
    return ce, ne


def test_head_phantom_gain_is_measured_not_assumed():
    rows, sweep = [], []
    for r in range(5):
        vol0, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=r)
        tc = np.array([s.center_ras for s in truth.seeds])
        for f in (3, 4):                                     # 2.1, 2.8 mm
            vol = _thick_slices(vol0, f)
            cands = _detect(vol)
            x = np.asarray(cands.centers_ras, dtype=float)
            ax = np.asarray(cands.axes_ras, dtype=float)
            d2t = _match(tc, x)
            m = d2t >= 0
            cov = slab_covariance_stand_in(vol.affine, len(x))
            kw = dict(cavity_center_ras=truth.cavity_center_ras,
                      spacing_mm=vol.spacing)
            raw_fit = fit_tiles_auto(x, ax, **kw)
            w_fit = fit_tiles_auto(x, ax, seed_cov=cov, **kw)
            post, _pc, _info = posterior_seed_positions(w_fit, x, cov)
            ce0, ne0 = _tile_errors(raw_fit, d2t, truth, x)
            ce1, ne1 = _tile_errors(w_fit, d2t, truth, post)
            common = sorted(set(ce0) & set(ce1))
            rows.append(dict(
                dz=0.7 * f, rng=r,
                raw=np.linalg.norm(x[m] - tc[d2t[m]], axis=1).mean(),
                post=np.linalg.norm(post[m] - tc[d2t[m]], axis=1).mean(),
                rawz=np.abs(x[m, 2] - tc[d2t[m], 2]).mean(),
                postz=np.abs(post[m, 2] - tc[d2t[m], 2]).mean(),
                c0=np.mean([ce0[t] for t in common]),
                c1=np.mean([ce1[t] for t in common]),
                n0=np.mean([ne0[t] for t in common]),
                n1=np.mean([ne1[t] for t in common])))
            for s in (0.1, 0.2, 0.3, 0.5):
                with mock.patch.object(auto_mod, "fit_deformable",
                                       partial(deform_mod.fit_deformable,
                                               slack_mm=s)):
                    res_s = fit_tiles_auto(x, ax, seed_cov=cov, **kw)
                ps = posterior_seed_positions(res_s, x, cov, slack_mm=s)[0]
                sweep.append((0.7 * f, s,
                              np.linalg.norm(ps[m] - tc[d2t[m]], axis=1).mean()))
    print("\nstage 5 head phantom (slab stand-in covariance, slack %.1f mm):"
          % SLACK_MM)
    for dz in (2.1, 2.8):
        rr = [row for row in rows if abs(row["dz"] - dz) < 1e-6]

        def mu(k):
            return float(np.mean([row[k] for row in rr]))
        print("  %.1f mm: 3D %.3f -> %.3f (%+.1f%%), z %.3f -> %.3f (%+.1f%%),"
              " tile centre %.3f -> %.3f, normal %.2f -> %.2f deg"
              % (dz, mu("raw"), mu("post"), 100 * (mu("post") / mu("raw") - 1),
                 mu("rawz"), mu("postz"), 100 * (mu("postz") / mu("rawz") - 1),
                 mu("c0"), mu("c1"), mu("n0"), mu("n1")))
        line = "    slack sweep 3D:"
        for s in (0.1, 0.2, 0.3, 0.5):
            line += " %.1f->%.3f" % (s, np.mean([v for d, ss, v in sweep
                                               if abs(d - dz) < 1e-6 and ss == s]))
        print(line)
        assert mu("post") <= mu("raw") + 0.02
        assert mu("c1") <= mu("c0") + 1e-9
        assert mu("n1") <= mu("n0") + 0.25
