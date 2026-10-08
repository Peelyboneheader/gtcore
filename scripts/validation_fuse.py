"""Stage 5 validation: hierarchical weighted least squares + posterior seeds.

Synthetic head phantom (``make_head_phantom(spacing=0.7, n_tiles=3,
rng_seed=r)``, r = 0..4, ``--seed-render binary|analytic``), slices
block-averaged along z to 0.7 / 1.4 / 2.1 / 2.8 mm
(``gtcore.phantom.seed_render.thick_slices``; the analytic rendering is
clipped at 3071 HU after slab averaging, as in ``validation_spacing.py``),
seeds detected with the pipeline's spacing-aware parameters.

Per-seed covariance (``--refine``):
  none      the analytic slab stand-in (``fuse.slab_covariance_stand_in``:
            diag over voxel axes of s^2/12 + (0.1 s)^2), seeds = detections;
  centroid  stage 2: ``gtcore.seeds.refine.refine_seed_candidates`` (grey-
            level centroid + analytic covariance), seeds = refined centres
            with the DETECTION axes (the refiner's default).

For every scan three seed sets are scored against truth:
  det   -- the raw detections;
  ref   -- the seeds handed to the tile fit (= det with --refine none);
  post  -- posterior positions from fit_tiles_auto / fit_tiles_prior
           (``--mode``) with ``seed_cov`` and fuse.posterior_seed_positions.
Also: partition correctness of the unweighted fit on ``ref`` and of the
weighted fit (every truth tile recovered by exactly one fitted tile,
supported + tentative), tile centre / normal error on the tiles both fits
partition correctly, NEES (3D: e'S^-1 e / 3; z: e_z^2 / S_zz) of the input
covariance and of the posterior in both covariance modes, runtime, the
slack sweep (``--slacks``) and, with ``--mc``, the model-only Monte-Carlo
of the 3-redundant-DOF ceiling.

Gate (docs/localization-notes.md, stage 5): posterior mean 3D error at
2.1 mm >= 8 % below the REFINED seeds (ceiling-adjusted bar).

Usage:
  python scripts/validation_fuse.py [--refine centroid] [--seed-render analytic]
         [--mode auto|prior] [--rng 0 1 2 3 4] [--slacks 0.2 0.3 0.5] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from functools import partial
from unittest import mock

import numpy as np
from scipy.optimize import linear_sum_assignment

from gtcore.phantom import make_head_phantom
from gtcore.phantom.seed_render import SATURATE_HU, thick_slices
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import ImplantPrior, fit_tiles_auto, fit_tiles_prior
from gtcore.tiles import auto as _auto
from gtcore.tiles import deform as _deform
from gtcore.tiles.deform import SLACK_MM
from gtcore.tiles.fuse import (
    posterior_seed_positions,
    slab_covariance_stand_in,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_SPACING = 0.7
Z_FACTORS = (1, 2, 3, 4)
RNG_SEEDS = (0, 1, 2, 3, 4)
N_TILES = 3
SLACKS = (0.2, 0.3, 0.5)
MATCH_MM = 2.0
GATE_GAIN_21 = 0.08          # ceiling-adjusted bar at 2.1 mm (notes)


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"],
                                    cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, text=True).strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_volume(r, seed_render="binary", metal_hu=None):
    if seed_render == "binary":
        return make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                                 rng_seed=r)
    return make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                             rng_seed=r, seed_render="analytic",
                             saturate_hu=None, metal_hu=metal_hu)


def slab_volume(vol0, f, seed_render="binary"):
    tv = thick_slices(vol0, f)
    if seed_render != "binary":
        tv = tv.copy_with(array=np.minimum(tv.array, np.float32(SATURATE_HU)))
    return tv


def detect(vol):
    p = seed_detection_params(vol.spacing)
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


def prepare(vol, refine="none"):
    """(x_det, x, axes, cov, n_fallback): detections, the seeds handed to
    the tile fit, their axes and covariance."""
    cands = detect(vol)
    x_det = np.asarray(cands.centers_ras, dtype=float)
    if refine == "none":
        return x_det, x_det, np.asarray(cands.axes_ras, dtype=float), \
            slab_covariance_stand_in(vol.affine, len(x_det)), 0
    from gtcore.seeds.refine import refine_seed_candidates
    ref = refine_seed_candidates(vol, cands, method=refine)
    status = ref.info.get("refine_status", np.array([]))
    n_fb = int(sum(1 for s in status if str(s) != "ok"))
    return x_det, np.asarray(ref.centers_ras, dtype=float), \
        np.asarray(ref.axes_ras, dtype=float), np.asarray(ref.cov_ras), n_fb


def match(truth_c, det_c):
    """Hungarian detection <-> truth match within MATCH_MM: det -> truth."""
    D = np.linalg.norm(det_c[:, None, :] - truth_c[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    out = np.full(len(det_c), -1, dtype=int)
    for i, j in zip(r, c):
        if D[i, j] < MATCH_MM:
            out[i] = j
    return out


def fit_tiles_mode(mode, x, a, cov, truth, vol, **kw):
    common = dict(cavity_center_ras=truth.cavity_center_ras,
                  spacing_mm=vol.spacing, seed_cov=cov)
    common.update(kw)
    if mode == "auto":
        return fit_tiles_auto(x, a, **common)
    return fit_tiles_prior(x, a, ImplantPrior(n_full=N_TILES), **common)


def partition_ok(res, d2t, truth):
    tid = np.array([s.tile_id for s in truth.seeds])
    tiles = res.all_tiles
    if len(tiles) != N_TILES:
        return False
    got = set()
    for p in tiles:
        g = {int(tid[d2t[i]]) if d2t[i] >= 0 else -1 for i in p.seed_indices}
        if len(g) != 1 or -1 in g:
            return False
        got |= g
    return len(got) == N_TILES


def plane_normal(pts):
    _u, _s, vt = np.linalg.svd(pts - pts.mean(axis=0))
    return vt[-1]


def tile_errors(res, d2t, truth, seeds_xyz):
    """Per fitted tile whose detected seeds all come from one truth tile:
    (centre error of the seed mean of ``seeds_xyz``, normal error deg of the
    tile's bent-tile fit vs the truth seeds' plane normal)."""
    tc = np.array([s.center_ras for s in truth.seeds])
    tid = np.array([s.tile_id for s in truth.seeds])
    ce, ne = {}, {}
    for p in res.all_tiles:
        g = {int(tid[d2t[i]]) if d2t[i] >= 0 else -1 for i in p.seed_indices}
        if len(g) != 1 or -1 in g or p.deform is None:
            continue
        t = g.pop()
        if len(p.seed_indices) != 4:
            continue
        truth_pts = tc[tid == t]
        ce[t] = float(np.linalg.norm(seeds_xyz[p.seed_indices].mean(axis=0)
                                     - truth_pts.mean(axis=0)))
        n_fit = np.asarray(p.deform.pose.normal, dtype=float)
        c = abs(float(n_fit @ plane_normal(truth_pts)))
        ne[t] = float(np.degrees(np.arccos(min(1.0, c))))
    return ce, ne


def nees(err, cov):
    if len(err) == 0:
        return np.nan, np.nan
    v3 = np.einsum("ki,kij,kj->k", err, np.linalg.inv(cov), err) / 3.0
    vz = err[:, 2] ** 2 / cov[:, 2, 2]
    return float(np.mean(v3)), float(np.mean(vz))


def redundant_fraction(x, d2t, truth):
    """Share of the seed-error energy (all 4 seeds of a truth tile
    detected) lying OUTSIDE the tangent space of the bent-tile model at the
    true seeds -- the part any tile fit can see at all (3 of 12 dims)."""
    from gtcore.tiles.deform import fit_deformable
    from gtcore.tiles.fuse import _fit_sensitivity
    tc = np.array([s.center_ras for s in truth.seeds])
    tid = np.array([s.tile_id for s in truth.seeds])
    num = den = 0.0
    for t in sorted(set(tid.tolist())):
        idx = [i for i in range(len(x)) if d2t[i] >= 0 and tid[d2t[i]] == t]
        if len(idx) != 4:
            continue
        T = tc[d2t[idx]]
        f = fit_deformable(T, None)
        G, _L = _fit_sensitivity(f, 4, np.repeat(np.eye(3)[None], 4, axis=0))
        e = (x[idx] - T).ravel()
        r = e - G @ np.linalg.lstsq(G, e, rcond=None)[0]
        num += float(r @ r)
        den += float(e @ e)
    return num / den if den > 0 else np.nan


def mc_redundancy(n_trials=400, dz=2.8, inplane=0.7, slack=SLACK_MM, seed=0):
    """Monte-Carlo with the model itself (no image): bent tiles of curvature
    kappa in random orientations, seeds displaced by d ~ N(0, slack^2 I) and
    e ~ N(0, Sigma_slab); weighted fit (no axes) + posterior.  Isolates the
    3-redundant-DOF limit from detection effects.  Returns
    {kappa: (raw 3D, post 3D, raw z, post z)} mean errors vs the TRUE seeds."""
    from scipy.spatial.transform import Rotation
    from gtcore.tiles.deform import DeformParams, deformed_seed_points, fit_deformable
    from gtcore.tiles.model import TilePose6
    rng = np.random.default_rng(seed)
    aff = np.diag([inplane, inplane, dz, 1.0])
    out = {}
    for kappa in (0.0, 0.03, 0.07, 0.15):
        acc = []
        for _ in range(n_trials):
            R = Rotation.random(random_state=rng).as_matrix()
            pose = TilePose6(R, rng.normal(scale=5.0, size=3), "full", 1.0)
            prm = DeformParams(kappa, kappa * rng.uniform(0.5, 1.0),
                               rng.uniform(-np.pi, np.pi))
            s_true = (deformed_seed_points(pose, prm)
                      + rng.normal(scale=slack, size=(4, 3)))
            cov = slab_covariance_stand_in(aff, 4)
            x = s_true + np.stack([rng.multivariate_normal(np.zeros(3), c)
                                   for c in cov])
            fit = fit_deformable(x, None, seed_cov=cov, slack_mm=slack)

            class _P:  # minimal pose stand-in for posterior_seed_positions
                tile_id, kind, seed_indices, deform = 0, "full", [0, 1, 2, 3], fit
            post, _c, _i = posterior_seed_positions([_P()], x, cov, slack_mm=slack)
            acc.append((np.linalg.norm(x - s_true, axis=1).mean(),
                        np.linalg.norm(post - s_true, axis=1).mean(),
                        np.abs(x - s_true)[:, 2].mean(),
                        np.abs(post - s_true)[:, 2].mean()))
        out[kappa] = tuple(np.mean(acc, axis=0))
    return out


def _err(e):
    return float(np.linalg.norm(e, axis=1).mean()), float(np.abs(e[:, 2]).mean())


def run_one(vol, truth, refine="none", mode="auto", slack=SLACK_MM):
    x_det, x, a, cov, n_fb = prepare(vol, refine)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = match(tc, x)
    m = d2t >= 0

    t0 = time.perf_counter()
    res_ref = fit_tiles_mode(mode, x, a, None, truth, vol)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    res_w = fit_tiles_mode(mode, x, a, cov, truth, vol)
    t_w = time.perf_counter() - t0
    t0 = time.perf_counter()
    post, pcov, info = posterior_seed_positions(res_w, x, cov, slack_mm=slack)
    t_post = time.perf_counter() - t0
    t0 = time.perf_counter()
    _p2, pcov_pev, _i2 = posterior_seed_positions(res_w, x, cov, slack_mm=slack,
                                                  cov_mode="pev")
    t_pev = time.perf_counter() - t0

    e_det = x_det[m] - tc[d2t[m]]
    e_ref = x[m] - tc[d2t[m]]
    e_post = post[m] - tc[d2t[m]]
    fused = (info["tile_of"] >= 0) & m
    ce0, ne0 = tile_errors(res_ref, d2t, truth, x)
    ce1, ne1 = tile_errors(res_w, d2t, truth, post)
    common = sorted(set(ce0) & set(ce1))
    (det3, detz), (ref3, refz), (post3, postz) = _err(e_det), _err(e_ref), _err(e_post)
    n_in3, n_inz = nees(x[fused] - tc[d2t[fused]], cov[fused])
    n_c3, n_cz = nees(post[fused] - tc[d2t[fused]], pcov[fused])
    n_p3, n_pz = nees(post[fused] - tc[d2t[fused]], pcov_pev[fused])
    row = dict(
        n_det=int(len(x)), n_match=int(m.sum()), n_fused=int(info["n_fused"]),
        n_fallback=n_fb,
        det3d=det3, detz=detz, ref3d=ref3, refz=refz, post3d=post3, postz=postz,
        part_ref=bool(partition_ok(res_ref, d2t, truth)),
        part_w=bool(partition_ok(res_w, d2t, truth)),
        n_sup_ref=len(res_ref.tiles), n_tent_ref=len(res_ref.tentative_tiles),
        n_sup_w=len(res_w.tiles), n_tent_w=len(res_w.tentative_tiles),
        cerr_ref=float(np.mean([ce0[t] for t in common])) if common else np.nan,
        cerr_post=float(np.mean([ce1[t] for t in common])) if common else np.nan,
        nerr_ref=float(np.mean([ne0[t] for t in common])) if common else np.nan,
        nerr_w=float(np.mean([ne1[t] for t in common])) if common else np.nan,
        nees_in3=n_in3, nees_inz=n_inz, nees_cond3=n_c3, nees_condz=n_cz,
        nees_pev3=n_p3, nees_pevz=n_pz,
        redundant_frac=redundant_fraction(x, d2t, truth),
        t_fit_ref=t_ref, t_fit_w=t_w, t_post=t_post, t_pev=t_pev,
        max_shift=float(info["shift_mm"].max()) if len(x) else 0.0,
    )
    flat = []
    for p in res_w.all_tiles:
        if p.deform is None:
            continue
        idx = [i for i in p.seed_indices if d2t[i] >= 0]
        if not idx:
            continue
        flat.append(dict(kappa=max(abs(p.deform.params.kappa1),
                                   abs(p.deform.params.kappa2)),
                         ref3d=np.linalg.norm(x[idx] - tc[d2t[idx]], axis=1).mean(),
                         post3d=np.linalg.norm(post[idx] - tc[d2t[idx]], axis=1).mean(),
                         refz=np.abs(x[idx, 2] - tc[d2t[idx], 2]).mean(),
                         postz=np.abs(post[idx, 2] - tc[d2t[idx], 2]).mean(),
                         tentative=p.tentative))
    return row, flat


def slack_sweep(vol, truth, refine="none", mode="auto", slacks=SLACKS):
    """Posterior mean 3D / |z| error, refitting with each slack."""
    _x_det, x, a, cov, _n = prepare(vol, refine)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = match(tc, x)
    m = d2t >= 0
    out = {}
    orig = _deform.fit_deformable
    for s in slacks:
        # fit and posterior with the same slack: swap the fit's default in
        # both modules that call it (auto; fit.py imports from deform lazily)
        patched = partial(orig, slack_mm=s)
        with mock.patch.object(_auto, "fit_deformable", patched), \
                mock.patch.object(_deform, "fit_deformable", patched):
            res = fit_tiles_mode(mode, x, a, cov, truth, vol)
        post, pcov, info = posterior_seed_positions(res, x, cov, slack_mm=s)
        _p2, pcov_pev, _i2 = posterior_seed_positions(res, x, cov, slack_mm=s,
                                                      cov_mode="pev")
        e = post[m] - tc[d2t[m]]
        fused = (info["tile_of"] >= 0) & m
        out[s] = dict(post3d=float(np.linalg.norm(e, axis=1).mean()),
                      postz=float(np.abs(e[:, 2]).mean()),
                      part=bool(partition_ok(res, d2t, truth)),
                      nees_cond=nees(post[fused] - tc[d2t[fused]], pcov[fused])[0],
                      nees_pev=nees(post[fused] - tc[d2t[fused]], pcov_pev[fused])[0])
    return out


def _mu(rows, k):
    return float(np.nanmean([r[k] for r in rows]))


def _pct(a, b):
    return 100.0 * (b / a - 1.0) if a > 0 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rng", type=int, nargs="*", default=list(RNG_SEEDS))
    ap.add_argument("--factors", type=int, nargs="*", default=list(Z_FACTORS))
    ap.add_argument("--refine", default="none", choices=("none", "centroid"))
    ap.add_argument("--seed-render", default="binary",
                    choices=("binary", "analytic"))
    ap.add_argument("--metal-hu", type=float, default=None,
                    help="analytic rendering contrast (default: printed)")
    ap.add_argument("--mode", default="auto", choices=("auto", "prior"),
                    help="tile fit: fit_tiles_auto or fit_tiles_prior(n_full=3)")
    ap.add_argument("--slacks", type=float, nargs="*", default=list(SLACKS))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--mc", action="store_true",
                    help="also run the model-only Monte-Carlo (flat vs curved)")
    args = ap.parse_args(argv)
    t_start = time.time()
    print("validation_fuse: commit %s; render %s; refine %s; mode %s; rng %s;"
          " slack %.1f" % (git_commit(), args.seed_render, args.refine, args.mode,
                           args.rng, SLACK_MM))
    rows, flats, sweeps = [], [], []
    for r in args.rng:
        vol0, truth = make_volume(r, args.seed_render, args.metal_hu)
        for f in args.factors:
            vol = slab_volume(vol0, f, args.seed_render)
            row, flat = run_one(vol, truth, refine=args.refine, mode=args.mode)
            row.update(dz=round(BASE_SPACING * f, 2), rng=r,
                       seed_render=args.seed_render, refine=args.refine,
                       mode=args.mode)
            rows.append(row)
            for fl in flat:
                fl.update(dz=row["dz"], rng=r)
            flats += flat
            print("dz %.1f rng %d: det %2d fused %2d fb %d | 3D %.3f -> %.3f -> %.3f"
                  " | z %.3f -> %.3f -> %.3f | part %s -> %s | nrm %.2f -> %.2f"
                  " | NEES in %.2f/%.2f cond %.2f/%.2f pev %.2f/%.2f"
                  " | fit %.2f/%.2f s post %.0f us"
                  % (row["dz"], r, row["n_det"], row["n_fused"], row["n_fallback"],
                     row["det3d"], row["ref3d"], row["post3d"], row["detz"],
                     row["refz"], row["postz"], row["part_ref"], row["part_w"],
                     row["nerr_ref"], row["nerr_w"], row["nees_in3"],
                     row["nees_inz"], row["nees_cond3"], row["nees_condz"],
                     row["nees_pev3"], row["nees_pevz"], row["t_fit_ref"],
                     row["t_fit_w"], 1e6 * row["t_post"]))
            if not args.no_sweep:
                sw = slack_sweep(vol, truth, refine=args.refine, mode=args.mode,
                                 slacks=args.slacks)
                for s, v in sw.items():
                    v.update(dz=row["dz"], rng=r, slack=s)
                    sweeps.append(v)

    print("\nper slice thickness (mean over realizations; errors in mm):")
    print("dz  | 3D det  ref  post  gain(post/ref) | z det  ref  post  gain |"
          " partition ref w | ctr ref post | nrm ref w (deg) |"
          " NEES input 3D/z cond 3D/z pev 3D/z | fit ref/w (s) | post us/seed |"
          " redundant | fallbacks")
    verdict = {}
    for dz in sorted(set(r["dz"] for r in rows)):
        rr = [r for r in rows if r["dz"] == dz]
        g3 = _pct(_mu(rr, "ref3d"), _mu(rr, "post3d"))
        gz = _pct(_mu(rr, "refz"), _mu(rr, "postz"))
        verdict[dz] = g3
        print("%.1f | %.3f %.3f %.3f %+5.1f%% | %.3f %.3f %.3f %+5.1f%% | %d/%d %d/%d"
              " | %.3f %.3f | %.2f %.2f | %.2f/%.2f %.2f/%.2f %.2f/%.2f"
              " | %.2f %.2f | %.0f | %.2f | %d"
              % (dz, _mu(rr, "det3d"), _mu(rr, "ref3d"), _mu(rr, "post3d"), g3,
                 _mu(rr, "detz"), _mu(rr, "refz"), _mu(rr, "postz"), gz,
                 sum(r["part_ref"] for r in rr), len(rr),
                 sum(r["part_w"] for r in rr), len(rr),
                 _mu(rr, "cerr_ref"), _mu(rr, "cerr_post"),
                 _mu(rr, "nerr_ref"), _mu(rr, "nerr_w"),
                 _mu(rr, "nees_in3"), _mu(rr, "nees_inz"),
                 _mu(rr, "nees_cond3"), _mu(rr, "nees_condz"),
                 _mu(rr, "nees_pev3"), _mu(rr, "nees_pevz"),
                 _mu(rr, "t_fit_ref"), _mu(rr, "t_fit_w"),
                 1e6 * _mu(rr, "t_post") / max(1.0, _mu(rr, "n_fused")),
                 _mu(rr, "redundant_frac"), sum(r["n_fallback"] for r in rr)))
    if 2.1 in verdict:
        ok = verdict[2.1] <= -100.0 * GATE_GAIN_21
        print("\nGATE (2.1 mm, posterior vs refined seeds, bar >= %.0f %%): %+.1f %% -> %s"
              % (100 * GATE_GAIN_21, verdict[2.1], "PASS" if ok else "FAIL"))
    if sweeps:
        print("\nslack sweep (posterior mean 3D / |z| error, partition, NEES cond / pev):")
        for dz in sorted(set(s["dz"] for s in sweeps)):
            line = "%.1f |" % dz
            for sl in args.slacks:
                ss = [s for s in sweeps if s["dz"] == dz and s["slack"] == sl]
                line += " s=%.1f: %.3f / %.3f %d/%d NEES %.2f / %.2f |" % (
                    sl, np.mean([s["post3d"] for s in ss]),
                    np.mean([s["postz"] for s in ss]),
                    sum(s["part"] for s in ss), len(ss),
                    np.nanmean([s["nees_cond"] for s in ss]),
                    np.nanmean([s["nees_pev"] for s in ss]))
            print(line)
    if flats:
        print("\nper fused tile: near-flat (max|kappa| < 0.03/mm) vs curved:")
        for dz in sorted(set(f["dz"] for f in flats)):
            for lab, sel in (("flat", lambda f: f["kappa"] < 0.03),
                             ("curved", lambda f: f["kappa"] >= 0.03)):
                ff = [f for f in flats if f["dz"] == dz and sel(f)]
                if not ff:
                    continue
                print("  dz %.1f %-6s n=%2d: 3D %.3f -> %.3f  z %.3f -> %.3f"
                      % (dz, lab, len(ff), np.mean([f["ref3d"] for f in ff]),
                         np.mean([f["post3d"] for f in ff]),
                         np.mean([f["refz"] for f in ff]),
                         np.mean([f["postz"] for f in ff])))
    if args.mc:
        print("\nmodel-only Monte-Carlo (400 tiles each, slab cov, slack %.1f):"
              % SLACK_MM)
        for dz in (0.7, 1.4, 2.1, 2.8):
            mc = mc_redundancy(dz=dz)
            for kappa, (r3, p3, rz, pz) in mc.items():
                print("  dz %.1f kappa %.2f: 3D %.3f -> %.3f (%+.1f%%)  z %.3f -> %.3f"
                      " (%+.1f%%)" % (dz, kappa, r3, p3, 100 * (p3 / r3 - 1), rz,
                                      pz, 100 * (pz / rz - 1)))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("\nwall %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
