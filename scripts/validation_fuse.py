"""Stage 5 validation: hierarchical weighted least squares + posterior seeds.

Synthetic head phantom (``make_head_phantom(spacing=0.7, n_tiles=3,
rng_seed=r)``, r = 0..4), slices block-averaged along z to 0.7 / 1.4 / 2.1 /
2.8 mm (``validation_spacing.thick_slices``), seeds detected with the
pipeline's spacing-aware parameters, per-seed covariance from the analytic
slab stand-in (``fuse.slab_covariance_stand_in``: diag over voxel axes of
s^2/12 + (0.1 s)^2) until the stage-2 grey-level covariance merges.

For every scan:
  raw   -- detected centres; tiles from the unweighted auto fit
  fused -- tiles from fit_tiles_auto(..., seed_cov), posterior seed
           positions from fuse.posterior_seed_positions

Reports mean 3D and |z| error vs truth (all matched detections), partition
correctness (every truth tile recovered by exactly one fitted tile, no mixed
tile; supported + tentative), tile centre / normal error, NEES of the
posterior (and of the prior) covariance, runtime, and the slack sweep.

Usage:  python scripts/validation_fuse.py [--rng 0 1 2 3 4] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from functools import partial
from unittest import mock

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_spacing import thick_slices  # noqa: E402

from gtcore.phantom import make_head_phantom  # noqa: E402
from gtcore.pipeline import filter_seed_shaped, seed_detection_params  # noqa: E402
from gtcore.seeds import detect_seed_candidates  # noqa: E402
from gtcore.tiles import auto as _auto  # noqa: E402
from gtcore.tiles import deform as _deform  # noqa: E402
from gtcore.tiles import fit_tiles_auto  # noqa: E402
from gtcore.tiles.deform import SLACK_MM  # noqa: E402
from gtcore.tiles.fuse import (  # noqa: E402
    posterior_seed_positions,
    slab_covariance_stand_in,
)

BASE_SPACING = 0.7
Z_FACTORS = (1, 2, 3, 4)
RNG_SEEDS = (0, 1, 2, 3, 4)
N_TILES = 3
SLACKS = (0.1, 0.2, 0.3, 0.5)
MATCH_MM = 2.0


def detect(vol):
    p = seed_detection_params(vol.spacing)
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


def match(truth_c, det_c):
    """Hungarian detection <-> truth match within MATCH_MM: det -> truth."""
    D = np.linalg.norm(det_c[:, None, :] - truth_c[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    out = np.full(len(det_c), -1, dtype=int)
    for i, j in zip(r, c):
        if D[i, j] < MATCH_MM:
            out[i] = j
    return out


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


def tile_errors(res, centers, d2t, truth, seeds_xyz):
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


def redundant_fraction(x, d2t, truth):
    """Share of the raw seed-error energy (all 4 seeds of a truth tile
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


def nees(err, cov):
    v = np.einsum("ki,kij,kj->k", err, np.linalg.inv(cov), err) / 3.0
    return float(np.mean(v))


def run_one(vol, truth, slack=SLACK_MM):
    cands = detect(vol)
    x = np.asarray(cands.centers_ras, dtype=float)
    a = np.asarray(cands.axes_ras, dtype=float)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = match(tc, x)
    m = d2t >= 0
    cov = slab_covariance_stand_in(vol.affine, len(x))
    cav = truth.cavity_center_ras

    t0 = time.perf_counter()
    res_raw = fit_tiles_auto(x, a, cavity_center_ras=cav,
                             spacing_mm=vol.spacing)
    t_raw = time.perf_counter() - t0
    t0 = time.perf_counter()
    res_w = fit_tiles_auto(x, a, cavity_center_ras=cav,
                           spacing_mm=vol.spacing, seed_cov=cov)
    t_w = time.perf_counter() - t0
    t0 = time.perf_counter()
    post, pcov, info = posterior_seed_positions(res_w, x, cov, slack_mm=slack)
    t_post = time.perf_counter() - t0
    t0 = time.perf_counter()
    _p2, pcov_pev, _i2 = posterior_seed_positions(res_w, x, cov, slack_mm=slack,
                                                  cov_mode="pev")
    t_pev = time.perf_counter() - t0

    e_raw = x[m] - tc[d2t[m]]
    e_post = post[m] - tc[d2t[m]]
    fused = (info["tile_of"] >= 0) & m
    ce_raw, ne_raw = tile_errors(res_raw, x, d2t, truth, x)
    ce_w, ne_w = tile_errors(res_w, x, d2t, truth, post)
    common = sorted(set(ce_raw) & set(ce_w))
    row = dict(
        n_det=int(len(x)), n_match=int(m.sum()),
        n_fused=int(info["n_fused"]),
        raw3d=float(np.linalg.norm(e_raw, axis=1).mean()),
        post3d=float(np.linalg.norm(e_post, axis=1).mean()),
        rawz=float(np.abs(e_raw[:, 2]).mean()),
        postz=float(np.abs(e_post[:, 2]).mean()),
        raw3d_fused=float(np.linalg.norm(x[fused] - tc[d2t[fused]], axis=1).mean())
        if fused.any() else np.nan,
        post3d_fused=float(np.linalg.norm(post[fused] - tc[d2t[fused]], axis=1).mean())
        if fused.any() else np.nan,
        part_raw=bool(partition_ok(res_raw, d2t, truth)),
        part_w=bool(partition_ok(res_w, d2t, truth)),
        n_sup_raw=len(res_raw.tiles), n_tent_raw=len(res_raw.tentative_tiles),
        n_sup_w=len(res_w.tiles), n_tent_w=len(res_w.tentative_tiles),
        cerr_raw=float(np.mean([ce_raw[t] for t in common])) if common else np.nan,
        cerr_post=float(np.mean([ce_w[t] for t in common])) if common else np.nan,
        nerr_raw=float(np.mean([ne_raw[t] for t in common])) if common else np.nan,
        nerr_w=float(np.mean([ne_w[t] for t in common])) if common else np.nan,
        nees_prior=nees(x[fused] - tc[d2t[fused]], cov[fused]) if fused.any() else np.nan,
        nees_post=nees(post[fused] - tc[d2t[fused]], pcov[fused]) if fused.any() else np.nan,
        nees_pev=nees(post[fused] - tc[d2t[fused]], pcov_pev[fused]) if fused.any() else np.nan,
        redundant_frac=redundant_fraction(x, d2t, truth),
        t_fit_raw=t_raw, t_fit_w=t_w, t_post=t_post, t_pev=t_pev,
        max_shift=float(info["shift_mm"].max()),
    )
    # flatness of each fused tile (for the 3-redundant-DOF paragraph)
    flat = []
    for p in res_w.all_tiles:
        if p.deform is None:
            continue
        idx = [i for i in p.seed_indices if d2t[i] >= 0]
        if not idx:
            continue
        er = np.linalg.norm(x[idx] - tc[d2t[idx]], axis=1).mean()
        ep = np.linalg.norm(post[idx] - tc[d2t[idx]], axis=1).mean()
        zr = np.abs(x[idx, 2] - tc[d2t[idx], 2]).mean()
        zp = np.abs(post[idx, 2] - tc[d2t[idx], 2]).mean()
        flat.append(dict(kappa=max(abs(p.deform.params.kappa1),
                                   abs(p.deform.params.kappa2)),
                         energy=p.deform.bending_energy,
                         nz=abs(float(p.deform.pose.normal[2])),
                         raw3d=er, post3d=ep, rawz=zr, postz=zp,
                         tentative=p.tentative))
    return row, flat


def slack_sweep(vol, truth, slacks=SLACKS):
    """Mean 3D / |z| posterior error, refitting with each slack."""
    cands = detect(vol)
    x = np.asarray(cands.centers_ras, dtype=float)
    a = np.asarray(cands.axes_ras, dtype=float)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = match(tc, x)
    m = d2t >= 0
    cov = slab_covariance_stand_in(vol.affine, len(x))
    out = {}
    for s in slacks:
        # fit and posterior with the same slack (the fit's default is the
        # module constant; auto.fit_deformable is swapped for the sweep only)
        with mock.patch.object(_auto, "fit_deformable",
                               partial(_deform.fit_deformable, slack_mm=s)):
            res = _auto.fit_tiles_auto(
                x, a, cavity_center_ras=truth.cavity_center_ras,
                spacing_mm=vol.spacing, seed_cov=cov)
        post, pcov, info = posterior_seed_positions(res, x, cov, slack_mm=s)
        e = post[m] - tc[d2t[m]]
        fused = (info["tile_of"] >= 0) & m
        out[s] = dict(post3d=float(np.linalg.norm(e, axis=1).mean()),
                      postz=float(np.abs(e[:, 2]).mean()),
                      part=bool(partition_ok(res, d2t, truth)),
                      nees=nees(post[fused] - tc[d2t[fused]], pcov[fused])
                      if fused.any() else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rng", type=int, nargs="*", default=list(RNG_SEEDS))
    ap.add_argument("--factors", type=int, nargs="*", default=list(Z_FACTORS))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--mc", action="store_true",
                    help="also run the model-only Monte-Carlo (flat vs curved)")
    args = ap.parse_args()
    rows, flats, sweeps = [], [], []
    for r in args.rng:
        vol0, truth = make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                                        rng_seed=r)
        for f in args.factors:
            vol = thick_slices(vol0, f)
            row, flat = run_one(vol, truth)
            row.update(dz=round(BASE_SPACING * f, 2), rng=r)
            rows.append(row)
            for fl in flat:
                fl.update(dz=row["dz"], rng=r)
            flats += flat
            print("dz %.1f rng %d: det %2d fused %2d | 3D %.3f -> %.3f | z %.3f"
                  " -> %.3f | part %s -> %s | ctr %.3f -> %.3f | nrm %.2f -> %.2f"
                  " deg | NEES prior %.2f post %.2f | fit %.2f/%.2f s post %.0f us"
                  % (row["dz"], r, row["n_det"], row["n_fused"], row["raw3d"],
                     row["post3d"], row["rawz"], row["postz"], row["part_raw"],
                     row["part_w"], row["cerr_raw"], row["cerr_post"],
                     row["nerr_raw"], row["nerr_w"], row["nees_prior"],
                     row["nees_post"], row["t_fit_raw"], row["t_fit_w"],
                     1e6 * row["t_post"]))
            if not args.no_sweep:
                sw = slack_sweep(vol, truth)
                for s, v in sw.items():
                    v.update(dz=row["dz"], rng=r, slack=s)
                    sweeps.append(v)
    print("\nper slice thickness (mean over realizations):")
    print("dz   | 3D raw  post  gain | z raw  post  gain | partition raw w |"
          " ctr raw post | nrm raw w (deg) | NEES prior post pev | fit raw w (s) |"
          " post us/seed | redundant share")
    for dz in sorted(set(r["dz"] for r in rows)):
        rr = [r for r in rows if r["dz"] == dz]

        def mu(k):
            return float(np.nanmean([r[k] for r in rr]))
        print("%.1f | %.3f %.3f %+5.1f%% | %.3f %.3f %+5.1f%% | %d/%d %d/%d |"
              " %.3f %.3f | %.2f %.2f | %.2f %.2f %.2f | %.2f %.2f | %.1f | %.2f"
              % (dz, mu("raw3d"), mu("post3d"),
                 100 * (mu("post3d") / mu("raw3d") - 1), mu("rawz"),
                 mu("postz"), 100 * (mu("postz") / mu("rawz") - 1),
                 sum(r["part_raw"] for r in rr), len(rr),
                 sum(r["part_w"] for r in rr), len(rr), mu("cerr_raw"),
                 mu("cerr_post"), mu("nerr_raw"), mu("nerr_w"),
                 mu("nees_prior"), mu("nees_post"), mu("nees_pev"),
                 mu("t_fit_raw"), mu("t_fit_w"),
                 1e6 * mu("t_post") / max(1.0, mu("n_fused")),
                 mu("redundant_frac")))
    if sweeps:
        print("\nslack sweep (posterior mean 3D / |z| error, partition, NEES):")
        for dz in sorted(set(s["dz"] for s in sweeps)):
            line = "%.1f |" % dz
            for sl in SLACKS:
                ss = [s for s in sweeps if s["dz"] == dz and s["slack"] == sl]
                line += " s=%.1f: %.3f / %.3f %d/%d NEES %.2f |" % (
                    sl, np.mean([s["post3d"] for s in ss]),
                    np.mean([s["postz"] for s in ss]),
                    sum(s["part"] for s in ss), len(ss),
                    np.nanmean([s["nees"] for s in ss]))
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
                      % (dz, lab, len(ff), np.mean([f["raw3d"] for f in ff]),
                         np.mean([f["post3d"] for f in ff]),
                         np.mean([f["rawz"] for f in ff]),
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


if __name__ == "__main__":
    main()
