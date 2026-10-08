"""Real-scan proxies for seed localization (docs/plan-localization.md,
"Verification": the no-truth track and the with-truth hook).

No-truth proxies
----------------
printed8   the 8-tile printed phantom (Philips 0.59 x 0.59 x 1.0 mm, 157
           slices, no skull, so the pipeline's vault filter fails open):
           ``reconstruct(vol, n_full_tiles="auto")``; seed and tile counts;
           per tile the bent-tile RMS (``pose.deform.rms_mm``), curvatures and
           within-tile seed-axis coherence; side chords (4 shortest of the 6
           seed distances) of near-flat tiles (max |kappa| <= KAPPA_FLAT) vs
           the nominal 10 mm pitch; real seed peak HU; and whether the two
           seeds ~4 mm apart (25/31) are partitioned the same way by the auto
           fit and the counted fit (``fit_tiles(..., 8,
           complete_degraded=True)``).
split      split-half repeatability: the 1 mm printed phantom decimated into
           two 2 mm volumes (even slices, odd slices; the affine's k step
           doubles, the odd origin moves one slice), detection (+ refinement)
           on each, every half matched to the full 1 mm result and to the
           other half (Hungarian, 3 mm).  Reported per voxel axis: odd-even
           disagreement / sqrt(2) (RMS of the difference over sqrt 2: the
           single-scan 2 mm precision when the halves' errors are independent)
           and the Bland-Altman bias and 95 % limits of agreement.
postop     PostOp CT (1 mm slices at irregular positions, rebuilt onto a 2 mm
           grid): candidate count, supported / tentative tiles, unassigned
           in-implant seeds (pipeline auto result, cross-checked with
           ``fit_tiles_prior(..., ImplantPrior())``).
negatives  the tile-free printed scan (``CT 3D printed``) and the pre-implant
           head CT (``DOEJOHNPOSTCT``): ``result.implant`` verdict (must not be
           "confirmed").

``--refine`` / ``--fuse`` are passed to ``reconstruct`` as ``refine_seeds`` /
``fuse_tiles`` when it accepts them; otherwise they are applied after the
pipeline (``refine_seed_candidates`` then the same auto tile fit;
``posterior_seed_positions`` on the fitted tiles) and the report says so; a
stage whose module is missing is skipped with the reason logged.

With-truth hook (``--truth <csv>``)
-----------------------------------
CSV with a header row and columns ``x,y,z`` in mm (any rigid frame: CAD,
caliper or a reference scan) plus an optional integer ``tile_id`` column;
other columns are ignored; one row per physical seed, e.g.::

    x,y,z,tile_id
    12.31,-4.02,7.85,0
    ...

The truth is registered to the detected printed-phantom seeds by rigid ICP
with a Hungarian assignment at every iteration (24 axis-permutation starts on
the principal axes, best RMS kept); reported: per-seed error after the one
global rigid fit (absolute accuracy) and after one rigid fit per tile
(``tile_id`` groups, else the detected tile partition: local accuracy with
the hand-placement error removed; note a 4-point rigid fit absorbs 6 of 12
degrees of freedom).  Pairs farther than 3 mm after registration count as
unmatched.

Caching: the slim pipeline result (seeds, tiles, implant verdict, timings,
volume meta; no masks or meshes) is pickled under ``--out``/cache, keyed by
scan, options, git commit and the working-tree diff; DICOM volumes reload in
~2 s and are not cached.  The printed-phantom pipeline needs ~5 GB RAM and
30-60 s.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
import hashlib
import inspect
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("GT_DATA_ROOT",
                           r"C:\Users\jacob\OneDrive\Documents")
SCANS = {
    "printed8": "3D-Printed Phantom-8tiles (223)",
    "postop": "PostOp CT",
    "printed_blank": "CT 3D printed",
    "doe": "DOEJOHNPOSTCT",
}
MATCH_MM = 3.0
# "near-flat": max |kappa| of the bent-tile fit.  Four seeds on a cylinder
# are coplanar, so flatness is not observable from positions alone and the
# fitted kappa (pulled up by noisy PCA axes: folds of 37-145 deg on the
# printed phantom's regular tiles) is all there is.  0.04 /mm (<= 46 deg over
# 20 mm; the 3 mm seed offset then shortens chords by <= 11 %) admits the
# least-bent tiles of that phantom; the chord residual against the bent-tile
# model is reported for every tile as the curvature-free variant.
KAPPA_FLAT = 0.04
PITCH_MM = 10.0
PAIR_25_31 = (25, 31)     # the ~4 mm seed pair of the printed phantom


# ------------------------------------------------------------------ helpers
def git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"],
                                    cwd=REPO, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO,
                                       text=True)
        return h, bool(diff.strip()), hashlib.sha1(
            diff.encode("utf-8", "replace")).hexdigest()[:10]
    except Exception:
        return "unknown", True, "nogit"


def match(a, b, gate=MATCH_MM):
    a = np.asarray(a, float).reshape(-1, 3)
    b = np.asarray(b, float).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = cdist(a, b)
    r, c = linear_sum_assignment(np.where(d <= gate, d, 1e6))
    ok = d[r, c] <= gate
    return r[ok], c[ok]


def stats3(E):
    """Per-axis and 3-D statistics of error rows (N, 3)."""
    E = np.asarray(E, float).reshape(-1, 3)
    if len(E) == 0:
        return {"n": 0}
    e3 = np.linalg.norm(E, axis=1)
    out = {"n": int(len(E)), "mean_3d": float(e3.mean()),
           "sd_3d": float(e3.std(ddof=1)) if len(e3) > 1 else 0.0,
           "p95_3d": float(np.percentile(e3, 95)), "max_3d": float(e3.max())}
    for a, name in enumerate("xyz"):
        out["bias_" + name] = float(E[:, a].mean())
        out["rms_" + name] = float(np.sqrt((E[:, a] ** 2).mean()))
    return out


def get_refiner():
    try:
        import gtcore.seeds.refine as refine_mod
    except Exception as exc:
        return None, "gtcore.seeds.refine not importable (%s)" % exc
    fn = getattr(refine_mod, "refine_seed_candidates", None)
    if fn is None:
        return None, "gtcore.seeds.refine has no refine_seed_candidates"
    return fn, None


def get_fuser():
    try:
        import gtcore.tiles.fuse as fuse_mod
    except Exception as exc:
        return None, "gtcore.tiles.fuse not importable (%s)" % exc
    fn = getattr(fuse_mod, "posterior_seed_positions", None)
    if fn is None:
        return None, "gtcore.tiles.fuse has no posterior_seed_positions"
    return fn, None


def _posterior_centers(out):
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("centers_ras", out.get("centers"))
    if hasattr(out, "centers_ras"):
        out = out.centers_ras
    return np.asarray(out, dtype=float).reshape(-1, 3)


def detect(vol, refiner=None, refine_method=None):
    from gtcore.pipeline import filter_seed_shaped, seed_detection_params
    from gtcore.seeds import detect_seed_candidates

    p = seed_detection_params(vol.spacing)
    c = filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])
    if refiner is not None and len(c):
        c = refiner(vol, c, method=refine_method)
    return c


def peaks_near(vol, centers, r_mm=2.0):
    out = []
    sp = vol.spacing
    for c in np.asarray(centers, float).reshape(-1, 3):
        ijk = vol.ras_to_index(c)
        lo = np.maximum(np.floor(ijk - r_mm / sp).astype(int), 0)
        hi = np.minimum(np.ceil(ijk + r_mm / sp).astype(int) + 1,
                        np.asarray(vol.shape_ijk))
        sub = vol.array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
        out.append(float(sub.max()) if sub.size else np.nan)
    return np.asarray(out)


# ------------------------------------------------------------ pipeline+cache
def _scan_signature(path):
    n, size = 0, 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.lower() == "desktop.ini":
                continue
            n += 1
            size += os.path.getsize(os.path.join(root, f))
    return "%d-%d" % (n, size)


class Context:
    def __init__(self, args):
        self.args = args
        self.commit, self.dirty, self.diff_hash = git_state()
        self.refiner, self.refine_reason = (None, None)
        self.fuser, self.fuse_reason = (None, None)
        if args.refine != "none":
            self.refiner, self.refine_reason = get_refiner()
        if args.fuse:
            self.fuser, self.fuse_reason = get_fuser()
        from gtcore.pipeline import reconstruct

        params = inspect.signature(reconstruct).parameters
        self.recon_refine = "refine_seeds" in params
        self.recon_fuse = "fuse_tiles" in params
        self.notes = []
        self.cache_dir = os.path.join(args.out, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def skipped(self):
        out = {}
        if self.refine_reason:
            out["refine"] = self.refine_reason
        if self.fuse_reason:
            out["fuse"] = self.fuse_reason
        return out


def load(name):
    from gtcore.io import load_volume

    path = os.path.join(DATA_ROOT, SCANS[name])
    if not os.path.isdir(path):
        return None, path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = load_volume(path)
    return vol, path


def run_pipeline(ctx, name, vol, path, tiles):
    """Slim pipeline result (dict), from the cache when the key matches."""
    from gtcore.pipeline import reconstruct

    args = ctx.args
    key = dict(scan=SCANS[name], data=_scan_signature(path),
               refine=args.refine if ctx.refiner else "none",
               fuse=bool(ctx.fuser), tiles=tiles, commit=ctx.commit,
               diff=ctx.diff_hash if ctx.dirty else "")
    fn = os.path.join(ctx.cache_dir, "%s_%s.pkl" % (
        name, hashlib.sha1(json.dumps(key, sort_keys=True).encode())
        .hexdigest()[:12]))
    if os.path.exists(fn) and not args.no_cache:
        with open(fn, "rb") as fh:
            slim = pickle.load(fh)
        if slim.get("key") == key:
            slim["cached"] = True
            return slim
    kw = {}
    if ctx.refiner is not None and ctx.recon_refine:
        kw["refine_seeds"] = args.refine
    if ctx.fuser is not None and ctx.recon_fuse:
        kw["fuse_tiles"] = True
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = reconstruct(vol, verbose=args.verbose,
                          n_full_tiles="auto" if tiles else None, **kw)
    wall = time.time() - t0
    seeds, tiles_res = res.seeds, res.tiles
    post_hoc = []
    if ctx.refiner is not None and not ctx.recon_refine and len(seeds):
        seeds = ctx.refiner(vol, seeds, method=args.refine)
        post_hoc.append("refine")
        if tiles:
            from gtcore.tiles import fit_tiles

            tiles_res = fit_tiles(seeds.centers_ras, seeds.axes_ras, "auto",
                                  0, cavity_center_ras=None,
                                  mesh=res.meshes.get("cavity"),
                                  spacing_mm=vol.spacing)
    fused = None
    if ctx.fuser is not None and tiles_res is not None:
        try:
            fused = _posterior_centers(ctx.fuser(
                tiles_res, seeds.centers_ras, getattr(seeds, "cov_ras", None)))
            if not ctx.recon_fuse:
                post_hoc.append("fuse")
        except Exception as exc:
            ctx.fuse_reason = "posterior_seed_positions failed: %r" % exc
    slim = dict(
        key=key, cached=False, wall_s=wall, post_hoc=post_hoc,
        centers=np.asarray(seeds.centers_ras), axes=np.asarray(seeds.axes_ras),
        volumes=np.asarray(seeds.volumes_mm3),
        cov=None if getattr(seeds, "cov_ras", None) is None
        else np.asarray(seeds.cov_ras),
        n_raw=len(res.seeds_raw), tiles=tiles_res, implant=res.implant,
        timings=dict(res.timings), meta={k: v for k, v in vol.meta.items()
                                         if k in ("vault_filter", "implant",
                                                  "seed_search",
                                                  "slices_interpolated",
                                                  "slices_on_grid",
                                                  "slices_present",
                                                  "interpolated_k")},
        fused=fused, shape=tuple(vol.array.shape),
        spacing=[float(s) for s in vol.spacing])
    del res
    gc.collect()
    with open(fn, "wb") as fh:
        pickle.dump(slim, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return slim


# ------------------------------------------------------------ printed phantom
def _mean_axis(axes):
    A = np.asarray(axes, float).reshape(-1, 3)
    w, v = np.linalg.eigh(A.T @ A)
    return v[:, -1]


def tile_report(slim, centers=None):
    """Per-tile bent-tile RMS, curvature, chords, axis coherence."""
    centers = slim["centers"] if centers is None else centers
    axes = slim["axes"]
    res = slim["tiles"]
    rows = []
    for pose in list(res.tiles) + list(getattr(res, "tentative_tiles", [])):
        idx = list(pose.seed_indices)
        row = dict(tile=int(pose.tile_id), kind=pose.kind,
                   confidence=getattr(pose, "confidence", "supported"),
                   seeds=idx, degraded=bool(pose.degraded),
                   inferred=pose.inferred_seed_ras is not None)
        fit = pose.deform
        if fit is not None:
            k1, k2 = float(fit.params.kappa1), float(fit.params.kappa2)
            row.update(rms_mm=float(fit.rms_mm), kappa1=k1, kappa2=k2,
                       kappa_max=max(abs(k1), abs(k2)),
                       fold_deg=float(fit.params.fold_deg),
                       axis_err_deg=float(fit.axis_err_deg))
        if len(idx) == 4:
            P = centers[idx]
            d = np.sort([np.linalg.norm(P[a] - P[b])
                         for a, b in itertools.combinations(range(4), 2)])
            row.update(sides=[float(x) for x in d[:4]],
                       diagonals=[float(x) for x in d[4:]])
            if fit is not None and len(fit.assignment) == 4:
                # sides by grid topology: canonical seeds 10 mm apart in uv;
                # residual = observed chord - the fitted bent tile's chord
                from gtcore.tiles.model import RigidTile

                uv = RigidTile("full").seed_uv
                M = fit.seed_points()
                asg = list(fit.assignment)
                obs, res = [], []
                for a, b in itertools.combinations(range(4), 2):
                    ca, cb = asg[a], asg[b]
                    if abs(np.linalg.norm(uv[ca] - uv[cb]) - PITCH_MM) < 1e-6:
                        o = float(np.linalg.norm(P[a] - P[b]))
                        obs.append(o)
                        res.append(o - float(np.linalg.norm(M[ca] - M[cb])))
                if len(obs) == 4:
                    row.update(sides=sorted(obs), chord_resid=res)
        A = axes[idx]
        m = _mean_axis(A)
        ang = np.degrees(np.arccos(np.clip(np.abs(A @ m), 0.0, 1.0)))
        row.update(axis_coherence=float(np.mean(np.abs(A @ m))),
                   axis_spread_deg=float(ang.mean()))
        rows.append(row)
    return rows


def pair_assignment(res, pair):
    """{seed: sorted member list of its tile} for the seeds in ``pair``."""
    out = {}
    for pose in list(res.tiles) + list(getattr(res, "tentative_tiles", [])):
        for s in pair:
            if s in pose.seed_indices:
                out[s] = sorted(int(i) for i in pose.seed_indices)
    return out


def printed_report(ctx, slim, vol):
    from gtcore.tiles import fit_tiles

    centers = slim["centers"]
    rep = dict(n_seeds=int(len(centers)), n_raw=int(slim["n_raw"]),
               wall_s=slim["wall_s"], cached=slim["cached"],
               post_hoc=slim["post_hoc"])
    res = slim["tiles"]
    rep.update(n_supported=len(res.tiles),
               n_tentative=len(getattr(res, "tentative_tiles", [])),
               n_unassigned=len(getattr(res, "unassigned_indices", [])),
               summary=res.summary() if hasattr(res, "summary") else "")
    pk = peaks_near(vol, centers, r_mm=1.0)
    rep.update(peak_median=float(np.median(pk)), peak_min=float(pk.min()),
               peak_sat=int((pk >= 3070.5).sum()))
    tiles = tile_report(slim)
    rep["tiles"] = tiles
    rms = [t["rms_mm"] for t in tiles if "rms_mm" in t]
    rep["rms_mean"] = float(np.mean(rms)) if rms else np.nan
    rep["rms_max"] = float(np.max(rms)) if rms else np.nan
    flat = [t for t in tiles if t.get("kappa_max", 1.0) <= KAPPA_FLAT
            and "sides" in t]
    sides_flat = np.concatenate([t["sides"] for t in flat]) if flat else \
        np.zeros(0)
    sides_all = np.concatenate([t["sides"] for t in tiles if "sides" in t])
    rep.update(n_flat=len(flat),
               chord_flat_mean=float(sides_flat.mean()) if flat else np.nan,
               chord_flat_sd=float(sides_flat.std(ddof=1))
               if len(sides_flat) > 1 else np.nan,
               chord_all_mean=float(sides_all.mean()),
               chord_all_sd=float(sides_all.std(ddof=1)))
    cres = np.concatenate([t["chord_resid"] for t in tiles
                           if "chord_resid" in t] or [np.zeros(0)])
    rep.update(chord_resid_mean=float(cres.mean()) if cres.size else np.nan,
               chord_resid_sd=float(cres.std(ddof=1)) if cres.size > 1
               else np.nan)
    coh = [t["axis_coherence"] for t in tiles]
    rep.update(axis_coherence_mean=float(np.mean(coh)),
               axis_spread_deg_mean=float(np.mean(
                   [t["axis_spread_deg"] for t in tiles])))
    if slim.get("fused") is not None:
        ft = tile_report(slim, centers=slim["fused"])
        fs = np.concatenate([t["sides"] for t in ft
                             if t.get("kappa_max", 1.0) <= KAPPA_FLAT
                             and "sides" in t] or [np.zeros(0)])
        rep.update(fused_chord_flat_mean=float(fs.mean()) if fs.size
                   else np.nan,
                   fused_chord_flat_sd=float(fs.std(ddof=1)) if fs.size > 1
                   else np.nan)
    # the close pair: the documented 25/31, and the closest pair found now
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    pair = tuple(sorted((int(i), int(j))))
    rep.update(closest_pair=list(pair), closest_pair_mm=float(D[i, j]))
    if len(centers) > max(PAIR_25_31):
        rep["pair_25_31_mm"] = float(np.linalg.norm(
            centers[PAIR_25_31[0]] - centers[PAIR_25_31[1]]))
    counted = fit_tiles(centers, slim["axes"], 8, complete_degraded=True)
    a_auto = pair_assignment(res, pair)
    a_cnt = pair_assignment(counted, pair)
    rep.update(pair_auto=a_auto, pair_counted=a_cnt,
               pair_consistent=bool(a_auto == a_cnt))
    return rep


def split_half(ctx, vol, ref_centers):
    from gtcore.volume import Volume

    halves = {}
    for name, start in (("even", 0), ("odd", 1)):
        aff = vol.affine.copy()
        aff[:3, 3] = vol.affine[:3, 3] + start * vol.affine[:3, 2]
        aff[:3, 2] = 2.0 * vol.affine[:3, 2]
        halves[name] = Volume(np.ascontiguousarray(vol.array[start::2]), aff,
                              {"modality": "CT"})
    dets = {}
    for name, hv in halves.items():
        c = detect(hv, ctx.refiner, ctx.args.refine)
        dets[name] = np.asarray(c.centers_ras, float)
    direction = vol.direction
    out = dict(n_ref=int(len(ref_centers)),
               spacing_half=[float(s) for s in halves["even"].spacing])
    per = {}
    for name in ("even", "odd"):
        r, c = match(ref_centers, dets[name])
        per[name] = {int(a): dets[name][int(b)] for a, b in zip(r, c)}
        E = np.array([dets[name][int(b)] - ref_centers[int(a)]
                      for a, b in zip(r, c)]).reshape(-1, 3) @ direction
        s = stats3(E)
        out[name] = dict(n_det=int(len(dets[name])), n_match=int(len(r)),
                         fp=int(len(dets[name]) - len(r)), vs_full=s)
    common = sorted(set(per["even"]) & set(per["odd"]))
    d = np.array([per["odd"][i] - per["even"][i] for i in common]).reshape(
        -1, 3) @ direction
    out["n_pairs"] = int(len(common))
    if len(common) > 1:
        sd = d.std(axis=0, ddof=1)
        bias = d.mean(axis=0)
        out["bland_altman"] = {
            ax: dict(bias=float(bias[a]), sd=float(sd[a]),
                     loa_lo=float(bias[a] - 1.96 * sd[a]),
                     loa_hi=float(bias[a] + 1.96 * sd[a]),
                     precision=float(np.sqrt((d[:, a] ** 2).mean())
                                     / np.sqrt(2.0)))
            for a, ax in enumerate("ijk")}
        e3 = np.linalg.norm(d, axis=1)
        out["diff_3d_mean"] = float(e3.mean())
        out["diff_3d_p95"] = float(np.percentile(e3, 95))
        out["precision_3d"] = float(np.sqrt((e3 ** 2).mean()) / np.sqrt(2.0))
    return out


# --------------------------------------------------------------------- truth
def _kabsch(P, Q):
    """R, t with P ~= R @ Q + t (rows are points), proper rotation."""
    from gtcore.tiles.model import kabsch

    R, t, _s = kabsch(P, Q)
    return R, t


def read_truth(path):
    pts, tid = [], []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in rd.fieldnames or []}
        for c in ("x", "y", "z"):
            if c not in cols:
                raise ValueError("truth CSV needs columns x,y,z (got %s)"
                                 % rd.fieldnames)
        for row in rd:
            pts.append([float(row[cols[c]]) for c in ("x", "y", "z")])
            if "tile_id" in cols and row[cols["tile_id"]].strip() != "":
                tid.append(int(float(row[cols["tile_id"]])))
            else:
                tid.append(None)
    pts = np.asarray(pts, float)
    tiles = None if any(t is None for t in tid) else np.asarray(tid, int)
    return pts, tiles


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        out.append(np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]))
    return out


def register_truth(truth, det, iters=50, gate=MATCH_MM, n_random=300):
    """Rigid ICP (Hungarian correspondences) of ``truth`` onto ``det``.

    A seed lining of a cavity is close to isotropic, so the principal axes
    alone do not fix the rotation: 24 signed permutations of the principal
    axes plus ``n_random`` uniformly random rotations are tried, each
    translated median-to-median and refined by trimmed ICP (Kabsch on the
    closest 80 % of the Hungarian pairs, so false positives and misses do not
    drag the fit).  The start with the most pairs within ``gate``, then the
    lowest RMS, wins.  Returns (R, t, rows_truth, cols_det).
    """
    truth = np.asarray(truth, float)
    det = np.asarray(det, float)
    tm, dm = np.median(truth, axis=0), np.median(det, axis=0)
    _, vt = np.linalg.eigh(np.cov((truth - truth.mean(axis=0)).T))
    _, vd = np.linalg.eigh(np.cov((det - det.mean(axis=0)).T))
    starts = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for a, (p, s) in enumerate(zip(perm, signs)):
                M[p, a] = s
            R0 = vd @ M @ vt.T
            if np.linalg.det(R0) > 0:
                starts.append(R0)
    starts += _random_rotations(n_random)
    n_keep = max(3, int(0.8 * min(len(truth), len(det))))
    best = None
    for R in starts:
        t = dm - R @ tm
        prev = None
        for _ in range(iters):
            D = cdist(truth @ R.T + t, det)
            r, c = linear_sum_assignment(D)
            order = np.argsort(D[r, c])[:n_keep]
            r, c = r[order], c[order]
            R, t = _kabsch(det[c], truth[r])
            key = (tuple(sorted(r)), tuple(sorted(c)))
            if key == prev:
                break
            prev = key
        D = cdist(truth @ R.T + t, det)
        r, c = linear_sum_assignment(D)
        ok = D[r, c] <= gate
        if ok.sum() >= 3:
            # final fit on every pair inside the gate
            R, t = _kabsch(det[c[ok]], truth[r[ok]])
            D = cdist(truth @ R.T + t, det)
            r, c = linear_sum_assignment(D)
            ok = D[r, c] <= gate
        rms = float(np.sqrt((D[r[ok], c[ok]] ** 2).mean())) if ok.any() \
            else np.inf
        score = (-int(ok.sum()), rms)
        if best is None or score < best[0]:
            best = (score, R, t, r[ok], c[ok])
    _, R, t, r, c = best
    return R, t, r, c


def truth_report(truth, truth_tiles, det, det_tiles):
    """Per-seed error after the global rigid fit and per-tile rigid fits."""
    R, t, r, c = register_truth(truth, det)
    moved = truth @ R.T + t
    E = det[c] - moved[r]
    out = dict(n_truth=int(len(truth)), n_det=int(len(det)),
               n_match=int(len(r)), global_fit=stats3(E))
    groups = {}
    if truth_tiles is not None:
        for a, b in zip(r, c):
            groups.setdefault(int(truth_tiles[a]), []).append((a, b))
        out["groups"] = "truth tile_id"
    else:
        for a, b in zip(r, c):
            g = det_tiles.get(int(b))
            if g is not None:
                groups.setdefault(g, []).append((a, b))
        out["groups"] = "detected tile partition"
    Et = []
    for pairs in groups.values():
        if len(pairs) < 3:
            continue
        ia = [p[0] for p in pairs]
        ib = [p[1] for p in pairs]
        Rg, tg = _kabsch(det[ib], truth[ia])
        Et.append(det[ib] - (truth[ia] @ Rg.T + tg))
    out["per_tile_fit"] = stats3(np.vstack(Et)) if Et else {"n": 0}
    out["n_groups"] = len(Et)
    return out


# ------------------------------------------------------------------ markdown
def fmt(x, nd=2):
    try:
        return "–" if x is None or not np.isfinite(x) else ("%%.%df" % nd) % x
    except TypeError:
        return str(x)


def to_markdown(rep, header):
    L = [header, ""]
    p = rep.get("printed8")
    if p:
        L += ["**Printed 8-tile phantom (1 mm).** %d seeds (%d raw blobs); "
              "%d supported + %d tentative tiles, %d unassigned; %s. Real seed "
              "peaks: %d/%d at 3071 HU (min %.0f). Pipeline %.0f s%s%s."
              % (p["n_seeds"], p["n_raw"], p["n_supported"], p["n_tentative"],
                 p["n_unassigned"], p["summary"], p["peak_sat"], p["n_seeds"],
                 p["peak_min"], p["wall_s"],
                 " (cached)" if p["cached"] else "",
                 (", post-hoc " + "+".join(p["post_hoc"]))
                 if p["post_hoc"] else ""), "",
              "| tile | kind / confidence | seeds | bent-tile RMS (mm) "
              "| κ1 / κ2 (1/mm) | fold (°) | sides (mm) | axis spread (°) |",
              "|---|---|---|---|---|---|---|---|"]
        for t in p["tiles"]:
            L.append("| %d | %s / %s%s | %s | %s | %s / %s | %s | %s | %s |" % (
                t["tile"], t["kind"], t["confidence"],
                ", degraded" if t["degraded"] else "",
                ",".join(map(str, t["seeds"])), fmt(t.get("rms_mm")),
                fmt(t.get("kappa1"), 3), fmt(t.get("kappa2"), 3),
                fmt(t.get("fold_deg"), 0),
                " ".join("%.2f" % x for x in t.get("sides", [])),
                fmt(t["axis_spread_deg"], 1)))
        L += ["",
              "- bent-tile RMS: mean %s mm, max %s mm"
              % (fmt(p["rms_mean"]), fmt(p["rms_max"])),
              "- side chords, near-flat tiles (max |κ| ≤ %.3f /mm, n = %d): "
              "%s ± %s mm vs nominal %.0f mm; all tiles: %s ± %s mm"
              % (KAPPA_FLAT, p["n_flat"], fmt(p["chord_flat_mean"]),
                 fmt(p["chord_flat_sd"]), PITCH_MM, fmt(p["chord_all_mean"]),
                 fmt(p["chord_all_sd"])),
              "- side chord − bent-tile model chord (all tiles): %s ± %s mm"
              % (fmt(p["chord_resid_mean"], 3), fmt(p["chord_resid_sd"], 3)),
              "- within-tile seed-axis coherence: mean |cos| %s, mean spread "
              "%s°" % (fmt(p["axis_coherence_mean"], 3),
                        fmt(p["axis_spread_deg_mean"], 1)),
              "- close pair: seeds %s at %.2f mm (25/31 at %s mm); auto "
              "partition %s vs counted %s -> %s"
              % (p["closest_pair"], p["closest_pair_mm"],
                 fmt(p.get("pair_25_31_mm")), p["pair_auto"],
                 p["pair_counted"],
                 "consistent" if p["pair_consistent"] else "INCONSISTENT")]
        if "fused_chord_flat_mean" in p:
            L.append("- fused side chords (near-flat): %s ± %s mm"
                     % (fmt(p["fused_chord_flat_mean"]),
                        fmt(p["fused_chord_flat_sd"])))
        L.append("")
    s = rep.get("split")
    if s:
        L += ["**Split-half repeatability** (1 mm phantom decimated to two "
              "%.1f mm volumes; reference = the %d full-resolution seeds; "
              "detection threshold of the 2 mm branch)."
              % (s["spacing_half"][2], s["n_ref"]), "",
              "| half | detected | matched to 1 mm | FP | vs 1 mm: bias i / j / "
              "k (mm) | RMS i / j / k (mm) | 3D mean ± SD | P95 | max |",
              "|---|---|---|---|---|---|---|---|---|"]
        for h in ("even", "odd"):
            v = s[h]["vs_full"]
            L.append("| %s | %d | %d | %d | %s / %s / %s | %s / %s / %s | "
                     "%s ± %s | %s | %s |" % (
                         h, s[h]["n_det"], s[h]["n_match"], s[h]["fp"],
                         fmt(v.get("bias_x"), 3), fmt(v.get("bias_y"), 3),
                         fmt(v.get("bias_z"), 3), fmt(v.get("rms_x")),
                         fmt(v.get("rms_y")), fmt(v.get("rms_z")),
                         fmt(v.get("mean_3d")), fmt(v.get("sd_3d")),
                         fmt(v.get("p95_3d")), fmt(v.get("max_3d"))))
        if "bland_altman" in s:
            L += ["", "Odd − even, %d seeds found in both halves:"
                  % s["n_pairs"], "",
                  "| axis | Bland–Altman bias (mm) | SD | 95 % LoA (mm) "
                  "| disagreement/√2 (mm) |", "|---|---|---|---|---|"]
            for ax, b in s["bland_altman"].items():
                L.append("| %s | %s | %s | %s to %s | %s |" % (
                    ax, fmt(b["bias"], 3), fmt(b["sd"], 3), fmt(b["loa_lo"]),
                    fmt(b["loa_hi"]), fmt(b["precision"], 3)))
            L.append("")
            L.append("3-D |odd − even|: mean %s mm, P95 %s mm; 3-D "
                     "disagreement/√2 = %s mm."
                     % (fmt(s["diff_3d_mean"]), fmt(s["diff_3d_p95"]),
                        fmt(s["precision_3d"])))
        L.append("")
    q = rep.get("postop")
    if q:
        L += ["**PostOp CT** (%s grid, %s of %s slices interpolated). %d raw "
              "blobs -> %d in-vault candidates; %d supported + %d tentative "
              "tiles, %d unassigned in-implant seeds (NN distances %s mm); "
              "fit_tiles_prior(ImplantPrior()) agrees: %s; implant %s. "
              "Peak HU (±1 mm) of the %d tile-assigned seeds: median %.0f, "
              "%d at 3071; %d of them centred on an interpolated slice. "
              "Pipeline %.0f s%s."
              % ("x".join("%.2f" % x for x in q["spacing"]),
                 q.get("slices_interpolated", "?"),
                 q.get("slices_on_grid", "?"), q["n_raw"], q["n_cand"],
                 q["n_supported"], q["n_tentative"], q["n_unassigned"],
                 ", ".join("%.1f" % x for x in q["unassigned_nn_mm"]),
                 q["prior_agrees"], q["implant"], q["n_implant"],
                 q["peak_median"], q["peak_sat"], q["n_on_interp"],
                 q["wall_s"], " (cached)" if q["cached"] else ""), ""]
    n = rep.get("negatives")
    if n:
        L += ["**Negative controls.**", "",
              "| scan | loaded | grid (mm) | candidates | verdict | reason |",
              "|---|---|---|---|---|---|"]
        for name, v in n.items():
            L.append("| %s | %s | %s | %s | %s | %s |" % (
                SCANS[name], v.get("loaded"), v.get("grid", "–"),
                v.get("n_cand", "–"), v.get("verdict", "–"),
                v.get("reason", "–")))
        L.append("")
    t = rep.get("truth")
    if t:
        g, pt = t["global_fit"], t["per_tile_fit"]
        L += ["**With truth** (%s; %d truth / %d detected / %d matched)."
              % (t["file"], t["n_truth"], t["n_det"], t["n_match"]), "",
              "| fit | n | bias x / y / z (mm) | RMS x / y / z (mm) "
              "| 3D mean ± SD | P95 | max |", "|---|---|---|---|---|---|---|"]
        for label, v in (("one global rigid fit", g),
                         ("per-tile rigid fits (%s)" % t["groups"], pt)):
            L.append("| %s | %d | %s / %s / %s | %s / %s / %s | %s ± %s | %s "
                     "| %s |" % (label, v.get("n", 0), fmt(v.get("bias_x"), 3),
                                 fmt(v.get("bias_y"), 3),
                                 fmt(v.get("bias_z"), 3), fmt(v.get("rms_x")),
                                 fmt(v.get("rms_y")), fmt(v.get("rms_z")),
                                 fmt(v.get("mean_3d")), fmt(v.get("sd_3d")),
                                 fmt(v.get("p95_3d")), fmt(v.get("max_3d"))))
        L.append("")
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refine", default="none",
                    choices=("none", "centroid", "model"))
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--truth", default=None,
                    help="CSV x,y,z[,tile_id] of the printed phantom seeds")
    ap.add_argument("--scans", default="printed8,split,postop,negatives")
    ap.add_argument("--out", default=os.path.join(
        REPO, "output", "validation_realdata_proxies"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.makedirs(args.out, exist_ok=True)
    ctx = Context(args)
    todo = [s.strip() for s in args.scans.split(",") if s.strip()]
    for k, why in ctx.skipped().items():
        print("SKIPPED %s: %s" % (k, why))
    t_start = time.time()
    rep = {}

    if {"printed8", "split"} & set(todo) or args.truth:
        vol, path = load("printed8")
        if vol is None:
            print("printed phantom not found at %s" % path)
        else:
            slim = run_pipeline(ctx, "printed8", vol, path, tiles=True)
            if "printed8" in todo:
                rep["printed8"] = printed_report(ctx, slim, vol)
                print("printed8: %d seeds, %d+%d tiles"
                      % (rep["printed8"]["n_seeds"],
                         rep["printed8"]["n_supported"],
                         rep["printed8"]["n_tentative"]))
            if "split" in todo:
                rep["split"] = split_half(ctx, vol, slim["centers"])
                print("split-half: %d pairs" % rep["split"]["n_pairs"])
            if args.truth:
                truth, ttiles = read_truth(args.truth)
                det_tiles = {}
                for pose in slim["tiles"].tiles:
                    for i in pose.seed_indices:
                        det_tiles[int(i)] = int(pose.tile_id)
                rep["truth"] = truth_report(truth, ttiles, slim["centers"],
                                            det_tiles)
                rep["truth"]["file"] = os.path.basename(args.truth)
            del vol
            gc.collect()

    if "postop" in todo:
        vol, path = load("postop")
        if vol is None:
            print("PostOp CT not found at %s" % path)
        else:
            from gtcore.tiles import ImplantPrior, fit_tiles_prior

            slim = run_pipeline(ctx, "postop", vol, path, tiles=True)
            res = slim["tiles"]
            prior = fit_tiles_prior(slim["centers"], slim["axes"],
                                    ImplantPrior(),
                                    spacing_mm=vol.spacing)
            agrees = (len(prior.tiles) == len(res.tiles)
                      and len(prior.tentative_tiles) == len(res.tentative_tiles)
                      and sorted(prior.unassigned_indices)
                      == sorted(res.unassigned_indices))
            C = slim["centers"]
            D = cdist(C, C)
            np.fill_diagonal(D, np.inf)
            un = list(res.unassigned_indices)
            implant_idx = sorted({i for p in res.all_tiles
                                  for i in p.seed_indices})
            pk = peaks_near(vol, C[implant_idx], r_mm=1.0) if implant_idx \
                else np.array([np.nan])
            interp_k = set(slim["meta"].get("interpolated_k", []))
            k_of = np.round(vol.ras_to_index(C[implant_idx])[:, 2]).astype(
                int) if implant_idx else np.zeros(0, int)
            on_interp = np.array([k in interp_k for k in k_of])
            rep["postop"] = dict(
                n_raw=int(slim["n_raw"]), n_cand=int(len(C)),
                n_supported=len(res.tiles), n_tentative=len(
                    res.tentative_tiles), n_unassigned=len(un),
                unassigned=un,
                unassigned_nn_mm=[float(D[i].min()) for i in un],
                prior_agrees=bool(agrees),
                implant=(slim["implant"] or {}).get("verdict"),
                summary=res.summary(), peak_median=float(np.nanmedian(pk)),
                peak_sat=int((pk >= 3070.5).sum()), n_implant=len(implant_idx),
                n_on_interp=int(on_interp.sum()),
                spacing=slim["spacing"], wall_s=slim["wall_s"],
                cached=slim["cached"], **{k: v for k, v in slim["meta"].items()
                                          if k in ("slices_interpolated",
                                                   "slices_on_grid")})
            print("postop: %s" % res.summary())
            del vol
            gc.collect()

    if "negatives" in todo:
        rep["negatives"] = {}
        for name in ("printed_blank", "doe"):
            vol, path = load(name)
            if vol is None:
                rep["negatives"][name] = dict(loaded=False)
                continue
            slim = run_pipeline(ctx, name, vol, path, tiles=False)
            imp = slim["implant"] or {}
            rep["negatives"][name] = dict(
                loaded=True, grid="x".join("%.2f" % s for s in vol.spacing)
                + " (%d slices)" % vol.array.shape[0],
                n_cand=int(len(slim["centers"])),
                # assess_implant gives no verdict key below 4 candidates
                verdict=imp.get("verdict") or (
                    "confirmed" if imp.get("present") else "absent"),
                reason=imp.get("reason"), wall_s=slim["wall_s"])
            print("%s: %s" % (name, imp.get("verdict")))
            del vol
            gc.collect()

    wall = time.time() - t_start
    skipped = ctx.skipped()
    tag = args.tag or ("%s%s" % (args.refine, "-fuse" if args.fuse else "")
                       + ("-skipped" if skipped else ""))
    cmd = "python scripts/validation_realdata_proxies.py " + " ".join(
        sys.argv[1:] if argv is None else argv)
    header = ("`%s` — commit %s%s, %s; refine %s, fuse %s; wall %.0f s"
              % (cmd, ctx.commit, " (dirty)" if ctx.dirty else "",
                 datetime.date.today().isoformat(),
                 args.refine if ctx.refiner else "none",
                 "on" if ctx.fuser else "off", wall))
    if skipped:
        header += "\n\n" + "\n".join("Skipped %s: %s" % kv
                                     for kv in skipped.items())
    md = to_markdown(rep, header)
    with open(os.path.join(args.out, "proxies_%s.md" % tag), "w",
              encoding="utf-8") as fh:
        fh.write(md + "\n")
    with open(os.path.join(args.out, "proxies_%s.json" % tag), "w",
              encoding="utf-8") as fh:
        json.dump(_jsonable(rep), fh, indent=1)
    print()
    print(md)
    runs = os.path.join(args.out, "runs.csv")
    new = not os.path.exists(runs)
    with open(runs, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "command", "tag", "commit", "dirty", "wall_s",
                        "skipped"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    cmd, tag, ctx.commit, int(ctx.dirty), "%.1f" % wall,
                    "; ".join("%s: %s" % kv for kv in skipped.items())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
