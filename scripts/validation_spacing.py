"""Validation study: seed detection & tile fitting vs slice spacing.

Simulates acquisition at coarser slice thickness by block-averaging the
0.7 mm synthetic head phantom along z (true partial-volume averaging), then
runs detection -> shape filter [-> seed refinement] -> tile fitting
[-> tile-model fusion] and scores against ground truth.  This quantifies the
degradation observed on the real 2 mm post-op export.

Options (docs/plan-localization.md, stage 0):

``--seed-render binary|analytic``
    ``binary`` = the historical phantom (voxel-painted 8000 HU capsules,
    unclipped); ``analytic`` = exactly integrated 4.5 x 0.8 mm capsules
    (``make_head_phantom(seed_render="analytic")``, contrast
    ``--metal-hu``, default METAL_HU_PRINTED) clipped at ``--saturate-hu``
    (default 3071) AFTER the slab averaging, as a scanner would.
``--refine none|centroid|model``
    ``gtcore.seeds.refine.refine_seed_candidates(vol, cands, method=...)``
    after the shape filter; skipped with a logged reason when absent.
``--fuse``
    counted bent-tile fit (``fit_tiles_prior`` with the true count, seed
    covariance forwarded when supported) then
    ``gtcore.tiles.fuse.posterior_seed_positions``; fused seed errors are
    reported next to the raw ones; skipped with a logged reason when absent.
``--realizations N``
    phantom rng seeds 0 .. N-1 (default 5).

Seeds are matched to truth by a Hungarian assignment within 2.0 mm (the
historical gate).  Outputs ``output/validation_spacing_<tag>.csv`` /
``.png`` / ``.md`` (tag = ``<render>-<refine>[-fuse]``, or ``--tag``); the
untagged ``output/validation_spacing.csv`` of earlier runs is never touched.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import os
import subprocess
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from gtcore.phantom import make_head_phantom
from gtcore.phantom.seed_render import METAL_HU_PRINTED, SATURATE_HU
from gtcore.phantom.seed_render import thick_slices  # noqa: F401  (re-export)
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import ImplantPrior, fit_tiles, fit_tiles_prior

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_SPACING = 0.7
Z_FACTORS = [1, 2, 3, 4]  # -> 0.7, 1.4, 2.1, 2.8 mm slices
N_TILES = 3
MATCH_MM = 2.0

FIXED_PARAMS = dict(hu_threshold=2000.0, min_mm3=1.0, max_mm3=15.0,
                    min_elong=1.5, max_elong=10.0)


# ------------------------------------------------------------ optional stages
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
    """Centres from whatever posterior_seed_positions returns."""
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("centers_ras", out.get("centers"))
    if hasattr(out, "centers_ras"):
        out = out.centers_ras
    return np.asarray(out, dtype=float).reshape(-1, 3)


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"],
                                    cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, text=True).strip()
        return h, bool(dirty)
    except Exception:
        return "unknown", True


# ------------------------------------------------------------------- scoring
def detect(vol, adaptive):
    p = seed_detection_params(vol.spacing) if adaptive else FIXED_PARAMS
    return filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])


def match(t, est, gate=MATCH_MM):
    if len(t) == 0 or len(est) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = cdist(t, est)
    r, c = linear_sum_assignment(np.where(d < gate, d, 1e6))
    ok = d[r, c] < gate
    return r[ok], c[ok]


def seed_errors(t, est, prefix):
    """Error statistics of matched seeds (RAS; the phantom grid is RAS)."""
    r, c = match(t, est)
    out = {prefix + "recall": len(r) / len(t)}
    if len(r) == 0:
        for k in ("err_mean", "err_sd", "err_p95", "err_max", "z_bias",
                  "z_rms", "xy_rms"):
            out[prefix + k] = np.nan
        out["_E" if not prefix else "_E_" + prefix.rstrip("_")] = \
            np.zeros((0, 3))
        return out, (r, c)
    E = est[c] - t[r]
    e3 = np.linalg.norm(E, axis=1)
    out.update({
        prefix + "err_mean": float(e3.mean()),
        prefix + "err_sd": float(e3.std(ddof=1)) if len(e3) > 1 else 0.0,
        prefix + "err_p95": float(np.percentile(e3, 95)),
        prefix + "err_max": float(e3.max()),
        prefix + "z_bias": float(E[:, 2].mean()),
        prefix + "z_rms": float(np.sqrt((E[:, 2] ** 2).mean())),
        prefix + "xy_rms": float(np.sqrt((E[:, :2] ** 2).sum(axis=1).mean())),
        "_E" if not prefix else "_E_" + prefix.rstrip("_"): E,
    })
    return out, (r, c)


def partition_check(fit, t_tile, d2t, truth):
    """(ok, mean tile-centre error mm, mean normal error deg) of a fit."""
    if not (fit.all_assigned and len(fit.tiles) == N_TILES):
        return False, np.nan, np.nan
    groups = []
    for tp in fit.tiles:
        groups.append(frozenset(d2t.get(i, -9) for i in tp.seed_indices))
    ok = all(len(g) == 1 and -1 not in g and -9 not in g for g in groups)
    if not ok:
        return False, np.nan, np.nan
    tiles = {tt.tile_id: tt for tt in truth.tiles}
    ce, ne = [], []
    for tp, g in zip(fit.tiles, groups):
        tt = tiles[next(iter(g))]
        ce.append(float(np.linalg.norm(tp.center_ras - tt.center_ras)))
        cosang = abs(float(np.dot(tp.normal_ras, tt.normal_ras)))
        ne.append(float(np.degrees(np.arccos(min(1.0, cosang)))))
    return True, float(np.mean(ce)), float(np.mean(ne))


def score(vol, truth, adaptive, refiner=None, refine_method=None, fuser=None,
          skipped=None):
    cands = detect(vol, adaptive)
    if refiner is not None and len(cands):
        cands = refiner(vol, cands, method=refine_method)
    t = np.array([s.center_ras for s in truth.seeds])
    t_tile = np.array([s.tile_id for s in truth.seeds])
    row = dict(n_det=len(cands))
    if len(cands) == 0:
        row.update(recall=0.0, err_mean=np.nan, err_max=np.nan,
                   partition_ok=False)
        return row
    est = np.asarray(cands.centers_ras, dtype=float)
    stats, (r, c) = seed_errors(t, est, "")
    row.update(stats)
    D = cdist(t, est)
    d2t = {}
    for j in range(len(cands)):
        ti = int(np.argmin(D[:, j]))
        d2t[j] = int(t_tile[ti]) if D[ti, j] < MATCH_MM else -1
    fit = fit_tiles(cands.centers_ras, cands.axes_ras, N_TILES,
                    cavity_center_ras=truth.cavity_center_ras)
    ok, ce, ne = partition_check(fit, t_tile, d2t, truth)
    row.update(partition_ok=ok, tile_center_err=ce, tile_normal_err_deg=ne)

    if fuser is not None:
        cov = getattr(cands, "cov_ras", None)
        try:
            kw = {}
            if "seed_cov" in inspect.signature(fit_tiles_prior).parameters:
                kw["seed_cov"] = cov
            pfit = fit_tiles_prior(cands.centers_ras, cands.axes_ras,
                                   ImplantPrior(n_full=N_TILES),
                                   cavity_center_ras=truth.cavity_center_ras,
                                   spacing_mm=vol.spacing, **kw)
            post = _posterior_centers(fuser(pfit, cands.centers_ras, cov))
            fstats, _ = seed_errors(t, post, "fused_")
            row.update(fstats)
            fok, fce, fne = partition_check(pfit, t_tile, d2t, truth)
            row.update(fused_partition_ok=fok, fused_tile_center_err=fce,
                       fused_tile_normal_err_deg=fne)
        except Exception as exc:  # report, never hide
            if skipped is not None:
                skipped["fuse"] = "posterior_seed_positions failed: %r" % exc
    return row


# ---------------------------------------------------------------------- main
def make_volume(seed, args):
    if args.seed_render == "binary":
        return make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                                 rng_seed=seed)
    return make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                             rng_seed=seed, seed_render="analytic",
                             saturate_hu=None, metal_hu=args.metal_hu)


def fmt(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else ("%%.%df" % nd) % x


def pooled(E):
    """Per-seed statistics of pooled error rows (N, 3)."""
    E = np.asarray(E, float).reshape(-1, 3)
    if len(E) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, p95=np.nan, max=np.nan,
                    z_bias=np.nan, z_rms=np.nan, xy_rms=np.nan)
    e3 = np.linalg.norm(E, axis=1)
    return dict(n=len(E), mean=float(e3.mean()),
                sd=float(e3.std(ddof=1)) if len(e3) > 1 else 0.0,
                p95=float(np.percentile(e3, 95)), max=float(e3.max()),
                z_bias=float(E[:, 2].mean()),
                z_rms=float(np.sqrt((E[:, 2] ** 2).mean())),
                xy_rms=float(np.sqrt((E[:, :2] ** 2).sum(axis=1).mean())))


def summary_md(rows, header, fuse):
    dzs = sorted(set(r["dz"] for r in rows))
    lines = [header, "",
             "| slices (mm) | recall adaptive / fixed | seeds | 3D mean ± SD "
             "(mm) | P95 | max | z bias | z RMS | xy RMS | partition | tile "
             "centre err (mm) | normal err (°) |" + (
                 " fused mean ± SD | fused z RMS | fused max | fused "
                 "partition |" if fuse else ""),
             "|---|---|---|---|---|---|---|---|---|---|---|---|" + (
                 "---|---|---|---|" if fuse else "")]
    for d in dzs:
        a = [r for r in rows if r["dz"] == d and r["mode"] == "adaptive"]
        f = [r for r in rows if r["dz"] == d and r["mode"] == "fixed"]

        def m(key, rr=a, fn=np.nanmean):
            v = [r.get(key, np.nan) for r in rr]
            v = [x for x in v if x is not None and np.isfinite(x)]
            return fn(v) if v else np.nan
        P = pooled(np.vstack([r["_E"] for r in a if "_E" in r]
                             or [np.zeros((0, 3))]))
        line = ("| %.1f | %s / %s | %d | %s ± %s | %s | %s | %s | %s | %s "
                "| %d/%d | %s | %s |" % (
                    d, fmt(m("recall")), fmt(m("recall", f)), P["n"],
                    fmt(P["mean"]), fmt(P["sd"]), fmt(P["p95"]),
                    fmt(P["max"]), fmt(P["z_bias"], 3), fmt(P["z_rms"]),
                    fmt(P["xy_rms"]),
                    sum(bool(r["partition_ok"]) for r in a), len(a),
                    fmt(m("tile_center_err")),
                    fmt(m("tile_normal_err_deg"), 1)))
        if fuse:
            F = pooled(np.vstack([r["_E_fused"] for r in a if "_E_fused" in r]
                                 or [np.zeros((0, 3))]))
            line += " %s ± %s | %s | %s | %d/%d |" % (
                fmt(F["mean"]), fmt(F["sd"]), fmt(F["z_rms"]), fmt(F["max"]),
                sum(bool(r.get("fused_partition_ok")) for r in a), len(a))
        lines.append(line)
    lines.append("")
    lines.append("Seed errors pooled over the matched seeds of all "
                 "realizations (adaptive detection): mean ± SD, P95, max of "
                 "the 3-D error; z = slice axis.  Tile errors: mean over the "
                 "correctly partitioned scans; the truth normal is the radial "
                 "direction from the cavity centre (generator definition), so "
                 "~7° is a floor, not a fit error.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refine", default="none",
                    choices=("none", "centroid", "model"))
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--seed-render", default="binary",
                    choices=("binary", "analytic"))
    ap.add_argument("--metal-hu", type=float, default=None,
                    help="analytic capsule contrast (default %.0f)"
                    % METAL_HU_PRINTED)
    ap.add_argument("--saturate-hu", default="auto",
                    help="clip after slab averaging: auto (binary none, "
                    "analytic 3071), none, or a number")
    ap.add_argument("--realizations", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "output"))
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if args.saturate_hu == "auto":
        sat = None if args.seed_render == "binary" else SATURATE_HU
    elif str(args.saturate_hu).lower() == "none":
        sat = None
    else:
        sat = float(args.saturate_hu)
    tag = args.tag or "%s-%s%s" % (args.seed_render, args.refine,
                                    "-fuse" if args.fuse else "")
    os.makedirs(args.out, exist_ok=True)
    skipped = {}

    refiner = None
    if args.refine != "none":
        refiner, why = get_refiner()
        if refiner is None:
            skipped["refine"] = why
    fuser = None
    if args.fuse:
        fuser, why = get_fuser()
        if fuser is None:
            skipped["fuse"] = why
    for k, why in skipped.items():
        print("SKIPPED %s: %s" % (k, why))
    if skipped and not args.tag:
        # never let a run with a missing stage overwrite a real result
        tag += "-skipped"

    commit, dirty = git_commit()
    t0 = time.time()
    rows = []
    for seed in range(args.realizations):
        vol, truth = make_volume(seed, args)
        for f in Z_FACTORS:
            tv = thick_slices(vol, f)
            if sat is not None:
                tv = tv.copy_with(array=np.minimum(tv.array, np.float32(sat)))
            for adaptive in (False, True):
                r = score(tv, truth, adaptive, refiner=refiner,
                          refine_method=args.refine, fuser=fuser,
                          skipped=skipped)
                r.update(dz=round(BASE_SPACING * f, 2), rng=seed,
                         mode="adaptive" if adaptive else "fixed",
                         seed_render=args.seed_render, refine=args.refine
                         if refiner is not None else "none",
                         fuse=bool(fuser is not None))
                rows.append(r)
                print("dz=%.1f rng=%d %-8s: det %2d/12 recall %.2f err %.2f/%.2f"
                      " mm partition %s" % (r["dz"], seed, r["mode"],
                                           r["n_det"], r["recall"],
                                           r["err_mean"], r["err_max"],
                                           r["partition_ok"]))
    wall = time.time() - t0

    keys = []
    for row in rows:
        for k in row:
            if k not in keys and not k.startswith("_"):
                keys.append(k)
    csv_path = os.path.join(args.out, "validation_spacing_%s.csv" % tag)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "validation_spacing_%s_seeds.csv" % tag),
              "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dz", "rng", "mode", "kind", "err_x", "err_y", "err_z"])
        for row in rows:
            for key, kind in (("_E", "raw"), ("_E_fused", "fused")):
                for e in row.get(key, []):
                    w.writerow([row["dz"], row["rng"], row["mode"], kind,
                                "%.5f" % e[0], "%.5f" % e[1], "%.5f" % e[2]])

    cmd = "python scripts/validation_spacing.py " + " ".join(
        sys.argv[1:] if argv is None else argv)
    header = ("`%s` — commit %s%s, %s; head phantom 0.7 mm, %d tiles, rng "
              "0-%d, slab-averaged ×1-4; render %s%s; refine %s; fuse %s; "
              "match Hungarian < %.0f mm; wall %.0f s"
              % (cmd, commit, " (dirty)" if dirty else "",
                 datetime.date.today().isoformat(), N_TILES,
                 args.realizations - 1, args.seed_render,
                 "" if args.seed_render == "binary" else
                 " (contrast %.0f HU, clip %s)" % (
                     args.metal_hu or METAL_HU_PRINTED,
                     "none" if sat is None else "%.0f" % sat),
                 args.refine if refiner is not None else "none",
                 "on" if fuser is not None else "off", MATCH_MM, wall))
    if skipped:
        header += "\n\n" + "\n".join("Skipped %s: %s" % kv
                                     for kv in skipped.items())
    md = summary_md(rows, header, fuser is not None)
    with open(os.path.join(args.out, "validation_spacing_%s.md" % tag), "w",
              encoding="utf-8") as fh:
        fh.write(md + "\n")
    print()
    print(md)

    dzs = sorted(set(r["dz"] for r in rows))

    def agg(key, mode, fn=np.mean):
        return [fn([r.get(key, np.nan) for r in rows
                    if r["dz"] == d and r["mode"] == mode]) for d in dzs]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(dzs, agg("recall", "fixed"), "o--", color="tab:gray",
             label="recall, fixed params")
    ax1.plot(dzs, agg("recall", "adaptive"), "o-", color="tab:blue",
             label="recall, adaptive params")
    ax1.plot(dzs, agg("partition_ok", "adaptive"), "s-", color="tab:green",
             label="tile partition accuracy (adaptive)")
    ax1.set_xlabel("slice spacing (mm)")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(dzs, agg("err_mean", "adaptive", np.nanmean), "^-",
             color="tab:red", label="localization error (adaptive)")
    ax2.set_ylabel("error (mm)", color="tab:red")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center left",
               fontsize=8)
    ax1.set_title("Seed detection & tile fitting vs slice spacing [%s]\n"
                  "(synthetic phantom, 3 tiles / 12 seeds, %d realizations)"
                  % (tag, args.realizations))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "validation_spacing_%s.png" % tag),
                dpi=150)

    runs = os.path.join(args.out, "validation_spacing_runs.csv")
    new = not os.path.exists(runs)
    with open(runs, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "command", "tag", "realizations", "commit",
                        "dirty", "wall_s", "skipped"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    cmd, tag, args.realizations, commit, int(dirty),
                    "%.1f" % wall, "; ".join("%s: %s" % kv
                                             for kv in skipped.items())])
    print("\nwrote %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
