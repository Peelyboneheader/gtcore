"""Validation campaign: automatic tile creation from the seed cloud alone.

Runs ``fit_tiles(..., "auto")`` -- no implant count -- on

1. the synthetic phantom: rng seeds x tile counts x voxel spacings
   (truth known: n chosen vs truth, exact partition, pose errors),
2. the physical 8-tile printed phantom (count known = 8, one crumpled tile),
3. the real post-op CT (27-seed cluster; count unknown, 4 tiles by
   saturation),

and writes ``output/validation_autogen.csv`` (one row per case),
``output/validation_autogen.png`` (score-saturation curves, the paper
figure) and a summary on stdout.  Real-scan folders are optional: pass
``--phantom8 <dir>`` / ``--postop <dir>`` or set the defaults below.

Run from the repo root:  python scripts/validation_autogen.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.spatial.distance import cdist  # noqa: E402

from gtcore.phantom import make_head_phantom  # noqa: E402
from gtcore.pipeline import filter_seed_shaped, seed_detection_params  # noqa: E402
from gtcore.seeds import detect_seed_candidates  # noqa: E402
from gtcore.tiles import fit_tiles  # noqa: E402
from gtcore.tiles.fit import _orient_normal, _plane_fit  # noqa: E402

DEFAULT_PHANTOM8 = r"C:\Users\jacob\OneDrive\Documents\3D-Printed Phantom-8tiles (223)"
DEFAULT_POSTOP = r"C:\Users\jacob\OneDrive\Documents\PostOp CT"


def _angle(a, b):
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def synthetic_case(rng_seed, n_tiles, spacing):
    vol, truth = make_head_phantom(spacing=spacing, n_tiles=n_tiles,
                                   rng_seed=rng_seed)
    p = seed_detection_params(vol.spacing)
    cands = filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])
    t0 = time.perf_counter()
    fit = fit_tiles(cands.centers_ras, cands.axes_ras, "auto",
                    cavity_center_ras=truth.cavity_center_ras)
    dt = time.perf_counter() - t0
    tc = np.array([s.center_ras for s in truth.seeds])
    D = cdist(cands.centers_ras, tc) if len(cands) else np.zeros((0, len(tc)))
    d2t = [int(D[i].argmin()) if D[i].min() < 2.0 else None
           for i in range(len(cands))]
    got = {frozenset(d2t[i] for i in p_.seed_indices) for p_ in fit.tiles}
    want = {frozenset(t.seed_ids) for t in truth.tiles}
    truth_by = {frozenset(t.seed_ids): t for t in truth.tiles}
    cerr, nerr = [], []
    for pose in fit.tiles:
        key = frozenset(d2t[i] for i in pose.seed_indices)
        t = truth_by.get(key)
        if t is None:
            continue
        cerr.append(float(np.linalg.norm(pose.center_ras - t.center_ras)))
        pts = tc[t.seed_ids]
        ref, _ = _plane_fit(pts)
        ref = _orient_normal(ref, pts.mean(axis=0), truth.cavity_center_ras)
        nerr.append(_angle(pose.normal_ras, ref))
    return dict(case="synthetic", rng=rng_seed, spacing=spacing,
                n_truth=n_tiles, n_seeds=len(cands), n_auto=fit.n_selected,
                exact=(got == want), n_degraded=sum(p_.degraded for p_ in fit.tiles),
                center_err_mean=float(np.mean(cerr)) if cerr else float("nan"),
                center_err_max=float(np.max(cerr)) if cerr else float("nan"),
                normal_err_mean=float(np.mean(nerr)) if nerr else float("nan"),
                seconds=dt,
                curve=[(p_.n, p_.score, p_.marginal) for p_ in fit.score_curve
                       if p_.feasible])


def real_case(name, path, n_truth):
    from gtcore.io import load_volume
    from gtcore.pipeline import reconstruct

    if not path or not os.path.isdir(path):
        print("  [skip] %s: folder not found (%s)" % (name, path))
        return None
    vol = load_volume(path)
    t0 = time.perf_counter()
    res = reconstruct(vol, verbose=False, n_full_tiles="auto")
    dt = time.perf_counter() - t0
    fit = res.tiles
    row = dict(case=name, rng=-1, spacing=float(np.max(vol.spacing)),
               n_truth=n_truth, n_seeds=len(res.seeds), n_auto=fit.n_selected,
               exact=(fit.n_selected == n_truth) if n_truth >= 0 else None,
               n_degraded=sum(p.degraded for p in fit.tiles),
               center_err_mean=float("nan"), center_err_max=float("nan"),
               normal_err_mean=float("nan"), seconds=dt,
               curve=[(p.n, p.score, p.marginal) for p in fit.score_curve
                      if p.feasible])
    print("  %s: %d seeds -> n=%d (%s); tiles %s" % (
        name, len(res.seeds), fit.n_selected, fit.summary(),
        [(p.seed_indices, "crumpled" if p.degraded else
          "%.2f mm" % p.residual_mm) for p in fit.tiles]))
    for p in fit.tiles:
        if p.deform is not None:
            print("    T%d rms %.2f mm  fold %.0f deg  E %.3f%s" % (
                p.tile_id, p.deform.rms_mm, p.deform.params.fold_deg,
                p.deform.bending_energy,
                ("  " + p.surface.verdict()) if p.surface is not None else ""))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller synthetic sweep")
    ap.add_argument("--phantom8", default=DEFAULT_PHANTOM8)
    ap.add_argument("--postop", default=DEFAULT_POSTOP)
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    rngs = [0, 1, 2] if args.quick else list(range(6))
    tiles = [1, 3, 5] if args.quick else [1, 2, 3, 4, 5]
    spacings = [0.8] if args.quick else [0.8, 1.2]
    print("synthetic sweep: %d cases" % (len(rngs) * len(tiles) * len(spacings)))
    for sp in spacings:
        for nt in tiles:
            for rng in rngs:
                r = synthetic_case(rng, nt, sp)
                rows.append(r)
                print("  sp %.1f n %d rng %d: auto n=%d exact=%s degraded=%d "
                      "centre %.2f/%.2f mm normal %.1f deg  %.2fs" % (
                          sp, nt, rng, r["n_auto"], r["exact"], r["n_degraded"],
                          r["center_err_mean"], r["center_err_max"],
                          r["normal_err_mean"], r["seconds"]))
    if not args.no_real:
        print("real scans:")
        for name, path, n in (("phantom8", args.phantom8, 8),
                              ("postop", args.postop, -1)):
            r = real_case(name, path, n)
            if r is not None:
                rows.append(r)

    # ---------------------------------------------------------------- csv
    fields = [k for k in rows[0].keys() if k != "curve"] + ["curve"]
    with open(os.path.join(out_dir, "validation_autogen.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["curve"] = ";".join("%d:%.2f:%.2f" % c for c in r["curve"])
            w.writerow(rr)

    # ------------------------------------------------------------- summary
    syn = [r for r in rows if r["case"] == "synthetic"]
    n_ok = sum(1 for r in syn if r["n_auto"] == r["n_truth"])
    n_exact = sum(1 for r in syn if r["exact"])
    print("\nsynthetic: count correct %d/%d, exact partition %d/%d, "
          "centre err mean %.2f mm (max %.2f), normal err mean %.1f deg, "
          "%.2f s/case" % (
              n_ok, len(syn), n_exact, len(syn),
              np.nanmean([r["center_err_mean"] for r in syn]),
              np.nanmax([r["center_err_max"] for r in syn]),
              np.nanmean([r["normal_err_mean"] for r in syn]),
              np.mean([r["seconds"] for r in syn])))
    for r in rows:
        if r["case"] != "synthetic":
            print("%s: n_auto=%d (truth %s), %d crumpled, %.1f s" % (
                r["case"], r["n_auto"], r["n_truth"] if r["n_truth"] >= 0
                else "unknown", r["n_degraded"], r["seconds"]))

    # -------------------------------------------------------------- figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from gtcore.tiles.auto import LAMBDA_FULL

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        ax = axes[0]
        for r in syn:
            if r["spacing"] != spacings[0] or r["rng"] != rngs[0]:
                continue
            ns = [c[0] for c in r["curve"]]
            marg = [c[2] for c in r["curve"]]
            ax.plot(ns[1:], marg[1:], "o-", label="truth n=%d" % r["n_truth"])
        ax.axhline(LAMBDA_FULL, color="k", ls="--", label="penalty lambda")
        ax.set_xlabel("tile count n")
        ax.set_ylabel("marginal score of the n-th tile")
        ax.set_title("synthetic phantom (%.1f mm): marginal gain saturates at "
                     "the true n" % spacings[0])
        ax.legend(fontsize=8)
        ax = axes[1]
        for r in rows:
            if r["case"] == "synthetic":
                continue
            ns = [c[0] for c in r["curve"]]
            marg = [c[2] for c in r["curve"]]
            lab = "%s (auto n=%d%s)" % (
                r["case"], r["n_auto"],
                "" if r["n_truth"] < 0 else ", truth %d" % r["n_truth"])
            ax.plot(ns[1:], marg[1:], "s-", label=lab)
        ax.axhline(LAMBDA_FULL, color="k", ls="--", label="penalty lambda")
        ax.set_xlabel("tile count n")
        ax.set_ylabel("marginal score of the n-th tile")
        ax.set_title("real scans: printed 8-tile phantom, post-op CT")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "validation_autogen.png"), dpi=150)
        print("figure: output/validation_autogen.png")
    except Exception as exc:  # matplotlib missing / headless trouble
        print("figure skipped:", exc)


if __name__ == "__main__":
    main()
