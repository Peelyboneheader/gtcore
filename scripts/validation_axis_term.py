"""Gate run for ``auto.DROP_AXIS_TERM_ON_COARSE`` (thick-slice axis term of
the default bent-tile fit; docs/localization-notes.md, stage 5 follow-up).

Head phantom, rng 0-4, slices 0.7 / 1.4 / 2.1 / 2.8 mm, both renderings
(``--seed-render``, default both), raw detections (``--refine centroid``
for refined seeds).  For every scan the auto fit (``fit_tiles_auto``) and
the counted prior (``fit_tiles_prior(n_full=3)``) are run with the switch
OFF (historical: axes always fitted) and ON, and compared:

* partition correctness (every truth tile = one fitted tile, supported +
  tentative), tile centre error (seed mean vs truth seed mean) and tile
  normal error (bent-tile normal vs the truth seeds' plane normal) on the
  correctly partitioned tiles;
* bit-identity of the poses on slices <= 1.2 mm.

Pre-declared gate: normal error at 2.1 / 2.8 mm improves; partition and
centre error not worse at ANY spacing; thin slices bit-identical.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_fuse import (  # noqa: E402
    BASE_SPACING,
    N_TILES,
    git_commit,
    make_volume,
    match,
    partition_ok,
    prepare,
    slab_volume,
    tile_errors,
)

from gtcore.tiles import ImplantPrior, fit_tiles_auto, fit_tiles_prior  # noqa: E402
from gtcore.tiles import auto as auto_mod  # noqa: E402


def _poses(res):
    return [(tuple(p.seed_indices), p.deform.pose.R.tobytes(),
             p.deform.pose.t.tobytes(), p.deform.params, p.confidence)
            for p in res.all_tiles]


def run(vol, truth, refine, mode, drop):
    _xd, x, a, _cov, _n = prepare(vol, refine)
    tc = np.array([s.center_ras for s in truth.seeds])
    d2t = match(tc, x)
    kw = dict(cavity_center_ras=truth.cavity_center_ras, spacing_mm=vol.spacing)
    with mock.patch.object(auto_mod, "DROP_AXIS_TERM_ON_COARSE", drop):
        t0 = time.perf_counter()
        res = fit_tiles_auto(x, a, **kw) if mode == "auto" else \
            fit_tiles_prior(x, a, ImplantPrior(n_full=N_TILES), **kw)
        dt = time.perf_counter() - t0
    ce, ne = tile_errors(res, d2t, truth, x)
    return dict(part=partition_ok(res, d2t, truth), ce=ce, ne=ne, poses=_poses(res),
                n_sup=len(res.tiles), n_tent=len(res.tentative_tiles),
                axes=[bool(p.deform.axes_fitted) for p in res.all_tiles
                      if p.deform is not None], t=dt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rng", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--factors", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--seed-render", nargs="*", default=["binary", "analytic"])
    ap.add_argument("--refine", default="none", choices=("none", "centroid"))
    ap.add_argument("--modes", nargs="*", default=["auto", "prior"])
    args = ap.parse_args(argv)
    t_start = time.time()
    print("validation_axis_term: commit %s; refine %s; rng %s" % (
        git_commit(), args.refine, args.rng))
    rows = []
    for render in args.seed_render:
        for r in args.rng:
            vol0, truth = make_volume(r, render)
            for f in args.factors:
                vol = slab_volume(vol0, f, render)
                dz = round(BASE_SPACING * f, 2)
                for mode in args.modes:
                    off = run(vol, truth, args.refine, mode, False)
                    on = run(vol, truth, args.refine, mode, True)
                    common = sorted(set(off["ce"]) & set(on["ce"]))
                    row = dict(render=render, rng=r, dz=dz, mode=mode,
                               part_off=off["part"], part_on=on["part"],
                               identical=off["poses"] == on["poses"],
                               axes_on=all(on["axes"]) if on["axes"] else None,
                               ce_off=np.mean([off["ce"][t] for t in common]) if common else np.nan,
                               ce_on=np.mean([on["ce"][t] for t in common]) if common else np.nan,
                               ne_off=np.mean([off["ne"][t] for t in common]) if common else np.nan,
                               ne_on=np.mean([on["ne"][t] for t in common]) if common else np.nan,
                               n_common=len(common), t_off=off["t"], t_on=on["t"],
                               tiles_off="%d+%d" % (off["n_sup"], off["n_tent"]),
                               tiles_on="%d+%d" % (on["n_sup"], on["n_tent"]))
                    rows.append(row)
                    print("%-8s rng %d dz %.1f %-5s: part %s -> %s | tiles %s -> %s"
                          " | ctr %.3f -> %.3f | nrm %.2f -> %.2f deg (n=%d)"
                          " | identical %s | %.2f -> %.2f s"
                          % (render, r, dz, mode, row["part_off"], row["part_on"],
                             row["tiles_off"], row["tiles_on"], row["ce_off"],
                             row["ce_on"], row["ne_off"], row["ne_on"],
                             row["n_common"], row["identical"], row["t_off"],
                             row["t_on"]))
    print("\nsummary (mean over realizations; OFF = axes always fitted, ON = dropped above 1.2 mm):")
    print("render   mode  dz  | partition off on | centre off on (mm) | normal off on (deg) | identical | fit s off on")
    verdict = True
    for render in args.seed_render:
        for mode in args.modes:
            for dz in sorted(set(r["dz"] for r in rows)):
                rr = [r for r in rows if r["render"] == render and r["mode"] == mode
                      and r["dz"] == dz]
                p_off, p_on = sum(r["part_off"] for r in rr), sum(r["part_on"] for r in rr)
                ce_off, ce_on = np.nanmean([r["ce_off"] for r in rr]), np.nanmean([r["ce_on"] for r in rr])
                ne_off, ne_on = np.nanmean([r["ne_off"] for r in rr]), np.nanmean([r["ne_on"] for r in rr])
                ident = sum(r["identical"] for r in rr)
                print("%-8s %-5s %.1f | %d/%d %d/%d | %.3f %.3f | %.2f %.2f | %d/%d | %.2f %.2f"
                      % (render, mode, dz, p_off, len(rr), p_on, len(rr), ce_off, ce_on,
                         ne_off, ne_on, ident, len(rr),
                         np.mean([r["t_off"] for r in rr]), np.mean([r["t_on"] for r in rr])))
                if dz <= 1.2 and ident != len(rr):
                    verdict = False
                    print("   GATE FAIL: thin slices not bit-identical")
                if p_on < p_off or (np.isfinite(ce_on) and np.isfinite(ce_off)
                                    and ce_on > ce_off + 1e-9):
                    verdict = False
                    print("   GATE FAIL: partition or centre worse")
                if dz >= 2.0 and np.isfinite(ne_on) and np.isfinite(ne_off) and ne_on >= ne_off:
                    verdict = False
                    print("   GATE FAIL: normal error not improved")
    print("\nGATE: %s" % ("PASS" if verdict else "FAIL"))
    print("wall %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
