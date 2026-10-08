"""Diagnostics behind the stage-2 real-data proxies (docs/localization-notes.md,
"Stage 2 real-data proxies").  Companion to validation_realdata_proxies.py.

``--part printed``
    Why the grey-level centroid (gtcore.seeds.refine) falls back on 31/32
    seeds of the printed 8-tile phantom: per-seed status, background-gradient
    shift, saturated-voxel count, background HU and noise under the default
    rules; the same with the rule thresholds relaxed (``max_bg_shift_mm``,
    ``max_shift_mm``), with the ellipsoid ROI and without neighbour masking;
    for every variant the auto tile fit's bent-tile RMS and chord residual,
    and the split-half repeatability (odd / even 1 mm slices as two 2 mm
    scans) when the estimator is allowed to engage.  Loads the phantom
    (~2 s) and runs detection; no full pipeline, no cache.

``--part thincut``
    PostOp (1 mm slabs every 2 mm, gaps interpolated) seeds against the
    seeds of the contiguous 1 mm thin-cut of the same acquisition
    (DOEJOHNPOSTCT) with the two sides refined independently, from the
    cached results of ``validation_realdata_proxies.py --refine none`` and
    ``--refine centroid`` (``--cache-dir``, default
    output/validation_realdata_proxies/cache, newest commit in the cache):
    the 2 x 2 table of PostOp-minus-thin-cut statistics, the per-seed
    before / after distance of every PostOp seed the refinement moved, and
    the thin-cut's own auto-fit bent-tile RMS before / after.
"""
from __future__ import annotations

import argparse
import functools
import glob
import importlib.util
import os
import pickle
import sys
import time
import types
import warnings

import numpy as np
from scipy.spatial.distance import cdist

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_proxies():
    spec = importlib.util.spec_from_file_location(
        "validation_realdata_proxies",
        os.path.join(REPO, "scripts", "validation_realdata_proxies.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hist(status):
    out = {}
    for s in status:
        out[s] = out.get(s, 0) + 1
    return ", ".join("%s x%d" % kv for kv in sorted(out.items()))


# ------------------------------------------------------------------ printed
VARIANTS = [
    ("default", {}),
    ("bg 0.3", dict(max_bg_shift_mm=0.3)),
    ("bg 1.0", dict(max_bg_shift_mm=1.0)),
    ("shift 3.0", dict(max_shift_mm=3.0)),
    ("bg 10, shift 10 (rule fallbacks off)", dict(max_bg_shift_mm=10.0,
                                                   max_shift_mm=10.0)),
    ("bg 10, shift 10, roi ellipsoid", dict(max_bg_shift_mm=10.0,
                                            max_shift_mm=10.0,
                                            roi="ellipsoid")),
    ("bg 10, shift 10, mask none", dict(max_bg_shift_mm=10.0,
                                        max_shift_mm=10.0, mask="none")),
]


def part_printed(vp):
    from gtcore.seeds.refine import refine_seed_candidates
    from gtcore.tiles import fit_tiles

    vol, path = vp.load("printed8")
    if vol is None:
        print("printed phantom not found at %s" % path)
        return 1
    print("volume %s, spacing %s, %s" % (vol.array.shape, vol.spacing, path))
    c0 = vp.detect(vol)
    print("detected %d" % len(c0))

    def auto_fit(centers, axes):
        res = fit_tiles(centers, axes, "auto", 0, spacing_mm=vol.spacing)
        r = [float(p.deform.rms_mm) for p in res.tiles
             if p.deform is not None] or [np.nan]
        rep = vp.tile_report(dict(centers=np.asarray(centers),
                                  axes=np.asarray(axes), tiles=res))
        cres = np.concatenate([t["chord_resid"] for t in rep
                               if "chord_resid" in t] or [np.zeros(0)])
        return (len(res.tiles), np.mean(r), np.max(r),
                cres.mean() if cres.size else np.nan,
                cres.std(ddof=1) if cres.size > 1 else np.nan)

    for name, kw in VARIANTS:
        t0 = time.time()
        r = refine_seed_candidates(vol, c0, **kw)
        dt = time.time() - t0
        info = r.info
        st = [str(s) for s in info["refine_status"]]
        ok = np.array([s == "ok" for s in st])
        shift = np.asarray(info["refine_shift_mm"], float)
        bgs = np.asarray(info["refine_bg_shift_mm"], float)
        nsat = np.asarray(info["refine_n_saturated"], int)
        bg = np.asarray(info["refine_background_hu"], float)
        sig = np.asarray(info["refine_sigma_noise_hu"], float)
        disp = np.linalg.norm(np.asarray(r.centers_ras)
                              - np.asarray(c0.centers_ras), axis=1)
        print("\n== %s: %d/%d ok (%.2f s); %s"
              % (name, ok.sum(), len(st), dt, _hist(st)))
        if ok.any():
            print("   ok seeds: shift median %.3f max %.3f mm; bg-gradient "
                  "shift median %.3f max %.3f mm; n_sat median %d; bg HU "
                  "median %.0f; sigma_noise median %.0f HU"
                  % (np.median(shift[ok]), shift[ok].max(),
                     np.nanmedian(bgs[ok]), np.nanmax(bgs[ok]),
                     np.median(nsat[ok]), np.median(bg[ok]),
                     np.median(sig[ok])))
        print("   all seeds: bg-gradient shift median %.3f max %.3f mm; "
              "n_sat median %d (max %d); bg HU median %.0f (min %.0f max "
              "%.0f); sigma_noise median %.0f (max %.0f)"
              % (np.nanmedian(bgs), np.nanmax(bgs), np.median(nsat),
                 nsat.max(), np.median(bg), bg.min(), bg.max(),
                 np.median(sig), sig.max()))
        n, m, mx, cm, cs = auto_fit(r.centers_ras, r.axes_ras)
        print("   auto fit: %d tiles, bent-tile RMS mean %.3f max %.3f; "
              "chord resid %.3f +/- %.3f mm; |refined - detected| median "
              "%.3f max %.3f" % (n, m, mx, cm, cs, np.median(disp),
                                 disp.max()))
        if name == "default":
            print("   per seed:")
            for i in range(len(st)):
                print("     %2d %-24s shift %.3f bgshift %.3f nsat %3d bg "
                      "%6.0f sig %4.0f vol %.2f mm3"
                      % (i, st[i], shift[i], bgs[i], nsat[i], bg[i], sig[i],
                         c0.volumes_mm3[i]))

    for name, kw in (("default", {}),
                     ("bg 10, shift 10", dict(max_bg_shift_mm=10.0,
                                              max_shift_mm=10.0))):
        ctx = types.SimpleNamespace(
            refiner=functools.partial(refine_seed_candidates, **kw),
            args=types.SimpleNamespace(refine="centroid"))
        ref = refine_seed_candidates(vol, c0, **kw).centers_ras
        s = vp.split_half(ctx, vol, np.asarray(ref))
        ba = s["bland_altman"]
        print("\n== split-half, %s: pairs %d; disagreement/sqrt2 i/j/k %.3f / "
              "%.3f / %.3f, 3D %.3f; BA bias %.3f / %.3f / %.3f, SD %.3f / "
              "%.3f / %.3f; even %d det %d FP, odd %d det %d FP; refined "
              "even %s, odd %s"
              % (name, s["n_pairs"], ba["i"]["precision"],
                 ba["j"]["precision"], ba["k"]["precision"],
                 s["precision_3d"], ba["i"]["bias"], ba["j"]["bias"],
                 ba["k"]["bias"], ba["i"]["sd"], ba["j"]["sd"],
                 ba["k"]["sd"], s["even"]["n_det"], s["even"]["fp"],
                 s["odd"]["n_det"], s["odd"]["fp"],
                 (s["refine"].get("even") or {}).get("n_ok"),
                 (s["refine"].get("odd") or {}).get("n_ok")))
        for h in ("even", "odd"):
            v = s[h]["vs_full"]
            print("   %s vs 1 mm: RMS %.2f / %.2f / %.2f, 3D %.2f +/- %.2f"
                  % (h, v["rms_x"], v["rms_y"], v["rms_z"], v["mean_3d"],
                     v["sd_3d"]))
    return 0


# ------------------------------------------------------------------ thincut
def _load_cache(cache_dir):
    slims = {}
    for fn in glob.glob(os.path.join(cache_dir, "*.pkl")):
        with open(fn, "rb") as fh:
            s = pickle.load(fh)
        k = s["key"]
        name = "postop" if os.path.basename(fn).startswith("postop_") else (
            "doe" if os.path.basename(fn).startswith("doe_") else None)
        if name is None or k.get("fuse"):
            continue
        slims.setdefault(k["commit"], {})[(name, k["refine"])] = s
    return slims


def part_thincut(vp, cache_dir, commit=None):
    from gtcore.tiles import fit_tiles

    by_commit = _load_cache(cache_dir)
    need = {("postop", "none"), ("postop", "centroid"), ("doe", "none"),
            ("doe", "centroid")}
    usable = [c for c, d in by_commit.items() if need <= set(d)]
    if commit is None:
        if not usable:
            print("no commit in %s has all four cached runs (postop / doe x "
                  "none / centroid); run validation_realdata_proxies.py "
                  "--refine none and --refine centroid first" % cache_dir)
            return 1
        commit = max(usable, key=lambda c: max(
            os.path.getmtime(f) for f in glob.glob(
                os.path.join(cache_dir, "*.pkl"))))
    slims = by_commit[commit]
    print("cached runs at commit %s" % commit)
    for pr in ("none", "centroid"):
        for tr in ("none", "centroid"):
            x = vp.postop_vs_thincut(slims[("postop", pr)], slims[("doe", tr)])
            for name in ("all", "assigned"):
                v = x["rows"]["detected"][name]
                print("PostOp %-8s thin-cut %-8s %-8s n %2d mean %6.3f %6.3f "
                      "%6.3f SD %.3f %.3f %.3f RMS %.3f %.3f %.3f 3D mean "
                      "%.2f median %.2f P95 %.2f max %.2f"
                      % (pr, tr, name, v["n"], *v["mean"], *v["sd"],
                         *v["rms"], v["mean_3d"], v["median_3d"],
                         v["p95_3d"], v["max_3d"]))
    P0 = np.asarray(slims[("postop", "none")]["centers_raw"], float)
    P1 = np.asarray(slims[("postop", "centroid")]["centers_raw"], float)
    st = slims[("postop", "centroid")]["refine"]["status"]
    T0 = np.asarray(slims[("doe", "none")]["centers_raw"], float)
    T1 = np.asarray(slims[("doe", "centroid")]["centers_raw"], float)
    stt = slims[("doe", "centroid")]["refine"]["status"]
    dt = np.linalg.norm(T1 - T0, axis=1)
    print("thin-cut refined %d/%d, |shift| median %.3f max %.3f mm; "
          "fallbacks %s" % (sum(s == "ok" for s in stt), len(stt),
                            np.median(dt[dt > 0]) if (dt > 0).any() else 0.0,
                            dt.max(), [s for s in stt if s != "ok"]))
    A = np.asarray(slims[("postop", "none")]["affine"], float)
    da = A[:3, :3] / np.linalg.norm(A[:3, :3], axis=0)
    interp = set(slims[("postop", "none")]["meta"].get("interpolated_k", []))
    Ainv = np.linalg.inv(A)
    moved = [i for i in range(len(st)) if st[i] == "ok"]
    print("PostOp refined %d/%d (i/j/k in PostOp voxel axes, mm):"
          % (len(moved), len(st)))
    for tl, T in (("vs thin-cut none", T0), ("vs thin-cut centroid", T1)):
        print("-- " + tl)
        for i in moved:
            j0 = int(np.argmin(cdist(P0[i:i + 1], T)))
            j1 = int(np.argmin(cdist(P1[i:i + 1], T)))
            e0 = (P0[i] - T[j0]) @ da
            e1 = (P1[i] - T[j1]) @ da
            k0 = (Ainv @ np.r_[P0[i], 1.0])[2]
            print("   seed %2d: |d| %.3f (%6.3f %6.3f %6.3f) -> %.3f (%6.3f "
                  "%6.3f %6.3f); moved %.3f mm; k %.2f, interpolated slices "
                  "within 1.5: %s"
                  % (i, np.linalg.norm(e0), *e0, np.linalg.norm(e1), *e1,
                     np.linalg.norm(P1[i] - P0[i]), k0,
                     sorted(k for k in interp if abs(k - k0) <= 1.5)))
    for lab, S in (("none", slims[("doe", "none")]),
                   ("centroid", slims[("doe", "centroid")])):
        res = fit_tiles(S["centers_raw"], S["axes"], "auto", 0,
                        spacing_mm=S["spacing"])
        r = [float(p.deform.rms_mm) for p in res.tiles
             if p.deform is not None]
        print("thin-cut auto fit (%s): %d tiles, bent-tile RMS %s, mean %.3f"
              % (lab, len(res.tiles), ["%.2f" % x for x in sorted(r)],
                 np.mean(r)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", default="printed", choices=("printed",
                                                           "thincut", "all"))
    ap.add_argument("--cache-dir", default=os.path.join(
        REPO, "output", "validation_realdata_proxies", "cache"))
    ap.add_argument("--commit", default=None,
                    help="thincut: cache commit to use (default newest)")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    vp = load_proxies()
    rc = 0
    if args.part in ("printed", "all"):
        rc |= part_printed(vp)
    if args.part in ("thincut", "all"):
        rc |= part_thincut(vp, args.cache_dir, args.commit)
    return rc


if __name__ == "__main__":
    sys.exit(main())
