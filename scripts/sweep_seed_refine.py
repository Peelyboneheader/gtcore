"""Stage 2 measurements (docs/localization-notes.md): grey-level centroid
refinement vs threshold detection.

Two parts, both with exact truth:

* ``head``: the synthetic head phantom (binary capsules painted on a 0.7 mm
  grid, 3 tiles / 12 seeds per realization), slices block-averaged to
  0.7 / 1.4 / 2.1 / 2.8 mm, detection as ``pipeline.reconstruct`` runs it;
  default refinement plus the ablation variants (ROI shape, masking,
  sampling term, shell, background check, PSF).
* ``threshold``: centre shift between 1200 and 2000 HU detections.
* ``analytic``: supersampled capsules (tests/test_seeds_refine.py
  ``_render_capsules``) on anisotropic grids standing in for the harness
  grids G1 (0.59x0.59x1.0), G2 (0.5x0.5x2.0), G4 (0.7x0.7x2.8) plus
  0.5x0.5x2.1 and 0.7 iso; 72 random seeds each, 20 HU noise; reports
  per-axis RMS and NEES for every sampling-term model.

    python scripts/sweep_seed_refine.py --part all
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gtcore.phantom import make_head_phantom  # noqa: E402
from gtcore.seeds import detect_seed_candidates, refine_seed_candidates  # noqa: E402
from test_seeds_refine import (  # noqa: E402
    _detect,
    _match,
    _random_layout,
    _render_capsules,
    _thick_slices,
)

VARIANTS = {
    "default": {},
    "roi=ellipsoid": dict(roi="ellipsoid"),
    "slab=exact": dict(slab="exact"),
    "slab=rule": dict(slab="rule"),
    "mask=none": dict(mask="none"),
    "mask=capsule": dict(mask="capsule"),
    "mask=both": dict(mask="both"),
    "bg_shell=1.0": dict(bg_shell_mm=1.0),
    "bg_shell=2.5": dict(bg_shell_mm=2.5),
    "max_bg_shift=0.05": dict(max_bg_shift_mm=0.05),
    "max_bg_shift=0.2": dict(max_bg_shift_mm=0.2),
    "max_bg_shift=off": dict(max_bg_shift_mm=None),
    "psf=0.35": dict(psf_sigma_mm=0.35),
    "psf=0.60": dict(psf_sigma_mm=0.60),
}
GRIDS = [  # spacing, supersampling, detection threshold
    ((0.59, 0.59, 1.0), (3, 3, 5), 1500.0),
    ((0.5, 0.5, 2.0), (3, 3, 10), 1000.0),
    ((0.5, 0.5, 2.1), (3, 3, 7), 1000.0),
    ((0.7, 0.7, 2.8), (3, 3, 8), 800.0),
    ((0.7, 0.7, 0.7), (3, 3, 3), 2000.0),
]


def head(rngs, variants):
    cases = []
    for r in rngs:
        vol0, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=r)
        t = np.array([s.center_ras for s in truth.seeds])
        for f in (1, 2, 3, 4):
            vol = _thick_slices(vol0, f)
            c = _detect(vol)
            cases.append((f, vol, c, t, _match(t, c.centers_ras)))
    for name in variants:
        kw = VARIANTS[name]
        acc = {f: dict(b=[], e=[], bz=[], ez=[], worst=[], fb=0, nees=[],
                       neesz=[], ms=[]) for f in (1, 2, 3, 4)}
        for f, vol, c, t, m in cases:
            t0 = time.perf_counter()
            rc = refine_seed_candidates(vol, c, **kw)
            a = acc[f]
            a["ms"].append(1e3 * (time.perf_counter() - t0) / max(1, len(c)))
            for ti, j in m.items():
                e0, e1 = c.centers_ras[j] - t[ti], rc.centers_ras[j] - t[ti]
                a["b"].append(np.linalg.norm(e0))
                a["e"].append(np.linalg.norm(e1))
                a["bz"].append(e0[2] ** 2)
                a["ez"].append(e1[2] ** 2)
                a["worst"].append(np.linalg.norm(e1) - np.linalg.norm(e0))
                if rc.info["refine_status"][j] == "ok":
                    C = rc.cov_ras[j]
                    a["nees"].append(e1 @ np.linalg.solve(C, e1) / 3.0)
                    a["neesz"].append(e1[2] ** 2 / C[2, 2])
                else:
                    a["fb"] += 1
        print("\n%s %s" % (name, kw))
        print("| slices | n | mean (mm) | gain | max | z RMS | worst seed vs "
              "detection | fallbacks | NEES 3D / z | ms/seed |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for f, a in acc.items():
            b, e = np.array(a["b"]), np.array(a["e"])
            print("| %.1f | %d | %.3f -> %.3f | %+.0f %% | %.3f -> %.3f | "
                  "%.3f -> %.3f | %+.3f | %d | %.0f / %.2f | %.2f |" % (
                      0.7 * f, len(b), b.mean(), e.mean(),
                      100 * (e.mean() / b.mean() - 1), b.max(), e.max(),
                      np.sqrt(np.mean(a["bz"])), np.sqrt(np.mean(a["ez"])),
                      max(a["worst"]), a["fb"], np.mean(a["nees"]),
                      np.mean(a["neesz"]), np.mean(a["ms"])))


def threshold(rngs):
    """Centre shift between 1200 and 2000 HU detections of the same seed
    (truth-matched at both thresholds), before and after refinement."""
    out = {}
    for r in rngs:
        vol0, truth = make_head_phantom(spacing=0.7, n_tiles=3, rng_seed=r)
        t = np.array([s.center_ras for s in truth.seeds])
        for f in (1, 2, 3, 4):
            vol = _thick_slices(vol0, f)
            lo, hi = _detect(vol, 1200.0), _detect(vol, 2000.0)
            rlo, rhi = (refine_seed_candidates(vol, lo),
                        refine_seed_candidates(vol, hi))
            mlo, mhi = _match(t, lo.centers_ras), _match(t, hi.centers_ras)
            for ti in set(mlo) & set(mhi):
                a, b = mlo[ti], mhi[ti]
                ok = (rlo.info["refine_status"][a] == "ok"
                      and rhi.info["refine_status"][b] == "ok")
                out.setdefault(f, []).append((
                    np.linalg.norm(lo.centers_ras[a] - hi.centers_ras[b]),
                    np.linalg.norm(rlo.centers_ras[a] - rhi.centers_ras[b]),
                    ok))
    print("\n| slices | seeds (both ok) | detection median / P90 | refined "
          "median / P90 | refined <= 0.1 mm |")
    print("|---|---|---|---|---|")
    for f, v in sorted(out.items()):
        v = np.asarray(v, dtype=float)
        print("| %.1f | %d (%d) | %.3f / %.3f | %.3f / %.3f | %.0f %% |" % (
            0.7 * f, len(v), int(v[:, 2].sum()), np.median(v[:, 0]),
            np.percentile(v[:, 0], 90), np.median(v[:, 1]),
            np.percentile(v[:, 1], 90), 100 * (v[:, 1] <= 0.1).mean()))


def analytic():
    print("\n| grid (mm) | slab | mean (mm) | RMS x/y/z after | NEES 3D | "
          "NEES x/y/z |")
    print("|---|---|---|---|---|---|")
    for sp, nf, thr in GRIDS:
        rng = np.random.default_rng(1)
        centers, axes = _random_layout(6, rng)
        vol = _render_capsules(sp, centers, axes, n_fine=nf, noise=20.0, rng=2)
        c = detect_seed_candidates(vol, hu_threshold=thr, min_mm3=0.2,
                                   max_mm3=200.0)
        D = cdist(c.centers_ras, centers)
        ok_det = D.min(axis=1) < 2.0
        ti = D.argmin(axis=1)[ok_det]
        e0 = c.centers_ras[ok_det] - centers[ti]
        for slab in ("bound", "exact", "rule"):
            rc = refine_seed_candidates(vol, c, slab=slab)
            e1 = rc.centers_ras[ok_det] - centers[ti]
            cv = rc.cov_ras[ok_det]
            good = rc.info["refine_status"][ok_det] == "ok"
            nees = np.array([v @ np.linalg.solve(C, v) / 3.0
                             for v, C in zip(e1[good], cv[good])])
            per = ((e1 ** 2) / cv[:, [0, 1, 2], [0, 1, 2]])[good].mean(axis=0)
            rms = np.sqrt((e1 ** 2).mean(axis=0))
            print("| %s | %s | %.3f -> %.3f | %.3f / %.3f / %.3f | %.2f | "
                  "%.2f / %.2f / %.2f |" % (
                      "x".join("%g" % v for v in sp), slab,
                      np.linalg.norm(e0, axis=1).mean(),
                      np.linalg.norm(e1, axis=1).mean(), *rms, nees.mean(),
                      *per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("head", "threshold", "analytic", "all"),
                    default="all")
    ap.add_argument("--rng", default="0,1,2,3,4")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    if args.part in ("head", "all"):
        head([int(x) for x in args.rng.split(",")], args.variants.split(","))
    if args.part in ("threshold", "all"):
        threshold([int(x) for x in args.rng.split(",")])
    if args.part in ("analytic", "all"):
        analytic()


if __name__ == "__main__":
    main()
