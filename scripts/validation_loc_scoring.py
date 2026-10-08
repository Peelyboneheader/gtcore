"""Stage 4 / stage 6 measurements of docs/plan-localization.md on the
physical 8-tile printed phantom (no truth).

1. ``--extract``: ``gtcore.io.load_volume`` + ``gtcore.pipeline.reconstruct``
   (no tile fit) once, seeds pickled to
   ``output/loc_scoring/cache/phantom8_seeds.pkl`` (needs ~5 GB; run alone).
2. default: from the cache, compare the counted fit with the chord score,
   the counted fit with the bent-tile (deformable) score and auto mode --
   partitions, which tile seeds 25 and 31 join, runtime -- and the partition
   margins of every tile (stage 6).
3. ``--calibration``: synthetic check of the Gauss-Newton centre covariance
   (``DeformableFit.compute_uncertainty``) against the empirical scatter of
   fitted centres, 200 noisy draws per row, for several axis-noise levels
   (the pooled ``s^2`` assumes one noise level for every residual row).

Run from the repo root:
    python scripts/validation_loc_scoring.py --extract
    python scripts/validation_loc_scoring.py
    python scripts/validation_loc_scoring.py --calibration
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_PHANTOM8 = r"C:\Users\jacob\OneDrive\Documents\3D-Printed Phantom-8tiles (223)"
CACHE = os.path.join("output", "loc_scoring", "cache", "phantom8_seeds.pkl")


def extract(path):
    from gtcore.io import load_volume
    from gtcore.pipeline import reconstruct

    t0 = time.perf_counter()
    vol = load_volume(path)
    res = reconstruct(vol, verbose=False)
    seeds = res.seeds
    payload = dict(
        centers_ras=np.asarray(seeds.centers_ras, float),
        axes_ras=np.asarray(seeds.axes_ras, float),
        spacing=tuple(float(s) for s in vol.spacing),
        source=path,
    )
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(payload, fh)
    print("extracted %d seeds (spacing %s) in %.1f s -> %s" % (
        len(payload["centers_ras"]), payload["spacing"],
        time.perf_counter() - t0, CACHE))


def _tile_of(result, seed):
    for p in result.tiles:
        if seed in p.seed_indices:
            return p.tile_id, tuple(sorted(p.seed_indices))
    return None, None


def _report(name, result, dt, margins_dt=None):
    print("\n== %s  (%.2f s%s)" % (name, dt, "" if margins_dt is None
                                   else ", margins +%.4f s" % margins_dt))
    print("   capped=%s  score_rule=%s" % (result.capped,
                                           getattr(result, "score_rule", "?")))
    for p in result.tiles:
        m = (result.partition_margins or {}).get(p.tile_id)
        rms = p.deform.rms_mm if p.deform is not None else float("nan")
        unc = ""
        if p.deform is not None:
            p.deform.compute_uncertainty()
            if p.deform.center_cov is not None:
                unc = "  centre sd %.2f mm  normal sd %.1f deg" % (
                    float(np.sqrt(np.trace(p.deform.center_cov))),
                    p.deform.normal_sigma_deg)
        print("   T%d %-18s rms %.2f mm%s  margin %s%s%s" % (
            p.tile_id, sorted(p.seed_indices), rms,
            " degraded" if p.degraded else "",
            "-" if m is None else ("inf" if not np.isfinite(m) else "%.2f" % m),
            "  AMBIGUOUS" if p.tile_id in (result.ambiguous_tiles or []) else "",
            unc))
        alt = (result.partition_alternatives or {}).get(p.tile_id)
        if alt is not None and m is not None and np.isfinite(m) and m < 3.0:
            mine = {tuple(sorted(q.seed_indices)) for q in result.tiles}
            print("      best alternative swaps in %s" % (
                [g for g in alt if tuple(sorted(g)) not in mine],))
    for seed in (25, 31):
        print("   seed %d -> tile %s %s" % ((seed,) + _tile_of(result, seed)))


def _timed_margins():
    """Wrap the two margin helpers so their own cost is measured."""
    import gtcore.tiles.auto as A
    import gtcore.tiles.fit as F

    acc = {"s": 0.0}

    def wrap(fn):
        def inner(*args, **kw):
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            acc["s"] += time.perf_counter() - t0
            return out
        return inner

    F._counted_margins = wrap(F._counted_margins)
    A._per_count_margins = wrap(A._per_count_margins)
    return acc


def _quad_detail(c, a):
    """Chord and bent-tile scores of the two competing readings of the
    25/31 pair, and how many quads pass the chord gates."""
    from gtcore.tiles.auto import deformable_score
    from gtcore.tiles.deform import fit_deformable
    from gtcore.tiles.fit import (DIAG_MAX_MM, SIDE_MIN_MM, _enumerate_quads,
                                  _normalize_axes)

    ax = _normalize_axes(a)
    dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    link = (dist >= SIDE_MIN_MM) & (dist <= DIAG_MAX_MM)
    quads = _enumerate_quads(c, ax, dist, link)
    chord = {frozenset(q[1]): q[0] for q in quads}
    print("\n%d quads pass the chord gates" % len(quads))
    readings = {"bent-tile (auto)": [(8, 16, 24, 25), (26, 29, 30, 31)],
                "chord (counted)": [(25, 26, 29, 30), (8, 16, 24, 31)]}
    for name, groups in readings.items():
        tot_c = tot_d = 0.0
        for g in groups:
            fit = fit_deformable(c[list(g)], ax[list(g)], kind="full")
            sc, sd = chord.get(frozenset(g), float("nan")), deformable_score(fit)
            tot_c += sc
            tot_d += sd
            print("   %-16s %s  chord %.2f  bent-tile %.2f (rms %.2f mm, "
                  "axis err %.1f deg, E %.3f)" % (name, g, sc, sd, fit.rms_mm,
                                                  fit.axis_err_deg,
                                                  fit.bending_energy))
        print("   %-16s total  chord %.2f  bent-tile %.2f" % (name, tot_c, tot_d))


def measure(n_full=8):
    from gtcore.tiles import ImplantPrior, fit_tiles, fit_tiles_prior

    with open(CACHE, "rb") as fh:
        d = pickle.load(fh)
    c, a, spacing = d["centers_ras"], d["axes_ras"], d["spacing"]
    print("printed phantom: %d seeds, spacing %s, |25-31| = %.2f mm" % (
        len(c), spacing, float(np.linalg.norm(c[25] - c[31]))))
    acc = _timed_margins()
    runs = [
        ("counted chord (fit_tiles n=8, score=chord)",
         lambda m: fit_tiles(c, a, n_full, 0, complete_degraded=True,
                             score="chord", margins=m)),
        ("counted deformable (fit_tiles n=8, score=deformable)",
         lambda m: fit_tiles(c, a, n_full, 0, complete_degraded=True,
                             score="deformable", margins=m)),
        ("auto (fit_tiles n='auto')",
         lambda m: fit_tiles(c, a, "auto", spacing_mm=spacing, margins=m)),
        ("planner, count given (fit_tiles_prior n_full=8)",
         lambda m: fit_tiles_prior(c, a, ImplantPrior(n_full=n_full),
                                   spacing_mm=spacing, margins=m)),
        ("planner, count unknown (fit_tiles_prior ImplantPrior())",
         lambda m: fit_tiles_prior(c, a, ImplantPrior(), spacing_mm=spacing,
                                   margins=m)),
    ]
    for name, fn in runs:
        dts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn(False)
            dts.append(time.perf_counter() - t0)
        acc["s"] = 0.0
        res = fn(True)
        dt_unc = 0.0
        for p in res.all_tiles if hasattr(res, "all_tiles") else res.tiles:
            if p.deform is not None and p.deform._solution is not None:
                p.deform.cov_x = None
                t0 = time.perf_counter()
                p.deform.compute_uncertainty()
                dt_unc += time.perf_counter() - t0
        _report(name, res, min(dts), acc["s"])
        print("   pose uncertainty for all tiles: %.2f ms" % (1000 * dt_unc))
    _quad_detail(c, a)


def calibration(n_draws=200, sigma=0.3):
    from scipy.spatial.transform import Rotation

    from gtcore.tiles.deform import (DeformParams, deformed_seed_axes,
                                     deformed_seed_points, fit_deformable)
    from gtcore.tiles.model import TilePose6

    R = Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix()
    pose = TilePose6(R, np.array([10.0, -20.0, 35.0]), "full", 1.0)
    print("seed noise %.1f mm per coordinate, %d draws per row; ratio = "
          "mean trace(center_cov) / trace(empirical cov of fitted centres)"
          % (sigma, n_draws))
    print("kappa  axis noise        ratio  normal sd pred / emp (deg)  ms/fit")
    for kappa in (0.0, 0.05):
        params = DeformParams(kappa, 0.6 * kappa, 0.3)
        P0 = deformed_seed_points(pose, params)
        A0 = deformed_seed_axes(pose, params)
        for label, sax in (("none", 0.0), ("sigma/w_axis 4.3deg", sigma / 4.0),
                           ("5 deg", np.radians(5.0)),
                           ("10 deg", np.radians(10.0))):
            rng = np.random.default_rng(1)
            cents, traces, nsig, nrm = [], [], [], []
            t0 = time.perf_counter()
            for _ in range(n_draws):
                P = P0 + rng.normal(0.0, sigma, P0.shape)
                A = []
                for ax in A0:
                    v = rng.normal(0.0, sax, 3) if sax > 0 else np.zeros(3)
                    v -= (v @ ax) * ax
                    A.append((ax + v) / np.linalg.norm(ax + v))
                f = fit_deformable(P, np.array(A), kind="full",
                                   hinge_starts=False).compute_uncertainty()
                cents.append(f.seed_points().mean(axis=0))
                traces.append(float(np.trace(f.center_cov)))
                nsig.append(f.normal_sigma_deg)
                nrm.append(f.pose.normal * np.sign(f.pose.normal @ R[:, 2]))
            ms = 1000.0 * (time.perf_counter() - t0) / n_draws
            emp = float(np.trace(np.cov(np.array(cents).T)))
            ang = np.degrees(np.arccos(np.clip(np.array(nrm) @ R[:, 2],
                                               -1.0, 1.0)))
            print("%.2f   %-18s %5.2f  %5.2f / %5.2f              %4.0f" % (
                kappa, label, float(np.mean(traces)) / emp,
                float(np.mean(nsig)), float(np.sqrt(np.mean(ang ** 2))), ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--phantom8", default=DEFAULT_PHANTOM8)
    args = ap.parse_args()
    if args.extract:
        extract(args.phantom8)
        return
    if args.calibration:
        calibration()
        return
    measure()


if __name__ == "__main__":
    main()
