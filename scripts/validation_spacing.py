"""Validation study: seed detection & tile fitting vs slice spacing.

Simulates acquisition at coarser slice thickness by block-averaging the
0.7 mm synthetic phantom along z (true partial-volume averaging), then runs
detection -> shape filter -> tile fitting and scores against ground truth.
This quantifies the degradation observed on the real 2 mm post-op export.

Outputs: output/validation_spacing.csv, output/validation_spacing.png,
and a summary table on stdout.
"""
from __future__ import annotations

import csv
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import filter_seed_shaped, seed_detection_params
from gtcore.seeds import detect_seed_candidates
from gtcore.tiles import fit_tiles
from gtcore.volume import Volume

BASE_SPACING = 0.7
Z_FACTORS = [1, 2, 3, 4]  # -> 0.7, 1.4, 2.1, 2.8 mm slices
RNG_SEEDS = [0, 1, 2, 3, 4]
N_TILES = 3


def thick_slices(vol, factor):
    """Block-average along k: partial-volume simulation of thick slices."""
    if factor == 1:
        return vol
    nk = (vol.array.shape[0] // factor) * factor
    arr = vol.array[:nk].reshape(-1, factor, *vol.array.shape[1:]).mean(axis=1)
    affine = vol.affine.copy()
    affine[:3, 2] *= factor
    # new slice centres sit at the mean of the merged slice centres
    affine[:3, 3] += vol.affine[:3, 2] * (factor - 1) / 2.0
    return Volume(arr.astype(np.float32), affine, dict(vol.meta))


FIXED_PARAMS = dict(hu_threshold=2000.0, min_mm3=1.0, max_mm3=15.0,
                    min_elong=1.5, max_elong=10.0)


def score(vol, truth, adaptive):
    p = seed_detection_params(vol.spacing) if adaptive else FIXED_PARAMS
    cands = filter_seed_shaped(
        detect_seed_candidates(vol, hu_threshold=p["hu_threshold"],
                               min_mm3=p["min_mm3"], max_mm3=p["max_mm3"]),
        min_mm3=p["min_mm3"], max_mm3=p["max_mm3"],
        min_elong=p["min_elong"], max_elong=p["max_elong"])
    t = np.array([s.center_ras for s in truth.seeds])
    if len(cands) == 0:
        return dict(n_det=0, recall=0.0, err_mean=np.nan, err_max=np.nan,
                    partition_ok=False)
    D = cdist(t, cands.centers_ras)
    errs, used = [], set()
    for ti in range(len(t)):
        for j in np.argsort(D[ti]):
            if int(j) not in used:
                if D[ti, int(j)] < 2.0:
                    used.add(int(j))
                    errs.append(D[ti, int(j)])
                break
    recall = len(errs) / len(t)
    fit = fit_tiles(cands.centers_ras, cands.axes_ras, N_TILES,
                    cavity_center_ras=truth.cavity_center_ras)
    # partition check: every truth tile recovered as one fitted tile
    ok = False
    if fit.all_assigned and len(fit.tiles) == N_TILES:
        det2truth = {}
        for j in range(len(cands)):
            ti = int(np.argmin(D[:, j]))
            det2truth[j] = truth.seeds[ti].tile_id if D[ti, j] < 2.0 else -1
        groups = [frozenset(det2truth.get(i, -9) for i in tp.seed_indices)
                  for tp in fit.tiles]
        ok = all(len(g) == 1 and -1 not in g and -9 not in g for g in groups)
    return dict(n_det=len(cands), recall=recall,
                err_mean=float(np.mean(errs)) if errs else np.nan,
                err_max=float(np.max(errs)) if errs else np.nan,
                partition_ok=ok)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in Z_FACTORS:
        for seed in RNG_SEEDS:
            vol, truth = make_head_phantom(spacing=BASE_SPACING, n_tiles=N_TILES,
                                           rng_seed=seed)
            tv = thick_slices(vol, f)
            for adaptive in (False, True):
                r = score(tv, truth, adaptive)
                r.update(dz=BASE_SPACING * f, rng=seed,
                         mode="adaptive" if adaptive else "fixed")
                rows.append(r)
                print("dz=%.1f rng=%d %-8s: det %2d/12 recall %.2f err %.2f/%.2f mm partition %s"
                      % (r["dz"], seed, r["mode"], r["n_det"], r["recall"],
                         r["err_mean"], r["err_max"], r["partition_ok"]))

    with open(os.path.join(out_dir, "validation_spacing.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dzs = sorted(set(r["dz"] for r in rows))

    def agg(key, mode, fn=np.mean):
        return [fn([r[key] for r in rows if r["dz"] == d and r["mode"] == mode])
                for d in dzs]

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
    ax2.plot(dzs, agg("err_mean", "adaptive", np.nanmean), "^-", color="tab:red",
             label="localization error (adaptive)")
    ax2.set_ylabel("error (mm)", color="tab:red")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=8)
    ax1.set_title("Seed detection & tile fitting vs slice spacing\n"
                  "(synthetic phantom, 3 tiles / 12 seeds, 5 realizations)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_spacing.png"), dpi=150)
    print("\nmean by spacing (mode=adaptive | fixed):")
    ra, rf = agg("recall", "adaptive"), agg("recall", "fixed")
    ea = agg("err_mean", "adaptive", np.nanmean)
    pa = agg("partition_ok", "adaptive")
    for d, a_, f_, e_, p_ in zip(dzs, ra, rf, ea, pa):
        print("  dz %.1f mm: recall %.2f | %.2f  err %.2f mm  partition %.2f"
              % (d, a_, f_, e_, p_))


if __name__ == "__main__":
    main()
