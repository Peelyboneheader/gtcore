"""Validation study: inter-seed and tile-carrier interference.

Quantifies what `gtcore.dose.interference` changes, on two geometries:

1. a synthetic three-tile layout (12 seeds on a 10 mm grid) where the seed
   arrangement is exactly known, and
2. the phantom's ground-truth implant, which has real wall-conformed seed
   poses rather than a flat grid.

For each it reports the dose ratio (corrected / plain TG-43) restricted to
clinically meaningful dose levels, contrasts capsules-only against
capsules-plus-carriers across a sweep of the unmeasured carrier density, and
times the overhead.

Outputs: output/validation_interference.csv, output/validation_interference.png,
and a summary table on stdout.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gtcore.dose import (
    InterferenceModel,
    compute_dose_grid,
    interference_report,
)
from gtcore.phantom import make_head_phantom

RX_CGY = 6000.0
GRID_MM = 2.0
PAD_MM = 50.0
#: Carrier densities to sweep; 1.0 is water-equivalent (the term vanishes).
CARRIER_DENSITIES = [0.15, 0.30, 0.50, 0.75, 1.00]


class _Tile:
    """Duck-typed stand-in for a PlacedTile (what TileCarrier.from_tile wants)."""

    def __init__(self, seed_centers, normal_ras, axis_ras, kind="full"):
        self.seed_centers = np.asarray(seed_centers, dtype=float)
        self.normal_ras = np.asarray(normal_ras, dtype=float)
        self.axis_ras = np.asarray(axis_ras, dtype=float)
        self.kind = kind


# ------------------------------------------------------------------ geometry
def flat_layout(n_tiles=3):
    """``n_tiles`` coplanar 4-seed tiles, 10 mm seed grid, 22 mm apart."""
    origins = [(0.0, 0.0), (22.0, 0.0), (0.0, 22.0), (22.0, 22.0)][:n_tiles]
    centers, axes, tiles = [], [], []
    for ox, oy in origins:
        seeds = np.array([[ox + dx, oy + dy, 0.0]
                          for dx in (-5.0, 5.0) for dy in (-5.0, 5.0)])
        centers.append(seeds)
        axes.append(np.tile([1.0, 0.0, 0.0], (4, 1)).astype(float))
        tiles.append(_Tile(seeds, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
    return np.vstack(centers), np.vstack(axes), tiles


def phantom_layout():
    """The phantom's ground-truth implant: wall-conformed, non-coplanar."""
    _vol, truth = make_head_phantom(spacing=0.7)
    centers = np.array([s.center_ras for s in truth.seeds])
    axes = np.array([s.axis_ras for s in truth.seeds])

    tiles = []
    for t in truth.tiles:
        ids = list(t.seed_ids)
        seeds = np.array([truth.seeds[i].center_ras for i in ids])
        # In-plane axis: the longest seed-to-seed direction within the tile,
        # projected off the tile normal.
        axis = seeds[-1] - seeds[0] if len(seeds) > 1 else axes[ids[0]]
        tiles.append(_Tile(seeds, t.normal_ras, axis, kind=t.kind))
    return centers, axes, tiles


# --------------------------------------------------------------------- study
def study(name, centers, axes, tiles, writer, rows):
    bounds = np.vstack([centers.min(axis=0) - PAD_MM,
                        centers.max(axis=0) + PAD_MM])

    t0 = time.perf_counter()
    free = compute_dose_grid(centers, axes, bounds, spacing_mm=GRID_MM)
    t_free = time.perf_counter() - t0

    caps = InterferenceModel.from_implant(centers, axes)
    t0 = time.perf_counter()
    corr = compute_dose_grid(centers, axes, bounds, spacing_mm=GRID_MM,
                             interference=caps)
    t_caps = time.perf_counter() - t0

    print("\n%s: %d seeds, %d tiles, grid %s, %.2f s -> %.2f s (+%.0f%%)"
          % (name, len(centers), len(tiles), free.array.shape,
             t_free, t_caps, 100.0 * (t_caps / t_free - 1.0)))
    print("  %-34s %9s %9s %9s %9s"
          % ("model", "mean %", "p05", "min", "n_vox"))

    for level, label in ((None, "all > 0"),
                         (0.25 * RX_CGY, ">= 25% rx"),
                         (RX_CGY, ">= 100% rx")):
        rep = interference_report(free, corr, level_cgy=level)
        print("  capsules only, %-19s %+8.2f%% %9.4f %9.4f %9d"
              % (label, rep["mean_percent_change"], rep["p05_ratio"],
                 rep["min_ratio"], rep["n_voxels"]))
        writer.writerow([name, "capsules", "", label,
                         "%.4f" % rep["mean_percent_change"],
                         "%.4f" % rep["p05_ratio"], "%.4f" % rep["min_ratio"],
                         rep["n_voxels"]])

    # Carrier sweep: the whole point is that this term is unpinned.
    sweep = []
    for rho in CARRIER_DENSITIES:
        model = InterferenceModel.from_implant(
            centers, axes, tiles=tiles, include_carriers=True,
            carrier_density_g_cm3=rho)
        both = compute_dose_grid(centers, axes, bounds, spacing_mm=GRID_MM,
                                 interference=model)
        rep = interference_report(free, both, level_cgy=0.25 * RX_CGY)
        sweep.append((rho, rep["mean_percent_change"]))
        print("  + carriers at rho=%.2f, >= 25%% rx      %+8.2f%% %9.4f %9.4f %9d"
              % (rho, rep["mean_percent_change"], rep["p05_ratio"],
                 rep["min_ratio"], rep["n_voxels"]))
        writer.writerow([name, "capsules+carriers", "%.2f" % rho, ">= 25% rx",
                         "%.4f" % rep["mean_percent_change"],
                         "%.4f" % rep["p05_ratio"], "%.4f" % rep["min_ratio"],
                         rep["n_voxels"]])

    caps_only = interference_report(free, corr, level_cgy=0.25 * RX_CGY)
    rows.append((name, caps_only["mean_percent_change"], sweep))
    return rows


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "validation_interference.csv")
    png_path = os.path.join(out_dir, "validation_interference.png")

    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["layout", "model", "carrier_density", "dose_level",
                         "mean_percent_change", "p05_ratio", "min_ratio",
                         "n_voxels"])
        c, a, t = flat_layout()
        study("flat 3-tile grid", c, a, t, writer, rows)
        c, a, t = phantom_layout()
        study("phantom truth implant", c, a, t, writer, rows)

    # ---- figure: the carrier density is the whole uncertainty
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for name, caps_pct, sweep in rows:
        rho = [r for r, _ in sweep]
        pct = [p for _, p in sweep]
        line, = ax.plot(rho, pct, "o-", label="%s, + carriers" % name)
        ax.axhline(caps_pct, ls="--", lw=1.0, color=line.get_color(),
                   label="%s, capsules only" % name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("assumed collagen carrier density (g/cm$^3$)")
    ax.set_ylabel("mean dose change at $\\geq$ 25% rx (%)")
    ax.set_title("Tile interference: the carrier density dominates the answer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)

    print("\nwrote %s" % csv_path)
    print("wrote %s" % png_path)


if __name__ == "__main__":
    main()
