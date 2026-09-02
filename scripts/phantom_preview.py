"""Visual sanity check for the synthetic head phantom.

Renders orthogonal slices through the resection-cavity centre in two windows
(soft tissue and metal) and prints the ground-truth seed table.

    python scripts/phantom_preview.py
"""
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtcore.phantom import make_head_phantom  # noqa: E402

SPACING = 0.8
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)
SOFT_WINDOW = (-100.0, 150.0)
METAL_WINDOW = (500.0, 4000.0)


def _panel(fig_title, soft, metal, extent, xlabel, ylabel, path):
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 9.6))
    for ax, img, win, tag in (
        (axes[0], soft, SOFT_WINDOW, "soft tissue [%g, %g] HU" % SOFT_WINDOW),
        (axes[1], metal, METAL_WINDOW, "metal [%g, %g] HU" % METAL_WINDOW),
    ):
        ax.imshow(
            img, cmap="gray", vmin=win[0], vmax=win[1], origin="lower",
            extent=extent, interpolation="nearest",
        )
        ax.set_title("%s -- %s" % (fig_title, tag), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote %s" % path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    vol, truth = make_head_phantom(spacing=SPACING, streaks=False)
    print(
        "generated %s at %.2f mm in %.2f s"
        % (vol.array.shape, SPACING, time.time() - t0)
    )

    c_ijk = np.round(vol.ras_to_index(truth.cavity_center_ras)).astype(int)
    ni, nj, nk = vol.shape_ijk
    i0 = int(np.clip(c_ijk[0], 0, ni - 1))
    j0 = int(np.clip(c_ijk[1], 0, nj - 1))
    k0 = int(np.clip(c_ijk[2], 0, nk - 1))
    print("cavity centre RAS %s -> voxel (i,j,k)=(%d,%d,%d)"
          % (np.round(truth.cavity_center_ras, 2), i0, j0, k0))

    lo = vol.index_to_ras([0, 0, 0])
    hi = vol.index_to_ras([ni - 1, nj - 1, nk - 1])
    ext_x = (lo[0], hi[0])
    ext_y = (lo[1], hi[1])
    ext_z = (lo[2], hi[2])

    arr = vol.array
    axial = arr[k0, :, :]           # rows = y (A), cols = x (R)
    coronal = arr[:, j0, :]         # rows = z (S), cols = x (R)
    sagittal = arr[:, :, i0]        # rows = z (S), cols = y (A)

    _panel("axial z=%.1f mm" % vol.index_to_ras([i0, j0, k0])[2], axial, axial,
           ext_x + ext_y, "x / R (mm)", "y / A (mm)",
           os.path.join(OUT_DIR, "phantom_axial.png"))
    _panel("coronal y=%.1f mm" % vol.index_to_ras([i0, j0, k0])[1], coronal,
           coronal, ext_x + ext_z, "x / R (mm)", "z / S (mm)",
           os.path.join(OUT_DIR, "phantom_coronal.png"))
    _panel("sagittal x=%.1f mm" % vol.index_to_ras([i0, j0, k0])[0], sagittal,
           sagittal, ext_y + ext_z, "y / A (mm)", "z / S (mm)",
           os.path.join(OUT_DIR, "phantom_sagittal.png"))

    print("")
    print("ground-truth seeds (%d seeds / %d tiles)"
          % (len(truth.seeds), len(truth.tiles)))
    print("%4s %5s %28s %28s %9s"
          % ("id", "tile", "center RAS (mm)", "axis RAS", "HU@center"))
    for s in truth.seeds:
        print(
            "%4d %5d  %8.2f %8.2f %8.2f   %8.3f %8.3f %8.3f  %8.0f"
            % (
                s.seed_id, s.tile_id,
                s.center_ras[0], s.center_ras[1], s.center_ras[2],
                s.axis_ras[0], s.axis_ras[1], s.axis_ras[2],
                vol.sample_ras(s.center_ras),
            )
        )
    print("")
    print("%4s %28s %28s" % ("tile", "center RAS (mm)", "normal RAS"))
    for t in truth.tiles:
        print(
            "%4d  %8.2f %8.2f %8.2f   %8.3f %8.3f %8.3f"
            % (
                t.tile_id,
                t.center_ras[0], t.center_ras[1], t.center_ras[2],
                t.normal_ras[0], t.normal_ras[1], t.normal_ras[2],
            )
        )
    print("")
    for name, mask in sorted(truth.masks.items()):
        print("mask %-7s %10d voxels" % (name, int(mask.sum())))


if __name__ == "__main__":
    main()
