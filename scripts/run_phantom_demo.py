"""End-to-end demo: phantom -> reconstruction -> seed localization -> meshes.

Produces, under output/:
  phantom_ct.nrrd            the synthetic post-implant CT
  brain.ply / cavity.ply / skull.ply   reconstructed surfaces (RAS mm)
  seeds_detected.csv         detected seed coordinates vs ground truth
  overview_*.png             orthogonal slices with truth (green) and
                             detected (red x) seed positions

Run:  python scripts/run_phantom_demo.py [--spacing 0.7] [--tiles 3]
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gtcore.io import save_volume
from gtcore.phantom import make_head_phantom
from gtcore.preprocess import inpaint_metal
from gtcore.seeds import detect_seed_candidates
from gtcore.segment import mask_to_mesh, segment_cavity, segment_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.7)
    ap.add_argument("--tiles", type=int, default=3)
    ap.add_argument("--streaks", action="store_true")
    args = ap.parse_args()

    out = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out, exist_ok=True)

    def stage(name, fn):
        t0 = time.perf_counter()
        result = fn()
        print("%-28s %6.2f s" % (name, time.perf_counter() - t0))
        return result

    vol, truth = stage(
        "phantom generation",
        lambda: make_head_phantom(spacing=args.spacing, n_tiles=args.tiles, streaks=args.streaks),
    )
    stage("save CT (nrrd)", lambda: save_volume(vol, os.path.join(out, "phantom_ct.nrrd")))
    cands = stage("seed detection", lambda: detect_seed_candidates(vol))
    clean = stage("metal inpainting", lambda: inpaint_metal(vol, cands.mask))
    masks = stage("head segmentation", lambda: segment_head(clean, metal_mask=cands.mask))
    cavity = stage(
        "cavity segmentation",
        lambda: segment_cavity(clean, masks["cranial_interior"], masks["brain"], cands.centers_ras),
    )
    meshes = stage(
        "surface meshes",
        lambda: {
            "brain": mask_to_mesh(masks["brain"], vol.affine),
            "skull": mask_to_mesh(masks["skull"], vol.affine),
            "cavity": mask_to_mesh(cavity, vol.affine),
        },
    )
    for name, mesh in meshes.items():
        mesh.export(os.path.join(out, name + ".ply"))

    # ---- seed accuracy table ---------------------------------------------
    from scipy.spatial.distance import cdist

    t_centers = np.array([s.center_ras for s in truth.seeds])
    D = cdist(t_centers, cands.centers_ras)
    rows, used = [], set()
    for ti, s in enumerate(truth.seeds):
        order = [j for j in np.argsort(D[ti]) if int(j) not in used]
        j = int(order[0]) if order else -1
        if j >= 0:
            used.add(j)
            err = D[ti, j]
            est = cands.centers_ras[j]
        else:
            err, est = np.nan, [np.nan] * 3
        rows.append([s.seed_id, s.tile_id] + list(s.center_ras) + list(est) + [err])
    with open(os.path.join(out, "seeds_detected.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_id", "tile_id", "tx", "ty", "tz", "ex", "ey", "ez", "err_mm"])
        w.writerows(rows)
    errs = np.array([r[-1] for r in rows], dtype=float)
    print("\nseeds: %d truth / %d detected | centroid error mean %.2f mm, max %.2f mm"
          % (len(t_centers), len(cands), np.nanmean(errs), np.nanmax(errs)))

    # ---- overview figures -------------------------------------------------
    cav_ijk = np.round(vol.ras_to_index(truth.cavity_center_ras)).astype(int)
    views = [
        ("axial", vol.array[cav_ijk[2]], (0, 1)),        # [j, i] -> x/y
        ("coronal", vol.array[:, cav_ijk[1]], (0, 2)),   # [k, i] -> x/z
        ("sagittal", vol.array[:, :, cav_ijk[0]], (1, 2)),  # [k, j] -> y/z
    ]
    sp = vol.spacing
    org = vol.origin_ras
    for name, sl, (ax_a, ax_b) in views:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        for ax, (vmin, vmax), title in zip(
            axes, [(-100, 150), (500, 4000)], ["soft tissue window", "metal window"]
        ):
            ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ta = (t_centers[:, ax_a] - org[ax_a]) / sp[ax_a]
            tb = (t_centers[:, ax_b] - org[ax_b]) / sp[ax_b]
            ea = (cands.centers_ras[:, ax_a] - org[ax_a]) / sp[ax_a]
            eb = (cands.centers_ras[:, ax_b] - org[ax_b]) / sp[ax_b]
            ax.plot(ta, tb, "o", mfc="none", mec="lime", ms=10, label="truth")
            ax.plot(ea, eb, "rx", ms=7, label="detected")
            ax.set_title("%s — %s" % (name, title))
            ax.legend(loc="lower right", fontsize=8)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "overview_%s.png" % name), dpi=140)
        plt.close(fig)
    print("outputs written to", os.path.abspath(out))


if __name__ == "__main__":
    main()
