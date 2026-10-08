"""Stage 3 (merge/split repair) real-data diagnosis and checks.

Usage (from the repo root, PYTHONPATH=.)::

    python scripts/localization_mergesplit_realdata.py diagnose  [--merge on|off]
    python scripts/localization_mergesplit_realdata.py check SCAN [--merge on|off]

``diagnose`` reconstructs the PostOp CT (cached under
output/loc_mergesplit/cache/), runs the count-free tile inference the
planner uses (``fit_tiles_prior`` with an empty ``ImplantPrior``) and dumps
every in-implant unassigned candidate with its nearest neighbour: separation
(RAS and voxel axes), axes, volumes, peak HU, minimum HU along the joining
segment ("bridge HU"), interpolated-slice status and whether the two came
from one thresholded blob.

``check`` runs one real scan (postop, phantom8, tilefree, doe) through
``reconstruct`` and prints the numbers the stage gate needs.  Run the real
scans one at a time (each needs a few GB).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
from scipy import ndimage

SCANS = {
    "postop": r"C:\Users\jacob\OneDrive\Documents\PostOp CT",
    "phantom8": r"C:\Users\jacob\OneDrive\Documents\3D-Printed Phantom-8tiles (223)",
    "tilefree": r"C:\Users\jacob\OneDrive\Documents\CT 3D printed",
    "doe": r"C:\Users\jacob\OneDrive\Documents\DOEJOHNPOSTCT",
}
CACHE = os.path.join("output", "loc_mergesplit", "cache")


def _merge_flag(s):
    return {"on": True, "off": False, "default": None}[s]


def load(scan):
    from gtcore.io import load_volume

    return load_volume(SCANS[scan])


def interpolated_slices(vol):
    """Bool per k: True when slice k was filled by interpolation.

    Uses ``vol.meta["interpolated_k"]`` when the loader records it, else
    re-derives it from the DICOM slice positions with the loader's rule.
    """
    nk = vol.array.shape[0]
    if "interpolated_k" in vol.meta:
        out = np.zeros(nk, bool)
        out[np.asarray(vol.meta["interpolated_k"], int)] = True
        return out
    if not vol.meta.get("z_gap_interpolated"):
        return np.zeros(nk, bool)
    import SimpleITK as sitk
    from gtcore.io.dicom import _slice_geometry

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(vol.meta["source"],
                                          vol.meta["series_id"])
    geo = _slice_geometry(files)
    z = geo["z"]
    keep = np.concatenate([[True], np.diff(z) > 1e-3])
    z = z[keep]
    dz = float(np.median(np.diff(z)))
    nz = int(round((z[-1] - z[0]) / dz)) + 1
    zt = z[0] + dz * np.arange(nz)
    out = np.zeros(nz, bool)
    for k, t in enumerate(zt):
        i1 = int(np.clip(np.searchsorted(z, t), 0, len(z) - 1))
        i0 = max(i1 - 1, 0)
        copied = i0 == i1 or min(abs(z[i1] - t), abs(z[i0] - t)) < 0.25 * dz
        out[k] = not copied
    assert nz == nk, (nz, nk)
    return out


def reconstruct_cached(scan, merge, legacy_split=False, **kw):
    """``legacy_split``: run detection with the pre-stage-3 split (median over
    all blobs, no halves guard) -- the baseline for before/after numbers."""
    import functools

    import gtcore.pipeline as pipeline

    tag = "%s_merge-%s%s%s" % (scan, {True: "on", False: "off", None: "default"}[merge],
                               "_legacy-split" if legacy_split else "",
                               "".join("_%s-%s" % kv for kv in sorted(kw.items())))
    path = os.path.join(CACHE, tag + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    vol = load(scan)
    t0 = time.perf_counter()
    kwargs = dict(kw)
    if merge is not None:
        kwargs["merge_fragments"] = merge
    orig = pipeline.detect_seed_candidates
    if legacy_split:
        pipeline.detect_seed_candidates = functools.partial(
            orig, split_window=False, split_guard=False, split_weighted=False)
    try:
        res = pipeline.reconstruct(vol, verbose=True, **kwargs)
    finally:
        pipeline.detect_seed_candidates = orig
    res.timings["total_wall"] = time.perf_counter() - t0
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return res


def suggest(res):
    """The planner's count-free suggest path."""
    from gtcore.tiles import fit_tiles_prior
    from gtcore.tiles.auto import ImplantPrior

    seeds = res.seeds
    mesh = res.meshes.get("cavity")
    cavity_center = None
    if mesh is not None and len(mesh.vertices):
        cavity_center = np.asarray(mesh.vertices, float).mean(axis=0)
    return fit_tiles_prior(seeds.centers_ras, seeds.axes_ras, ImplantPrior(),
                           cavity_center_ras=cavity_center, mesh=mesh,
                           spacing_mm=res.volume.spacing)


def bridge_hu(vol, a, b, n=41):
    t = np.linspace(0.0, 1.0, n)[:, None]
    return vol.sample_ras(a[None, :] * (1 - t) + b[None, :] * t)


def peak_hu(vol, c, r_mm=1.5):
    ijk = vol.ras_to_index(c)
    sp = vol.spacing
    rad = np.ceil(r_mm / sp).astype(int)
    lo = np.maximum(np.round(ijk).astype(int) - rad, 0)
    hi = np.minimum(np.round(ijk).astype(int) + rad + 1,
                    np.array(vol.shape_ijk))
    sub = vol.array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    return float(sub.max()) if sub.size else float("nan")


def blob_label_at(labels, vol, c):
    """Label of the thresholded blob nearest to RAS point ``c``."""
    ijk = np.round(vol.ras_to_index(c)).astype(int)
    k, j, i = ijk[2], ijk[1], ijk[0]
    lab = labels[k, j, i]
    if lab:
        return int(lab)
    sl = tuple(slice(max(x - 2, 0), x + 3) for x in (k, j, i))
    sub = labels[sl]
    nz = np.argwhere(sub > 0)
    if not len(nz):
        return 0
    ctr = np.array([min(k, 2), min(j, 2), min(i, 2)])
    best = nz[np.argmin(((nz - ctr) ** 2).sum(axis=1))]
    return int(sub[tuple(best)])


def local_background(vol, c, r_in=6.0, r_out=10.0, n=200):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(r_in, r_out, size=(n, 1))
    return float(np.median(vol.sample_ras(c[None, :] + r * d)))


def diagnose(args):
    from gtcore.pipeline import seed_detection_params
    from gtcore.seeds.detect import SEED_CONNECTIVITY

    res = reconstruct_cached("postop", _merge_flag(args.merge),
                             legacy_split=args.legacy_split)
    vol = res.volume
    seeds = res.seeds
    fit = suggest(res)
    print("\nsuggest: %s" % fit.summary())
    print("  supported tiles: %d  tentative: %d  unassigned: %d  clutter: %d"
          % (len(fit.tiles), len(fit.tentative_tiles),
             len(fit.unassigned_indices), len(fit.clutter_indices)))
    for p in fit.tiles:
        print("    supported T%d seeds %s" % (p.tile_id, list(p.seed_indices)))
    for p in fit.tentative_tiles:
        print("    tentative T%d seeds %s" % (p.tile_id, list(p.seed_indices)))
    params = seed_detection_params(vol.spacing)
    thr = params["hu_threshold"]
    hot = np.asarray(vol.array) > thr
    labels, _n = ndimage.label(
        hot, structure=ndimage.generate_binary_structure(3, SEED_CONNECTIVITY))
    interp = interpolated_slices(vol)
    print("spacing %s  thr %.0f HU  interpolated %d/%d slices"
          % (np.round(vol.spacing, 3), thr, int(interp.sum()), len(interp)))
    A = vol.affine[:3, :3]
    Ainv = np.linalg.inv(A)
    C = seeds.centers_ras
    src = None if seeds.info is None else seeds.info.get("source_blob")
    rows = []
    assigned = {i for p in fit.all_tiles for i in p.seed_indices}
    nrm = A[:, 2] / np.linalg.norm(A[:, 2])
    for u in fit.unassigned_indices:
        d = np.linalg.norm(C - C[u], axis=1)
        d[u] = np.inf
        v = int(np.argmin(d))
        sep = C[v] - C[u]
        sep_vox = Ainv @ sep
        ku = vol.ras_to_index(C[u])[2]
        kv = vol.ras_to_index(C[v])[2]
        br = bridge_hu(vol, C[u], C[v])
        pu, pv = peak_hu(vol, C[u]), peak_hu(vol, C[v])
        lu, lv = blob_label_at(labels, vol, C[u]), blob_label_at(labels, vol, C[v])
        uhat = sep / np.linalg.norm(sep)
        ang_u = np.degrees(np.arccos(min(1.0, abs(float(seeds.axes_ras[u] @ uhat)))))
        ang_v = np.degrees(np.arccos(min(1.0, abs(float(seeds.axes_ras[v] @ uhat)))))
        along = abs(float(sep @ nrm))
        inplane = float(np.linalg.norm(sep - (sep @ nrm) * nrm))
        bg = local_background(vol, 0.5 * (C[u] + C[v]))
        rows.append(dict(
            u=u, v=v, v_status=("tile" if v in assigned else
                                "unassigned" if v in fit.unassigned_indices
                                else "clutter"),
            dist=float(d[v]), sep=sep, sep_vox=sep_vox, along_n=along,
            inplane=inplane, ax_u=seeds.axes_ras[u], ax_v=seeds.axes_ras[v],
            ang_u=ang_u, ang_v=ang_v,
            vol_u=float(seeds.volumes_mm3[u]), vol_v=float(seeds.volumes_mm3[v]),
            elong_u=float(seeds.elongations[u]), elong_v=float(seeds.elongations[v]),
            peak_u=pu, peak_v=pv, bridge=float(br.min()), bg=bg,
            k_u=float(ku), k_v=float(kv),
            interp_u=bool(interp[int(round(ku))]),
            interp_v=bool(interp[int(round(kv))]),
            lab_u=lu, lab_v=lv,
            src_u=None if src is None else int(src[u]),
            src_v=None if src is None else int(src[v]),
        ))
    for r in rows:
        frac = (r["bridge"] - r["bg"]) / max(min(r["peak_u"], r["peak_v"]) - r["bg"], 1.0)
        print("\n#%d <-> #%d (%s)  |d|=%.2f mm"
              % (r["u"], r["v"], r["v_status"], r["dist"]))
        print("  sep RAS %s  sep vox(i,j,k) %s  along-normal %.2f  in-plane %.2f"
              % (np.round(r["sep"], 2), np.round(r["sep_vox"], 2), r["along_n"],
                 r["inplane"]))
        print("  axis u %s (%.0f deg to sep)  axis v %s (%.0f deg)"
              % (np.round(r["ax_u"], 2), r["ang_u"], np.round(r["ax_v"], 2),
                 r["ang_v"]))
        print("  vol %.2f / %.2f mm3  elong %.1f / %.1f  peak %.0f / %.0f HU  "
              "bridge min %.0f HU  bg %.0f HU  bridge frac %.2f"
              % (r["vol_u"], r["vol_v"], r["elong_u"], r["elong_v"], r["peak_u"],
                 r["peak_v"], r["bridge"], r["bg"], frac))
        print("  k %.2f / %.2f  interpolated %s / %s   blob label %d / %d  "
              "(same=%s)  source_blob %s / %s"
              % (r["k_u"], r["k_v"], r["interp_u"], r["interp_v"], r["lab_u"],
                 r["lab_v"], r["lab_u"] == r["lab_v"] and r["lab_u"] > 0,
                 r["src_u"], r["src_v"]))
    return res, fit, rows


def check(args):
    scan = args.scan
    kw = {}
    if scan == "phantom8":
        kw["n_full_tiles"] = "auto"
    res = reconstruct_cached(scan, _merge_flag(args.merge),
                             legacy_split=args.legacy_split, **kw)
    vol = res.volume
    print("\n[%s merge=%s%s] spacing %s  raw blobs %d -> seeds %d  implant: %s (%s)"
          % (scan, args.merge, " legacy-split" if args.legacy_split else "",
             np.round(vol.spacing, 3), len(res.seeds_raw),
             len(res.seeds), res.implant.get("verdict"), res.implant.get("reason")))
    sm = vol.meta.get("seed_merge", {})
    print("  seed_merge: enabled=%s (%s), %s pair(s) rejoined"
          % (sm.get("enabled"), sm.get("reason"), sm.get("n_merged")))
    info = res.seeds_raw.info or {}
    if "split_k" in info:
        print("  split blobs: %d candidates come from k>1 splits"
              % int((np.asarray(info["split_k"]) > 1).sum()))
    if res.tiles is not None:
        t = res.tiles
        print("  tiles: %d supported, %d tentative; %s"
              % (len(t.tiles), len(getattr(t, "tentative_tiles", [])), t.summary()))
    fit = suggest(res)
    print("  suggest: %d supported + %d tentative, %d unassigned, %d clutter; %s"
          % (len(fit.tiles), len(fit.tentative_tiles), len(fit.unassigned_indices),
             len(fit.clutter_indices), fit.summary()))
    for p in fit.tiles:
        print("    supported T%d seeds %s centre %s"
              % (p.tile_id, list(p.seed_indices), np.round(p.center_ras, 1)))
    print("  timings: %s" % {k: round(v, 2) for k, v in res.timings.items()})
    return res, fit


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--merge", default="default", choices=["on", "off", "default"])
    d.add_argument("--legacy-split", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("scan", choices=sorted(SCANS))
    c.add_argument("--merge", default="default", choices=["on", "off", "default"])
    c.add_argument("--legacy-split", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "diagnose":
        diagnose(args)
    else:
        check(args)


if __name__ == "__main__":
    sys.exit(main())
