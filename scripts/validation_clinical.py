"""Clinical validation: gtcore vs the treatment-planning system on one case.

Inputs (DICOM): the post-implant CT series, the RTSTRUCT exported from the
TPS (physician contours: ``Cav_Post`` resection cavity, ``CTV_GT`` the
GammaTile clinical target, ``CTV_Post``, ``Seeds``) and the RTDOSE grid of
the clinical plan.  The script

1. runs the gtcore pipeline on the CT (seed detection, segmentation),
2. rasterises the TPS contours onto the CT grid,
3. compares the gtcore cavity with ``Cav_Post`` and the gtcore HR-CTV
   (``gtcore.dose.hrctv``) with ``CTV_GT`` / ``CTV_Post`` (Dice, volumes),
4. matches detected seeds to the TPS ``Seeds`` contours,
5. evaluates D90 / D50 / Dmin / V100 / V150 on every structure from the
   clinical RTDOSE and from the gtcore TG-43 engine, and the TG-43 / RTDOSE
   dose ratio away from the seeds (which exposes the seed strength the TPS
   actually used).

Usage::

    python scripts/validation_clinical.py <ct_dir> <rtstruct.dcm|dir> <rtdose.dcm|dir> [--rx 6000] [--sk 3.5]

Writes nothing; prints the tables that docs/clinical-validation-notes.md
quotes.  Needs pydicom and scikit-image (both in the project environment).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtcore.dose import dose_at_points                      # noqa: E402
from gtcore.dose.hrctv import build_hrctv, hrctv_stats     # noqa: E402
from gtcore.io import load_volume                           # noqa: E402
from gtcore.pipeline import reconstruct                     # noqa: E402
from gtcore.volume import Volume                            # noqa: E402


def _dcm(path):
    import pydicom
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "*.dcm")))[0]
    return pydicom.dcmread(path)


def rasterise(rs, roi_name, vol):
    """Fill one RTSTRUCT ROI's planar contours onto the CT grid (bool [k,j,i]).
    DICOM contour points are LPS; gtcore is RAS (x, y negated)."""
    from skimage.draw import polygon
    names = {r.ROINumber: r.ROIName for r in rs.StructureSetROISequence}
    nk, nj, ni = vol.array.shape
    m = np.zeros(vol.array.shape, dtype=bool)
    for c in rs.ROIContourSequence:
        if names[c.ReferencedROINumber] != roi_name:
            continue
        for cs in getattr(c, "ContourSequence", []):
            p = np.array(cs.ContourData, dtype=float).reshape(-1, 3)
            p[:, :2] *= -1.0
            ijk = vol.ras_to_index(p)
            k = int(np.rint(ijk[:, 2].mean()))
            if 0 <= k < nk:
                rr, cc = polygon(ijk[:, 1], ijk[:, 0], (nj, ni))
                m[k, rr, cc] ^= True        # XOR: inner contours are holes
    return m


def rtdose_volume(rd):
    """RTDOSE as a gtcore Volume in cGy, RAS affine."""
    dose = rd.pixel_array.astype(np.float64) * float(rd.DoseGridScaling)
    if str(rd.DoseUnits).upper() == "GY":
        dose *= 100.0
    ipp = np.array(rd.ImagePositionPatient, dtype=float)
    ps = [float(x) for x in rd.PixelSpacing]
    gz = np.array(rd.GridFrameOffsetVector, dtype=float)
    A = np.eye(4)
    A[0, 0] = ps[1]
    A[1, 1] = ps[0]
    A[2, 2] = gz[1] - gz[0]
    A[:3, 3] = ipp
    A = np.diag([-1.0, -1.0, 1.0, 1.0]) @ A     # LPS -> RAS
    return Volume(dose, A, {"units": "cGy", "kind": "rtdose"})


def dice(a, b):
    return 2.0 * (a & b).sum() / max(1, a.sum() + b.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ct")
    ap.add_argument("rtstruct")
    ap.add_argument("rtdose")
    ap.add_argument("--rx", type=float, default=6000.0, help="prescription [cGy]")
    ap.add_argument("--sk", type=float, default=3.5, help="assumed S_K per seed [U]")
    args = ap.parse_args(argv)

    vol = load_volume(args.ct)
    res = reconstruct(vol, verbose=False)
    arr = vol.array
    vmm3 = float(np.prod(vol.spacing))
    rs = _dcm(args.rtstruct)
    rd = _dcm(args.rtdose)
    dv = rtdose_volume(rd)

    S = {n: rasterise(rs, n, vol) for n in ("Cav_Post", "CTV_GT", "CTV_Post", "Seeds")}
    print("TPS structures:")
    for n, m in S.items():
        print("  %-9s %7.1f cc" % (n, m.sum() * vmm3 / 1000.0))

    cav = res.cavity_mask
    print("gtcore cavity %.1f cc: Dice vs Cav_Post %.3f, covers %.0f%% of it, %.0f%% of ours inside it"
          % (cav.sum() * vmm3 / 1000.0, dice(cav, S["Cav_Post"]),
             100.0 * (cav & S["Cav_Post"]).sum() / max(1, S["Cav_Post"].sum()),
             100.0 * (cav & S["Cav_Post"]).sum() / max(1, cav.sum())))

    pts_all = vol.index_to_ras(np.stack(
        [c.ravel() for c in np.indices(arr.shape)[::-1]], axis=1).astype(float))
    h = build_hrctv(res.meshes["cavity"], inside_mask=cav, inside_affine=vol.affine,
                    keep_mask=res.masks["cranial_interior"], keep_affine=vol.affine)
    hr = (np.asarray(h.volume.sample_ras(pts_all, order=0, fill=0.0)) > 0.5).reshape(arr.shape)
    print("gtcore HR-CTV %.1f cc: Dice vs CTV_GT %.3f, vs CTV_Post %.3f"
          % (hr.sum() * vmm3 / 1000.0, dice(hr, S["CTV_GT"]), dice(hr, S["CTV_Post"])))
    dist = ndimage.distance_transform_edt(~S["Cav_Post"], sampling=vol.spacing[::-1])
    rind = (dist <= 5.0) & ~S["Cav_Post"] & res.masks["cranial_interior"]
    print("5 mm rind of Cav_Post %.1f cc: Dice vs CTV_GT %.3f, vs CTV_Post %.3f"
          % (rind.sum() * vmm3 / 1000.0, dice(rind, S["CTV_GT"]), dice(rind, S["CTV_Post"])))

    lab, n = ndimage.label(S["Seeds"])
    cs = np.array(ndimage.center_of_mass(S["Seeds"], lab, range(1, n + 1)))[:, ::-1]
    cs_ras = vol.index_to_ras(cs)
    axes = []
    for i in range(1, n + 1):
        q = vol.index_to_ras(np.argwhere(lab == i)[:, ::-1].astype(float))
        q = q - q.mean(axis=0)
        _w, v = np.linalg.eigh(q.T @ q)
        axes.append(v[:, -1])
    axes = np.array(axes)
    ours = res.seeds.centers_ras
    dd, _ = cKDTree(cs_ras).query(ours)
    d2, _ = cKDTree(ours).query(cs_ras)
    print("seeds: TPS %d, detected %d; detected->nearest TPS seed median %.2f mm, p95 %.2f, max %.2f; "
          "TPS seeds with no detection within 2 mm: %d"
          % (n, len(ours), np.median(dd), np.percentile(dd, 95), dd.max(), int((d2 > 2.0).sum())))

    def stats(mask, d):
        s = hrctv_stats(d, args.rx, vmm3)
        return "D90 %6.0f  D50 %6.0f  Dmin %6.0f  V100 %3.0f%%  V150 %3.0f%%  (%.1f cc)" % (
            s["D90"], s["D50"], s["Dmin"], 100 * s["V100"], 100 * s["V150"], s["volume_cc"])

    def clin(mask):
        return dv.sample_ras(pts_all[np.flatnonzero(mask.ravel())], order=1, fill=0.0)

    def tg(mask, c, a, sk):
        return dose_at_points(c, a, pts_all[np.flatnonzero(mask.ravel())], sk_per_seed_u=sk)

    print("clinical RTDOSE (rx %.0f cGy):" % args.rx)
    for nme in ("CTV_GT", "CTV_Post", "Cav_Post"):
        print("  RTDOSE on %-12s %s" % (nme, stats(S[nme], clin(S[nme]))))
    print("  RTDOSE on %-12s %s" % ("gtcore HR-CTV", stats(hr, clin(hr))))
    print("  RTDOSE on %-12s %s" % ("rind(Cav_Post)", stats(rind, clin(rind))))

    p = pts_all[np.flatnonzero(S["CTV_GT"].ravel())][::5]
    dcl = dv.sample_ras(p, order=1, fill=0.0)
    dtg = dose_at_points(cs_ras, axes, p, sk_per_seed_u=args.sk)
    ds, _ = cKDTree(cs_ras).query(p)
    ok = (ds > 5.0) & (dcl > 100.0)
    r = dtg[ok] / dcl[ok]
    sk_fit = args.sk / float(np.median(r))
    print("TG-43 (%.2f U) / RTDOSE at CTV_GT voxels > 5 mm from a seed: median %.3f, p5 %.3f, p95 %.3f "
          "(n=%d) -> implied S_K %.2f U" % (args.sk, np.median(r), np.percentile(r, 5),
                                             np.percentile(r, 95), int(ok.sum()), sk_fit))
    print("gtcore TG-43:")
    for sk in (args.sk, sk_fit):
        print("  TPS seeds   %.2f U on CTV_GT        %s" % (sk, stats(S["CTV_GT"], tg(S["CTV_GT"], cs_ras, axes, sk))))
        print("  our seeds   %.2f U on CTV_GT        %s" % (sk, stats(S["CTV_GT"], tg(S["CTV_GT"], ours, res.seeds.axes_ras, sk))))
        print("  our seeds   %.2f U on gtcore HR-CTV %s" % (sk, stats(hr, tg(hr, ours, res.seeds.axes_ras, sk))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
