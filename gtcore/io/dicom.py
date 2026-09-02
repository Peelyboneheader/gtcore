"""DICOM / medical-image I/O for gtcore.

Everything crossing this boundary is converted **once**: SimpleITK and DICOM
speak LPS, ``gtcore`` speaks RAS. The flip matrix ``diag(-1, -1, 1, 1)`` is
applied here and nowhere else.

Array layout is SimpleITK's native ``GetArrayFromImage`` order, which is
already ``[k, j, i]`` -- the same order :class:`gtcore.volume.Volume` wants.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import SimpleITK as sitk

from ..volume import Volume

__all__ = [
    "volume_from_sitk",
    "sitk_from_volume",
    "load_dicom_series",
    "load_volume",
    "save_volume",
]

# LPS <-> RAS.  Self-inverse, so the same matrix converts both ways.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# Cs-131 seeds are 4.5 mm x 0.8 mm; anything coarser than this cannot resolve
# them and localization becomes guesswork.
_MAX_USABLE_SPACING_MM = 2.5

_VOLUME_SUFFIXES = (".nrrd", ".nhdr", ".nii", ".nii.gz", ".mha", ".mhd")


def volume_from_sitk(img):
    """Convert a ``SimpleITK.Image`` to a :class:`Volume` with an RAS affine."""
    array = sitk.GetArrayFromImage(img)  # already [k, j, i]
    if array.ndim != 3:
        raise ValueError(
            "expected a 3D image, got array shape %s" % (array.shape,)
        )

    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.array(img.GetSpacing(), dtype=float)
    origin = np.array(img.GetOrigin(), dtype=float)

    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin
    affine_ras = _LPS_TO_RAS @ affine_lps

    return Volume(array, affine_ras, {})


def sitk_from_volume(vol):
    """Convert a :class:`Volume` back to a ``SimpleITK.Image`` (LPS geometry)."""
    affine_lps = _LPS_TO_RAS @ np.asarray(vol.affine, dtype=float)

    linear = affine_lps[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0):
        raise ValueError("degenerate affine: zero spacing along some axis")
    direction = linear / spacing

    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.array))
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in affine_lps[:3, 3]))
    img.SetDirection(tuple(float(d) for d in direction.reshape(-1)))
    return img


def _series_meta(first_file):
    """Pull a few descriptive tags without ever touching the pixel data."""
    meta = {}
    try:
        import pydicom
    except ImportError:  # pragma: no cover - pydicom is a hard dependency
        return meta

    try:
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
    except Exception as exc:  # pragma: no cover - unreadable header
        warnings.warn("could not read DICOM header %s: %s" % (first_file, exc))
        return meta

    for key, tag in (
        ("modality", "Modality"),
        ("kvp", "KVP"),
        ("manufacturer", "Manufacturer"),
        ("slice_thickness", "SliceThickness"),
        ("series_description", "SeriesDescription"),
        ("convolution_kernel", "ConvolutionKernel"),
    ):
        value = getattr(ds, tag, None)
        if value is None:
            continue
        try:
            if key in ("kvp", "slice_thickness"):
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            value = str(value)
        meta[key] = value
    return meta


def _slice_geometry(files):
    """Per-file z position along the slice normal, from pydicom headers.

    Returns None when the headers lack positioning tags (then the SimpleITK
    path is used unchanged).
    """
    try:
        import pydicom
    except ImportError:  # pragma: no cover
        return None
    zs, ipps, iop, ps = [], [], None, None
    try:
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is None or getattr(ds, "ImageOrientationPatient", None) is None:
                return None
            if iop is None:
                iop = np.array([float(v) for v in ds.ImageOrientationPatient])
                ps = np.array([float(v) for v in ds.PixelSpacing])
            n = np.cross(iop[:3], iop[3:])
            p = np.array([float(v) for v in ipp])
            ipps.append(p)
            zs.append(float(np.dot(p, n)))
    except Exception:
        return None
    order = np.argsort(zs)
    return {
        "z": np.asarray(zs)[order],
        "files": [files[i] for i in order],
        "iop": iop,
        "ps": ps,
        "ipp": np.asarray(ipps)[order],
    }


def _z_nonuniform(z, tol_frac=0.15):
    if len(z) < 3:
        return False
    dz = np.diff(z)
    med = float(np.median(dz))
    if med <= 1e-6:
        return False
    return bool(np.any(np.abs(dz - med) > tol_frac * med))


def _sheared(geo, tol_mm=0.5):
    """True when slice origins drift IN-PLANE along the stack (tilt/shear).

    SimpleITK reduces the stack to (origin, orthogonal direction, scalar
    spacing) and sets the spacing to the 3D step NORM, so a sheared stack is
    silently both stretched along z and displaced in-plane -- millimetre-scale
    georeferencing error, fatal for seed localization. Detected here so such
    series go through the rebuild path, whose k-axis is the true step vector.
    """
    ipp = geo["ipp"]
    if len(ipp) < 2:
        return False
    n = _unit_normal(geo["iop"])
    rel = ipp - ipp[0]
    inplane = rel - np.outer(rel @ n, n)
    return bool(np.max(np.linalg.norm(inplane, axis=1)) > tol_mm)


def _unit_normal(iop):
    n = np.cross(iop[:3], iop[3:])
    return n / np.linalg.norm(n)


def _load_series_resampled(geo):
    """Rebuild a series with missing/irregular slices onto its TRUE geometry.

    SimpleITK spreads N slices uniformly over the scanned extent, which
    silently distorts z by up to the largest gap -- fatal for millimetre seed
    localization. Here every present slice is placed at its actual
    ImagePositionPatient and the gaps are filled by linear interpolation
    between the bracketing slices, with the interpolated fraction reported in
    ``meta`` and via a warning.
    """
    import pydicom

    z, files = geo["z"], geo["files"]
    keep = np.concatenate([[True], np.diff(z) > 1e-3])  # drop duplicate positions
    z = z[keep]
    files = [f for f, k in zip(files, keep) if k]
    ipp = geo["ipp"][keep]

    if len(z) < 2:
        return None  # jittered duplicates collapsed to one slice -> sitk path
    dz = float(np.median(np.diff(z)))
    if not np.isfinite(dz) or dz <= 0:
        return None
    stack = []
    try:
        for f in files:
            ds = pydicom.dcmread(f)
            a = ds.pixel_array.astype(np.float32)
            a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(
                getattr(ds, "RescaleIntercept", 0.0)
            )
            stack.append(a)
    except Exception as exc:  # e.g. compressed syntax pydicom cannot decode
        warnings.warn("slice rebuild fell back to SimpleITK (pixel decode: %s)"
                      % exc)
        return None
    stack = np.stack(stack)  # [n_present, rows, cols]

    nz = int(round((z[-1] - z[0]) / dz)) + 1
    zt = z[0] + dz * np.arange(nz)
    out = np.empty((nz,) + stack.shape[1:], dtype=np.float32)
    n_copied = 0
    for k, t in enumerate(zt):
        i1 = int(np.clip(np.searchsorted(z, t), 0, len(z) - 1))
        i0 = max(i1 - 1, 0)
        if i0 == i1 or min(abs(z[i1] - t), abs(z[i0] - t)) < 0.25 * dz:
            src = i1 if abs(z[i1] - t) <= abs(z[i0] - t) else i0
            out[k] = stack[src]
            n_copied += 1
        else:
            w = (t - z[i0]) / (z[i1] - z[i0])
            out[k] = (1.0 - w) * stack[i0] + w * stack[i1]

    iop, ps = geo["iop"], geo["ps"]
    row_dir, col_dir = iop[:3], iop[3:]
    normal = _unit_normal(iop)
    # k axis = TRUE mean per-slice step vector: for a tilted/sheared stack the
    # slice origins drift in-plane, and using normal*dz would drop that drift
    # (millimetre-scale error). For a straight stack this equals normal*dz.
    step = (ipp[-1] - ipp[0]) / (nz - 1) if nz > 1 else normal * dz
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_dir * ps[1]  # i (column index) steps along the row
    affine_lps[:3, 1] = col_dir * ps[0]  # j (row index) steps down the column
    affine_lps[:3, 2] = step
    affine_lps[:3, 3] = ipp[0]

    vol = Volume(out, _LPS_TO_RAS @ affine_lps)
    n_interp = nz - n_copied
    vol.meta.update({
        "z_gap_interpolated": True,
        "slices_present": int(len(z)),
        "slices_on_grid": int(nz),
        "slices_interpolated": int(n_interp),
    })
    if n_interp > 0.5 * nz:
        vol.meta["z_gap_unreliable"] = True
        warnings.warn(
            "UNRELIABLE volume: %d of %d slices are interpolated -- most of "
            "this anatomy is invented; do not trust seed localization from "
            "this series" % (n_interp, nz)
        )
    warnings.warn(
        "non-uniform slice positions: %d slices present, rebuilt onto a %d-slice "
        "grid at %.2f mm (%d slices linearly interpolated). Check the export -- "
        "the source series may be incompletely copied/synced."
        % (len(z), nz, dz, n_interp)
    )
    return vol


def load_dicom_series(directory, series_id=None):
    """Load one DICOM series from ``directory`` as an RAS :class:`Volume`.

    If ``series_id`` is omitted the series with the most slices wins, which is
    almost always the axial reconstruction rather than a scout or a dose
    report. Series with missing/irregular slices are rebuilt onto their true
    geometry (see :func:`_load_series_resampled`).
    """
    directory = str(directory)
    if not os.path.isdir(directory):
        raise NotADirectoryError("not a directory: %s" % directory)

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise FileNotFoundError("no DICOM series found in %s" % directory)

    if series_id is not None:
        if series_id not in series_ids:
            raise ValueError(
                "series %r not in %s (available: %s)"
                % (series_id, directory, list(series_ids))
            )
        chosen = series_id
        files = reader.GetGDCMSeriesFileNames(directory, chosen)
    else:
        chosen, files = None, []
        for sid in series_ids:
            candidate = reader.GetGDCMSeriesFileNames(directory, sid)
            if len(candidate) > len(files):
                chosen, files = sid, candidate

    if not files:
        raise FileNotFoundError(
            "series %r in %s contains no files" % (chosen, directory)
        )

    geo = _slice_geometry(files)
    vol = None
    if geo is not None and (_z_nonuniform(geo["z"]) or _sheared(geo)):
        vol = _load_series_resampled(geo)  # may fall back with None
    if vol is None:
        reader.SetFileNames(files)
        img = reader.Execute()
        vol = volume_from_sitk(img)

    vol.meta.update(_series_meta(files[0]))
    vol.meta["series_id"] = chosen
    vol.meta["num_slices"] = len(files)
    vol.meta["source"] = directory
    vol.meta["available_series"] = list(series_ids)

    modality = str(vol.meta.get("modality", "")).upper()
    if modality and modality != "CT":
        warnings.warn(
            "expected a CT series, got modality %r -- HU-based seed "
            "localization will not be meaningful" % modality
        )

    max_spacing = float(np.max(vol.spacing))
    if max_spacing > _MAX_USABLE_SPACING_MM:
        warnings.warn(
            "max voxel spacing %.2f mm exceeds %.1f mm; Cs-131 seeds are only "
            "4.5 x 0.8 mm, so individual seeds may be unresolvable"
            % (max_spacing, _MAX_USABLE_SPACING_MM)
        )
    return vol


def load_volume(path):
    """Load a volume from a DICOM directory or a single image file."""
    path = str(path)
    if os.path.isdir(path):
        return load_dicom_series(path)
    if not os.path.exists(path):
        raise FileNotFoundError("no such file or directory: %s" % path)

    img = sitk.ReadImage(path)
    vol = volume_from_sitk(img)
    vol.meta["source"] = path
    return vol


def save_volume(vol, path):
    """Write ``vol`` to ``path`` (format inferred from the extension)."""
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    sitk.WriteImage(sitk_from_volume(vol), path, useCompression=True)
    return path
