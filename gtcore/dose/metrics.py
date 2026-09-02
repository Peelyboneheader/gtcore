"""Clinical dose metrics on top of the TG-43 engine.

GammaTile is prescribed as a dose at depth: 60 Gy to a point 5 mm from the
tile (cavity-wall) surface. The quantities a planner needs to report are
therefore (a) the dose reaching a *rind* of tissue around the cavity, as a
DVH with D90 / V100 / V150 / V200, and (b) the dose *on the wall at depth*,
as an area-weighted coverage fraction. Both are pure numpy over the
``Volume`` / ``trimesh`` objects the rest of the pipeline already produces.

Conventions: dose in cGy, distances in mm, ``rx_cgy`` the prescription.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import ndimage

from ..volume import Volume
from .engine import TG43Engine, dose_at_points

__all__ = ["DVH", "dvh", "dose_metrics", "resample_mask_to", "rind_mask",
           "wall_dose", "surface_coverage"]


# ------------------------------------------------------------------------ DVH
@dataclass
class DVH:
    """Cumulative dose-volume histogram of one structure.

    Stores the sorted voxel doses rather than a binned curve, so ``D`` and
    ``V`` are exact order statistics and the binned ``curve`` is derived
    on demand at whatever resolution the caller wants.
    """

    doses_sorted: np.ndarray        # ascending, cGy, one entry per voxel
    voxel_volume_mm3: float

    @property
    def n_voxels(self) -> int:
        return int(self.doses_sorted.size)

    @property
    def volume_cc(self) -> float:
        return self.n_voxels * self.voxel_volume_mm3 / 1000.0

    def D(self, percent):
        """Dose [cGy] received by at least ``percent`` % of the volume.

        D90 is the classic coverage statistic; D100 is the minimum dose.
        """
        if self.n_voxels == 0:
            return float("nan")
        p = np.asarray(percent, dtype=float)
        if np.any(p < 0.0) or np.any(p > 100.0):
            raise ValueError("percent must be within [0, 100]")
        out = np.quantile(self.doses_sorted, 1.0 - p / 100.0,
                          method="inverted_cdf")
        return float(out) if out.ndim == 0 else out

    def V(self, dose_cgy, relative=True):
        """Volume receiving at least ``dose_cgy``: fraction (or cc)."""
        d = np.asarray(dose_cgy, dtype=float)
        n_ge = self.n_voxels - np.searchsorted(self.doses_sorted, d,
                                               side="left")
        if relative:
            out = n_ge / self.n_voxels if self.n_voxels else np.full_like(
                d, np.nan, dtype=float)
        else:
            out = n_ge * self.voxel_volume_mm3 / 1000.0
        return float(out) if np.ndim(out) == 0 else out

    def curve(self, n_bins=256, max_dose_cgy=None):
        """Binned cumulative curve: ``(dose_axis, fraction >= dose)``."""
        top = float(max_dose_cgy) if max_dose_cgy is not None else (
            float(self.doses_sorted[-1]) if self.n_voxels else 1.0)
        axis = np.linspace(0.0, top, int(n_bins))
        return axis, self.V(axis)


def dvh(dose_volume: Volume, mask=None) -> DVH:
    """DVH of ``dose_volume`` inside ``mask`` (bool ``[k, j, i]`` on the
    dose grid; ``None`` = the whole grid)."""
    arr = np.asarray(dose_volume.array, dtype=float)
    if mask is None:
        vals = arr.ravel()
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != arr.shape:
            raise ValueError("mask shape %s != dose grid shape %s"
                             % (m.shape, arr.shape))
        vals = arr[m]
    return DVH(np.sort(vals), float(np.prod(dose_volume.spacing)))


def dose_metrics(dose_volume: Volume, mask, rx_cgy: float) -> Dict[str, float]:
    """Standard permanent-implant statistics for one structure.

    Returns D90/D100/D50/Dmean/Dmax [cGy], V100/V150/V200 (fractions of the
    structure at >= 100/150/200 % of ``rx_cgy``) and the structure volume.
    """
    h = dvh(dose_volume, mask)
    rx = float(rx_cgy)
    if h.n_voxels == 0:
        nan = float("nan")
        return {"volume_cc": 0.0, "D90": nan, "D100": nan, "D50": nan,
                "Dmean": nan, "Dmax": nan, "V100": nan, "V150": nan,
                "V200": nan, "rx_cgy": rx}
    return {
        "volume_cc": h.volume_cc,
        "D90": h.D(90.0),
        "D100": h.D(100.0),
        "D50": h.D(50.0),
        "Dmean": float(h.doses_sorted.mean()),
        "Dmax": float(h.doses_sorted[-1]),
        "V100": h.V(rx),
        "V150": h.V(1.5 * rx),
        "V200": h.V(2.0 * rx),
        "rx_cgy": rx,
    }


# ------------------------------------------------------------- structures
def resample_mask_to(dose_volume: Volume, mask, affine):
    """Nearest-neighbour resample of a bool mask (``[k, j, i]`` with its own
    ``affine``) onto the dose grid. Returns bool ``[k, j, i]`` on the grid."""
    src = Volume(np.asarray(mask, dtype=np.uint8), np.asarray(affine, float))
    nk, nj, ni = dose_volume.array.shape
    kk, jj, ii = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni),
                             indexing="ij")
    ijk = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(float)
    pts = dose_volume.index_to_ras(ijk)
    vals = src.sample_ras(pts, order=0, fill=0.0)
    return (np.asarray(vals) > 0.5).reshape(nk, nj, ni)


def rind_mask(dose_volume: Volume, structure_mask, structure_affine,
              depth_mm=5.0, exclude_mask=None):
    """Shell of tissue within ``depth_mm`` outside a structure, on the dose
    grid -- the GammaTile target volume for a resection cavity.

    Parameters
    ----------
    dose_volume : Volume
        Defines the output grid.
    structure_mask, structure_affine
        The cavity (or any structure) mask in its own grid.
    depth_mm : float
        Rind thickness measured from the structure surface (Euclidean
        distance in physical mm, so anisotropic grids are handled).
    exclude_mask : bool array on the dose grid, optional
        Voxels to drop from the rind, e.g. skull or air outside the head.
    """
    if depth_mm <= 0.0:
        raise ValueError("depth_mm must be positive")
    inside = resample_mask_to(dose_volume, structure_mask, structure_affine)
    if not inside.any():
        return np.zeros_like(inside)
    # Distance from every voxel to the nearest structure voxel, in mm.
    dist = ndimage.distance_transform_edt(~inside,
                                          sampling=dose_volume.spacing[::-1])
    rind = (dist <= float(depth_mm)) & ~inside
    if exclude_mask is not None:
        rind &= ~np.asarray(exclude_mask, dtype=bool)
    return rind


# ------------------------------------------------------------- wall dose
def wall_dose(mesh, depth_mm, seed_centers=None, seed_axes=None,
              sk_per_seed_u=TG43Engine.DEFAULT_SK_U, engine=None,
              elapsed_hours=None, dose_volume: Optional[Volume] = None):
    """Dose [cGy] at ``depth_mm`` along each vertex normal of ``mesh``.

    The cavity mesh from :func:`gtcore.segment.mask_to_mesh` carries normals
    pointing out of the cavity into the surrounding brain, so the offset
    points sit ``depth_mm`` deep in the wall -- the prescription point of a
    GammaTile implant. Dose comes from the exact engine over the given
    seeds, or by trilinear sampling of ``dose_volume`` if one is supplied.
    Returns a ``(n_vertices,)`` array.
    """
    if len(mesh.vertices) == 0:
        return np.zeros(0)
    pts = np.asarray(mesh.vertices, float) \
        + float(depth_mm) * np.asarray(mesh.vertex_normals, float)
    if dose_volume is not None:
        return np.asarray(dose_volume.sample_ras(pts), dtype=float)
    if seed_centers is None or seed_axes is None:
        raise ValueError("give seed_centers/seed_axes or a dose_volume")
    return dose_at_points(seed_centers, seed_axes, pts, sk_per_seed_u,
                          engine=engine, elapsed_hours=elapsed_hours)


def surface_coverage(mesh, vertex_dose, rx_cgy):
    """Area fraction of ``mesh`` whose (face-mean) dose is >= ``rx_cgy``."""
    faces = np.asarray(mesh.faces)
    if faces.shape[0] == 0:
        return float("nan")
    vd = np.asarray(vertex_dose, dtype=float)
    if vd.shape != (len(mesh.vertices),):
        raise ValueError("vertex_dose must have one value per vertex")
    face_dose = vd[faces].mean(axis=1)
    area = np.asarray(mesh.area_faces, dtype=float)
    total = area.sum()
    if total <= 0.0:
        return float("nan")
    return float(area[face_dose >= float(rx_cgy)].sum() / total)
