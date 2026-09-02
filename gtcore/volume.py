"""Core 3D volume representation.

Conventions
-----------
- ``array`` is indexed ``[k, j, i]`` -- numpy slice/row/column order.
- ``affine`` is a 4x4 matrix mapping homogeneous voxel index ``(i, j, k, 1)``
  to physical **RAS** millimetres (right-anterior-superior). RAS keeps every
  export comparable with the earlier GammaView work and with common
  neuro-imaging tools.
- DICOM headers are LPS; ``gtcore.io`` converts to RAS at load time, exactly
  once, so nothing downstream ever thinks about LPS again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy import ndimage


def apply_affine(matrix, points):
    """Apply a 4x4 affine to one point ``(3,)`` or many points ``(N, 3)``."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    mat = np.asarray(matrix, dtype=float)
    out = np.atleast_2d(pts) @ mat[:3, :3].T + mat[:3, 3]
    return out[0] if single else out


@dataclass
class Volume:
    """A 3D scalar volume with physical-space geometry (see module docstring)."""

    array: np.ndarray
    affine: np.ndarray
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.array = np.asarray(self.array)
        self.affine = np.asarray(self.affine, dtype=float)
        if self.array.ndim != 3:
            raise ValueError(
                "Volume.array must be 3D [k, j, i], got shape %s" % (self.array.shape,)
            )
        if self.affine.shape != (4, 4):
            raise ValueError("Volume.affine must be 4x4, got %s" % (self.affine.shape,))

    # ------------------------------------------------------------- geometry
    @property
    def shape_ijk(self):
        """(ni, nj, nk) -- axis order matching the affine's columns."""
        nk, nj, ni = self.array.shape
        return (ni, nj, nk)

    @property
    def spacing(self):
        """Per-axis voxel spacing (si, sj, sk) in mm."""
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @property
    def direction(self):
        """3x3 matrix whose columns are the unit i/j/k axis directions in RAS."""
        return self.affine[:3, :3] / self.spacing

    @property
    def origin_ras(self):
        """RAS position of the centre of voxel (0, 0, 0)."""
        return self.affine[:3, 3].copy()

    def index_to_ras(self, ijk):
        return apply_affine(self.affine, ijk)

    def ras_to_index(self, ras):
        return apply_affine(np.linalg.inv(self.affine), ras)

    def bounds_ras(self):
        """(2, 3) array of min/max RAS coordinates over the volume corners."""
        ni, nj, nk = self.shape_ijk
        corners = np.array(
            [[i, j, k] for i in (0, ni - 1) for j in (0, nj - 1) for k in (0, nk - 1)],
            dtype=float,
        )
        ras = self.index_to_ras(corners)
        return np.vstack([ras.min(axis=0), ras.max(axis=0)])

    # ------------------------------------------------------------- sampling
    def sample_ras(self, points, order=1, fill=0.0):
        """Interpolate voxel values at RAS points ``(N, 3)`` (or a single point)."""
        pts = np.asarray(points, dtype=float)
        single = pts.ndim == 1
        ijk = np.atleast_2d(self.ras_to_index(pts))
        coords_kji = ijk[:, ::-1].T  # map_coordinates uses array index order
        vals = ndimage.map_coordinates(
            self.array.astype(np.float32, copy=False), coords_kji, order=order,
            mode="constant", cval=fill,
        )
        return float(vals[0]) if single else vals

    def copy_with(self, array=None, affine=None):
        return Volume(
            self.array.copy() if array is None else array,
            self.affine.copy() if affine is None else affine,
            dict(self.meta),
        )
