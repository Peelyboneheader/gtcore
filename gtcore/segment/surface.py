"""Binary mask -> physical-space triangle mesh.

Meshes are what the front-end actually renders and what the tile-placement
interaction runs against, so this is the hand-off point from voxels to
geometry.
"""
from __future__ import annotations

import numpy as np
import trimesh
from skimage import measure

from ..volume import apply_affine


def mask_to_mesh(mask, affine, smooth_iterations=10, largest_only=True, step_size=1):
    """Marching-cubes a bool mask into a ``trimesh.Trimesh`` in RAS mm.

    Parameters
    ----------
    mask : ndarray of bool, ``[k, j, i]``
    affine : (4, 4) array
        Voxel-index ``(i, j, k, 1)`` -> RAS mm, i.e. ``Volume.affine``.
    smooth_iterations : int
        Taubin smoothing passes.  Taubin (not Laplacian) because it is
        shrink-free: a Laplacian pass would pull a cavity surface inwards, and
        a cavity that quietly shrinks by a millimetre per smoothing pass moves
        every tile we later snap to it.
    largest_only : bool
        Keep only the largest connected component.  Speckle from thresholding
        would otherwise show up as floating debris.

    Returns
    -------
    trimesh.Trimesh -- empty if the mask is empty.

    Notes
    -----
    ``skimage.measure.marching_cubes`` returns vertices in the array's own
    axis order, ``[k, j, i]``; the affine expects ``(i, j, k)``.  The flip
    below is the single place that conversion happens.

    Normals are fixed to point **out of the solid**.  For a cavity mask -- a
    pocket of air/fluid -- "out of the solid" means out of the cavity and into
    the surrounding brain tissue, which is exactly the direction a GammaTile
    is pressed when it is laid against the wall.  The future snap-to-wall
    interaction consumes these normals directly as the tile's seating
    orientation, so their sign is load-bearing, not cosmetic.
    """
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine, dtype=float)

    if not mask.any():
        return trimesh.Trimesh()

    verts_kji, faces, _normals, _values = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, step_size=int(step_size)
    )

    verts_ijk = verts_kji[:, ::-1]  # [k, j, i] -> (i, j, k)
    verts_ras = apply_affine(affine, verts_ijk)

    mesh = trimesh.Trimesh(vertices=verts_ras, faces=faces, process=True)

    if largest_only:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda m: len(m.faces))

    if smooth_iterations and len(mesh.faces):
        trimesh.smoothing.filter_taubin(mesh, iterations=int(smooth_iterations))

    if len(mesh.faces):
        mesh.fix_normals()
    return mesh
