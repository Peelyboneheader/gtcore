"""Resection-cavity segmentation, optionally guided by the detected seed cloud.

The seed-cloud prior
--------------------
GammaTiles are collagen tiles carrying Cs-131 seeds that the surgeon lays
against the *wall of the resection cavity* -- that is what the device is for.
So once ``gtcore.seeds.detect`` has found the seeds, their centres are not
merely an output, they are a strong spatial prior on where the cavity is: the
cavity is the low-density pocket that the seed cloud surrounds.

That prior does real work.  A post-op brain contains several dark pockets --
ventricles, sulcal CSF, pneumocephalus far from the surgical bed, a
contralateral cyst -- and intensity alone cannot tell you which one the
surgeon operated in.  With no seeds supplied we fall back to the largest
candidate component, which is the usual heuristic and usually right but not
reliably so.

The seed *sheet* prior (four or more seeds)
-------------------------------------------
Picking the dark component that touches the seed cloud is not enough on a
real post-operative scan.  The cavity contents there are a speckled mixture
of fluid, clot, debris and air, and the low-density class runs on without a
break into peri-cavity oedema and from there into the lateral ventricle, so
"the component the seeds touch" came back as cavity + oedema + both
ventricles (193 cc for a ~25 cc cavity on the first clinical case).  Two
facts about the device fix that:

1. The seeds lie *inside* the cavity, 3 mm in from the tissue face of the
   tile (``gtcore.tiles.model``), so the convex hull of the seed cloud is a
   solid core of cavity, and nothing more than ``sheet_tol_mm`` (3 mm) beyond
   the sheet of seeds -- on the far side of the tiles -- can be cavity.
2. Where the wall is not tiled, the cavity may still extend beyond the hull,
   but only through low-density voxels geodesically connected to it, and not
   far: ``hull_reach_mm``.

So the cavity is grown from the seed hull through low-density voxels, with a
geodesic reach, clipped at the seed sheet, then lightly closed and filled.
Against the clinical structure set of the first case this gives 41 cc (Dice
0.65 to the physician's 35 cc cavity) instead of 193 cc, and 0.85 Dice on the
synthetic phantom (docs/clinical-validation-notes.md); the old component rule
stays as the fallback for fewer than four non-coplanar seeds.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._morph import close_mm, largest_cc

AIR_HU = -250.0          # pneumocephalus / trapped air
FLUID_HI_HU = 26.0       # CSF and serosanguinous fluid sit around 0-20 HU
BRAIN_ENVELOPE_MM = 10.0  # closing radius used to build the "inside brain" hull
CAVITY_CLOSE_MM = 5.0     # tidies the final mask across thin septa / clot flecks
# seed-sheet prior (see module docstring)
HULL_REACH_MM = 14.0      # geodesic growth beyond the seed hull through low density
# (sweep in docs/clinical-validation-notes.md: phantom Dice rises with reach,
#  the clinical cavity leaks into oedema beyond ~12 mm; 14 mm balances both)
SHEET_TOL_MM = 3.0        # cavity ends this far beyond the seed sheet (tile tissue face)
SHEET_CLOSE_MM = 3.0      # gentler closing: 5 mm bridged speckle into oedema (+40 cc)
_SHEET_MIN_SEEDS = 4


def segment_cavity(vol, interior_mask, brain_mask, seed_centers_ras=None,
                   seed_radius_mm=14.0, hull_reach_mm=HULL_REACH_MM,
                   sheet_tol_mm=SHEET_TOL_MM):
    """Segment the resection cavity inside an already-segmented brain.

    Parameters
    ----------
    vol : gtcore.volume.Volume
        The CT, HU in ``array`` indexed ``[k, j, i]``.
    interior_mask : ndarray of bool
        ``cranial_interior`` from :func:`gtcore.segment.head.segment_head`.
    brain_mask : ndarray of bool
        ``brain`` from the same call.
    seed_centers_ras : array_like, shape (N, 3), optional
        Seed centres in RAS mm from ``detect_seed_candidates``.  See module
        docstring.
    seed_radius_mm : float
        Fallback component rule only: how far from the seed cloud a cavity
        component may be and still be accepted.  14 mm is roughly a tile
        diagonal plus the wall thickness a tile is pressed into.
    hull_reach_mm : float
        Seed-sheet rule: geodesic distance the cavity may grow beyond the
        convex hull of the seeds through low-density voxels (untiled wall).
    sheet_tol_mm : float
        Seed-sheet rule: how far beyond the sheet of seeds (away from the
        implant centre) a voxel may lie and still be cavity -- the 3 mm from
        the seed plane to the tile's tissue face.

    Returns
    -------
    ndarray of bool, shaped like ``vol.array``.
    """
    arr = np.asarray(vol.array, dtype=np.float32)
    spacing = vol.spacing
    interior = np.asarray(interior_mask, dtype=bool)
    brain = np.asarray(brain_mask, dtype=bool)

    # ------------------------------------------------ intensity candidates
    air = (arr < AIR_HU) & interior
    fluid = (arr < FLUID_HI_HU) & interior & ~brain & ~air

    # Restrict to holes *within* the brain envelope.  Closing the brain mask
    # by 10 mm produces a solid hull whose surface follows the cortex; this
    # both excludes the subarachnoid CSF rim between brain and skull (which is
    # fluid-density and touches everything) and re-admits the cavity itself,
    # which was punched out of the brain mask by the intensity threshold.
    envelope = close_mm(brain, spacing, BRAIN_ENVELOPE_MM)
    envelope = ndimage.binary_fill_holes(envelope)

    # ------------------------------------------------- seed-sheet rule
    if seed_centers_ras is not None:
        centers = np.atleast_2d(np.asarray(seed_centers_ras, dtype=float))
        if centers.shape[0] >= _SHEET_MIN_SEEDS:
            # no brain-envelope mask here: a cavity under the craniotomy
            # lies largely outside the closed brain hull, and the sheet clip
            # plus the reach already bound where the growth can go
            out = _cavity_from_seed_sheet(
                vol, arr, air | fluid, interior, centers,
                float(hull_reach_mm), float(sheet_tol_mm))
            if out is not None:
                return out

    cand = ndimage.binary_opening(air | fluid, iterations=1) & envelope

    if not cand.any():
        return np.zeros(arr.shape, dtype=bool)

    # --------------------------------------------------- seed-guided select
    selected = None
    if seed_centers_ras is not None:
        centers = np.atleast_2d(np.asarray(seed_centers_ras, dtype=float))
        if centers.size:
            seed_vol = np.zeros(arr.shape, dtype=bool)
            ijk = np.rint(vol.ras_to_index(centers)).astype(int)
            nk, nj, ni = arr.shape
            # ras_to_index gives (i, j, k); the array is [k, j, i] -- flip.
            kji = ijk[:, ::-1]
            ok = np.all((kji >= 0) & (kji < np.array([nk, nj, ni])), axis=1)
            kji = kji[ok]
            if len(kji):
                seed_vol[kji[:, 0], kji[:, 1], kji[:, 2]] = True
                cloud = ndimage.distance_transform_edt(
                    ~seed_vol, sampling=(spacing[2], spacing[1], spacing[0])
                ) <= float(seed_radius_mm)

                labels, n = ndimage.label(cand)
                if n:
                    hit = np.unique(labels[cloud & cand])
                    hit = hit[hit > 0]
                    if hit.size:
                        selected = np.isin(labels, hit)

    if selected is None:
        selected = largest_cc(cand)

    # --------------------------------------------------------------- tidy
    out = close_mm(selected, spacing, CAVITY_CLOSE_MM) & interior
    out = ndimage.binary_fill_holes(out)
    return out


def _cavity_from_seed_sheet(vol, arr, low, interior, centers, reach_mm,
                            tol_mm):
    """Cavity = low-density voxels grown geodesically from the convex hull
    of the seeds, clipped at the seed sheet, closed and filled.  Returns
    None when the seeds are degenerate (coplanar / collinear) so the caller
    falls back to the component rule."""
    from scipy.spatial import Delaunay, QhullError, cKDTree

    try:
        tri = Delaunay(centers)
    except (QhullError, ValueError):
        return None

    spacing = vol.spacing
    nk, nj, ni = arr.shape
    # work on the bounding box of what can possibly be cavity
    pad = reach_mm + tol_mm + 2.0 * float(np.max(spacing))
    lo = np.floor(vol.ras_to_index(centers.min(axis=0) - pad)).astype(int)
    hi = np.ceil(vol.ras_to_index(centers.max(axis=0) + pad)).astype(int)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    lo = np.clip(lo, 0, [ni - 1, nj - 1, nk - 1])
    hi = np.clip(hi, 0, [ni - 1, nj - 1, nk - 1])
    sl = (slice(lo[2], hi[2] + 1), slice(lo[1], hi[1] + 1), slice(lo[0], hi[0] + 1))
    kk, jj, ii = np.indices(tuple(s.stop - s.start for s in sl))
    ijk = np.stack([ii.ravel() + lo[0], jj.ravel() + lo[1], kk.ravel() + lo[2]],
                   axis=1).astype(float)
    ras = vol.index_to_ras(ijk)
    shape_box = tuple(s.stop - s.start for s in sl)

    hull = (tri.find_simplex(ras) >= 0).reshape(shape_box)
    if not hull.any():
        return None

    # seed sheet: for each voxel the nearest seed and the local normal of
    # the sheet there; beyond tol_mm along it is tile / tissue.  The normal
    # is the plane through the seed's nearest neighbours (a tile's four seeds
    # are coplanar; neighbouring tiles bend gently), oriented away from the
    # implant centre.  Using the sheet normal rather than the direction from
    # the centre matters for one-sided implants: at the rim of a flat tile
    # the radial direction runs along the sheet and would clip the cavity.
    outward = _sheet_normals(centers, vol, low)
    _d, idx = cKDTree(centers).query(ras)
    proj = np.einsum("ij,ij->i", ras - centers[idx], outward[idx])
    inside_sheet = (proj <= tol_mm).reshape(shape_box)

    low_box = np.asarray(low, dtype=bool)[sl]
    interior_box = np.asarray(interior, dtype=bool)[sl]
    # the growth starts from the hull padded by the sheet tolerance: seeds
    # sit at the wall (1-3 mm either side of it), so a one-sided implant's
    # hull is a thin slab that need not itself contain cavity voxels
    pad_vox = max(1, int(round(tol_mm / float(np.min(spacing)))))
    padded = ndimage.binary_dilation(hull, iterations=pad_vox) & low_box
    start = (hull | padded) & inside_sheet
    grow_mask = (low_box | hull) & inside_sheet
    iters = max(1, int(round(reach_mm / float(np.min(spacing)))))
    grown = ndimage.binary_dilation(start, iterations=iters, mask=grow_mask)

    tidy = close_mm(grown, spacing, SHEET_CLOSE_MM) & interior_box & inside_sheet
    tidy = ndimage.binary_fill_holes(tidy)

    out = np.zeros(arr.shape, dtype=bool)
    out[sl] = tidy
    return out


_SHEET_K = 6            # neighbours used for the local sheet plane
_SHEET_NEIGHBOUR_MM = 16.0   # a tile diagonal plus a little: same or adjacent tile
_SHEET_PROBE_MM = (3.0, 5.0, 7.0)   # where the cavity / tissue test samples


def _sheet_normals(centers, vol, low):
    """Unit normal of the local seed sheet at every seed, pointing OUT of
    the cavity (towards the tile and the tissue behind it).

    Direction: the plane through the seed and up to ``_SHEET_K`` neighbours
    within ``_SHEET_NEIGHBOUR_MM`` (a tile's four seeds are coplanar and
    neighbouring tiles bend gently); a seed with fewer than two usable
    neighbours takes the radial direction from the implant centre.

    Sign: the side with more low-density voxels at 3-7 mm along the normal
    is the cavity, so the normal points the other way.  The implant centre
    cannot decide this for a one-sided implant (three tiles on one wall put
    the centre in the sheet itself), and the intensity evidence can; the
    radial direction only breaks ties.
    """
    from scipy.spatial import cKDTree

    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    centre = centers.mean(axis=0)
    radial = centers - centre
    rn = np.linalg.norm(radial, axis=1)
    rn[rn < 1e-9] = 1.0
    radial /= rn[:, None]
    tree = cKDTree(centers)
    k = min(_SHEET_K + 1, n)
    dist, idx = tree.query(centers, k=k)
    if n == 1:
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    low = np.asarray(low, dtype=bool)
    nk, nj, ni = low.shape

    def low_fraction(points):
        ijk = np.rint(vol.ras_to_index(points)).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array([ni, nj, nk])), axis=1)
        if not ok.any():
            return 0.0
        ijk = ijk[ok]
        return float(low[ijk[:, 2], ijk[:, 1], ijk[:, 0]].mean())

    out = np.empty_like(centers)
    probes = np.asarray(_SHEET_PROBE_MM, dtype=float)
    for i in range(n):
        near = idx[i][(dist[i] <= _SHEET_NEIGHBOUR_MM)]
        if near.size < 3:
            nrm = radial[i]
        else:
            q = centers[near] - centers[near].mean(axis=0)
            _w, v = np.linalg.eigh(q.T @ q)
            nrm = v[:, 0]
        plus = low_fraction(centers[i] + probes[:, None] * nrm)
        minus = low_fraction(centers[i] - probes[:, None] * nrm)
        if plus > minus:          # cavity lies along +nrm: point the other way
            nrm = -nrm
        elif plus == minus and np.dot(nrm, radial[i]) < 0.0:
            nrm = -nrm
        out[i] = nrm
    return out
