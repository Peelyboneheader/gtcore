"""Candidate generation (section 3 A). Owner: A1 Geometry, branch ``plan/candidates``.

Implements, with the signatures frozen in ``gtcore.plan.__init__``:

- :func:`build_candidates`: farthest-point anchor sampling at spacing
  ``h_mm`` on the eligible faces, the spin set (full: ``n_spins`` over
  [0, 90); half: over [0, 180)), ``snap_to_wall`` + ``conform_tile`` per
  (anchor, spin, kind), rejection of hanging (more than one ray-cast
  fallback) and detached (seed further than ``detached_mm`` off its 3 mm
  offset) tiles, rejection counts, ``anchor_ids`` so "same anchor, other
  spin" is recoverable.
- :func:`visible_faces`: first-hit faces from an interior point (the inner
  wall of a closed shell mesh).
- :func:`recommend_tile_count`: the section 10 manufacturer rule.

Read-only use of ``gtcore.interact``; never edit it.  Tests in
``tests/test_plan_candidates.py``.

Detecting the conformer's fallback
----------------------------------
``interact._project_to_wall`` casts every grid point (seeds and corners)
from its flat tangent-plane position along the anchor normal ``+-n`` and,
when the ray misses or only hits further than ``interact._MAX_SAG_MM``,
falls back to the nearest mesh point.  The fallback is internal, so this
module replicates the cast exactly (same origins, directions, intersector
and sag rule) in one chunked batch for every (anchor, spin, kind) BEFORE
conforming, and only conforms the candidates that pass the hanging rule.
This is the conformer's own decision, and it saves the ``conform_tile``
call for every hanging candidate.

The cheaper "detect it from the conformed tile" alternative -- recover each
grid point's wall point as the nearest mesh point of its conformed 3 mm
offset point and flag a lateral deviation from the ``+-n`` ray above
0.5 mm -- was measured and rejected: the conformer offsets along the
SMOOTH interpolated normal, which differs from the anchor normal near
edges and on bumps, so true hits showed lateral deviations up to 1.9 mm
(p99 1.26 mm) on the synthetic cavity and 2.3 mm on the flat wall's edge
strip, and the rule rejected 30 of 63 good tiles.  It survives here as
:func:`grid_fallback_flags` for diagnostics only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from ..interact import (
    SEED_WALL_OFFSET_MM,
    TILE_HALF_SIZE_MM,
    PlacedTile,
    _MAX_SAG_MM,
    _grid_offsets,
    _rodrigues,
    conform_tile,
    snap_to_wall,
)
from . import (
    DEFAULT_H_MM,
    DEFAULT_N_SPINS_FULL,
    DEFAULT_N_SPINS_HALF,
    DETACHED_MM,
    TILE_AREA_CM2,
    CandidateSet,
    TileCountRecommendation,
)

FALLBACK_LATERAL_TOL_MM = 0.5
# Diagnostic only (grid_fallback_flags): lateral deviation of a recovered
# wall point from its +-n ray above which the point looks like a fallback.

RAY_CHUNK = 1000
# Grid points per broad-phase batch: bounds the transient (point, triangle)
# pair arrays (~2000 pairs per point on a 1 mm marching-cubes mesh) and the
# reference replica's per-ray candidate lists.

DENSE_SAMPLES_PER_H2 = 20.0
# Dense surface samples per h^2 of eligible area feeding the farthest-point
# sampler: dense spacing ~ h / 4.5, well below the anchor spacing.

DENSE_SAMPLES_MIN = 500
DENSE_SAMPLES_MAX = 200_000

VISIBLE_TOL_MM = 0.5
# visible_faces: the first hit counts as the face itself when the hit point
# is within this distance of the face centroid (grazing an edge).

ELLIPSOID_P = 1.6075
# Knud Thomsen approximation exponent for the ellipsoid surface area.

MIN_FACES = 4
# Fewer faces than a tetrahedron is not a surface a tile can sit on.

_TILE_DIAGONAL_MM = {
    "full": 2.0 * math.sqrt(2.0) * TILE_HALF_SIZE_MM,                          # 28.28
    "half": math.sqrt((2.0 * TILE_HALF_SIZE_MM) ** 2 + TILE_HALF_SIZE_MM ** 2),  # 22.36
}


def tile_diagonal_mm(kind: str) -> float:
    """Footprint diagonal of a ``kind`` tile, mm (28.3 full, 22.4 half)."""
    try:
        return _TILE_DIAGONAL_MM[str(kind)]
    except KeyError:
        raise ValueError("kind must be 'full' or 'half', got %r" % (kind,))


# ------------------------------------------------------------ validation
def _check_mesh(mesh, who: str = "build_candidates") -> np.ndarray:
    """Raise ``ValueError`` for a degenerate mesh; return the face areas."""
    if mesh is None:
        raise ValueError("%s: mesh is None" % who)
    try:
        faces = np.asarray(mesh.faces)
        verts = np.asarray(mesh.vertices)
    except AttributeError:
        raise ValueError("%s: mesh must be a trimesh.Trimesh" % who)
    if faces.size == 0 or verts.size == 0:
        raise ValueError("%s: degenerate mesh (empty: no faces)" % who)
    if faces.shape[0] < MIN_FACES:
        raise ValueError("%s: degenerate mesh (%d faces < %d)"
                         % (who, faces.shape[0], MIN_FACES))
    area_faces = np.asarray(mesh.area_faces, dtype=float)
    if not np.isfinite(area_faces).all() or area_faces.sum() <= 0.0:
        raise ValueError("%s: degenerate mesh (zero area)" % who)
    return area_faces


def _eligible_mask(mesh, eligible_faces) -> np.ndarray:
    """``(F,)`` bool from a bool mask, an index array, or None (all faces)."""
    n_faces = int(np.asarray(mesh.faces).shape[0])
    if eligible_faces is None:
        return np.ones(n_faces, dtype=bool)
    mask = np.asarray(eligible_faces)
    if mask.dtype != bool:
        idx = mask.astype(int).reshape(-1)
        mask = np.zeros(n_faces, dtype=bool)
        mask[idx] = True
    mask = mask.reshape(-1)
    if mask.shape[0] != n_faces:
        raise ValueError("eligible_faces must have one entry per face (%d), got %d"
                         % (n_faces, mask.shape[0]))
    return mask


# ----------------------------------------------------------------- anchors
def _farthest_point_indices(points: np.ndarray, h: float, start: int) -> np.ndarray:
    """Greedy farthest-point subset of ``points`` with pairwise spacing >= h.

    Starts from ``start`` and keeps adding the point farthest from the
    current subset until that distance drops below ``h``.  O(n_anchors * n).
    """
    chosen = [int(start)]
    mind = np.linalg.norm(points - points[start][None, :], axis=1)
    while True:
        i = int(np.argmax(mind))
        if mind[i] < h:
            break
        chosen.append(i)
        d = np.linalg.norm(points - points[i][None, :], axis=1)
        np.minimum(mind, d, out=mind)
    return np.asarray(chosen, dtype=int)


def sample_anchors(mesh, h_mm: float, eligible: np.ndarray, rng_seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Anchor points at spacing ~``h_mm`` on the eligible faces of ``mesh``.

    Dense seeded ``trimesh.sample.sample_surface`` restricted to the
    eligible faces (area weighted), then farthest-point sampling from the
    dense point nearest the eligible region's centroid until the largest
    remaining gap is below ``h_mm`` (so anchors are pairwise >= h apart).
    Returns ``(anchors (A, 3), face_ids (A,), method)`` in FPS order.
    """
    area_faces = np.asarray(mesh.area_faces, dtype=float)
    weights = np.where(eligible, area_faces, 0.0)
    area = float(weights.sum())
    n_dense = int(math.ceil(DENSE_SAMPLES_PER_H2 * area / float(h_mm) ** 2))
    n_dense = int(min(max(n_dense, DENSE_SAMPLES_MIN), DENSE_SAMPLES_MAX))
    pts, fid = trimesh.sample.sample_surface(mesh, n_dense, face_weight=weights,
                                             seed=int(rng_seed))
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    fid = np.asarray(fid, dtype=int).reshape(-1)
    if pts.shape[0] == 0:
        raise ValueError("build_candidates: eligible region yielded no surface samples")
    centre = pts.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(pts - centre[None, :], axis=1)))
    keep = _farthest_point_indices(pts, float(h_mm), start)
    method = "farthest_point(sample_surface n=%d, seed=%d)" % (n_dense, int(rng_seed))
    return pts[keep], fid[keep], method


# ------------------------------------------------------------- axis hints
def _axis_hint(normal: np.ndarray) -> np.ndarray:
    """Deterministic unit tangent: the global axis least aligned with
    ``normal``, projected onto the tangent plane."""
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    e = np.zeros(3)
    e[int(np.argmin(np.abs(n)))] = 1.0
    t = e - float(e @ n) * n
    return t / float(np.linalg.norm(t))


def spin_set(kind: str, n_spins: Optional[int]) -> np.ndarray:
    """Spin angles [deg] for ``kind``: ``n`` steps over [0, 90) (full) or
    [0, 180) (half); ``n_spins`` None -> the module defaults."""
    if kind == "full":
        n = DEFAULT_N_SPINS_FULL if n_spins is None else int(n_spins)
        period = 90.0
    elif kind == "half":
        n = DEFAULT_N_SPINS_HALF if n_spins is None else int(n_spins)
        period = 180.0
    else:
        raise ValueError("kind must be 'full' or 'half', got %r" % (kind,))
    if n < 1:
        raise ValueError("n_spins must be >= 1, got %d" % n)
    return np.arange(n, dtype=float) * (period / n)


def _conform_frame(inward_normal: np.ndarray, axis_hint: np.ndarray):
    """The ``(n, t1, t2)`` frame ``conform_tile`` builds
    (``interact._tangent_frame``, including its parallel-hint fallback)."""
    n = np.asarray(inward_normal, dtype=float)
    n = n / float(np.linalg.norm(n))
    hint = np.asarray(axis_hint, dtype=float)
    t1 = hint - float(hint @ n) * n
    if float(np.linalg.norm(t1)) < 1e-6:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(helper @ n)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n, helper)
    t1 = t1 / float(np.linalg.norm(t1))
    t2 = np.cross(n, t1)
    t2 = t2 / float(np.linalg.norm(t2))
    return n, t1, t2


# ------------------------------------------------------- fallback detection
def count_ray_fallbacks_trimesh(mesh, flat_points: np.ndarray, normals: np.ndarray,
                                chunk: int = RAY_CHUNK) -> np.ndarray:
    """Reference replica of the conformer's fallback decision through
    ``mesh.ray.intersects_location`` itself (the call ``_project_to_wall``
    makes), chunked.  Slow on oblique rays (trimesh's broad phase boxes the
    whole ray through the mesh bounds); used by the tests to check
    :func:`count_ray_fallbacks`.
    """
    flat = np.asarray(flat_points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=float).reshape(-1, 3)
    g = flat.shape[0]
    best = np.full(g, np.inf)
    for lo in range(0, g, int(chunk)):
        hi = min(g, lo + int(chunk))
        f = flat[lo:hi]
        m = f.shape[0]
        origins = np.vstack([f, f])
        dirs = np.vstack([nrm[lo:hi], -nrm[lo:hi]])
        try:
            locs, ray_idx, _tri = mesh.ray.intersects_location(
                ray_origins=origins, ray_directions=dirs, multiple_hits=True)
        except Exception:
            continue                       # the conformer treats this as all-miss
        locs = np.atleast_2d(np.asarray(locs, dtype=float)).reshape(-1, 3)
        ray_idx = np.asarray(ray_idx, dtype=int).reshape(-1)
        if ray_idx.size == 0:
            continue
        pi = ray_idx % m
        t_abs = np.linalg.norm(locs - f[pi], axis=1)
        np.minimum.at(best, lo + pi, t_abs)
    return ~np.isfinite(best) | (best > _MAX_SAG_MM)


def count_ray_fallbacks(mesh, flat_points: np.ndarray, normals: np.ndarray,
                        chunk: int = RAY_CHUNK) -> np.ndarray:
    """``(G,)`` bool: for each flat grid point, whether ``interact._project_to_wall``
    would fall back to the nearest mesh point (no hit along ``+-normal``
    within ``interact._MAX_SAG_MM``).

    Same decision as the conformer's ``mesh.ray.intersects_location`` call,
    computed with trimesh's own narrow phase (``intersections.planes_lines``
    + barycentric containment at ``tol.zero``) but a broad phase clipped to
    the sag limit: a hit beyond ``_MAX_SAG_MM`` is a fallback anyway, so
    only triangles whose centroid lies within ``_MAX_SAG_MM`` + their
    circumradius of the grid point (cKDTree ball query; oversized triangles
    are paired with every point) and whose centroid is within one
    circumradius of the ``+-n`` line can change the answer.  O(G) ball
    queries; ``chunk`` points per batch bounds the transient pair arrays.
    """
    from scipy.spatial import cKDTree
    from trimesh import intersections as _tm_int
    from trimesh import triangles as _tm_tri
    from trimesh.constants import tol as _tol

    flat = np.asarray(flat_points, dtype=float).reshape(-1, 3)
    nrm = np.asarray(normals, dtype=float).reshape(-1, 3)
    g = flat.shape[0]
    best = np.full(g, np.inf)
    if g == 0:
        return np.zeros(0, dtype=bool)

    tris = np.asarray(mesh.triangles, dtype=float)
    cent = tris.mean(axis=1)
    r_tri = np.linalg.norm(tris - cent[:, None, :], axis=2).max(axis=1)
    face_n = np.asarray(mesh.face_normals, dtype=float)
    r_cut = 3.0 * float(np.median(r_tri))
    large = r_tri > r_cut
    if large.mean() > 0.2:                 # not a few outliers: one ball for all
        r_cut = float(r_tri.max())
        large = np.zeros(r_tri.size, dtype=bool)
    small_idx = np.flatnonzero(~large)
    large_idx = np.flatnonzero(large)
    tree = cKDTree(cent[small_idx]) if small_idx.size else None
    radius = _MAX_SAG_MM + r_cut + 1e-6

    for lo in range(0, g, int(chunk)):
        hi = min(g, lo + int(chunk))
        f = flat[lo:hi]
        n = nrm[lo:hi]
        m = f.shape[0]
        pi_parts: List[np.ndarray] = []
        ti_parts: List[np.ndarray] = []
        if tree is not None:
            lists = tree.query_ball_point(f, radius, return_sorted=False)
            lens = np.fromiter((len(l) for l in lists), dtype=int, count=m)
            if lens.sum():
                pi_parts.append(np.repeat(np.arange(m), lens))
                ti_parts.append(small_idx[np.concatenate([np.asarray(l, dtype=int)
                                                          for l in lists if len(l)])])
        if large_idx.size:
            pi_parts.append(np.repeat(np.arange(m), large_idx.size))
            ti_parts.append(np.tile(large_idx, m))
        if not pi_parts:
            continue
        pi = np.concatenate(pi_parts)
        ti = np.concatenate(ti_parts)
        # sound prefilter: centroid within one circumradius of the line and
        # within sag + circumradius along it
        d = cent[ti] - f[pi]
        along = (d * n[pi]).sum(axis=1)
        lat2 = (d * d).sum(axis=1) - along * along
        rr = r_tri[ti] + 1e-6
        keep = (lat2 <= rr * rr) & (np.abs(along) <= _MAX_SAG_MM + rr)
        if not keep.any():
            continue
        pi = pi[keep]
        ti = ti[keep]
        loc, valid = _tm_int.planes_lines(plane_origins=tris[ti, 0, :],
                                          plane_normals=face_n[ti],
                                          line_origins=f[pi],
                                          line_directions=n[pi])
        if not valid.any():
            continue
        pi = pi[valid]
        ti = ti[valid]
        bary = _tm_tri.points_to_barycentric(tris[ti], loc)
        hit = (bary > -_tol.zero).all(axis=1) & (bary < 1.0 + _tol.zero).all(axis=1)
        if not hit.any():
            continue
        t_abs = np.linalg.norm(loc[hit] - f[pi[hit]], axis=1)
        np.minimum.at(best, lo + pi[hit], t_abs)
    return ~np.isfinite(best) | (best > _MAX_SAG_MM)


def grid_fallback_flags(tile: PlacedTile, wall_points: np.ndarray,
                        tol_mm: float = FALLBACK_LATERAL_TOL_MM) -> np.ndarray:
    """Diagnostic: ``(G,)`` bool of grid points (seeds then corners, the
    order of ``interact._grid_offsets``) whose recovered ``wall_points``
    deviate laterally from the ``+-n`` ray through their flat position by
    more than ``tol_mm``.  NOT used for rejection (see the module docstring).
    """
    n, t1, t2 = _conform_frame(tile.normal_ras, tile.axis_ras)
    uv = np.vstack(_grid_offsets(tile.kind))
    flat = tile.anchor_ras[None, :] + uv @ np.vstack([t1, t2])
    wall = np.asarray(wall_points, dtype=float).reshape(-1, 3)
    if wall.shape[0] != flat.shape[0]:
        raise ValueError("wall_points must have one row per grid point")
    delta = wall - flat
    along = delta @ n
    lateral = np.linalg.norm(delta - along[:, None] * n[None, :], axis=1)
    return lateral > float(tol_mm)


# ------------------------------------------------------------------- build
def build_candidates(mesh, h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
                     kinds: Sequence[str] = ("full",), eligible_faces=None,
                     rng_seed: int = 0, detached_mm: float = DETACHED_MM,
                     min_fraction_on_wall: Optional[float] = None) -> CandidateSet:
    """Sample anchors on ``mesh`` and conform one tile per (anchor, spin, kind).

    See ``gtcore.plan.build_candidates`` for the parameter contract.  Output
    order: anchor-major, then ``kinds`` in the order given, then spin
    ascending.  Deterministic for identical inputs and ``rng_seed``.

    Pipeline: anchors (farthest point) -> ``snap_to_wall`` once per anchor
    -> flat grid points of every (anchor, kind, spin) -> one chunked ray
    batch (the conformer's own fallback rule) -> hanging rejection ->
    ``conform_tile`` for the survivors -> one batched proximity query of
    every conformed seed -> detached rejection.  ``n_rejected`` counts
    candidates (``"ineligible"`` = dropped anchors x spins x kinds).

    Raises ``ValueError`` (with a reason) for a degenerate mesh, an empty
    eligible region, or when every candidate was rejected.
    """
    area_faces = _check_mesh(mesh)
    h = float(h_mm)
    if not (h > 0.0):
        raise ValueError("h_mm must be positive, got %r" % (h_mm,))
    kinds = tuple(str(k) for k in kinds)
    if len(kinds) == 0:
        raise ValueError("kinds must name at least one of 'full', 'half'")
    for k in kinds:
        if k not in ("full", "half"):
            raise ValueError("kind must be 'full' or 'half', got %r" % (k,))
    if min_fraction_on_wall is not None and not (0.0 <= float(min_fraction_on_wall) <= 1.0):
        raise ValueError("min_fraction_on_wall must be in [0, 1] or None")
    eligible = _eligible_mask(mesh, eligible_faces)
    wall_area = float(area_faces[eligible].sum())
    if not eligible.any() or wall_area <= 0.0:
        raise ValueError("build_candidates: eligible region is empty "
                         "(%d of %d faces eligible, area %.3g mm^2)"
                         % (int(eligible.sum()), eligible.size, wall_area))

    spins = {k: spin_set(k, n_spins) for k in kinds}
    n_spin_total = int(sum(len(s) for s in spins.values()))
    grids = {k: np.vstack(_grid_offsets(k)) for k in kinds}      # seeds then corners

    anchors_raw, _fid, method = sample_anchors(mesh, h, eligible, rng_seed)
    n_rejected: Dict[str, int] = {"ineligible": 0, "hanging": 0, "detached": 0,
                                  "conform_error": 0}

    # drop anchors whose nearest face is ineligible
    _surf, _dist, tid = trimesh.proximity.closest_point(mesh, anchors_raw)
    tid = np.asarray(tid, dtype=int).reshape(-1)
    ok = eligible[tid]
    n_rejected["ineligible"] = int((~ok).sum()) * n_spin_total
    anchor_idx = np.flatnonzero(ok)
    if anchor_idx.size == 0:
        raise ValueError("build_candidates: every sampled anchor (%d) lies on an "
                         "ineligible face" % anchors_raw.shape[0])

    # snap once per anchor; enumerate (anchor, kind, spin) with their flat grids
    specs = []            # (anchor_id, surf, n_in, hint, kind, theta)
    flat_blocks: List[np.ndarray] = []
    nrm_blocks: List[np.ndarray] = []
    for a_id in anchor_idx:
        surf, n_in = snap_to_wall(mesh, anchors_raw[a_id])
        hint0 = _axis_hint(n_in)
        for kind in kinds:
            uv = grids[kind]
            for theta in spins[kind]:
                if theta == 0.0:
                    hint = hint0
                else:
                    hint = _rodrigues(hint0, n_in, math.radians(float(theta)))
                n, t1, t2 = _conform_frame(n_in, hint)
                specs.append((int(a_id), surf, n_in, hint, kind, float(theta)))
                flat_blocks.append(surf[None, :] + uv @ np.vstack([t1, t2]))
                nrm_blocks.append(np.tile(n, (uv.shape[0], 1)))
    n_grid = np.asarray([b.shape[0] for b in flat_blocks], dtype=int)
    offsets = np.concatenate([[0], np.cumsum(n_grid)])

    # the conformer's own fallback decision, batched
    fallback = count_ray_fallbacks(mesh, np.vstack(flat_blocks), np.vstack(nrm_blocks))
    n_fb = np.add.reduceat(fallback.astype(int), offsets[:-1])
    if min_fraction_on_wall is None:
        hanging = n_fb > 1
    else:
        hanging = (1.0 - n_fb / n_grid.astype(float)) < float(min_fraction_on_wall)
    n_rejected["hanging"] = int(hanging.sum())

    # conform the survivors
    tiles: List[PlacedTile] = []
    spins_out: List[float] = []
    anchor_ids: List[int] = []
    for c in np.flatnonzero(~hanging):
        a_id, surf, n_in, hint, kind, theta = specs[c]
        try:
            tile = conform_tile(mesh, surf, n_in, hint, kind=kind)
        except Exception:
            n_rejected["conform_error"] += 1
            continue
        if not (np.isfinite(tile.seed_centers).all() and np.isfinite(tile.corners_ras).all()):
            n_rejected["conform_error"] += 1
            continue
        tiles.append(tile)
        spins_out.append(theta)
        anchor_ids.append(a_id)

    if not tiles:
        raise ValueError("build_candidates: every candidate was rejected before "
                         "conforming (%d enumerated; rejections %r; h=%g mm, kinds=%r)"
                         % (len(specs), n_rejected, h, kinds))

    # one batched proximity query for every conformed seed: detached test
    seeds_all = np.vstack([t.seed_centers for t in tiles])
    n_seeds = np.asarray([t.seed_centers.shape[0] for t in tiles], dtype=int)
    s_off = np.concatenate([[0], np.cumsum(n_seeds)])
    pq = trimesh.proximity.ProximityQuery(mesh)
    _wall, dist_all, _tid = pq.on_surface(seeds_all)
    dev = np.abs(np.asarray(dist_all, dtype=float) - SEED_WALL_OFFSET_MM)
    detached = np.add.reduceat((dev > float(detached_mm)).astype(int), s_off[:-1]) > 0
    n_rejected["detached"] = int(detached.sum())

    kept = np.flatnonzero(~detached)
    if kept.size == 0:
        raise ValueError("build_candidates: every candidate was rejected "
                         "(%d enumerated; rejections %r; h=%g mm, kinds=%r)"
                         % (len(specs), n_rejected, h, kinds))

    return CandidateSet.from_tiles(
        [tiles[i] for i in kept],
        spins_deg=np.asarray(spins_out, dtype=float)[kept],
        anchor_ids=np.asarray(anchor_ids, dtype=int)[kept],
        eligible=np.ones(kept.size, dtype=bool),
        method=method, h_mm=h, n_spins=int(len(spins[kinds[0]])),
        n_rejected=n_rejected, wall_area_mm2=wall_area,
    )


# ---------------------------------------------------------- visible faces
def visible_faces(mesh, center_ras) -> np.ndarray:
    """``(F,)`` bool: faces whose centroid is the FIRST hit of a ray from
    ``center_ras`` toward it (the hit triangle is the face itself, or the
    hit point lies within ``VISIBLE_TOL_MM`` of the centroid).  O(F) rays.
    """
    _check_mesh(mesh, "visible_faces")
    centre = np.asarray(center_ras, dtype=float).reshape(3)
    cents = np.asarray(mesh.triangles_center, dtype=float)
    n_faces = cents.shape[0]
    dirs = cents - centre[None, :]
    lengths = np.linalg.norm(dirs, axis=1)
    good = lengths > 1e-9
    out = np.zeros(n_faces, dtype=bool)
    if not good.any():
        return out
    face_of_ray = np.flatnonzero(good)
    origins = np.tile(centre, (face_of_ray.size, 1))
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs[good], multiple_hits=True)
    locs = np.atleast_2d(np.asarray(locs, dtype=float)).reshape(-1, 3)
    ray_idx = np.asarray(ray_idx, dtype=int).reshape(-1)
    tri_idx = np.asarray(tri_idx, dtype=int).reshape(-1)
    if ray_idx.size == 0:
        return out
    # first hit per ray = the smallest distance from the centre
    t_hit = np.linalg.norm(locs - centre[None, :], axis=1)
    order = np.lexsort((t_hit, ray_idx))
    ray_sorted = ray_idx[order]
    first = order[np.concatenate([[True], ray_sorted[1:] != ray_sorted[:-1]])]
    f = face_of_ray[ray_idx[first]]
    self_hit = tri_idx[first] == f
    near = np.linalg.norm(locs[first] - cents[f], axis=1) <= VISIBLE_TOL_MM
    out[f[self_hit | near]] = True
    return out


# ------------------------------------------------------------ tile count
def ellipsoid_area_mm2(diameters_mm, p: float = ELLIPSOID_P) -> float:
    """Knud Thomsen approximation of an ellipsoid's surface area from its
    three diameters (relative error < 1.1 %)."""
    a, b, c = (float(d) / 2.0 for d in np.asarray(diameters_mm, dtype=float).reshape(3))
    s = ((a ** p) * (b ** p) + (a ** p) * (c ** p) + (b ** p) * (c ** p)) / 3.0
    return float(4.0 * math.pi * s ** (1.0 / p))


def _n_tiles(area_mm2: float) -> int:
    return int(math.ceil(max(float(area_mm2), 0.0) / (TILE_AREA_CM2 * 100.0) - 1e-9))


def recommend_tile_count(mesh, contraction_pct: float = 0.0, untreated_pct: float = 0.0,
                         eligible_faces=None) -> TileCountRecommendation:
    """Manufacturer-rule tile count (section 10).

    ``area_mm2`` is the eligible (or full) mesh area; the treatable area
    deducts ``contraction_pct`` and ``untreated_pct``; ``n_tiles`` is the
    treatable area over 4 cm^2 rounded up.  The pre-operative ellipsoid
    estimate uses the extents of the eligible vertices along their
    principal (PCA) axes as the three diameters and the Knud Thomsen area
    formula, with the same deductions.
    """
    area_faces = _check_mesh(mesh, "recommend_tile_count")
    eligible = _eligible_mask(mesh, eligible_faces)
    if not eligible.any():
        raise ValueError("recommend_tile_count: eligible region is empty")
    area = float(area_faces[eligible].sum())
    for name, pct in (("contraction_pct", contraction_pct), ("untreated_pct", untreated_pct)):
        if not (0.0 <= float(pct) < 100.0):
            raise ValueError("%s must be in [0, 100), got %r" % (name, pct))
    deduct = (1.0 - float(contraction_pct) / 100.0) * (1.0 - float(untreated_pct) / 100.0)
    treatable = area * deduct

    faces = np.asarray(mesh.faces, dtype=int)
    vidx = np.unique(faces[eligible].reshape(-1))
    verts = np.asarray(mesh.vertices, dtype=float)[vidx]
    centred = verts - verts.mean(axis=0, keepdims=True)
    if centred.shape[0] >= 3:
        cov = centred.T @ centred / float(centred.shape[0])
        _w, vecs = np.linalg.eigh(cov)
        proj = centred @ vecs
        diameters = proj.max(axis=0) - proj.min(axis=0)
    else:
        diameters = centred.max(axis=0) - centred.min(axis=0)
    diameters = np.sort(np.asarray(diameters, dtype=float))[::-1]
    ell_area = ellipsoid_area_mm2(diameters)

    volume = None
    try:
        if bool(mesh.is_watertight):
            volume = float(abs(mesh.volume))
    except Exception:
        volume = None

    source = ("GammaTile Cavity Surface Area Calculator "
              "(gammatile.com/hcp/medical-physics/surface-area-calculator): "
              "ellipsoid surface area from three diameters (medial-lateral, "
              "anterior-posterior, superior-inferior), minus 'estimated surgical "
              "cavity contraction (%)' and 'estimated surface area not requiring "
              "GammaTiles (%)', then 'divide the tumor bed calculator result by 4, "
              "rounding up to the nearest whole number (GammaTile = 4 cm^2)'. "
              "With a measured mesh the area is the mesh area itself "
              "(docs/plan-tile-optimize.md section 10).")
    return TileCountRecommendation(
        n_tiles=_n_tiles(treatable), area_mm2=area, treatable_area_mm2=treatable,
        contraction_pct=float(contraction_pct), untreated_pct=float(untreated_pct),
        ellipsoid_area_mm2=ell_area, n_tiles_ellipsoid=_n_tiles(ell_area * deduct),
        diameters_mm=diameters, volume_mm3=volume, source=source,
    )


__all__ = [
    "FALLBACK_LATERAL_TOL_MM", "RAY_CHUNK", "VISIBLE_TOL_MM", "ELLIPSOID_P",
    "tile_diagonal_mm", "sample_anchors", "spin_set", "count_ray_fallbacks",
    "count_ray_fallbacks_trimesh",
    "grid_fallback_flags", "build_candidates", "visible_faces",
    "ellipsoid_area_mm2", "recommend_tile_count",
]
