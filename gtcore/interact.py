"""Tile drag-and-drop geometry: snap to the cavity wall and conform to it.

Pure numpy/trimesh -- NO rendering imports here.  This module is the geometry
engine behind the interactive planner (:mod:`gtcore.planner`): every mouse
action becomes one of the pure functions below, so all of the physics is unit
testable without a window.

Conforming model
----------------
A GammaTile is a 20x20x4 mm collagen square carrying four Cs-131 seeds on a
10 mm grid (half tiles: 10x20 mm, two seeds 10 mm apart).  Laid inside a
resection cavity it drapes onto the wall, so the seeds do not stay coplanar:
each seed sits on the *local* wall, pulled ``SEED_WALL_OFFSET_MM`` off it into
the cavity.  The phantom generator does this analytically
(:func:`gtcore.phantom.generate._build_tiles` re-projects every seed onto the
lumpy wall and insets it by ``SEED_INSET_MM``); here the wall is only known as
a triangle mesh, so each grid offset is laid out in the flat tangent plane at
the seating point and cast onto the mesh along the seating normal (a
semi-rigid tile pressed against the wall bridges small concave dips rather
than draping into them), and the inset direction is the smoothly interpolated
local surface normal.

Normal-orientation robustness: pipeline cavity meshes have normals pointing
out of the cavity into tissue, but nothing here trusts that -- every local
normal is re-oriented toward the cavity interior (the mesh centroid) before
use, so either winding works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh

__all__ = [
    "SEED_WALL_OFFSET_MM",
    "TILE_SEED_PITCH_MM",
    "TILE_HALF_SIZE_MM",
    "PlacedTile",
    "snap_to_wall",
    "conform_tile",
    "translate_on_wall",
    "rotate_on_wall",
    "tiles_to_seed_arrays",
    "find_overlapping_tiles",
]

SEED_WALL_OFFSET_MM = 2.0     # seed centre this far off the wall, cavity side
TILE_SEED_PITCH_MM = 10.0     # seed grid pitch (seeds at +/- pitch/2)
TILE_HALF_SIZE_MM = 10.0      # full tile is 20x20 mm -> corners at +/- 10

_SEED_HALF = TILE_SEED_PITCH_MM / 2.0   # 5.0
_MAX_SAG_MM = 12.0                      # max plausible wall sag under a tile


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return v / n


@dataclass
class PlacedTile:
    """One interactively placed tile, fully conformed to the cavity wall.

    Attributes
    ----------
    kind : str
        ``"full"`` (4 seeds) or ``"half"`` (2 seeds).
    center_ras : (3,) ndarray
        Centroid of the conformed seed centres (matches the phantom's
        ``TileTruth.center_ras`` convention).
    normal_ras : (3,) ndarray
        Unit local wall normal at the tile's wall anchor, pointing INTO the
        cavity (toward the interior -- the direction seeds are offset).
    axis_ras : (3,) ndarray
        Unit tangent giving the tile's in-plane orientation (the ``t1`` of
        the conforming frame; exactly perpendicular to ``normal_ras``).
    seed_centers : (N, 3) ndarray
        Conformed seed centres in RAS mm (N = 4 full, 2 half).
    seed_axes : (N, 3) ndarray
        Unit seed long axes, tangent to the local wall.
    corners_ras : (4, 3) ndarray
        The tile's four corners draped onto the wall (loop order, suitable
        for rendering as two triangles ``(0,1,2)`` and ``(0,2,3)``).
    anchor_ras : (3,) ndarray
        The wall point the tile was conformed around (ON the mesh).  Used by
        :func:`translate_on_wall` / :func:`rotate_on_wall` so repeated
        gestures do not drift.
    """

    kind: str
    center_ras: np.ndarray
    normal_ras: np.ndarray
    axis_ras: np.ndarray
    seed_centers: np.ndarray
    seed_axes: np.ndarray
    corners_ras: np.ndarray
    anchor_ras: Optional[np.ndarray] = None

    def __post_init__(self):
        self.center_ras = np.asarray(self.center_ras, dtype=float).reshape(3)
        self.normal_ras = np.asarray(self.normal_ras, dtype=float).reshape(3)
        self.axis_ras = np.asarray(self.axis_ras, dtype=float).reshape(3)
        self.seed_centers = np.asarray(self.seed_centers, dtype=float).reshape(-1, 3)
        self.seed_axes = np.asarray(self.seed_axes, dtype=float).reshape(-1, 3)
        self.corners_ras = np.asarray(self.corners_ras, dtype=float).reshape(4, 3)
        if self.anchor_ras is None:
            self.anchor_ras = self.center_ras.copy()
        else:
            self.anchor_ras = np.asarray(self.anchor_ras, dtype=float).reshape(3)


# ----------------------------------------------------------------- mesh queries
def _interior_reference(mesh):
    """A point inside the cavity used to orient normals (the mesh centroid).

    The cavity is star-shaped to good approximation (lumpy ellipsoid with
    bounded bump amplitude), so "toward the centroid" and "into the cavity"
    agree everywhere on the wall.
    """
    try:
        c = np.asarray(mesh.centroid, dtype=float).reshape(3)
        if np.all(np.isfinite(c)):
            return c
    except Exception:
        pass
    return np.asarray(mesh.vertices, dtype=float).mean(axis=0)


def _closest_with_normals(mesh, points):
    """Nearest surface points plus smooth INWARD local normals.

    Returns ``(surface_points (N,3), inward_normals (N,3))``.  Normals are
    barycentric interpolations of the vertex normals (falling back to face
    normals), then re-oriented toward the cavity interior regardless of the
    mesh's winding.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    surf, _dist, tid = trimesh.proximity.closest_point(mesh, pts)
    surf = np.asarray(surf, dtype=float)
    tid = np.asarray(tid)

    try:
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tid], surf)
        vnorm = np.asarray(mesh.vertex_normals)[np.asarray(mesh.faces)[tid]]
        normals = (bary[:, :, None] * vnorm).sum(axis=1)
        norms = np.linalg.norm(normals, axis=1)
        bad = ~np.isfinite(norms) | (norms < 1e-8)
        if bad.any():
            normals[bad] = np.asarray(mesh.face_normals)[tid[bad]]
            norms = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(norms, 1e-12)[:, None]
    except Exception:
        normals = np.asarray(mesh.face_normals)[tid].astype(float).copy()
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]

    inward_ref = _interior_reference(mesh)
    to_interior = inward_ref[None, :] - surf
    flip = (normals * to_interior).sum(axis=1) < 0.0
    normals[flip] *= -1.0
    return surf, normals


def snap_to_wall(mesh, point_ras):
    """Snap an arbitrary RAS point onto the cavity wall.

    Returns ``(surface_point, inward_normal)`` -- the nearest point on the
    mesh and the unit local normal pointing into the cavity interior
    (robust to either mesh winding).
    """
    surf, normals = _closest_with_normals(mesh, np.asarray(point_ras, dtype=float))
    return surf[0], normals[0]


# -------------------------------------------------------------- tangent frames
def _tangent_frame(inward_normal, axis_hint):
    """Orthonormal ``(n, t1, t2)`` with ``t1`` = hint projected tangent."""
    n = _unit(inward_normal)
    hint = np.asarray(axis_hint, dtype=float)
    t1 = hint - float(hint @ n) * n
    if np.linalg.norm(t1) < 1e-6:
        # hint (near-)parallel to the normal: pick any stable tangent
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(helper @ n)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n, helper)
    t1 = _unit(t1)
    t2 = _unit(np.cross(n, t1))
    return n, t1, t2


def _grid_offsets(kind):
    """(seed_uv (N,2), corner_uv (4,2)) in the (t1, t2) tangent plane, mm.

    Full tile: seeds at (+-5, +-5), corners at (+-10, +-10).
    Half tile: one seed column -- seeds at (0, +-5) along t2, corners at
    (+-5, +-10) (a 10 x 20 mm strip).
    """
    s = _SEED_HALF
    if kind == "full":
        seed_uv = np.array([[-s, -s], [-s, s], [s, -s], [s, s]], dtype=float)
        c = TILE_HALF_SIZE_MM
        corner_uv = np.array([[-c, -c], [c, -c], [c, c], [-c, c]], dtype=float)
    elif kind == "half":
        seed_uv = np.array([[0.0, -s], [0.0, s]], dtype=float)
        cu, cv = _SEED_HALF, TILE_HALF_SIZE_MM
        corner_uv = np.array([[-cu, -cv], [cu, -cv], [cu, cv], [-cu, cv]],
                             dtype=float)
    else:
        raise ValueError("kind must be 'full' or 'half', got %r" % (kind,))
    return seed_uv, corner_uv


def _project_to_wall(mesh, anchor, n, t1, t2, uv):
    """Press flat tangent-plane offsets ``uv (M,2)`` onto the wall.

    Physical model: the collagen tile is pressed flat against the wall along
    the seating direction, so each grid point sits at its nominal in-plane
    offset ``u*t1 + v*t2`` from the anchor and is cast onto the mesh **along
    the anchor normal** ``n`` (both ways; nearest hit wins).  A semi-rigid
    tile bridges concave dips instead of draping into them, which this
    reproduces -- and it is also what the phantom generator's radial
    re-projection does, so the two agree to well under a millimetre at tile
    scales.  Points whose ray misses (or only hits implausibly far away)
    fall back to the nearest point on the mesh.

    Returns ``(wall_points (M,3), local_inward_normals (M,3))``.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    anchor = np.asarray(anchor, dtype=float).reshape(3)
    flat = anchor[None, :] + uv @ np.vstack([t1, t2])
    m = flat.shape[0]

    wall = np.full((m, 3), np.nan)
    best = np.full(m, np.inf)
    try:
        origins = np.vstack([flat, flat])
        dirs = np.vstack([np.tile(n, (m, 1)), np.tile(-n, (m, 1))])
        locs, ray_idx, _tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs, multiple_hits=True
        )
        for loc, ri in zip(np.atleast_2d(locs), np.atleast_1d(ray_idx)):
            pi = int(ri) % m
            t_abs = float(np.linalg.norm(loc - flat[pi]))
            if t_abs < best[pi]:
                best[pi] = t_abs
                wall[pi] = loc
    except Exception:
        pass

    # reject hits far from the tangent plane (folds / far wall) and misses
    bad = ~np.isfinite(best) | (best > _MAX_SAG_MM)
    if bad.any():
        surf_fb, _ = _closest_with_normals(mesh, flat[bad])
        wall[bad] = surf_fb

    # smooth interior-oriented normals at the wall points
    _surf, loc_n = _closest_with_normals(mesh, wall)
    return wall, loc_n


def conform_tile(mesh, surface_point, inward_normal, axis_hint_ras, kind="full"):
    """Drape one tile onto the cavity wall around ``surface_point``.

    Mirrors the phantom generator's physics: each seed grid position and each
    tile corner is offset in the tangent plane, projected back onto the wall,
    then pulled ``SEED_WALL_OFFSET_MM`` off the wall INTO the cavity along the
    local (smooth, interior-oriented) normal.  Seed long axes are ``t1``
    projected onto the local tangent plane at each seed, so seeds twist with
    the wall exactly like the phantom's truth tiles do.
    """
    surface_point = np.asarray(surface_point, dtype=float).reshape(3)
    n, t1, t2 = _tangent_frame(inward_normal, axis_hint_ras)
    seed_uv, corner_uv = _grid_offsets(kind)

    uv = np.vstack([seed_uv, corner_uv])
    wall_pts, loc_n = _project_to_wall(mesh, surface_point, n, t1, t2, uv)
    conformed = wall_pts + SEED_WALL_OFFSET_MM * loc_n

    n_seeds = seed_uv.shape[0]
    seed_centers = conformed[:n_seeds]
    corners = conformed[n_seeds:]

    # Tangent-project the seed axes against the normal at each seed's OWN
    # nearest wall point (re-queried from the final 2 mm-offset position, not
    # the wall point it was draped from): in tight concave bumps those two
    # normals can differ by 15+ degrees, and the seed physically lies flat
    # against the wall it is nearest to.
    _surf_s, seed_n = _closest_with_normals(mesh, seed_centers)
    seed_axes = np.empty((n_seeds, 3))
    for i in range(n_seeds):
        ax = t1 - float(t1 @ seed_n[i]) * seed_n[i]
        if np.linalg.norm(ax) < 1e-6:
            ax = t2 - float(t2 @ seed_n[i]) * seed_n[i]
        seed_axes[i] = _unit(ax)

    return PlacedTile(
        kind=kind,
        center_ras=seed_centers.mean(axis=0),
        normal_ras=n,
        axis_ras=t1,
        seed_centers=seed_centers,
        seed_axes=seed_axes,
        corners_ras=corners,
        anchor_ras=surface_point,
    )


# ------------------------------------------------------------------- gestures
def translate_on_wall(mesh, tile, delta_ras):
    """Slide a tile along the wall: move it by ``delta``, re-snap, re-conform.

    The gesture moves the tile's wall anchor (its on-surface attachment
    point) so repeated small moves -- and a move followed by its inverse --
    do not accumulate drift from the seed centroid sitting 2 mm off the wall.
    """
    target = tile.anchor_ras + np.asarray(delta_ras, dtype=float).reshape(3)
    surf, n_in = snap_to_wall(mesh, target)
    return conform_tile(mesh, surf, n_in, tile.axis_ras, kind=tile.kind)


def _rodrigues(v, axis, angle):
    """Rotate ``v`` about unit ``axis`` by ``angle`` radians."""
    axis = _unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1.0 - c)


def rotate_on_wall(mesh, tile, angle_rad):
    """Spin a tile in place about its local wall normal.

    Rotates the tile's in-plane axis about ``tile.normal_ras`` (the frame the
    axis is exactly perpendicular to) and re-conforms at the same anchor, so
    a full 2*pi rotation is an exact identity.
    """
    hint = _rodrigues(tile.axis_ras, tile.normal_ras, float(angle_rad))
    return conform_tile(mesh, tile.anchor_ras, tile.normal_ras, hint,
                        kind=tile.kind)


def tiles_to_seed_arrays(tiles):
    """Stack all placed tiles' seeds into ``(centers (M,3), axes (M,3))``."""
    if not tiles:
        return np.zeros((0, 3)), np.zeros((0, 3))
    centers = np.vstack([t.seed_centers for t in tiles])
    axes = np.vstack([t.seed_axes for t in tiles])
    return centers, axes


# ------------------------------------------------------------------- export
def export_plan_csv(path, tiles, detected_centers=None, detected_axes=None,
                    rx_cgy=None):
    """Write every seed on the board to ``path`` as CSV; returns the count.

    Columns: ``source`` (detected/placed), ``tile`` (1-based placed tile
    number, blank for detected seeds), ``kind``, ``seed`` (index within the
    tile), RAS centre ``x_mm y_mm z_mm`` and unit long axis ``ax ay az``.
    A ``# rx_cgy=...`` comment line records the prescription when given.
    """
    import csv

    det_c = np.zeros((0, 3)) if detected_centers is None \
        else np.asarray(detected_centers, dtype=float).reshape(-1, 3)
    det_a = np.zeros((0, 3)) if detected_axes is None \
        else np.asarray(detected_axes, dtype=float).reshape(-1, 3)
    n = 0
    with open(path, "w", newline="") as fh:
        if rx_cgy is not None:
            fh.write("# rx_cgy=%.1f\n" % float(rx_cgy))
        w = csv.writer(fh)
        w.writerow(["source", "tile", "kind", "seed",
                    "x_mm", "y_mm", "z_mm", "ax", "ay", "az"])
        for i, (c, a) in enumerate(zip(det_c, det_a)):
            w.writerow(["detected", "", "", i] + ["%.3f" % v for v in c]
                       + ["%.4f" % v for v in a])
            n += 1
        for t_i, tile in enumerate(tiles):
            for s_i, (c, a) in enumerate(zip(tile.seed_centers, tile.seed_axes)):
                w.writerow(["placed", t_i + 1, tile.kind, s_i]
                           + ["%.3f" % v for v in c] + ["%.4f" % v for v in a])
                n += 1
    return n


# ------------------------------------------------------------ overlap detection
_OVERLAP_GRID_N = 7          # NxN footprint sampling grid
_OVERLAP_NORMAL_DOT = 0.5    # local normals must agree this much (same wall)


def _footprint_surface(tile, n_grid=_OVERLAP_GRID_N):
    """Sample a placed tile's conformed collagen footprint from its own fields.

    The physical footprint is the 20x20 mm (full) / 10x20 mm (half) collagen
    sheet draped on the wall.  ``find_overlapping_tiles`` gets no mesh, so the
    curved sheet is reconstructed from the conformed points the tile already
    carries: the 4 corners, the seed centres, and the anchor pushed
    ``SEED_WALL_OFFSET_MM`` along the anchor normal (all of which lie ON the
    conformed sheet, 2 mm off the wall).  In the anchor tangent frame
    ``(t1, t2, n)`` a quadratic height field ``z(u, v)`` is least-squares
    fitted through those 7-9 points -- a bilinear patch over the 4 corners
    alone is NOT good enough here: on this cavity's curvature the corners sag
    several mm below the tangent plane, so a bilinear centre would float mm
    off the real sheet, while the quadratic tracks it to well under 1 mm.

    Returns ``(points (n_grid**2, 3), normals (n_grid**2, 3))`` with normals
    unit and oriented INTO the cavity (same sense as ``tile.normal_ras``).
    """
    n = _unit(tile.normal_ras)
    t1 = _unit(tile.axis_ras - float(tile.axis_ras @ n) * n)
    t2 = _unit(np.cross(n, t1))
    anchor = tile.anchor_ras

    pts = np.vstack([
        tile.corners_ras,
        tile.seed_centers,
        anchor + SEED_WALL_OFFSET_MM * n,
    ])
    rel = pts - anchor[None, :]
    u = rel @ t1
    v = rel @ t2
    z = rel @ n

    # quadratic height field z(u, v) = c . [1, u, v, u^2, u*v, v^2], fitted in
    # coordinates scaled by the tile half-size for conditioning.  rcond=1e-6
    # (not None) is essential: at a symmetric placement the u^2 and v^2
    # columns coincide across every fit point (corners/seeds all at |u|=|v|),
    # and keeping the resulting near-zero singular value would amplify fp
    # noise into wildly curved surfaces; the cutoff drops it and lstsq falls
    # back to the benign minimum-norm fit.
    sc = TILE_HALF_SIZE_MM
    us, vs = u / sc, v / sc
    basis = np.column_stack(
        [np.ones_like(us), us, vs, us * us, us * vs, vs * vs]
    )
    coef, _res, _rank, _sv = np.linalg.lstsq(basis, z, rcond=1e-6)

    # sample the (u, v) domain as the bilinear span of the 4 corners' own
    # tangent coordinates (loop order c0, c1, c2, c3 -> bilinear corners
    # p00, p10, p11, p01), so a warped/rotated quad is covered exactly
    cu, cv = u[:4], v[:4]
    s = np.linspace(0.0, 1.0, n_grid)
    S, T = np.meshgrid(s, s, indexing="ij")
    S = S.reshape(-1)
    T = T.reshape(-1)
    w00, w10, w11, w01 = ((1 - S) * (1 - T), S * (1 - T), S * T, (1 - S) * T)
    gu = w00 * cu[0] + w10 * cu[1] + w11 * cu[2] + w01 * cu[3]
    gv = w00 * cv[0] + w10 * cv[1] + w11 * cv[2] + w01 * cv[3]

    gus, gvs = gu / sc, gv / sc
    gz = (coef[0] + coef[1] * gus + coef[2] * gvs
          + coef[3] * gus * gus + coef[4] * gus * gvs + coef[5] * gvs * gvs)
    points = (anchor[None, :] + gu[:, None] * t1[None, :]
              + gv[:, None] * t2[None, :] + gz[:, None] * n[None, :])

    # surface normal of z - f(u, v) = 0 is (n - f_u t1 - f_v t2), which keeps
    # a positive component along the inward anchor normal n
    fu = (coef[1] + 2.0 * coef[3] * gus + coef[4] * gvs) / sc
    fv = (coef[2] + coef[4] * gus + 2.0 * coef[5] * gvs) / sc
    normals = (n[None, :] - fu[:, None] * t1[None, :] - fv[:, None] * t2[None, :])
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
    return points, normals


def _grid_triangles(n_grid):
    """Vertex-index triples triangulating an ``n_grid x n_grid`` point grid."""
    a, b = np.meshgrid(np.arange(n_grid - 1), np.arange(n_grid - 1),
                       indexing="ij")
    i00 = (a * n_grid + b).reshape(-1)
    i10 = i00 + n_grid
    i01 = i00 + 1
    i11 = i10 + 1
    return np.vstack([
        np.column_stack([i00, i10, i01]),
        np.column_stack([i10, i11, i01]),
    ])


def _point_triangle_dist(P, tri):
    """Min distance from each point to each triangle: ``(p, t)`` array.

    Vectorized closest-point-on-triangle (Ericson, *Real-Time Collision
    Detection* 5.1.5) -- exact, no iteration, sizes here are ~49 x 72.
    """
    A = tri[None, :, 0, :]
    B = tri[None, :, 1, :]
    C = tri[None, :, 2, :]
    Pp = P[:, None, :]
    ab = B - A
    ac = C - A
    ap = Pp - A
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = Pp - B
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = Pp - C
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _safe_div(num, den):
        return num / np.where(np.abs(den) < 1e-30, 1e-30, den)

    # candidate closest points for every region, then pick by first-match
    close = np.empty(np.broadcast_shapes(Pp.shape, A.shape), dtype=float)
    done = np.zeros(d1.shape, dtype=bool)

    def _take(mask, value):
        m = mask & ~done
        if m.any():
            close[m] = np.broadcast_to(value, close.shape)[m]
            done[m] = True

    _take((d1 <= 0) & (d2 <= 0), A)                              # vertex A
    _take((d3 >= 0) & (d4 <= d3), B)                             # vertex B
    _take((d6 >= 0) & (d5 <= d6), C)                             # vertex C
    t_ab = _safe_div(d1, d1 - d3)[..., None]
    _take((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + t_ab * ab)      # edge AB
    t_ac = _safe_div(d2, d2 - d6)[..., None]
    _take((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + t_ac * ac)      # edge AC
    t_bc = _safe_div(d4 - d3, (d4 - d3) + (d5 - d6))[..., None]
    _take((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
          B + t_bc * (C - B))                                    # edge BC
    denom = _safe_div(np.ones_like(va), va + vb + vc)
    _take(np.ones_like(done),
          A + (vb * denom)[..., None] * ab + (vc * denom)[..., None] * ac)

    return np.linalg.norm(P[:, None, :] - close, axis=-1)


def _directed_hit(pts_a, nrm_a, pts_b, nrm_b, tris_b, tree_b, threshold,
                  slack):
    """True if a sample of A lies within ``threshold`` of B's footprint AND
    the local normals agree (same wall, not the opposite one).

    Two stages: sample-to-sample distances (an upper bound on the surface
    distance) accept clear overlaps outright; only samples in the ambiguous
    band ``(threshold, threshold + slack]`` -- within half a grid diagonal of
    a possible surface contact -- go through the exact point-to-triangle
    stage.  Returns ``None`` (instead of False) when A has no sample within
    ``threshold + slack`` of B's samples at all, letting the caller skip the
    reverse direction too.
    """
    d, idx = tree_b.query(pts_a, distance_upper_bound=threshold + slack)
    cand = np.isfinite(d)
    if not cand.any():
        return None
    dots = (nrm_a[cand] * nrm_b[idx[cand]]).sum(axis=1)
    agree = dots > _OVERLAP_NORMAL_DOT
    if bool((agree & (d[cand] <= threshold)).any()):
        return True                                    # definite overlap
    sub = np.flatnonzero(cand)[agree]
    if sub.size == 0:
        return False
    d_exact = _point_triangle_dist(pts_a[sub], pts_b[tris_b]).min(axis=1)
    return bool((d_exact <= threshold).any())


def find_overlapping_tiles(tiles, threshold_mm=1.0):
    """Detect pairs of placed tiles whose collagen footprints collide.

    Two tiles overlap when their physical footprints (20x20 mm full,
    10x20 mm half, conformed to the wall) intersect or come within
    ``threshold_mm`` of each other.  Tiles seated on OPPOSITE walls of a
    narrow cavity are not flagged: a close approach only counts where the
    local footprint normals agree (``dot > 0.5``), i.e. the tiles share a
    wall.  Runs on every drag step, so it is all vectorized numpy + one
    cKDTree per tile (well under 50 ms for 10 tiles).

    Parameters
    ----------
    tiles : sequence of PlacedTile
        Any mix of full and half tiles; empty and single-element sequences
        are fine.
    threshold_mm : float
        Footprints closer than this are flagged.  Edge-to-edge abutment
        (~1-3 mm real gap) is legal at the default 1.0.

    Returns
    -------
    list of (i, j) tuples with ``i < j``, indices into ``tiles``.
    """
    from scipy.spatial import cKDTree

    tiles = list(tiles)
    if len(tiles) < 2:
        return []
    threshold = float(threshold_mm)

    pts = []
    nrm = []
    trees = []
    centers = []
    radii = []
    for t in tiles:
        p, nn = _footprint_surface(t)
        pts.append(p)
        nrm.append(nn)
        trees.append(cKDTree(p))
        c = p.mean(axis=0)
        centers.append(c)
        radii.append(float(np.linalg.norm(p - c[None, :], axis=1).max()))
    tris = _grid_triangles(_OVERLAP_GRID_N)
    # samples are at most half a grid diagonal from the surface they discretize
    slack = float(max(
        np.linalg.norm(p[_OVERLAP_GRID_N + 1] - p[0]) for p in pts
    ))

    out = []
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            gap = float(np.linalg.norm(centers[i] - centers[j]))
            if gap > radii[i] + radii[j] + threshold:
                continue                           # bounding spheres clear
            hit = _directed_hit(pts[i], nrm[i], pts[j], nrm[j], tris,
                                trees[j], threshold, slack)
            if hit is None:
                continue                           # clouds clear even w/ slack
            if hit or _directed_hit(pts[j], nrm[j], pts[i], nrm[i], tris,
                                    trees[i], threshold, slack):
                out.append((i, j))
    return out
