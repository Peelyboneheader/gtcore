"""Stub fixtures for ``gtcore.plan`` tests (section 6: A3-A6 develop against
these until the real candidate / influence modules land).

Not a test module itself; import from test files as ``import plan_fixtures``
(tests run from the repo root with ``tests/`` on ``sys.path`` via pytest's
rootdir conftest-less import mode).

Fixtures
--------
``flat_wall_mesh``      closed box whose TOP face (z = 0 plane) is the wall;
                        the box interior (z < 0) is the "cavity", so the
                        inward normal of the wall is -z and the +5 mm target
                        shell lies at z = +5 (outside the box, "tissue").
``flat_wall_top_faces`` face mask of that top face (an eligibility mask).
``sphere_cap_mesh``     closed icosphere (a whole spherical cavity); its
                        +5 mm shell has analytic radius ``r + 5``.
``analytic_point_dose`` toy inverse-square dose, engine independent.
``toy_instance``        complete fake (CandidateSet with real ``conform_tile``
                        tiles, InfluenceMatrix from the analytic dose,
                        ConflictGraph from the planner's overlap rule) so
                        solver tests do not wait for A1/A2.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import trimesh

from gtcore.interact import conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.plan import (
    DEFAULT_RX_CGY,
    CandidateSet,
    ConflictGraph,
    InfluenceMatrix,
    TargetSet,
)

ANALYTIC_RX_DISTANCE_MM = 8.0
# A seed 3 mm off the wall is 8 mm from the +5 mm shell point above it, so
# the toy dose is calibrated to give exactly rx there.


# ------------------------------------------------------------------ meshes
def flat_wall_mesh(size_mm: float = 60.0, step_mm: float = 2.0,
                   depth_mm: float = 40.0) -> trimesh.Trimesh:
    """Closed box ``[-size/2, size/2]^2 x [-depth, 0]`` with its top face
    (z = 0) triangulated at ``step_mm`` so ``conform_tile`` has triangles to
    hit.  Side walls and bottom are coarse but share the top's boundary
    vertices, so the mesh is watertight.  ``snap_to_wall`` orients normals
    toward the centroid (z = -depth/2), so the wall's inward normal is -z.
    """
    half = float(size_mm) / 2.0
    n = max(1, int(round(float(size_mm) / float(step_mm))))
    xs = np.linspace(-half, half, n + 1)

    verts = []
    faces = []

    def add_vertex(p):
        verts.append(tuple(float(v) for v in p))
        return len(verts) - 1

    # top grid (z = 0)
    top = np.empty((n + 1, n + 1), dtype=int)
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            top[j, i] = add_vertex((x, y, 0.0))
    for j in range(n):
        for i in range(n):
            a, b, c, d = top[j, i], top[j, i + 1], top[j + 1, i + 1], top[j + 1, i]
            faces.append((a, b, c))   # CCW seen from +z -> outward normal +z
            faces.append((a, c, d))

    # bottom grid boundary + centre (z = -depth); interior of bottom is a fan
    zb = -float(depth_mm)
    bot = np.full((n + 1, n + 1), -1, dtype=int)
    for j, y in enumerate(xs):
        for i, x in enumerate(xs):
            if j in (0, n) or i in (0, n):
                bot[j, i] = add_vertex((x, y, zb))
    centre = add_vertex((0.0, 0.0, zb))

    # boundary loop of the grid in CCW order (seen from +z)
    loop = []
    for i in range(n):
        loop.append((0, i))
    for j in range(n):
        loop.append((j, n))
    for i in range(n, 0, -1):
        loop.append((n, i))
    for j in range(n, 0, -1):
        loop.append((j, 0))

    for k in range(len(loop)):
        (j0, i0), (j1, i1) = loop[k], loop[(k + 1) % len(loop)]
        t0, t1 = top[j0, i0], top[j1, i1]
        b0, b1 = bot[j0, i0], bot[j1, i1]
        # side quad: outward normal must point away from the box axis
        faces.append((t1, t0, b0))
        faces.append((t1, b0, b1))
        # bottom fan: outward normal -z (CW seen from +z)
        faces.append((b0, centre, b1))

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=int), process=False)
    mesh.fix_normals()
    return mesh


def flat_wall_top_faces(mesh: trimesh.Trimesh, tol: float = 1e-6) -> np.ndarray:
    """``(F,)`` bool: faces lying in the z = 0 plane (the wall)."""
    tri = np.asarray(mesh.triangles, dtype=float)
    return np.all(np.abs(tri[:, :, 2]) < tol, axis=1)


def sphere_cap_mesh(radius_mm: float = 25.0, subdivisions: int = 4) -> trimesh.Trimesh:
    """Closed icosphere of radius ``radius_mm`` (a whole spherical cavity)."""
    return trimesh.creation.icosphere(subdivisions=int(subdivisions),
                                      radius=float(radius_mm))


# -------------------------------------------------------------------- dose
def analytic_point_dose(points, seed_centers, rx_scale: float = DEFAULT_RX_CGY
                        ) -> np.ndarray:
    """Toy inverse-square "dose" ``sum_s k / max(|p - s|, 1)^2`` [cGy-like].

    ``k`` is chosen so that a single seed ``ANALYTIC_RX_DISTANCE_MM`` (8 mm)
    away delivers exactly ``rx_scale``.  Engine independent; for solver
    tests only.  Returns ``(M,)`` float64.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    seeds = np.asarray(seed_centers, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0 or seeds.shape[0] == 0:
        return np.zeros(pts.shape[0], dtype=float)
    k = float(rx_scale) * ANALYTIC_RX_DISTANCE_MM ** 2
    d = np.linalg.norm(pts[:, None, :] - seeds[None, :, :], axis=2)
    d = np.maximum(d, 1.0)
    return (k / d ** 2).sum(axis=1)


# ---------------------------------------------------------------- instance
def toy_instance(n_candidates: int = 30, n_targets: int = 200, rng_seed: int = 0,
                 rx_cgy: float = DEFAULT_RX_CGY, pitch_mm: float = 10.0,
                 target_offset_mm: float = 5.0) -> Dict[str, object]:
    """A complete fake optimization instance on the flat wall.

    - ``candidates``: the first ``n_candidates`` anchors of a regular
      ``pitch_mm`` grid (row-major, centred) on the top face, spin 0 (axis
      hint +x), full tiles conformed with the real ``conform_tile``.  The
      wall is sized so every tile lies at least 10 mm inside the edge.
    - ``target``: ``n_targets`` seeded-uniform points on the +``target_offset_mm``
      shell (z = +offset) over the anchor region +- one tile, equal weights.
    - ``influence``: ``analytic_point_dose`` per candidate, float32.
    - ``conflicts``: the planner's own rule, ``find_overlapping_tiles``
      (threshold 1 mm) on the conformed tiles -- abutting tiles on a 10 mm
      grid therefore conflict; anchors >= 30 mm apart along an axis do not.
      Cliques: for each candidate, the candidates with anchor Chebyshev
      distance <= ``pitch_mm``, kept only if they form a true clique.

    Returns ``{"mesh", "candidates", "target", "influence", "conflicts",
    "rx_cgy", "anchor_grid"}``.
    """
    n_side = int(np.ceil(np.sqrt(n_candidates)))
    half_span = pitch_mm * (n_side - 1) / 2.0
    size = 2.0 * (half_span + 10.0 + 10.0)        # tile half-side + margin
    mesh = flat_wall_mesh(size_mm=size, step_mm=2.0, depth_mm=40.0)
    xs = -half_span + pitch_mm * np.arange(n_side)
    grid = np.array([(x, y) for y in xs for x in xs], dtype=float)[:n_candidates]

    tiles = []
    for x, y in grid:
        surf, n_in = snap_to_wall(mesh, np.array([x, y, 0.0]))
        tiles.append(conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]),
                                  kind="full"))
    cand = CandidateSet.from_tiles(
        tiles, spins_deg=np.zeros(len(tiles)), anchor_ids=np.arange(len(tiles)),
        method="toy_grid", h_mm=float(pitch_mm), n_spins=1,
        n_rejected={}, wall_area_mm2=float(size * size))

    rng = np.random.default_rng(int(rng_seed))
    lo, hi = -half_span - 10.0, half_span + 10.0
    pts = np.column_stack([
        rng.uniform(lo, hi, size=n_targets),
        rng.uniform(lo, hi, size=n_targets),
        np.full(n_targets, float(target_offset_mm)),
    ])
    target = TargetSet.from_points(pts, name="toy_shell+%gmm" % target_offset_mm)

    dose = np.empty((len(tiles), n_targets), dtype=np.float32)
    for c, t in enumerate(tiles):
        dose[c] = analytic_point_dose(target.points, t.seed_centers, rx_cgy)
    influence = InfluenceMatrix(dose=dose, target=target,
                                target_index=np.arange(n_targets), rx_cgy=rx_cgy,
                                sk_per_seed_u=float("nan"), kernel="analytic")

    c = len(tiles)
    pairs = sp.lil_matrix((c, c), dtype=bool)
    for i, j in find_overlapping_tiles(tiles, threshold_mm=1.0):
        pairs[i, j] = True
        pairs[j, i] = True
    pairs = pairs.tocsr()
    cliques = []
    for i in range(c):
        cheb = np.max(np.abs(grid - grid[i][None, :]), axis=1)
        q = np.flatnonzero(cheb <= pitch_mm + 1e-9)
        sub = pairs[q][:, q].toarray()
        if q.size >= 2 and np.all(sub[~np.eye(q.size, dtype=bool)]):
            cliques.append(q)
    conflicts = ConflictGraph(n=c, pairs=pairs, cliques=cliques, gap_mm=0.0)

    return {"mesh": mesh, "candidates": cand, "target": target,
            "influence": influence, "conflicts": conflicts, "rx_cgy": rx_cgy,
            "anchor_grid": grid}
