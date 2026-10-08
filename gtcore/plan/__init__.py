"""Automated GammaTile placement optimizer (opt-in; ``gtcore.plan``).

Phase 0 interface freeze (see ``docs/plan-tile-optimize.md`` section 6).
This module defines the data contracts and the function signatures that
every ``plan/*`` branch builds against.  Dataclasses and their small
helpers are implemented here; every algorithmic function raises
``NotImplementedError`` naming the branch that implements it.  Nothing in
this subpackage is imported by the pipeline, the planner, or the engine.

Problem statement (``docs/plan-tile-optimize.md`` section 2, verbatim)
-----------------------------------------------------------------------

**Given**
- cavity wall mesh *S* (trimesh) and eligibility region *E ⊆ S* (default
  all of *S*; callers may pass an exclusion mask);
- target sample set *T* = {(p_m, w_m)}: points with area weights. Default:
  vertices of the +5 mm shell weighted by vertex area (one third of adjacent
  face areas). Alternative when an RTSTRUCT exists: voxel centres of the
  clinical HR-CTV, equal weights;
- optional OAR sample sets *O_j* with limits *L_j*;
- prescription *rx* (cGy); per-seed *S_K* and total-decay integration as in
  the engine;
- inventory: *N_full*, *N_half* (fixed-N form) or free (minimum-N form).

**Decision variables** per tile *i*: anchor *a_i ∈ E*, spin *θ_i* about the
local normal (full: *θ ∈ [0, 90°)* by the 2×2 symmetry; half:
*[0, 180°)*), kind ∈ {full, half}. Seed poses follow deterministically from
`conform_tile`.

**Dose** is additive over seeds: *D(p) = Σ_i Σ_s d(p; seed_{i,s})*, TG-43U1S2
line source in water. Interseed attenuation off during optimization; may be
on for final reporting with the difference stated.

**Metrics**
- *V100(T)* = Σ_m w_m·[D_m ≥ rx] / Σ_m w_m — primary.
- *D90(T)* = weighted 10th percentile of {D_m}.
- *V150(T), V200(T)* — hot-spot measures.
- *Dmax(O_j)* or *D0.1cc(O_j)*.
- *N*.

**Constraints:** no two tiles conflict (hard, §3 C); anchors in *E* (hard);
count = inventory (fixed-N).

**Problem forms**
- **P1 (fixed N):** maximize
  *V100 − λ_hot·max(0, V200 − v200_tol) − Σ_j λ_oar·max(0, Dmax(O_j) − L_j)*.
  Defaults λ_hot = 0.5, v200_tol = 0.10, λ_oar = 1e3 per cGy over limit.
  Every λ is a named, swept parameter.
- **P2 (minimum N):** smallest *N* with the P1 optimum satisfying
  *D90(T) ≥ rx* (also report the *V100 ≥ 0.90* criterion). Solved by
  sweeping P1 over *N* and reporting the full coverage-vs-N curve.

**Discretization.** Once anchors and spins are sampled, the problem is a
combinatorial selection with pairwise conflicts and a coverage objective,
which admits an exact MILP reference on reduced instances and therefore a
measured optimality gap. The discretization error is itself measured
(§4 V4).

Conventions
-----------
RAS mm everywhere; dose in cGy (total decay at nominal S_K unless stated);
areas in mm^2; angles in degrees; ``selection`` arguments are either an
integer id array into a :class:`CandidateSet` or a ``(C,)`` boolean mask;
deterministic outputs for identical inputs and seeds.  numpy / scipy /
trimesh only.
"""
from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from ..interact import PlacedTile

# --------------------------------------------------------------- constants
# Every tunable is a named constant with a one-line justification (section 5).
# They are collected again, with the same justifications, in the parameter
# table of docs/optimize-notes.md.

DEFAULT_H_MM = 2.5
# Anchor spacing on the wall: 1/8 of a tile side, so a tile can be re-sited
# in steps well below its own size without exploding the candidate count.

DEFAULT_N_SPINS_FULL = 6
# Full tile spins 0, 15, ..., 75 deg: the 2x2 seed grid has 90 deg symmetry.

DEFAULT_N_SPINS_HALF = 12
# Half tile spins 0, 15, ..., 165 deg: a 10 x 20 strip only has 180 deg symmetry.

DETACHED_MM = 1.5
# Reject a candidate if any conformed seed is more than this off its 3 mm
# wall offset (half the manufacturer's 2.25-3.75 mm hydrated spread).

DEFAULT_RX_CGY = 6000.0
# GammaTile prescription: 60 Gy to the 5 mm shell (STaRT trial convention).

TARGET_SHELL_OFFSET_MM = 5.0
# Default target: the +5 mm shell (the HR-CTV margin used in the clinic).

M_OPT_MAX = 4000
# Target points kept for optimization: C x M x 4 bytes stays under ~100 MB
# for C ~ 6000 candidates and an objective evaluation stays ~1 ms.

TAU_FRACTION = 0.05
# Soft-coverage sigmoid width tau = 0.05 * rx: moves that change a point's
# dose by a few percent of rx get a graded, non-zero gain for annealing.

LAMBDA_HOT = 0.5
# P1 hot-spot penalty per unit of V200 above tolerance (section 2 default).

V200_TOL = 0.10
# P1 tolerated V200 fraction before the hot-spot penalty applies (section 2).

LAMBDA_OAR = 1e3
LAMBDA_TAIL = 1.0
# Weight of the lower-tail coverage term (objective.tail_mean: weighted mean
# of min(D, rx) / rx over the coldest TAIL_Q of the target).  It pulls the
# optimizer towards the clinical endpoint D90 when N is too small for full
# coverage (V100 alone clusters tiles and leaves one cold patch; see
# docs/optimize-notes.md "Objective: lower-tail term").  0 restores pure V100.
TAIL_Q = 0.10
# P1 OAR penalty per cGy over the limit: any violation dominates coverage.

LOCAL_RADIUS_MM = 10.0
# Local-search re-siting radius: half a tile side, so moves stay "local".

SA_ALPHA = 0.95
# Geometric cooling factor per sweep (section 3 E3).

SA_MOVES_PER_TILE_PER_SWEEP = 50
# One sweep = 50 * N proposed moves, so each tile is visited ~50 times/sweep.

SA_N_SWEEPS = 40
# 40 sweeps at alpha = 0.95 cools T to ~13 % of T0, past the freeze point.

SA_N_RESTARTS = 3
# One greedy start plus two random feasible starts (section 3 E3).

MILP_TIME_LIMIT_S = 60.0
# HiGHS time limit per reduced instance; the bound, not the incumbent, is the
# reference when the limit is hit (section 3 E4).

TILE_AREA_CM2 = 4.0
# Manufacturer: a GammaTile is 2 x 2 cm = 4 cm^2 (tile-count rule, section 10).

CONFLICT_GAP_MM = 0.0
# Minimum edge gap between footprints beyond the planner's overlap rule.


# ---------------------------------------------------------------- helpers
def _as_ids(selection, n: int) -> np.ndarray:
    """Normalize ``selection`` (int ids or (n,) bool mask) to sorted unique ids."""
    sel = np.asarray(selection)
    if sel.size == 0:
        return np.zeros(0, dtype=int)
    if sel.dtype == bool:
        if sel.shape != (n,):
            raise ValueError("boolean selection must have shape (%d,)" % n)
        return np.flatnonzero(sel)
    ids = np.unique(sel.astype(int).reshape(-1))
    if ids.size and (ids.min() < 0 or ids.max() >= n):
        raise IndexError("selection id out of range [0, %d)" % n)
    return ids


def _jsonable(obj):
    """Recursively convert numpy / dataclass objects to JSON-serializable ones."""
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if sp.issparse(obj):
        return {"sparse_shape": list(obj.shape), "nnz": int(obj.nnz)}
    return obj


def _flatten_rows(obj) -> List[Tuple[str, Any]]:
    """Flat ``(key, value)`` rows: dicts recurse with dotted keys, scalars pass
    through, and lists / arrays become one JSON string so a row is one line."""
    rows: List[Tuple[str, Any]] = []

    def rec(o, key):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, (key + "." if key else "") + str(k))
        elif isinstance(o, (list, tuple)):
            rows.append((key, json.dumps(o)))
        else:
            rows.append((key, o))

    rec(obj, "")
    return rows


# ----------------------------------------------------------------- targets
@dataclass
class TargetSet:
    """Weighted target sample set *T*.

    Attributes
    ----------
    points : (M, 3) float
        Sample points, RAS mm.
    weights : (M,) float
        Area weights (mm^2 when built by :meth:`from_shell`; arbitrary
        positive scale otherwise -- only ratios matter).
    name : str
        Label for reports (e.g. ``"shell+5mm"``, ``"HR-CTV"``).
    """

    points: np.ndarray
    weights: np.ndarray
    name: str = "target"

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if self.weights.shape[0] != self.points.shape[0]:
            raise ValueError("weights must have one entry per point")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def total_weight(self) -> float:
        """Sum of the weights (mm^2 for shell targets)."""
        return float(self.weights.sum())

    @classmethod
    def from_shell(cls, mesh, offset_mm: float = TARGET_SHELL_OFFSET_MM,
                   name: Optional[str] = None) -> "TargetSet":
        """Vertices of ``shell_points(mesh, offset_mm)`` weighted by vertex area.

        The vertex area is one third of the areas of its adjacent faces on
        the OFFSET shell (the mesh faces carried onto the shell vertices),
        so the weights sum to the shell's area, not the wall's.
        """
        from ..dose.dvh import shell_points
        import trimesh

        pts = np.asarray(shell_points(mesh, float(offset_mm)), dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        shell = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
        area_faces = np.asarray(shell.area_faces, dtype=float)
        w = np.zeros(pts.shape[0], dtype=float)
        third = area_faces / 3.0
        for k in range(3):
            np.add.at(w, faces[:, k], third)
        if name is None:
            name = "shell%+gmm" % float(offset_mm)
        return cls(points=pts, weights=w, name=name)

    @classmethod
    def from_points(cls, points, weights=None, name: str = "points") -> "TargetSet":
        """Arbitrary points; equal weights (1.0 each) when ``weights`` is None."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if weights is None:
            w = np.ones(pts.shape[0], dtype=float)
        else:
            w = np.asarray(weights, dtype=float).reshape(-1)
        return cls(points=pts, weights=w, name=name)

    def subsample(self, m_max: int, rng_seed: int = 0
                  ) -> Tuple["TargetSet", np.ndarray]:
        """Deterministic weighted sampling without replacement, ``m_max`` points.

        Systematic (probability-proportional-to-size) sampling over the
        cumulative weights after a seeded permutation.  Points whose weight
        exceeds the sampling step are taken with certainty (they cannot be
        drawn twice), the rest are drawn systematically.  Sampled points get
        equal weights ``remaining_weight / n_drawn`` (certainty points keep
        theirs), so the subsample's ``total_weight`` equals the full set's and
        weighted fractions such as V100 are unbiased estimates.

        Returns ``(subset, index)`` with ``index`` (m,) int into ``self``;
        returns ``(self, arange(M))`` unchanged when ``M <= m_max``.
        """
        m_full = len(self)
        m_max = int(m_max)
        if m_full <= m_max:
            return self, np.arange(m_full)
        rng = np.random.default_rng(int(rng_seed))
        perm = rng.permutation(m_full)
        w = self.weights[perm]
        certain = np.zeros(m_full, dtype=bool)
        # peel off certainty units until no remaining weight exceeds the step
        while True:
            n_rest = m_max - int(certain.sum())
            rest_w = np.where(certain, 0.0, w)
            if n_rest <= 0 or rest_w.sum() <= 0:
                break
            step = rest_w.sum() / n_rest
            new = (~certain) & (w > step)
            if not new.any():
                break
            certain |= new
        n_rest = m_max - int(certain.sum())
        chosen = np.flatnonzero(certain)
        out_w = w[chosen].copy()
        if n_rest > 0:
            rest_idx = np.flatnonzero(~certain & (w > 0))
            rest_w = w[rest_idx]
            total_rest = float(rest_w.sum())
            step = total_rest / n_rest
            start = rng.uniform(0.0, step)
            ticks = start + step * np.arange(n_rest)
            cum = np.cumsum(rest_w)
            pick = np.searchsorted(cum, ticks, side="right")
            pick = np.minimum(pick, rest_idx.size - 1)
            pick = np.unique(pick)            # safety; weights <= step so unique
            drawn = rest_idx[pick]
            chosen = np.concatenate([chosen, drawn])
            out_w = np.concatenate([out_w, np.full(drawn.size,
                                                   total_rest / drawn.size)])
        order = np.argsort(perm[chosen], kind="stable")
        index = perm[chosen][order]
        out_w = out_w[order]
        sub = TargetSet(points=self.points[index], weights=out_w,
                        name=self.name)
        return sub, index


# -------------------------------------------------------------- candidates
@dataclass
class CandidateSet:
    """Discrete placement candidates (anchor, spin, kind), conformed to the wall.

    ``tiles`` (the :class:`~gtcore.interact.PlacedTile` objects) is the source
    of truth; the arrays are convenience views built from them.

    Attributes
    ----------
    anchors : (C, 3) float
        Wall anchor of each candidate, RAS mm (on the mesh).
    spins_deg : (C,) float
        Spin about the local inward normal, degrees (full: [0, 90); half: [0, 180)).
    kinds : (C,) object
        ``"full"`` or ``"half"``.
    n_seeds : (C,) int
        4 for full, 2 for half.
    seed_centers : (C, 4, 3) float
        Conformed seed centres, RAS mm; rows beyond ``n_seeds[c]`` are NaN.
    seed_axes : (C, 4, 3) float
        Unit seed long axes; NaN beyond ``n_seeds[c]``.
    corners : (C, 4, 3) float
        Conformed footprint corners (loop order), RAS mm.
    eligible : (C,) bool
        Anchor inside the eligibility region *E*.
    tiles : list of PlacedTile
        The conformed tiles, ``len == C``.
    method : str
        Anchor sampling method (e.g. ``"even"``, ``"farthest_point"``).
    h_mm : float
        Nominal anchor spacing, mm.
    n_spins : int
        Spins per anchor and kind.
    n_rejected : dict
        Rejection counts by reason (``"hanging"``, ``"detached"``, ``"ineligible"`` ...).
    wall_area_mm2 : float
        Area of the (eligible) wall the anchors were sampled on, mm^2.
    anchor_ids : (C,) int
        Index of the sampled anchor each candidate came from (same id =
        same anchor, different spin / kind).
    """

    anchors: np.ndarray
    spins_deg: np.ndarray
    kinds: np.ndarray
    n_seeds: np.ndarray
    seed_centers: np.ndarray
    seed_axes: np.ndarray
    corners: np.ndarray
    eligible: np.ndarray
    tiles: List[PlacedTile]
    method: str = "unknown"
    h_mm: float = DEFAULT_H_MM
    n_spins: int = DEFAULT_N_SPINS_FULL
    n_rejected: Dict[str, int] = field(default_factory=dict)
    wall_area_mm2: float = float("nan")
    anchor_ids: np.ndarray = None

    def __post_init__(self):
        c = len(self.tiles)
        self.anchors = np.asarray(self.anchors, dtype=float).reshape(c, 3)
        self.spins_deg = np.asarray(self.spins_deg, dtype=float).reshape(c)
        self.kinds = np.asarray(self.kinds, dtype=object).reshape(c)
        self.n_seeds = np.asarray(self.n_seeds, dtype=int).reshape(c)
        self.seed_centers = np.asarray(self.seed_centers, dtype=float).reshape(c, 4, 3)
        self.seed_axes = np.asarray(self.seed_axes, dtype=float).reshape(c, 4, 3)
        self.corners = np.asarray(self.corners, dtype=float).reshape(c, 4, 3)
        self.eligible = np.asarray(self.eligible, dtype=bool).reshape(c)
        if self.anchor_ids is None:
            self.anchor_ids = np.arange(c)
        self.anchor_ids = np.asarray(self.anchor_ids, dtype=int).reshape(c)

    @classmethod
    def from_tiles(cls, tiles: Sequence[PlacedTile], spins_deg, anchor_ids=None,
                   eligible=None, **meta) -> "CandidateSet":
        """Build the array views from conformed tiles (helper for builders)."""
        tiles = list(tiles)
        c = len(tiles)
        anchors = np.zeros((c, 3))
        kinds = np.empty(c, dtype=object)
        n_seeds = np.zeros(c, dtype=int)
        seed_centers = np.full((c, 4, 3), np.nan)
        seed_axes = np.full((c, 4, 3), np.nan)
        corners = np.zeros((c, 4, 3))
        for i, t in enumerate(tiles):
            anchors[i] = t.anchor_ras
            kinds[i] = t.kind
            k = int(t.seed_centers.shape[0])
            n_seeds[i] = k
            seed_centers[i, :k] = t.seed_centers
            seed_axes[i, :k] = t.seed_axes
            corners[i] = t.corners_ras
        if eligible is None:
            eligible = np.ones(c, dtype=bool)
        return cls(anchors=anchors, spins_deg=np.asarray(spins_deg, dtype=float),
                   kinds=kinds, n_seeds=n_seeds, seed_centers=seed_centers,
                   seed_axes=seed_axes, corners=corners, eligible=eligible,
                   tiles=tiles, anchor_ids=anchor_ids, **meta)

    def __len__(self) -> int:
        return len(self.tiles)

    def ids(self, selection) -> np.ndarray:
        """Sorted unique candidate ids for an id array or boolean mask."""
        return _as_ids(selection, len(self))

    def seeds_of(self, selection) -> Tuple[np.ndarray, np.ndarray]:
        """``(centers (k, 3), axes (k, 3))`` of the selected candidates' seeds.

        Candidates are taken in ascending id order (independent of the order
        ids appear in ``selection``); within a candidate, seeds keep the
        tile's own order.  Only the valid (non-NaN) seeds are included.
        """
        ids = self.ids(selection)
        if ids.size == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        centers = [self.seed_centers[i, :self.n_seeds[i]] for i in ids]
        axes = [self.seed_axes[i, :self.n_seeds[i]] for i in ids]
        return np.vstack(centers), np.vstack(axes)

    def tiles_of(self, selection) -> List[PlacedTile]:
        """The selected :class:`PlacedTile` objects in ascending id order."""
        return [self.tiles[int(i)] for i in self.ids(selection)]

    def subset(self, mask_or_ids) -> "CandidateSet":
        """A new CandidateSet restricted to ``mask_or_ids`` (ascending id order).

        ``anchor_ids`` are preserved (they index the original anchor sample).
        """
        ids = self.ids(mask_or_ids)
        return CandidateSet(
            anchors=self.anchors[ids], spins_deg=self.spins_deg[ids],
            kinds=self.kinds[ids], n_seeds=self.n_seeds[ids],
            seed_centers=self.seed_centers[ids], seed_axes=self.seed_axes[ids],
            corners=self.corners[ids], eligible=self.eligible[ids],
            tiles=[self.tiles[int(i)] for i in ids], method=self.method,
            h_mm=self.h_mm, n_spins=self.n_spins,
            n_rejected=dict(self.n_rejected), wall_area_mm2=self.wall_area_mm2,
            anchor_ids=self.anchor_ids[ids],
        )


# --------------------------------------------------------------- influence
@dataclass
class InfluenceMatrix:
    """Per-candidate dose at the target points, *D[c, m]*.

    Attributes
    ----------
    dose : (C, M) float32
        Total-decay dose [cGy] at nominal ``sk_per_seed_u`` from candidate
        ``c``'s seeds at target point ``m``.
    target : TargetSet
        The M points actually used (possibly a subsample of the full target).
    target_index : (M,) int
        Index of each used point into the full target set.
    rx_cgy : float
        Prescription [cGy] the matrix was built for (metrics reference).
    sk_per_seed_u : float
        Air-kerma strength per seed [U] used for ``dose``.
    oar : dict
        ``name -> (C, Mo) float32`` dose [cGy] at each OAR sample set.
    oar_limits : dict
        ``name -> limit`` [cGy] (Dmax limits).
    kernel : str
        ``"tabulated"`` or ``"exact"`` engine kernel.
    build_seconds : float
        Wall-clock time to build, s.
    """

    dose: np.ndarray
    target: TargetSet
    target_index: np.ndarray
    rx_cgy: float = DEFAULT_RX_CGY
    sk_per_seed_u: float = float("nan")
    oar: Dict[str, np.ndarray] = field(default_factory=dict)
    oar_limits: Dict[str, float] = field(default_factory=dict)
    kernel: str = "tabulated"
    build_seconds: float = 0.0

    def __post_init__(self):
        self.dose = np.asarray(self.dose, dtype=np.float32)
        if self.dose.ndim != 2:
            raise ValueError("dose must be (C, M)")
        self.target_index = np.asarray(self.target_index, dtype=int).reshape(-1)
        if self.dose.shape[1] != len(self.target):
            raise ValueError("dose columns must match the target points")
        if self.target_index.shape[0] != self.dose.shape[1]:
            raise ValueError("target_index must have M entries")

    @property
    def n_candidates(self) -> int:
        return int(self.dose.shape[0])

    @property
    def n_targets(self) -> int:
        return int(self.dose.shape[1])

    def dose_of(self, selection) -> np.ndarray:
        """Total dose ``(M,)`` float64 [cGy] of a selection (``dose[sel].sum(0)``)."""
        ids = _as_ids(selection, self.n_candidates)
        if ids.size == 0:
            return np.zeros(self.n_targets, dtype=np.float64)
        return self.dose[ids].astype(np.float64).sum(axis=0)

    def oar_dose_of(self, name: str, selection) -> np.ndarray:
        """Total dose ``(Mo,)`` float64 [cGy] at OAR ``name`` for a selection."""
        mat = self.oar[name]
        ids = _as_ids(selection, self.n_candidates)
        if ids.size == 0:
            return np.zeros(mat.shape[1], dtype=np.float64)
        return np.asarray(mat[ids], dtype=np.float64).sum(axis=0)


# --------------------------------------------------------------- conflicts
@dataclass
class ConflictGraph:
    """Pairwise footprint conflicts plus clique constraints.

    Attributes
    ----------
    n : int
        Number of candidates C.
    pairs : scipy.sparse.csr_matrix (C, C) bool
        Symmetric; ``pairs[i, j]`` True when candidates i and j conflict
        (footprints overlap as :func:`gtcore.interact.find_overlapping_tiles`
        defines it, plus ``gap_mm``).  Diagonal False.
    cliques : list of (k,) int arrays
        Candidate id sets of which at most one may be selected.
    gap_mm : float
        Minimum edge gap used, mm.
    """

    n: int
    pairs: Any
    cliques: List[np.ndarray] = field(default_factory=list)
    gap_mm: float = CONFLICT_GAP_MM

    def __post_init__(self):
        self.n = int(self.n)
        self.pairs = sp.csr_matrix(self.pairs, dtype=bool)
        if self.pairs.shape != (self.n, self.n):
            raise ValueError("pairs must be (n, n)")
        self.cliques = [np.asarray(q, dtype=int).reshape(-1) for q in self.cliques]

    def conflicts(self, i: int, j: int) -> bool:
        """True when candidates ``i`` and ``j`` may not both be selected."""
        return bool(self.pairs[int(i), int(j)])

    def neighbors(self, i: int) -> np.ndarray:
        """Sorted ids of the candidates conflicting with ``i``."""
        return np.sort(self.pairs.indices[self.pairs.indptr[int(i)]:
                                          self.pairs.indptr[int(i) + 1]])

    def is_feasible(self, selection) -> bool:
        """True when no two selected candidates conflict."""
        ids = _as_ids(selection, self.n)
        if ids.size < 2:
            return True
        sub = self.pairs[ids][:, ids]
        return int(sub.count_nonzero()) == 0

    def compatible_mask(self, selection) -> np.ndarray:
        """``(C,)`` bool: candidates that conflict with no selected candidate.

        Selected candidates themselves are False.
        """
        ids = _as_ids(selection, self.n)
        mask = np.ones(self.n, dtype=bool)
        if ids.size:
            hit = np.asarray(self.pairs[ids].sum(axis=0)).reshape(-1) > 0
            mask &= ~hit
            mask[ids] = False
        return mask

    def count_pairs(self) -> int:
        """Number of unordered conflicting pairs."""
        return int(self.pairs.count_nonzero() // 2)


# --------------------------------------------------------------- objective
@dataclass
class Objective:
    """P1 objective over selections of an :class:`InfluenceMatrix`.

    Attributes
    ----------
    influence : InfluenceMatrix
    conflicts : ConflictGraph
    rx_cgy : float
        Prescription [cGy].
    lambda_hot : float
        Penalty per unit V200 above ``v200_tol``.
    v200_tol : float
        Tolerated V200 (fraction of target weight).
    lambda_oar : float
        Penalty per cGy of OAR Dmax above its limit.
    tau_cgy : float
        Soft-coverage sigmoid width [cGy]; default ``TAU_FRACTION * rx_cgy``.
    lambda_tail : float
        Weight of the lower-tail coverage term ``tail_mean`` (weighted mean
        of ``min(D, rx) / rx`` over the coldest ``tail_q`` of the target
        weight, in [0, 1]; a lower bound on ``min(D90, rx) / rx``).  Added
        to both ``hard`` and ``soft``; 0 restores the pure V100 objective.
    tail_q : float
        Tail fraction of the target weight (0.10 pairs with D90).
    """

    influence: InfluenceMatrix
    conflicts: ConflictGraph
    rx_cgy: float = DEFAULT_RX_CGY
    lambda_hot: float = LAMBDA_HOT
    v200_tol: float = V200_TOL
    lambda_oar: float = LAMBDA_OAR
    tau_cgy: Optional[float] = None
    lambda_tail: float = LAMBDA_TAIL
    tail_q: float = TAIL_Q

    def __post_init__(self):
        self.rx_cgy = float(self.rx_cgy)
        if self.tau_cgy is None:
            self.tau_cgy = TAU_FRACTION * self.rx_cgy
        self.tau_cgy = float(self.tau_cgy)
        self.lambda_tail = float(self.lambda_tail)
        self.tail_q = float(self.tail_q)
        if not (0.0 < self.tail_q <= 1.0):
            raise ValueError("tail_q must lie in (0, 1], got %r" % (self.tail_q,))

    def dose_of(self, selection) -> np.ndarray:
        """Dose ``(M,)`` float64 [cGy] at the influence target for ``selection``.

        O(N * M).  Implemented on branch plan/influence.
        """
        from .objective import dose_of
        return dose_of(self, selection)

    def metrics(self, selection) -> Dict[str, float]:
        """Weighted target metrics for ``selection``.

        Keys: ``V100``, ``V150``, ``V200`` (weighted fractions of the target
        at >= 100/150/200 % rx), ``D90`` (weighted 10th percentile [cGy]),
        ``Dmean`` (weighted mean [cGy]) and ``oar_dmax_<name>`` [cGy] for
        every OAR set.  O(N * M + M log M).  Implemented on branch
        plan/influence.
        """
        from .objective import metrics
        return metrics(self, selection)

    def hard(self, selection) -> float:
        """P1 objective: ``V100 + lambda_tail * tail_mean - lambda_hot *
        max(0, V200 - v200_tol) - sum_j lambda_oar * max(0, Dmax(O_j) - L_j)``
        (dimensionless; the OAR term is per cGy).  Implemented on branch
        plan/influence; the tail term was added 2026-10-08.
        """
        from .objective import hard
        return hard(self, selection)

    def soft(self, selection) -> float:
        """Smooth surrogate for annealing: ``sum_m w_m sigma((D_m - rx) / tau)
        / sum_m w_m`` plus the same ``lambda_tail * tail_mean`` term and minus
        the same hot-spot / OAR penalties as :meth:`hard`.  Implemented on
        branch plan/influence.
        """
        from .objective import soft
        return soft(self, selection)

    def gain(self, selection, candidate: int) -> float:
        """``hard(selection + [candidate]) - hard(selection)`` without
        recomputing the base dose (incremental, O(M)).  Implemented on branch
        plan/influence.
        """
        from .objective import gain
        return gain(self, selection, candidate)

    def gains_all(self, selection, dose_vec=None) -> np.ndarray:
        """Hard gain of adding EACH candidate to ``selection``, ``(C,)``;
        vectorized fast path for greedy (module function on plan/influence,
        bound here so solvers find it via ``getattr``)."""
        from .objective import gains_all
        return gains_all(self, selection, dose_vec)

    def soft_gains_all(self, selection, dose_vec=None) -> np.ndarray:
        """Soft-objective counterpart of :meth:`gains_all`, ``(C,)``."""
        from .objective import soft_gains_all
        return soft_gains_all(self, selection, dose_vec)


# ----------------------------------------------------------------- results
@dataclass
class SolverResult:
    """Outcome of one solver run.

    Attributes
    ----------
    selection : (N,) int
        Selected candidate ids (ascending).
    objective : float
        Hard P1 objective of ``selection``.
    metrics : dict
        :meth:`Objective.metrics` of ``selection``.
    history : list of (step, value)
        Best-so-far hard objective per iteration / sweep.
    runtime_s : float
    solver : str
        ``"greedy"``, ``"local"``, ``"sa"``, ``"milp"`` ...
    seed : int or None
        RNG seed (stochastic solvers).
    status : str
        ``"ok"``, ``"infeasible"``, ``"time_limit"``, ``"error"`` ...
    reason : str
        Human-readable detail (mandatory when ``status != "ok"``).
    bound : float or None
        MILP upper bound on the objective (coverage only), if any.
    mip_gap : float or None
        Relative MIP gap reported by the solver, if any.
    feasible : bool
        ``selection`` violates no conflict and has the requested size.
    extra : dict
        Solver-specific diagnostics (e.g. marginal-gain curve, restarts).
    """

    selection: np.ndarray
    objective: float = float("nan")
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Tuple[int, float]] = field(default_factory=list)
    runtime_s: float = 0.0
    solver: str = ""
    seed: Optional[int] = None
    status: str = "ok"
    reason: str = ""
    bound: Optional[float] = None
    mip_gap: Optional[float] = None
    feasible: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.selection = np.asarray(self.selection, dtype=int).reshape(-1)


@dataclass
class SweepResult:
    """Coverage-vs-N curve (P2).

    Attributes
    ----------
    rows : list of dict
        One per N: ``N, V100, D90, V150, V200, objective, runtime_s, solver``
        plus ``oar_dmax_<name>`` for every OAR.
    min_n : dict
        Criterion -> smallest N satisfying it (None if never), criteria
        ``"D90>=rx"`` and ``"V100>=0.90"``.
    results : list of SolverResult
        The per-N solver results, in N order.
    """

    rows: List[Dict[str, Any]] = field(default_factory=list)
    min_n: Dict[str, Optional[int]] = field(default_factory=dict)
    results: List[SolverResult] = field(default_factory=list)


@dataclass
class TileCountRecommendation:
    """Manufacturer-rule tile count for a cavity (section 10).

    Attributes
    ----------
    n_tiles : int
        ``ceil(treatable_area_mm2 / (TILE_AREA_CM2 * 100))``.
    area_mm2 : float
        Measured (eligible) wall area, mm^2.
    treatable_area_mm2 : float
        ``area_mm2 * (1 - contraction_pct/100) * (1 - untreated_pct/100)``.
    contraction_pct : float
        Estimated surgical cavity contraction, % (0 for a post-resection scan).
    untreated_pct : float
        Estimated surface not requiring tiles, %.
    ellipsoid_area_mm2 : float
        Surface area of the ellipsoid with the cavity's principal diameters
        (the pre-operative calculator's input), mm^2.
    n_tiles_ellipsoid : int
        The same rule applied to ``ellipsoid_area_mm2``.
    diameters_mm : (3,) float
        Principal extents of the cavity, mm (descending).
    volume_mm3 : float or None
        Enclosed volume when the mesh is watertight.
    source : str
        Citation of the rule.
    """

    n_tiles: int
    area_mm2: float
    treatable_area_mm2: float
    contraction_pct: float
    untreated_pct: float
    ellipsoid_area_mm2: float
    n_tiles_ellipsoid: int
    diameters_mm: np.ndarray
    volume_mm3: Optional[float] = None
    source: str = ("GammaTile Cavity Surface Area Calculator "
                   "(gammatile.com/hcp/medical-physics/surface-area-calculator): "
                   "ellipsoid area minus contraction % and untreated %, "
                   "divided by 4 cm^2 per tile, rounded up")

    def __post_init__(self):
        self.diameters_mm = np.asarray(self.diameters_mm, dtype=float).reshape(3)

    def describe(self) -> str:
        """Multi-line human-readable summary."""
        d = self.diameters_mm
        lines = [
            "Recommended tiles: %d (mesh area %.0f mm^2, treatable %.0f mm^2 "
            "after %.0f%% contraction and %.0f%% untreated; %.0f mm^2 per tile)"
            % (self.n_tiles, self.area_mm2, self.treatable_area_mm2,
               self.contraction_pct, self.untreated_pct, TILE_AREA_CM2 * 100.0),
            "Ellipsoid estimate: %d tiles (diameters %.1f x %.1f x %.1f mm, "
            "area %.0f mm^2)" % (self.n_tiles_ellipsoid, d[0], d[1], d[2],
                                 self.ellipsoid_area_mm2),
        ]
        if self.volume_mm3 is not None:
            lines.append("Cavity volume: %.0f mm^3 (%.1f cc)"
                         % (self.volume_mm3, self.volume_mm3 / 1000.0))
        lines.append("Rule: " + self.source)
        return "\n".join(lines)


@dataclass
class OptimizeReport:
    """Everything reported for one configuration (section 3 G).

    Attributes
    ----------
    tiles : list of PlacedTile
        The configuration.
    solver : SolverResult or None
    parameters : dict
        Every named parameter used (h, n_spins, tau, lambdas, schedule ...).
    candidate_stats : dict
        Candidate counts, rejections, wall area, build time.
    metrics_influence : dict
        Metrics from the influence matrix (optimization-time estimate).
    metrics_grid : dict
        From ``compute_dose_grid`` + ``shell_report``: ``{offset_mm: dvh_stats}``
        (keys are the shell offsets as strings in JSON); the reported numbers.
    metrics_grid_interference : dict or None
        Same with the interseed-attenuation model on, if requested.
    overlaps : list of (i, j)
        ``find_overlapping_tiles`` on ``tiles`` (must be empty for optimizer output).
    shadowing : list
        ``find_shadowing_tiles`` output.
    runtime : dict
        Stage -> seconds.
    gtcore_version : str
    seed : int or None
    wall_clock_s : float
    notes : list of str
    """

    tiles: List[PlacedTile] = field(default_factory=list)
    solver: Optional[SolverResult] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    candidate_stats: Dict[str, Any] = field(default_factory=dict)
    metrics_influence: Dict[str, Any] = field(default_factory=dict)
    metrics_grid: Dict[Any, Any] = field(default_factory=dict)
    metrics_grid_interference: Optional[Dict[Any, Any]] = None
    overlaps: List[Tuple[int, int]] = field(default_factory=list)
    shadowing: List[Any] = field(default_factory=list)
    runtime: Dict[str, float] = field(default_factory=dict)
    gtcore_version: str = ""
    seed: Optional[int] = None
    wall_clock_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.gtcore_version:
            from .. import __version__
            self.gtcore_version = __version__

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (numpy converted, tiles as field dicts)."""
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d["tiles"] = [dataclasses.asdict(t) for t in self.tiles]
        d["n_tiles"] = len(self.tiles)
        return _jsonable(d)

    def to_json(self, path) -> None:
        """Write :meth:`to_dict` as indented JSON."""
        with open(str(path), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)

    def to_csv(self, path) -> None:
        """Write flat ``key,value`` rows (nested keys dotted; arrays as JSON)."""
        rows = _flatten_rows(self.to_dict())
        with open(str(path), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "value"])
            for k, v in rows:
                w.writerow([k, "" if v is None else v])

    def summary(self) -> str:
        """Short human-readable summary (tiles, solver, grid shell metrics)."""
        n_full = sum(1 for t in self.tiles if t.kind == "full")
        n_half = len(self.tiles) - n_full
        lines = ["%d tiles (%d full, %d half); gtcore %s; %.1f s"
                 % (len(self.tiles), n_full, n_half, self.gtcore_version,
                    self.wall_clock_s)]
        if self.solver is not None:
            s = self.solver
            lines.append("solver %s: objective %.4f, status %s%s"
                         % (s.solver, s.objective, s.status,
                            (" (" + s.reason + ")") if s.reason else ""))
        for off, stats in self.metrics_grid.items():
            st = stats.get("stats", stats) if isinstance(stats, dict) else stats
            try:
                lines.append("shell %s: V100 %.3f  D90 %.0f cGy  V150 %.3f  V200 %.3f"
                             % (off, st["V100"], st["D90"], st["V150"], st["V200"]))
            except (KeyError, TypeError):
                lines.append("shell %s: %r" % (off, st))
        if self.overlaps:
            lines.append("overlaps: %r" % (list(self.overlaps),))
        if self.shadowing:
            lines.append("shadowing: %d" % len(self.shadowing))
        for n in self.notes:
            lines.append("note: " + str(n))
        return "\n".join(lines)


# ------------------------------------------------------------ stub helpers
def _stub(name: str, branch: str):
    return NotImplementedError("%s: implemented on branch plan/%s" % (name, branch))


# --------------------------------------------------------------- functions
def build_candidates(mesh, h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
                     kinds: Sequence[str] = ("full",), eligible_faces=None,
                     rng_seed: int = 0, detached_mm: float = DETACHED_MM,
                     min_fraction_on_wall: Optional[float] = None) -> CandidateSet:
    """Sample anchors on ``mesh`` and conform one tile per (anchor, spin, kind).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Cavity wall, RAS mm (either winding).
    h_mm : float
        Target anchor spacing, mm (even surface / farthest-point sampling on
        a densified vertex set; the method is recorded in ``method``).
    n_spins : int or None
        Spins per anchor; None -> ``DEFAULT_N_SPINS_FULL`` for full,
        ``DEFAULT_N_SPINS_HALF`` for half (spin step 90/n or 180/n degrees).
    kinds : sequence of "full" / "half"
    eligible_faces : (F,) bool or None
        Face mask on ``mesh`` defining *E*; anchors on other faces are dropped
        (counted under ``n_rejected["ineligible"]``).  None = all faces.
    rng_seed : int
        Seed for the anchor sampling.
    detached_mm : float
        Reject when any conformed seed is further than this from its 3 mm
        wall offset.
    min_fraction_on_wall : float or None
        Reject when fewer than this fraction of the conformer's grid points
        hit the wall by ray casting (None -> the section 3 A rule: at most
        one fallback point).

    Returns
    -------
    CandidateSet with ``len == C`` accepted candidates; every rejection is
    counted in ``n_rejected``.  Complexity O(C) conformer calls, each a
    handful of ray casts and nearest-point queries.
    """
    from .candidates import build_candidates as _impl
    return _impl(mesh, h_mm=h_mm, n_spins=n_spins, kinds=kinds,
                 eligible_faces=eligible_faces, rng_seed=rng_seed,
                 detached_mm=detached_mm, min_fraction_on_wall=min_fraction_on_wall)


def visible_faces(mesh, center_ras) -> np.ndarray:
    """``(F,)`` bool: faces whose centroid is the FIRST hit of a ray from
    ``center_ras`` (RAS mm) toward it.

    Eligibility helper to isolate the inner wall of a closed shell mesh
    (e.g. the printed phantom's ``meshes["body"]``) from its outer surface.
    O(F) ray casts.
    """
    from .candidates import visible_faces as _impl
    return _impl(mesh, center_ras)


def build_influence(candidates: CandidateSet, target: TargetSet,
                    rx_cgy: float = DEFAULT_RX_CGY,
                    sk_per_seed_u: Optional[float] = None, m_opt: int = M_OPT_MAX,
                    oars: Optional[Dict[str, TargetSet]] = None,
                    oar_limits: Optional[Dict[str, float]] = None, engine=None,
                    kernel: str = "tabulated", rng_seed: int = 0) -> InfluenceMatrix:
    """Dose [cGy] from every candidate at every (subsampled) target point.

    ``target`` is subsampled to at most ``m_opt`` points (``TargetSet.subsample``
    with ``rng_seed``); ``sk_per_seed_u`` None -> ``TG43Engine.DEFAULT_SK_U``;
    ``kernel`` selects ``dose_at_points(exact=False)`` ("tabulated") or
    ``exact=True``.  OAR sets are evaluated in full (no subsampling).
    Complexity O(C * 4 * M) kernel evaluations, chunked; memory C*M*4 bytes.
    """
    from .influence import build_influence as _impl
    return _impl(candidates, target, rx_cgy=rx_cgy, sk_per_seed_u=sk_per_seed_u,
                 m_opt=m_opt, oars=oars, oar_limits=oar_limits, engine=engine,
                 kernel=kernel, rng_seed=rng_seed)


def build_conflicts(candidates: CandidateSet, gap_mm: float = CONFLICT_GAP_MM,
                    robust: bool = True) -> ConflictGraph:
    """Pairwise conflicts (as ``interact.find_overlapping_tiles`` defines
    overlap, plus ``gap_mm``) and anchor-neighbourhood cliques.

    ``robust=True`` (default) also adds the geometric proxy conflicts
    (``conflicts.tile_pair_proxy_conflict``: same wall, anchors closer than
    18 mm or seeds closer than 9 mm) that the planner's footprint fit can
    miss; ``robust=False`` is the planner rule verbatim.
    O(C^2) bounding-sphere tests, exact footprint tests only for close pairs.
    """
    from .conflicts import build_conflicts as _impl
    return _impl(candidates, gap_mm=gap_mm, robust=robust)


def tiles_conflict(tiles: Sequence[PlacedTile], robust: bool = True) -> List[Tuple[int, int]]:
    """Robust counterpart of ``interact.find_overlapping_tiles`` for any
    placed tiles: the planner's overlapping pairs unioned with the geometric
    proxy pairs (``robust=True``), or the planner's pairs alone.  Sorted
    ``(i, j)`` with ``i < j``; empty for fewer than two tiles.
    """
    from .conflicts import tiles_conflict as _impl
    return _impl(tiles, robust=robust)


def make_objective(influence: InfluenceMatrix, conflicts: ConflictGraph,
                   **weights) -> Objective:
    """Construct an :class:`Objective`; ``weights`` override ``lambda_hot``,
    ``v200_tol``, ``lambda_oar``, ``tau_cgy``, ``rx_cgy``."""
    from .objective import make_objective as _impl
    return _impl(influence, conflicts, **weights)


def evaluate(objective: Objective, selection) -> Dict[str, Any]:
    """``{"hard", "soft", "metrics", "feasible"}`` for ``selection``. O(N * M)."""
    from .objective import evaluate as _impl
    return _impl(objective, selection)


def solve_greedy(objective: Objective, n_tiles: int, fixed: Sequence[int] = (),
                 kinds_required: Optional[Dict[str, int]] = None, **kw) -> SolverResult:
    """E1 greedy forward selection: add the compatible candidate with the
    largest hard-objective gain until ``n_tiles`` (ties -> lowest id).

    ``fixed`` candidates are pre-selected and never removed;
    ``kinds_required`` (``{"full": n, "half": k}``) constrains the count per
    kind.  Deterministic.  ``extra["marginal_gain"]`` records the gain curve.
    Complexity O(N * C * M).  Fails with ``status="infeasible"`` when no
    compatible candidate remains.
    """
    from .solvers import solve_greedy as _impl
    return _impl(objective, n_tiles, fixed=fixed, kinds_required=kinds_required, **kw)


def solve_local(objective: Objective, n_tiles: int, start, radius_mm: float = LOCAL_RADIUS_MM,
                candidates: Optional[CandidateSet] = None) -> SolverResult:
    """E2 first-improvement local search from ``start`` (selection) over
    moves: re-site within ``radius_mm`` of the tile's anchor, re-spin in
    place (same ``anchor_id``), re-site anywhere.  Stops at a local optimum
    of the hard objective.  ``candidates`` supplies anchors / anchor_ids for
    the move neighbourhoods.  O(iterations * N * C * M) worst case.
    """
    from .solvers import solve_local as _impl
    return _impl(objective, n_tiles, start, radius_mm=radius_mm, candidates=candidates)


def solve_sa(objective: Objective, n_tiles: int, seed: int = 0,
             n_sweeps: int = SA_N_SWEEPS, n_restarts: int = SA_N_RESTARTS,
             start=None, candidates: Optional[CandidateSet] = None) -> SolverResult:
    """E3 simulated annealing on the soft objective; keeps the best HARD
    objective seen.  Move mix 60 % local re-site / 20 % spin / 20 % global;
    T0 from trial moves (~50 % uphill acceptance); geometric cooling
    ``SA_ALPHA`` per sweep of ``SA_MOVES_PER_TILE_PER_SWEEP * n_tiles`` moves;
    restarts = greedy start (or ``start``) + random feasible starts.
    Reproducible from ``seed``; ``history`` = best-so-far per sweep.
    O(n_restarts * n_sweeps * 50 * N * M).
    """
    from .solvers import solve_sa as _impl
    return _impl(objective, n_tiles, seed=seed, n_sweeps=n_sweeps, n_restarts=n_restarts,
                 start=start, candidates=candidates)


def solve_milp(objective: Objective, n_tiles: int,
               time_limit_s: float = MILP_TIME_LIMIT_S, exact_n: bool = True,
               mip_rel_gap: float = 1e-4, **kw) -> SolverResult:
    """E4 exact reference via ``scipy.optimize.milp`` (HiGHS): maximize
    weighted coverage ``sum_m w_m y_m`` s.t. ``D x >= rx y``, clique
    constraints, ``sum x = N`` (``exact_n``) or ``<= N``, OAR rows.
    Coverage only (no hot-spot term).  ``bound`` and ``mip_gap`` are always
    filled; ``status="time_limit"`` when the limit was hit (then ``bound`` is
    the reference, not ``selection``).  Exponential worst case; intended for
    reduced instances.
    """
    from .milp import solve_milp as _impl   # lazy: keeps gtcore.plan import light
    return _impl(objective, n_tiles, time_limit_s=time_limit_s, exact_n=exact_n,
                 mip_rel_gap=mip_rel_gap, **kw)


def solve_enumeration(objective: Objective, n_tiles: int,
                      time_limit_s: float = MILP_TIME_LIMIT_S, exact_n: bool = True,
                      **kw) -> SolverResult:
    """Exact reference by depth-first enumeration branch-and-bound over
    conflict-free N-subsets (the default ``solve_milp(method="auto")`` path
    for C <= 600, N <= 8; see ``gtcore.plan.milp``).  ``bound`` is a valid
    upper bound on V100 when the time limit is hit."""
    from .milp import solve_enumeration as _impl
    return _impl(objective, n_tiles, time_limit_s=time_limit_s, exact_n=exact_n, **kw)


def solve_continuous(mesh, candidates: CandidateSet, target: TargetSet, rx_cgy: float,
                     n_tiles: int, **kw):
    """Direct continuous multi-start solver over each tile's (u, v, theta)
    through ``conform_tile`` (promoted from the §7.4 scout).  Returns
    ``(List[PlacedTile], SolverResult)``; the tiles are continuous poses, not
    candidates.  See ``gtcore.plan.solvers.solve_continuous`` for the
    keywords (seed, n_starts, n_passes, time_budget_s, start, ...)."""
    from .solvers import solve_continuous as _impl
    return _impl(mesh, candidates, target, rx_cgy, n_tiles, **kw)


def refine_continuous(mesh, candidates: CandidateSet, selection, target: TargetSet,
                      rx_cgy: float = DEFAULT_RX_CGY, **kw) -> Tuple[List[PlacedTile], Dict[str, Any]]:
    """E5 continuous polish: Nelder-Mead over each selected tile's (u, v,
    theta) through ``conform_tile`` (as ``tiles.surface.fit_on_surface``
    parameterizes it), coordinate descent over tiles, conflicts re-checked
    with ``find_overlapping_tiles`` after each accepted step.

    Returns ``(tiles, info)`` with ``info`` holding the hard objective before
    and after (the discretization-error estimate, section 4 V4) and the
    number of accepted steps.  O(tiles * NM evaluations * conformer calls).
    """
    from .solvers import refine_continuous as _impl
    return _impl(mesh, candidates, selection, target, rx_cgy=rx_cgy, **kw)


def sweep_n(mesh, target: TargetSet, n_max: int, rx_cgy: float = DEFAULT_RX_CGY,
            solver: str = "sa", seed: int = 0,
            candidates: Optional[CandidateSet] = None,
            influence: Optional[InfluenceMatrix] = None,
            conflicts: Optional[ConflictGraph] = None, **kw) -> SweepResult:
    """P2: run P1 for N = 1 .. ``n_max`` with ``solver`` (warm-started from
    N-1) and report the coverage-vs-N curve plus the minimum N per criterion
    (``"D90>=rx"``, ``"V100>=0.90"``).  Builds candidates / influence /
    conflicts when not supplied.  O(n_max * solver cost).
    """
    from .sweep import sweep_n as _impl
    return _impl(mesh, target, n_max, rx_cgy=rx_cgy, solver=solver, seed=seed,
                 candidates=candidates, influence=influence, conflicts=conflicts, **kw)


def final_report(mesh, tiles: Sequence[PlacedTile], rx_cgy: float = DEFAULT_RX_CGY,
                 target: Optional[TargetSet] = None, cavity_mask=None,
                 cavity_affine=None, interference: bool = False,
                 grid_mm: float = 1.0, margin_mm: float = 50.0,
                 solver_result: Optional[SolverResult] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> OptimizeReport:
    """Section 3 G reporting for any configuration: ``compute_dose_grid``
    (exact kernel, ``grid_mm``, cavity bounds + ``margin_mm``), ``shell_report``
    on the full target shells, rind DVH when ``cavity_mask`` / ``cavity_affine``
    are given, optional second grid with the interference model, conflicts
    re-verified with ``find_overlapping_tiles``, shadowing via
    ``find_shadowing_tiles``.  O(grid voxels * seeds).
    """
    from .report import final_report as _impl
    return _impl(mesh, tiles, rx_cgy=rx_cgy, target=target, cavity_mask=cavity_mask,
                 cavity_affine=cavity_affine, interference=interference,
                 grid_mm=grid_mm, margin_mm=margin_mm, solver_result=solver_result,
                 parameters=parameters)


def recommend_tile_count(mesh, contraction_pct: float = 0.0, untreated_pct: float = 0.0,
                         eligible_faces=None) -> TileCountRecommendation:
    """Manufacturer-rule tile count (section 10): treatable area / 4 cm^2,
    rounded up, from the measured (eligible) mesh area; the ellipsoid
    estimate from the cavity's principal extents is reported alongside.
    O(F).
    """
    from .candidates import recommend_tile_count as _impl
    return _impl(mesh, contraction_pct=contraction_pct, untreated_pct=untreated_pct,
                 eligible_faces=eligible_faces)


def optimize(mesh, n_full: int, n_half: int = 0, rx_cgy: float = DEFAULT_RX_CGY,
             target: Optional[TargetSet] = None, solver: str = "sa", seed: int = 0,
             h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
             eligible_faces=None, oars: Optional[Dict[str, TargetSet]] = None,
             oar_limits: Optional[Dict[str, float]] = None, refine: bool = False,
             report: bool = True, verbose: bool = False, **kw
             ) -> Tuple[List[PlacedTile], OptimizeReport]:
    """Entry point (section 3 H): candidates -> influence -> conflicts ->
    ``solver`` (``"greedy"`` / ``"local"`` / ``"sa"`` / ``"milp"``) for
    ``n_full`` full and ``n_half`` half tiles -> optional E5 ``refine`` ->
    :func:`final_report` (when ``report``).  ``target`` None -> the +5 mm
    shell.  Returns ordinary :class:`PlacedTile` objects for the planner.
    Fails loudly (``ValueError`` with a reason) rather than returning a worse
    plan silently (section 4 V8).

    Implemented in :mod:`gtcore.plan.api` (branch plan/ui); extra keywords
    (``fixed_tiles``, ``candidates``, ``log``) are forwarded there.
    """
    from .api import optimize as _impl
    return _impl(mesh, n_full, n_half=n_half, rx_cgy=rx_cgy, target=target,
                 solver=solver, seed=seed, h_mm=h_mm, n_spins=n_spins,
                 eligible_faces=eligible_faces, oars=oars, oar_limits=oar_limits,
                 refine=refine, report=report, verbose=verbose, **kw)


def suggest_next(mesh, placed_tiles: Sequence[PlacedTile], rx_cgy: float = DEFAULT_RX_CGY,
                 target: Optional[TargetSet] = None, kind: str = "full",
                 h_mm: float = DEFAULT_H_MM, n_spins: Optional[int] = None,
                 eligible_faces=None, candidates: Optional[CandidateSet] = None,
                 **kw) -> Tuple[PlacedTile, Dict[str, Any]]:
    """One greedy step: the ``kind`` candidate compatible with ``placed_tiles``
    that most increases the hard objective.  Returns the tile and a dict with
    the objective before / after and the candidate id.  O(C * M).

    Implemented in :mod:`gtcore.plan.api` (branch plan/ui).
    """
    from .api import suggest_next as _impl
    return _impl(mesh, placed_tiles, rx_cgy=rx_cgy, target=target, kind=kind,
                 h_mm=h_mm, n_spins=n_spins, eligible_faces=eligible_faces,
                 candidates=candidates, **kw)


__all__ = [
    # constants
    "DEFAULT_H_MM", "DEFAULT_N_SPINS_FULL", "DEFAULT_N_SPINS_HALF", "DETACHED_MM",
    "DEFAULT_RX_CGY", "TARGET_SHELL_OFFSET_MM", "M_OPT_MAX", "TAU_FRACTION",
    "LAMBDA_HOT", "V200_TOL", "LAMBDA_OAR", "LAMBDA_TAIL", "TAIL_Q", "LOCAL_RADIUS_MM", "SA_ALPHA",
    "SA_MOVES_PER_TILE_PER_SWEEP", "SA_N_SWEEPS", "SA_N_RESTARTS",
    "MILP_TIME_LIMIT_S", "TILE_AREA_CM2", "CONFLICT_GAP_MM",
    # dataclasses
    "TargetSet", "CandidateSet", "InfluenceMatrix", "ConflictGraph", "Objective",
    "SolverResult", "SweepResult", "TileCountRecommendation", "OptimizeReport",
    # functions
    "build_candidates", "visible_faces", "build_influence", "build_conflicts",
    "tiles_conflict",
    "make_objective", "evaluate", "solve_greedy", "solve_local", "solve_sa",
    "solve_milp", "solve_enumeration", "solve_continuous", "refine_continuous",
    "sweep_n", "final_report", "recommend_tile_count", "optimize", "suggest_next",
]
