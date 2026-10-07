"""Candidate generation (section 3 A). Owner: A1 Geometry, branch ``plan/candidates``.

Must implement, with the signatures frozen in ``gtcore.plan.__init__``:

- ``build_candidates(mesh, h_mm, n_spins, kinds, eligible_faces, rng_seed,
  detached_mm, min_fraction_on_wall) -> CandidateSet``: even / farthest-point
  anchor sampling at spacing ``h_mm`` on the eligible faces, the spin set
  (full: ``n_spins`` over [0, 90); half: over [0, 180)), ``snap_to_wall`` +
  ``conform_tile`` per (anchor, spin, kind), rejection of hanging (more than
  one ray-cast fallback) and detached (seed > ``DETACHED_MM`` off its 3 mm
  offset) tiles, rejection counts, ``anchor_ids`` so "same anchor, other
  spin" is recoverable.
- ``visible_faces(mesh, center_ras) -> (F,) bool``: first-hit faces from an
  interior point (inner wall of a closed shell mesh).
- ``recommend_tile_count(mesh, contraction_pct, untreated_pct,
  eligible_faces) -> TileCountRecommendation`` (section 10 manufacturer rule).

Read-only use of ``gtcore.interact``; never edit it.  Tests in
``tests/test_plan_candidates.py``.
"""
