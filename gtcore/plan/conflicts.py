"""Conflict graph (section 3 C). Owner: A1 Geometry, branch ``plan/candidates``.

Must implement ``build_conflicts(candidates, gap_mm) -> ConflictGraph``:

- pairwise conflicts defined exactly as ``gtcore.interact.find_overlapping_tiles``
  defines overlap (call the same geometry on the two ``PlacedTile`` objects),
  plus the optional minimum edge gap ``gap_mm``;
- a sparse symmetric boolean matrix, diagonal False;
- clique constraints for the MILP: for every anchor neighbourhood
  (candidates whose anchors lie within one tile diagonal) "at most one of
  these" -- every listed clique must be a true clique of the pairwise graph
  (test: pairwise and clique forms agree).

Required tests: symmetry; a candidate conflicts with its own other spins;
two tiles with anchors > 30 mm apart on a flat wall never conflict.
"""
