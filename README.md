# IntraOp GammaTile — Algorithm Core (`gtcore`)

Post-implant brain CT → 3D reconstruction → Cs-131 seed localization →
tile-configuration inference → interactive replanning with TG-43U1 dosimetry.

Standalone pure-Python algorithm core: **no 3D Slicer, no host application.**
`slicer`/`vtk`/`qt` are imported nowhere in the algorithm; the optional 3D
front-end (`gtcore/viz.py`, `gtcore/planner.py`) uses PyVista and is the only
place allowed to touch a rendering stack.

## Quick start

From this folder (`gt.bat` wraps the project venv — no activation needed):

```
.\gt view  <dicom-folder-or-file>   load a CT, run the pipeline, open the 3D viewer
.\gt view  <scan> --tiles auto      ... and infer the tile configuration with NO implant count
.\gt plan  <dicom-folder-or-file>   pipeline + interactive tile planner (drag/drop + isodoses)
.\gt plan  <scan> --suggest         ... starting from the auto-inferred tiles (planner key 'g')
.\gt view                           either command, phantom mode (synthetic ground truth)
.\gt demo                           full phantom demo -> output\ (NRRD, PLY meshes, figures, CSV)
.\gt test                           run the test suite (206 tests)
```

Planner controls: pick on the cavity wall to drop a conformed tile (`h` = half
tile), `tab` select, arrows translate along the wall, `[` `]` rotate, `x`
delete, `g` = suggest tiles from the detected seeds (auto count),
`u` = recompute TG-43 dose, redraw 100/50/25% isodose shells and
report the cavity-wall area fraction receiving ≥ rx at 5 mm depth.

## Pipeline (and why the order matters)

1. **`gtcore.io`** — DICOM series → RAS-space `Volume` (LPS→RAS exactly once).
   Detects missing/irregular slices and rebuilds the volume on its TRUE z
   grid (SimpleITK's uniform assumption silently distorts z by up to the
   largest gap — observed at ~9 mm on a real export).
2. **`gtcore.seeds`** — seed-candidate detection runs FIRST: threshold →
   26-connected components → intensity-weighted subvoxel centroids → PCA long
   axes → merged-blob splitting (population-median volume + k-means).
   Detection parameters adapt to slice spacing (`seed_detection_params`):
   partial volume halves seed peak HU at 2 mm slices, and elongation is
   degenerate when a capsule spans one slice.
3. **`gtcore.preprocess`** — seed-scale metal inpainting (bloom removal) and
   isotropic resampling.
4. **`gtcore.segment`** — skull/brain (craniotomy sealed by escalating
   physical-radius closing), then resection cavity using the seed cloud as a
   spatial prior (tiles line the cavity wall by definition); marching-cubes
   surface meshes with outward normals.
5. **`gtcore.tiles`** — tile-configuration inference: quad/pair enumeration
   with deformation-tolerant gates, exact branch-and-bound assignment for the
   known implant count (full + sliced 2×1 half tiles), per-tile pose +
   residual; everything unassigned is rejected (clips, bone, streaks).
   **Automatic mode (`n_full_tiles="auto"`, no count):** rigid nominal
   tile model (`tiles.model`), developable bent-tile fit with the 3 mm
   seed-offset metric that explains wall-conformed *and* folded tiles
   (`tiles.deform`), count-free model selection with a per-tile penalty
   and a score-saturation curve (`tiles.auto`), and a stick-to-surface
   cross-check when a cavity mesh exists (`tiles.surface`: footprint on
   the wall, attached / detached verdict). Geometry constants are cited in
   `gtcore.geometry` (seed plane 3.0 mm from the tissue face). Notes:
   `docs/autogen-notes.md`.
6. **`gtcore.dose`** — TG-43U1 line-source engine for IsoRay Proxcelan CS-1
   Rev2 (`engine.TG43Engine`, vectorized `compute_dose_grid` /
   `dose_at_points`, sub-voxel `isodose_surfaces`). The five physics defects
   of the AAPM-era port are fixed and regression-pinned
   (`docs/tg43-port-notes.md`); the capsule interior projects to the nearest
   surface point (continuous field), the far field keeps falling beyond the
   10 cm data domain, and a tabulated (ln r, θ) kernel makes 1 mm grids
   interactive (≤0.1 % vs the analytic rate outside the capsule). S_K per
   seed is an explicit parameter to be fed from the assay certificate, with
   `sk_decayed` / `delivered_fraction` for assay→implant decay and dose at a
   given time. `dose.metrics` adds the clinical readouts: DVH (D90/V100/
   V150/V200), the 5 mm cavity rind, and wall coverage at depth.
7. **`gtcore.interact` / `gtcore.planner`** — snap-to-wall + curvature
   conforming for placed tiles (pure geometry, unit-tested against phantom
   truth) and the interactive planner on top.

`gtcore.pipeline.reconstruct(vol, n_full_tiles=..., n_half_tiles=...)` runs
1→5 in one call; `gtcore.phantom` provides the synthetic ground-truth head
(skull + craniotomy + lumpy cavity + wall-conformed tiles) that validates
every stage.

## Validation snapshot (2026-09-02 overnight run + dose-engine refinement; 206 tests green)

| Claim | Measured |
|---|---|
| Seed localization (0.7 mm phantom) | 12/12 seeds, mean 0.16 mm, max 0.36 mm |
| Brain / cavity segmentation | Dice 0.961 / 0.881 |
| Tile partition accuracy (sweeps, 120 configs) | 117/120 exact; failures = physically overlapping seeds (<1 mm gap) |
| Tile pose | centre ≤0.14 mm mean, normal ≤0.9° mean; fit <10 ms |
| TG-43 v2 vs independent quadrature | ≤1e-9 relative on G_L; tabulated kernel ≤1e-3 vs analytic (r ≥ 2.5 mm); grid 12 seeds/2 mm/100 mm in 0.17 s, 1 mm in 1.1 s |
| Isodose surface placement (log-dose marching cubes, 2 mm grid) | 0.04 mm rms, 0.09 mm max vs analytic isodose radius |
| Slice-spacing robustness (adaptive params) | recall 1.00 at 1.4/2.1/2.8 mm (fixed params: 0.58/0/0); figure `output/validation_spacing.png` |
| Real post-op CT (degraded export: 2 mm + gaps) | 4 complete tiles recovered, grid residuals 0.28–0.94 mm |
| Physical 8-tile printed phantom (157 slices, 1 mm, O-MAR) | **32/32 seeds, 8/8 tiles**, residuals 0.32–1.38 mm; one physically crumpled tile recovered via the count constraint and flagged degraded |
| **Automatic tile creation, no count** (`scripts/validation_autogen.py`) | synthetic 54/60 exact (30/30 at 0.8 mm; the 6 misses at 1.2 mm are seed-detection misses the counted fit shares), centre 0.12 mm mean, normal 2.4°; printed phantom **8/8 incl. the crumpled tile** (0.46 mm rms, 288° fold, no count fallback); post-op cluster n = 4 by score saturation |

Per-dataset findings and data-quality caveats: `docs/data-notes.md`.

## Layout

```
gtcore/            algorithm core (pure numpy/scipy/scikit-image/SimpleITK/trimesh)
gtcore/viz.py      optional PyVista viewer   (only files allowed to render)
gtcore/planner.py  optional PyVista planner
gtcore/cli.py      the `gt` command
scripts/           demo + validation studies
tests/             206 tests, all stages scored against phantom ground truth
docs/              TG-43 physics notes, data notes
output/            generated volumes, meshes, figures (gitignored)
```

## Conventions

- Voxel arrays `[k, j, i]`; `Volume.affine` maps `(i, j, k, 1)` → **RAS** mm.
- DICOM/SimpleITK (LPS) converted once, in `gtcore.io`.
- Python ≥3.9-compatible core; venv at `~\.venvs\gammatile`.
