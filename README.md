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
.\gt plan  <scan> --suggest         ... starting from the inferred tiles (planner key 'T')
.\gt plan  <scan> --tiles 6 --half 0 --seeds 24   ... telling it what the OR team knows (all optional)
.\gt optimize <scan> --tiles 8      opt-in placement optimizer (gtcore.plan): recommends a tile count, then
                                    places N tiles -> output\optimize_<timestamp>\ (JSON, CSV, seed CSV);
                                    --tiles omitted = the recommended count; --solver greedy|local|sa|milp|continuous
.\gt view                           either command, phantom mode (synthetic ground truth)
.\gt demo                           full phantom demo -> output\ (NRRD, PLY meshes, figures, CSV)
.\gt test                           run the test suite (424 tests)
```

Planner controls (the same legend is on screen; `?` collapses it):

| Group | Key / mouse | Action |
|---|---|---|
| Place | hover the blue wall | white **ghost tile** previews the next drop (red = would overlap) |
| | right-click or `P` | drop the tile there; `H` = next tile full / half |
| | `T` | **suggest tiles**: infer the implant configuration from the detected seeds (`gt plan --suggest` starts this way). With `--tiles N` the OR count is a trusted input; without it the count is inferred, **no half tiles are assumed** and every tile has 4 seeds. Gold outline = supported by the calibrated fit; **orange outline = tentative** (cover pass / triplet completion: verify it); a hollow orange sphere is a seed detection missed, inferred from its 3 tile-mates; **magenta seeds** are detected seeds inside the implant no tile explains (place one by hand). Suggestions are ordinary tiles afterwards, and they **own** the detected seeds they were built from: those seeds move with the tile (they are not drawn separately) and are counted once in dose and export; deleting the tile releases them |
| | `O` | **optimize placement** (opt-in `gtcore.plan`, see `docs/plan-tile-optimize.md`): first recommends a tile count from the wall area by the manufacturer's rule, treatable surface area / 4 cm² per tile rounded up (GammaTile Cavity Surface Area Calculator; on a measured post-resection mesh contraction defaults to 0 %), then prompts for N in the status line: digits / BackSpace edit, `H` edits the half-tile count, `M` flips the mode, `S` cycles the solver, Enter runs, Esc cancels. Mode **replace** (default): this session's hand-placed tiles are replaced (one undo step) while tiles fitted from the scan stay as fixed obstacles; **add** keeps everything on the board fixed. The result lands as ordinary tiles, **violet until touched** (drag, nudge or rotate); the status line reports the solver, wall time and V100/D90 on the +5 mm shell before/after. `Shift+O` (or `gt plan --optimizer greedy\|sa\|continuous`) cycles the solver; fixed tiles always use greedy. The planner builds its candidates at **h = 4 mm / 3 spins** (`planner.PLANNER_H_MM/_N_SPINS`, the validation grid) so a run stays interactive; the library default `gtcore.plan.DEFAULT_H_MM` = 2.5 mm / 6 spins takes minutes on a 1 mm mesh and remains the `gt optimize --h/--spins` default. The status line prints the candidate count and build time |
| | `N` | **suggest next tile**: one greedy step of the same optimizer given the current board, for the current full/half kind; drops the tile (violet) and reports the coverage gain |
| Adjust | left-drag on a tile | grab it by its quad or seed capsules (no modifier) and slide it along the wall; hovered tile lights up. On a scan with no cavity or shell mesh (e.g. a degraded export) tiles move rigidly in free space instead |
| | Ctrl + left-drag on the wall | slide the *selected* tile from anywhere (forgiving mode); a press that grabs nothing says so in the status bar |
| | gold outline | tile fitted **from the scan** (the implant); green = placed by hand. Backspace clears only hand-placed tiles |
| | `Tab`, arrows, `[` `]` | select next tile, nudge 2 mm, rotate 10 deg |
| | `X` / `Del`, `Backspace`, `Z` | delete tile, delete all hand-placed tiles, undo (50 steps) |
| Dose | `U` | TG-43 dose grid, 100/50/25 % isodoses (red/orange/yellow) and the **dose panel** (D90/D50/Dmin/V100/V150 on the cavity wall and +5/+10 mm tissue shells + shell DVH; flagged STALE when a tile moves) |
| | `+` `-` | prescription +/- 100 cGy, isodoses re-cut from the grid |
| | `A` | inter-seed attenuation on/off for the next `U` (capsule shadowing; carriers excluded, see `docs/interference-notes.md`); the panel also reports the cavity-wall area fraction receiving >= rx at 5 mm depth |
| | `I`, `C`, `D` | isodoses on/off, clear isodoses, dose panel on/off (also clickable buttons above the DVH chart) |
| Export | `S` | save every seed (detected + placed, RAS mm + axis) to `output/plan_<timestamp>.csv` |
| View | left/right/middle-drag, `R`, `G`, `B` | rotate (off tiles) / zoom / pan, reset camera, ghost preview on/off, background colour |

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
   the wall, attached / detached verdict). **Cover pass:** the calibrated
   selection is conservative (thin-cut gates, 3.5 penalty), so on coarse or
   gappy exports a second, explicitly *tentative* tier runs on the seeds it
   leaves inside the implant region: relaxed quads with spacing-scaled
   gates, then **triplet completion** (3 seeds forming an L of the tile
   square are completed to a full tile, the 4th seed inferred and flagged;
   1- and 3-seed tiles do not exist). What remains is reported as
   unassigned, never dropped silently. `ImplantPrior` carries the OR's
   counts when known (`fit_tiles_prior`); unknown means 0 half tiles.
   Geometry constants are cited in `gtcore.geometry` (seed plane 3.0 mm
   from the tissue face). Notes: `docs/autogen-notes.md`.
6. **`gtcore.dose`** — TG-43U1 line-source engine for IsoRay Proxcelan CS-1
   Rev2 (`engine.TG43Engine`, vectorized `compute_dose_grid` /
   `dose_at_points`, sub-voxel `isodose_surfaces`). The five physics defects
   of the AAPM-era port are fixed and regression-pinned
   (`docs/tg43-port-notes.md`). Two verified parameter sets: the
   **AAPM+GEC-ESTRO TG-43U1S2 consensus** (default: CON Λ = 1.056, CON L =
   0.40 cm, CON g_L / CON F from Tables AII/AXIII) and the **CLRP v2** MC
   dataset (`TG43Engine("clrp_v2")`), with TG-43U1S1 interpolation and
   extrapolation rules (bilinear F, log-linear g_L, exponential tail beyond
   10 cm). The capsule interior projects to the nearest surface point
   (continuous field) and a tabulated (ln r, θ) kernel makes 1 mm grids
   interactive (≤0.1 % vs the analytic rate outside the capsule). S_K per
   seed is an explicit parameter to be fed from the assay certificate, with
   `sk_decayed` / `delivered_fraction` for assay→implant decay and dose at a
   given time. `dose.metrics` adds the clinical readouts: DVH (D90/V100/
   V150/V200), the 5 mm cavity rind, and wall coverage at depth.
   `dose.InterferenceModel` adds what superposition cannot express — seeds
   and tile carriers attenuating each other along the line of sight
   (`docs/interference-notes.md`). It is **opt-in**
   (`compute_dose_grid(..., interference=model)`), so bare TG-43 stays the
   default and stays regression-pinned; seed capsules cost -0.3 to -0.7% of
   mean dose in the treated volume with 15% local shadows, while the collagen
   carrier term is off entirely because it rests on an unmeasured density.
   `dose.find_shadowing_tiles` turns the same physics into a planner check:
   which placed tiles stand in each other's line of fire at prescription
   depth, the dosimetric counterpart to `interact.find_overlapping_tiles`.
7. **`gtcore.interact` / `gtcore.planner`** — snap-to-wall + curvature
   conforming for placed tiles (pure geometry, unit-tested against phantom
   truth) and the interactive planner on top; `gtcore.dose.dvh` scores
   cavity-wall shells (offset outward into tissue) for the planner's dose
   panel.

`gtcore.pipeline.reconstruct(vol, n_full_tiles=..., n_half_tiles=...)` runs
1→5 in one call; `gtcore.phantom` provides the synthetic ground-truth head
(skull + craniotomy + lumpy cavity + wall-conformed tiles) that validates
every stage.

## Validation snapshot (2026-09-02 overnight run + dose-engine refinement + tile interference + planner UI v3; 424 tests green)

| Claim | Measured |
|---|---|
| Seed localization (0.7 mm phantom) | 12/12 seeds, mean 0.16 mm, max 0.36 mm |
| Brain / cavity segmentation | Dice 0.961 / 0.881 |
| Tile partition accuracy (sweeps, 120 configs) | 117/120 exact; failures = physically overlapping seeds (<1 mm gap) |
| Tile pose | centre ≤0.14 mm mean, normal ≤0.9° mean; fit <10 ms |
| TG-43 v2 vs independent quadrature | ≤1e-9 relative on G_L; tabulated kernel ≤1e-3 vs analytic (r ≥ 2.5 mm); grid 12 seeds/2 mm/100 mm in 0.17 s, 1 mm in 1.1 s |
| Isodose surface placement (log-dose marching cubes, 2 mm grid) | 0.04 mm rms, 0.09 mm max vs analytic isodose radius |
| Slice-spacing robustness (adaptive params) | recall 1.00 at 1.4/2.1/2.8 mm (fixed params: 0.58/0/0); figure `output/validation_spacing.png` |
| Inter-seed attenuation (12 seeds, capsules only) | mean -0.27% (flat grid) / -0.65% (conformed) at >=25% rx; worst voxel 0.84; +14-20% runtime |
| Tile shadowing flag (prescription depth, 5 mm) | coplanar tiles <0.1%, conformed phantom implant 0.56% (both quiet); a tile stacked 4 mm behind another 5.7% (flagged) |
| Tile-carrier term (unmeasured density) | swings +19% to 0% over rho 0.15->1.00 g/cm^3 — **off by default**; figure `output/validation_interference.png` |
| Real post-op CT (degraded export: 2 mm + gaps) | 4 complete tiles recovered, grid residuals 0.28–0.94 mm; cover pass adds 2 tentative tiles by triplet completion (1 seed each inferred) and reports 5 in-implant seeds unassigned (split-blob duplicates + singles), 26 far candidates as clutter |
| Physical 8-tile printed phantom (157 slices, 1 mm, O-MAR) | **32/32 seeds, 8/8 tiles**, residuals 0.32–1.38 mm; one physically crumpled tile recovered via the count constraint and flagged degraded |
| **Automatic tile creation, no count** (`scripts/validation_autogen.py`) | synthetic 54/60 exact (30/30 at 0.8 mm; the 6 misses at 1.2 mm are seed-detection misses the counted fit shares), centre 0.12 mm mean, normal 2.4°; printed phantom **8/8 incl. the crumpled tile** (0.46 mm rms, 288° fold, no count fallback); post-op cluster n = 4 by score saturation; **cover pass** (tentative tier, `tests/test_tiles_cover.py`): any single missed seed on the phantom is completed to a full tile (inferred seed 0.3–2.1 mm from truth), 1.4 mm thick-slice phantom 3/3 tiles (2 supported + 1 tentative), printed phantom unchanged at 8/8 with 0 tentative |

Per-dataset findings and data-quality caveats: `docs/data-notes.md`.

## Layout

```
gtcore/            algorithm core (pure numpy/scipy/scikit-image/SimpleITK/trimesh)
gtcore/viz.py      optional PyVista viewer   (only files allowed to render)
gtcore/planner.py  optional PyVista planner
gtcore/cli.py      the `gt` command
scripts/           demo + validation studies
tests/             424 tests, all stages scored against phantom ground truth
docs/              TG-43 physics notes, interference notes, data notes
output/            generated volumes, meshes, figures (gitignored)
```

## Conventions

- Voxel arrays `[k, j, i]`; `Volume.affine` maps `(i, j, k, 1)` → **RAS** mm.
- DICOM/SimpleITK (LPS) converted once, in `gtcore.io`.
- Python ≥3.9-compatible core; venv at `~\.venvs\gammatile`.
