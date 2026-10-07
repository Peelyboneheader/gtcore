# Scout 7.4 — does discretization error dominate? Discrete greedy + E5 vs direct continuous

Branch `plan/scout-continuous` (scout; evidence only, never merged). Script:
`scripts/scout_continuous.py`; raw numbers: `output/scout_continuous/results.csv`
(+ `log.txt`; `output/` is git-ignored, the table below is the committed copy).

## Question

How large is the discretization error of the candidate grid (anchor spacing
*h*, spin count) in the discrete tile-placement optimizer, and does a direct
continuous optimization over each tile's (u, v, θ) match or beat
"discrete greedy + E5 Nelder–Mead polish" at equal wall time?

## Method

Three synthetic cavities (`make_head_phantom(spacing=1.0, n_tiles=3,
rng_seed=k)`, k = 1..3; wall mesh = `mask_to_mesh(truth.masks["cavity"],
vol.affine)`), N = 6 full tiles, rx = 6000 cGy, eligibility = whole wall.
Target = +5 mm shell vertices (`shell_points`), subsampled to 1500 points
with probability ∝ vertex area (one third of the adjacent face areas),
equal weights (unbiased for the area-weighted metrics). Hard objective V100;
D90 logged. Dose rows via `dose_at_points(exact=False)` (tabulated kernel,
nominal S_K, total decay).

1. **Discrete greedy.** FPS anchors on the mesh vertices until spacing < *h*;
   spin set rotated about the local normal from a fixed global hint;
   `snap_to_wall` → `conform_tile`; reject candidates with any seed more than
   1.5 mm from its 3 mm offset. Forward greedy with lexicographic gain
   (hard V100, then soft coverage, then lowest id). Conflicts are
   `find_overlapping_tiles` on the pair, bounding-sphere prefiltered, evaluated
   lazily in gain order (first non-conflicting candidate wins). The 6-spin
   candidate set is built once per *h* and the 2-/3-spin subsets re-use it;
   a subset's build time is the FPS time plus its own candidates' conform +
   dose time (rejected candidates charged pro rata).
2. **E5 polish** of the h = 3 mm / 6-spin greedy: coordinate descent over
   tiles, Nelder–Mead on (u, v, θ) with `fit_on_surface`'s parameterization
   (anchor = surf0 + u·t1 + v·t2, re-snapped; hint = t1 rotated θ about n0),
   soft objective Σ w σ((D − rx)/τ), τ = 0.05 rx; the other tiles' dose rows
   cached; infeasible points (detached seeds, overlap with any other tile)
   cost +1e6; a tile's step accepted only if hard V100 does not drop and the
   soft objective improves. 2 passes.
3. **Direct continuous multi-start.** 3 random feasible starts (random
   vertex + random spin, rejection on detachment / overlap) + the same
   coordinate descent (first pass with a wider simplex), each start capped at
   one third of wall(1)+wall(2) at h = 3 / 6 spins; best-of-3 reported.
4. **FD gradient.** Same starts and cap; central differences on (u, v, θ)
   (0.5 mm, 0.5 mm, 2°), projected gradient ascent on the soft objective with
   backtracking (2 mm initial step, 5 halvings), same acceptance rule.

## Parameters

| parameter | value |
|---|---|
| cavities | rng_seed 1, 2, 3; spacing 1.0 mm |
| N tiles / kind | 6 / full |
| rx, τ | 6000 cGy, 300 cGy |
| target | +5 mm shell, 1500 pts, area-proportional sample |
| h (mm) | 4, 3, 2 |
| spins | 2 (0/45), 3 (0/30/60), 6 (0..75 step 15) |
| detached rejection | any seed \|d − 3 mm\| > 1.5 mm |
| overlap | `find_overlapping_tiles`, 1.0 mm threshold, sphere prefilter +2 mm |
| NM simplex (polish / global) | 1.5 mm, 10° / 4 mm, 20°; maxfev 100, xatol 0.05, fatol 1e-7 |
| E5 passes | 2 |
| continuous starts | 3, each capped at wall(1+2)/3 |
| FD | step 0.5 mm / 2°, θ scaled 10 mm/rad, step0 2 mm |
| match tolerance | 0.25 pp V100 |
| seeds | target rng 12345+k; starts rng 777+100k+start |

## Results

Cavity wall areas 41.8 / 45.6 / 44.4 cm^2 (seeds 1/2/3; the manufacturer's
4 cm^2 rule would call for 11-12 tiles, so N = 6 is deliberately
under-tiled and V100 is packing-limited at ~50 %). No candidate was rejected
as detached on any grid. Machine: this laptop, single process; total run
677 s.

### Results A: discrete greedy grid (V100 % / D90 cGy / build+greedy s)

| cavity | h mm | spins=2 | spins=3 | spins=6 |
|---|---|---|---|---|
| 1 | 4 | 52.67 / 3017.2 / 4.64 | 50.53 / 2880.1 / 6.78 | 50.27 / 2918.4 / 12.98 |
| 1 | 3 | 54.27 / 2993.9 / 8.68 | 50.87 / 2658.7 / 11.76 | 50.00 / 2749.3 / 23.5 |
| 1 | 2 | 49.47 / 2664.3 / 19.03 | 50.00 / 2811.8 / 28.64 | 50.20 / 2484.1 / 56.55 |
| 2 | 4 | 47.60 / 2544.3 / 6.29 | 46.73 / 2209.0 / 8.76 | 44.93 / 2242.5 / 18.03 |
| 2 | 3 | 43.13 / 2399.9 / 11.25 | 43.93 / 2068.8 / 17.13 | 46.13 / 1996.1 / 33.07 |
| 2 | 2 | 42.47 / 2375.0 / 23.23 | 48.00 / 2125.3 / 33.82 | 48.00 / 2125.3 / 67.51 |
| 3 | 4 | 43.33 / 2333.2 / 6.24 | 49.60 / 2330.6 / 9.37 | 48.47 / 2361.6 / 18.2 |
| 3 | 3 | 43.33 / 2333.2 / 11.27 | 49.33 / 2351.8 / 16.75 | 44.00 / 2368.4 / 33.18 |
| 3 | 2 | 45.27 / 2241.8 / 21.26 | 47.87 / 2465.4 / 31.59 | 46.93 / 2289.8 / 62.61 |

### Results B: discrete + E5 vs continuous (h = 3 mm / 6 spins reference; cap = build + greedy + E5)

| cavity | method | V100 % | D90 cGy | time s | notes |
|---|---|---|---|---|---|
| 1 | greedy h3/6 | 50.00 | 2749.3 | 23.5 | build 22.94 + greedy 0.56 |
| 1 | greedy+E5 (cap) | 52.67 | 2823.7 | 48.32 | E5 24.8s, 2 passes |
| 1 | random starts 0/1/2 | 44.87 / 47.93 / 49.20 | 4434.8 / 4044.2 / 4163.2 | 0 | before optimization |
| 1 | cont NM starts 0/1/2 | 58.20 / 59.33 / 56.40 | 3738.1 / 3939.6 / 4166.4 | 16.13 / 16.13 / 16.13 | 2 passes; 2 passes; 2 passes |
| 1 | cont FD starts 0/1/2 | 53.40 / 58.33 / 49.53 | 4552.1 / 3523.8 / 4146.1 | 5.93 / 4.65 / 4.03 | 6 sweeps; 5 sweeps; 3 sweeps |
| 1 | **cont NM best** | **59.33** | 3939.6 | 16.13 | best of 3 starts, total 48.4s cap 48.3s |
| 1 | **cont FD best** | **58.33** | 3523.8 | 4.65 | best of 3 starts, total 14.0s cap 48.3s |
| 2 | greedy h3/6 | 46.13 | 1996.1 | 33.07 | build 32.46 + greedy 0.61 |
| 2 | greedy+E5 (cap) | 48.87 | 2131.5 | 60.38 | E5 27.3s, 2 passes |
| 2 | random starts 0/1/2 | 37.67 / 36.73 / 33.67 | 3930.5 / 3940.4 / 3887.7 | 0 | before optimization |
| 2 | cont NM starts 0/1/2 | 51.47 / 44.27 / 41.67 | 3131.9 / 3605.4 / 3383.5 | 20.14 / 20.14 / 20.16 | 2 passes; 2 passes; 2 passes |
| 2 | cont FD starts 0/1/2 | 49.60 / 41.93 / 34.40 | 3269.3 / 3722.8 / 3856.6 | 19.48 / 8.98 / 4.72 | 12 sweeps; 6 sweeps; 3 sweeps |
| 2 | **cont NM best** | **51.47** | 3131.9 | 20.14 | best of 3 starts, total 60.4s cap 60.4s |
| 2 | **cont FD best** | **49.60** | 3269.3 | 19.48 | best of 3 starts, total 58.4s cap 60.4s |
| 3 | greedy h3/6 | 44.00 | 2368.4 | 33.18 | build 32.65 + greedy 0.52 |
| 3 | greedy+E5 (cap) | 47.53 | 2475.0 | 57.3 | E5 24.1s, 2 passes |
| 3 | random starts 0/1/2 | 37.53 / 41.40 / 49.80 | 4041.1 / 3806.4 / 3266.1 | 0 | before optimization |
| 3 | cont NM starts 0/1/2 | 43.00 / 45.60 / 57.33 | 3897.8 / 3648.4 / 2968.8 | 19.11 / 19.11 / 19.1 | 2 passes; 2 passes; 2 passes |
| 3 | cont FD starts 0/1/2 | 43.93 / 44.67 / 56.40 | 3742.2 / 3559.1 / 3105.3 | 8.52 / 7.88 / 10.01 | 7 sweeps; 6 sweeps; 6 sweeps |
| 3 | **cont NM best** | **57.33** | 2968.8 | 19.1 | best of 3 starts, total 57.3s cap 57.3s |
| 3 | **cont FD best** | **56.40** | 3105.3 | 10.01 | best of 3 starts, total 30.0s cap 57.3s |

## Verdict against the pre-declared criteria (plan 7.4)

| cavity | greedy h3/6 | greedy h2/6 | dh (2 vs 3 mm) | E5 gain | dominates? | cont NM best | cont FD best | vs discrete+E5 |
|---|---|---|---|---|---|---|---|---|
| 1 | 50.00 | 50.20 | +0.20 | **+2.67** | yes | 59.33 | 58.33 | beat / beat |
| 2 | 46.13 | 48.00 | +1.87 | **+2.73** | yes | 51.47 | 49.60 | beat / beat |
| 3 | 44.00 | 46.93 | +2.93 | **+3.53** | yes | 57.33 | 56.40 | beat / beat |

- **Discretization error dominates on 3/3 cavities**: the E5 gain
  (+2.7 to +3.5 pp) exceeds both the h = 3 -> 2 mm greedy difference and
  1 pp everywhere.
- **Continuous methods pass the promotion criterion on 3/3 cavities**
  (needed 2/3): multi-start Nelder-Mead beats discrete + E5 by +6.7, +2.6
  and +9.8 pp at equal (strictly capped) wall time; FD projected gradient
  by +5.7, +0.7 and +8.9 pp while using only 25-95 % of the cap (it stops at
  a local optimum).
- The discrete greedy is **non-monotone in h and spins** (cavity 1:
  h = 3 mm / 2 spins 54.3 > h = 2 mm / 6 spins 50.2; cavity 3: h = 4 / 3
  spins 49.6 > h = 2 / 6 spins 46.9). The grid-to-grid scatter of the
  greedy (2-6 pp) is as large as the E5 gain, so refining the grid buys
  nothing reliable at 2-3x the build cost; a finer grid is NOT the remedy.
- Random feasible starts alone score 34-50 %, i.e. within a few pp of
  greedy's 42-54 %: on an under-tiled cavity V100 is decided by how tightly
  the 6 tiles abut (a 1-2 mm gap drops the +5 mm shell below rx), which a
  3 mm grid cannot express and which the greedy cannot repair.
- Continuous multi-start has high start-to-start variance (cavity 2:
  41.7 / 44.3 / 51.5; cavity 3: 43.0 / 45.6 / 57.3) -> it needs several
  starts (or a greedy start among them); each start used only one third of
  the cap and only 2 coordinate-descent passes, so there is headroom.
- D90 of the continuous solutions is 600-1200 cGy higher than greedy's
  (the soft surrogate spreads dose); hot-spot terms were not scored.

**Not claimed.** The discrete solver here is greedy only (no SA/MILP), so
the 7.4 symptom "E5 gain > SA-vs-greedy difference" is untested; wall time
for the discrete arm includes the candidate build (conform_tile at ~7 ms
dominates), which a cached/vectorized conformer would shrink; only one N
(under-tiled) and three cavities; metrics from the tabulated kernel on the
1500-point subsample, not from a final dose grid.

## Recommendation

1. **Promote direct continuous optimization** over (u, v, theta) per tile
   (coordinate-descent Nelder-Mead through `conform_tile`, soft surrogate,
   hard-V100 acceptance) to a first-class solver in `gtcore.plan`, run as a
   multi-start (>= 3 random feasible starts **plus** the greedy solution as a
   start) and keep the best hard V100. The FD-gradient variant is a cheaper
   local polisher (converges in 5-20 s) but is not better than NM.
2. **E5 is worth its cost**: +2.7 to +3.5 pp V100 for ~25 s (2 passes).
   Any discrete solution should be polished before reporting.
3. **Default grid: h = 4 mm, 3 spins** (0/30/60) for the discrete start.
   With continuous polish/optimization behind it, the grid only has to
   supply a decent start; h = 4 / 3 spins is never worse than h = 3 / 6 spins
   by more than noise here (cavity 1: 50.5 vs 50.0; 2: 46.7 vs 46.1;
   3: 49.6 vs 44.0) and builds 3.5x faster (7-9 s vs 23-33 s). Do not
   default to h = 2 mm.
4. Re-test at N near the manufacturer count (11-12 tiles here) before
   fixing the defaults: when coverage saturates the grid may matter less.

## Reproduce

    cd <worktree root>
    C:/Users/jacob/.venvs/gammatile/Scripts/python.exe scripts/scout_continuous.py

Writes `output/scout_continuous/results.csv`, `log.txt`, cached
`cavity_seed{1,2,3}.ply`. Fixed seeds; total wall 677 s on this machine.
Commit that produced the numbers above: `9232124`.
