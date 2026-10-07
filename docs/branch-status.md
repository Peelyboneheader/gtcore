# Branch status (snapshot 2026-10-07, optimizer overnight run)

Remote: https://github.com/Peelyboneheader/gtcore.git
Main branch lives in `C:\Users\jacob\OneDrive\Desktop\AlgorithmProject` (Desktop is OneDrive-synced on this machine).
Each feature branch is checked out in its own worktree under `C:\Users\jacob\OneDrive\Desktop\gt-worktrees\<name>`.

## Optimizer build (`gtcore.plan`, docs/plan-tile-optimize.md)

Baseline before the build: da71f76 (tile-selection work committed, 436 tests).
Interface freeze: cd73082. All branches below are fully merged into main;
the suite on main is 671 passed / 14 skipped (14 = interface stub checks for
wired functions + one `GT_E2E=1`-gated planner test).

| Branch | Worktree | Owner | State |
|---|---|---|---|
| plan/candidates | gt-worktrees/plan-candidates | A1 | merged (candidates, visible_faces angular broad phase, robust conflicts, tile-count rule) |
| plan/influence | gt-worktrees/plan-influence | A2 | merged (influence matrix, objective, §3 B gate) |
| plan/solvers | gt-worktrees/plan-solvers | A3 | merged (greedy, local, SA, E5, continuous multi-start, N-sweep) |
| plan/milp | gt-worktrees/plan-milp | A4 | merged (enumeration B&B reference, HiGHS MILP, LP bound, brute force) |
| plan/validation | gt-worktrees/plan-validation | A5 | merged (final_report, campaign script, quick-run results); full campaign results merged when its commit lands |
| plan/ui | gt-worktrees/plan-ui | A6 | merged (api, `gt optimize`, planner O/N keys with capacity-aware prompt) |
| plan/review | gt-worktrees/plan-review | A7 | merged (independent reference metrics + adversarial tests, 16/16 pass) |
| plan/scout-objective | gt-worktrees/plan-scout-objective | scout 7.1 | NOT merged by design; report + script copied into main (docs/scout-objective.md) |
| plan/scout-milp-bound | gt-worktrees/plan-scout-milp-bound | scout 7.3 | NOT merged by design; report + script copied (docs/scout-milp-bound.md) |
| plan/scout-continuous | gt-worktrees/plan-scout-continuous | scout 7.4 | NOT merged by design; report + script copied (docs/scout-continuous.md) |

Older feature branches (dose-engine, reconstruction, tile-autogen, tile-inference,
tile-selection, ui-interaction, validation) are all contained in main and can be
deleted with their worktrees (`git worktree remove <path>`, `git branch -d <name>`).
The plan/* worktrees can be removed the same way once the campaign commit is merged.

## To resume

- Test command: `C:\Users\jacob\.venvs\gammatile\Scripts\python.exe -m pytest -q -p no:cacheprovider` (about 7 min; run it alone — the off-screen VTK planner tests hang when several suites share the GPU).
- Open decision for Jacob: `interact._footprint_surface` rcond (docs/optimize-notes.md, "Open decisions"); the optimizer works around it with the geometric proxy.
- Validation campaign: `python scripts/validation_optimize.py` (3-3.5 h, needs ~11 GB free); `--quick` for a smoke run.
