# Branch status (snapshot 2026-09-01, evening)

Remote: https://github.com/Peelyboneheader/gtcore.git
Main branch lives in `C:\Users\jacob\OneDrive\Desktop\AlgorithmProject` (Desktop is OneDrive-synced on this machine, so "Desktop" and "OneDrive\Desktop" are the same folder).
Each feature branch is checked out in its own worktree under `C:\Users\jacob\OneDrive\Desktop\gt-worktrees\<name>`.
The worktrees were moved here from `C:\Users\jacob\gt-worktrees` and re-linked with `git worktree repair`.

| Branch | Worktree | vs main | State |
|---|---|---|---|
| main | AlgorithmProject | - | Integration branch. 419 tests passed at 32b20c2; dose-engine merged on top (eabe76d). Has uncommitted edits to tiles/auto.py, tiles/fit.py, planner.py, pipeline.py, cli.py, tests/test_tiles_suggest.py (tile-selection work in progress) |
| feature/tile-selection | gt-worktrees/tile-selection | 0 ahead, 0 behind | New, at main. No commits of its own yet |
| feature/dose-engine | gt-worktrees/dose-engine | 0 ahead, 2 behind | Merged into main at eabe76d (TG-43U1S2 default dataset, U1S1 interpolation, source verification). 424 tests passed on the branch |
| feature/tile-inference | gt-worktrees/tile-inference | 0 ahead, 2 behind | Fully merged. Tile-config inference + shadowing check |
| feature/tile-autogen | gt-worktrees/tile-autogen | 0 ahead, 7 behind | Fully merged. Steps 0-5: rigid tile model, Kabsch fit, count-free search, fold deformation, stick-to-surface fit, pipeline/CLI/planner integration |
| feature/ui-interaction | gt-worktrees/ui-interaction | 0 ahead, 14 behind | Fully merged. Planner UI v3 |
| feature/reconstruction | gt-worktrees/reconstruction | 0 ahead, 25 behind | Fully merged, stale. Original suite commit (169 tests) |
| feature/validation | gt-worktrees/validation | 0 ahead, 25 behind | Fully merged, stale. Same commit as reconstruction; never diverged |

Every feature branch is fully contained in main. The only live work is the uncommitted tile-selection edits in the main worktree.

## To resume

- Tile selection: the edits sit uncommitted in the main worktree. Either commit them on `feature/tile-selection` (from that worktree) or on main directly.
- All six older feature branches can be deleted along with their worktrees (`git worktree remove <path>` then `git branch -d <name>`), or kept as bookmarks.
- Test command: `C:\Users\jacob\.venvs\gammatile\Scripts\python.exe -m pytest -q` (about 3.5 min).
