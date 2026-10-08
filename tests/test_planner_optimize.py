"""Planner + CLI integration of the opt-in placement optimizer (A6).

Headless: an off-screen ``_PlannerApp`` on the phantom scene (the fixture
``tests/test_planner_interaction.py`` uses), driven through the real VTK
key-event pipeline.  ``gtcore.plan.optimize`` / ``suggest_next`` /
``recommend_tile_count`` are monkeypatched with fakes that build tiles with
``conform_tile`` on the planner's own wall, so these tests do not depend on
the solver branches and keep passing once the real functions land.

Covered: the tile-count prompt state machine (digits, numpad, BackSpace,
Enter, Esc, H/M/S, no leakage into other bindings), optimizer tiles landing
as ordinary tiles on the undo stack with no overlap flagged (section 4 V1),
the violet tint dropped once touched, before/after in the status line, the
N key adding exactly one tile, replace vs add mode, graceful
NotImplementedError handling, Shift+O, and ``gt optimize`` argument parsing
plus a run on the synthetic phantom.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

pytest.importorskip("pyvista")

import gtcore.plan as plan
from gtcore.interact import conform_tile, find_overlapping_tiles, snap_to_wall
from gtcore.phantom import make_head_phantom
from gtcore.pipeline import reconstruct
from gtcore.plan import OptimizeReport, SolverResult, TileCountRecommendation
from gtcore.planner import (
    HELP_TEXT,
    OPTIMIZER_SOLVERS,
    _OPTIMIZED_COLOR,
    _OPTIMIZED_OUTLINE,
    _TILE_STYLE,
    _PlannerApp,
    wall_mesh_for,
)


@pytest.fixture(scope="module")
def result():
    vol, _truth = make_head_phantom(spacing=1.0)
    res = reconstruct(vol, verbose=False)
    if "cavity" not in res.meshes or not len(res.meshes["cavity"].vertices):
        pytest.skip("pipeline found no cavity on this phantom")
    return res


@pytest.fixture()
def app(result):
    try:
        planner = _PlannerApp(result, off_screen=True)
        planner.pl.render()
    except Exception as exc:  # headless CI without OpenGL etc.
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    yield planner
    planner.close()


# ------------------------------------------------------------------- fakes
def _spread_tiles(mesh, n, avoid=(), kind="full", n_probe=40):
    """``n`` wall-conformed, mutually non-overlapping tiles (also clear of
    ``avoid``), picked by farthest-point sampling of the mesh vertices."""
    verts = np.asarray(mesh.vertices, dtype=float)
    order = [0]
    d = np.linalg.norm(verts - verts[0], axis=1)
    for _ in range(1, min(n_probe, len(verts))):
        order.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(verts - verts[order[-1]], axis=1))
    kept = list(avoid)
    out = []
    for i in order:
        surf, n_in = snap_to_wall(mesh, verts[i])
        tile = conform_tile(mesh, surf, n_in, np.array([1.0, 0.0, 0.0]), kind=kind)
        if find_overlapping_tiles(kept + [tile], threshold_mm=1.0):
            continue
        kept.append(tile)
        out.append(tile)
        if len(out) == n:
            break
    if len(out) < n:
        raise ValueError("fake optimizer: only %d of %d tiles fit" % (len(out), n))
    return out


def _install_fakes(monkeypatch, calls, n_recommended=12, capacity=10):
    from gtcore.plan import api as plan_api

    def packing_capacity(mesh, rx_cgy=6000.0, target=None, kind="full", h_mm=2.5,
                         n_spins=None, eligible_faces=None, fixed_tiles=(), n_max=40,
                         candidates=None):
        calls.append(("capacity", len(list(fixed_tiles or ())), h_mm, n_spins))
        return capacity, {"seconds": 0.4, "candidates_s": 0.0, "n_candidates": 300,
                          "status": "infeasible", "at_least": False, "n_fixed": 0}

    monkeypatch.setattr(plan_api, "packing_capacity", packing_capacity)

    def optimize(mesh, n_full, n_half=0, **kw):
        calls.append(("optimize", n_full, n_half, dict(kw)))
        fixed = list(kw.get("fixed_tiles", ()) or ())
        tiles = _spread_tiles(mesh, n_full, avoid=fixed)
        if n_half:
            tiles += _spread_tiles(mesh, n_half, avoid=fixed + tiles, kind="half")
        res = SolverResult(selection=np.arange(len(tiles)), objective=0.9,
                           metrics={"V100": 0.9, "D90": 6100.0}, solver=kw.get("solver"))
        rep = OptimizeReport(tiles=fixed + tiles, solver=res,
                             metrics_influence={"V100": 0.9, "D90": 6100.0, "hard": 0.9},
                             runtime={"solver": 0.01}, wall_clock_s=0.02)
        return tiles, rep

    def suggest_next(mesh, placed_tiles, **kw):
        calls.append(("suggest_next", len(list(placed_tiles)), dict(kw)))
        tile = _spread_tiles(mesh, 1, avoid=list(placed_tiles), kind=kw.get("kind", "full"))[0]
        return tile, {"gain": 0.12, "V100_before": 0.5, "V100_after": 0.62,
                      "D90_before": 4000.0, "D90_after": 4800.0, "n_candidates": 300,
                      "n_compatible": 120, "seconds": 0.3}

    def recommend_tile_count(mesh, contraction_pct=0.0, untreated_pct=0.0,
                             eligible_faces=None):
        calls.append(("recommend", eligible_faces is None))
        return TileCountRecommendation(
            n_tiles=n_recommended, area_mm2=4800.0, treatable_area_mm2=4800.0,
            contraction_pct=0.0, untreated_pct=0.0, ellipsoid_area_mm2=5000.0,
            n_tiles_ellipsoid=n_recommended + 1, diameters_mm=[40.0, 38.0, 36.0])

    monkeypatch.setattr(plan, "optimize", optimize)
    monkeypatch.setattr(plan, "suggest_next", suggest_next)
    monkeypatch.setattr(plan, "recommend_tile_count", recommend_tile_count)


@pytest.fixture()
def fakes(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls)
    return calls


def _key(planner, sym, shift=0):
    """Deliver one key press through VTK exactly as the window would."""
    iren = planner.pl.iren.interactor
    code = sym if len(sym) == 1 else " "
    iren.SetKeyEventInformation(0, int(shift), code, 0, sym)
    iren.InvokeEvent("KeyPressEvent")


def _shell_result(result):
    """The printed-phantom situation: no segmented cavity, only a closed
    shell mesh under ``meshes["body"]`` (here the cavity mesh re-labelled)."""
    meshes = {k: v for k, v in result.meshes.items() if k != "cavity"}
    meshes["body"] = result.meshes["cavity"]
    return dataclasses.replace(result, meshes=meshes)


def _wall_point(planner, k=0):
    verts = np.asarray(planner.cavity.vertices)
    return verts[k]


# ------------------------------------------------------------------ prompt
def test_o_key_opens_the_prompt_with_the_recommendation(app, fakes):
    _key(app, "o")
    assert app._prompt is not None
    s = app._last_status
    assert "OPTIMIZE" in s and "recommended 12 (4 cm^2 rule)" in s and "Enter = run" in s
    assert "fits at this grid: 10" in s and "capacity greedy 0.4 s" in s
    assert "[10_]" in s, "pre-filled with min(recommendation, capacity)"
    assert app._prompt.prefill == 10 and app._prompt.value("full") == 10
    assert fakes[0][0] == "recommend" and fakes[0][1] is True  # all faces: real cavity
    cap = [c for c in fakes if c[0] == "capacity"][0]
    assert cap[1] == 0 and cap[2] == 4.0 and cap[3] == 3, "planner grid, no fixed tiles"
    _key(app, "Escape")


def test_prompt_prefills_the_recommendation_when_it_is_smaller(app, monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=3, capacity=10)
    _key(app, "o")
    assert "[3_]" in app._last_status and app._prompt.prefill == 3
    _key(app, "Escape")
    # capacity is computed against the fixed tiles of the current mode
    app.drop_at(_wall_point(app))
    app._opt_mode = "add"
    _key(app, "o")
    assert [c for c in calls if c[0] == "capacity"][-1][1] == 1
    _key(app, "Escape")


def test_infeasible_run_reopens_the_prompt_with_the_count_that_fits(app, monkeypatch):
    from gtcore.plan import api as plan_api

    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=12, capacity=10)
    real_optimize = plan.optimize

    def infeasible(mesh, n_full, n_half=0, **kw):
        calls.append(("optimize", n_full, n_half, dict(kw)))
        raise plan_api.InfeasibleError(
            "greedy solver placed 4 of %d tiles: 4 fit at this grid" % n_full,
            n_full, 4, "greedy")

    monkeypatch.setattr(plan, "optimize", infeasible)
    _key(app, "o")
    _key(app, "Return")            # runs with the pre-filled 10
    assert [c for c in calls if c[0] == "optimize"][0][1] == 10
    assert len(app.tiles) == 0 and not app._history
    s = app._last_status
    assert "optimize failed" in s and "only 4 tiles fit at this grid" in s
    assert "prompt reopened with 4" in s
    assert app._prompt is not None and app._prompt.text["full"] == "4"
    assert "[4_]" in s
    # Enter now runs with 4 (fake restored); Esc would cancel
    monkeypatch.setattr(plan, "optimize", real_optimize)
    _key(app, "Return")
    assert len(app.tiles) == 4 and app._prompt is None
    assert [c for c in calls if c[0] == "optimize"][-1][1] == 4

    # a zero-capacity failure does not loop the prompt
    def nothing_fits(mesh, n_full, n_half=0, **kw):
        raise plan_api.InfeasibleError("greedy solver placed 0 of 3 tiles", 3, 0, "greedy")

    monkeypatch.setattr(plan, "optimize", nothing_fits)
    app.run_optimize(3)
    assert app._prompt is None and "optimize failed" in app._last_status


def test_prompt_state_machine_and_no_key_leakage(app, fakes):
    _key(app, "o")
    for d in "12":
        _key(app, d)
    assert app._prompt.text["full"] == "12" and "[12_]" in app._last_status
    _key(app, "BackSpace")
    assert app._prompt.text["full"] == "1" and len(app.tiles) == 0
    _key(app, "KP_8")
    assert app._prompt.text["full"] == "18"
    _key(app, "9")
    _key(app, "9")  # a fourth digit is ignored
    assert app._prompt.text["full"] == "189"
    # other bindings are inert while the prompt is open
    n_hist = len(app._history)
    for sym in ("p", "u", "Tab", "x", "t", "z", "Delete", "BackSpace"):
        _key(app, sym)
    assert len(app.tiles) == 0 and len(app._history) == n_hist
    assert app._dose_volume is None and app._prompt is not None
    assert app._prompt.text["full"] == "18", "BackSpace edits the prompt, not the board"
    # H switches to the half-tile count, M flips the mode, S cycles the solver
    _key(app, "h")
    assert app._prompt.field == "half"
    _key(app, "2")
    assert app._prompt.text["half"] == "2" and "[2_]" in app._last_status
    assert app.next_kind == "full", "H must not reach the next-drop toggle"
    _key(app, "m")
    assert app._opt_mode == "add" and "mode (add)" in app._last_status
    _key(app, "m")
    assert app._opt_mode == "replace"
    _key(app, "s")
    assert app._opt_solver == "continuous" and "solver (continuous)" in app._last_status
    app._opt_solver = "greedy"
    _key(app, "Escape")
    assert app._prompt is None and "cancelled" in app._last_status
    assert len(app.tiles) == 0 and not app._history
    assert not any(c[0] == "optimize" for c in fakes)
    # the board works again
    app.drop_at(_wall_point(app))
    assert len(app.tiles) == 1


def test_enter_runs_the_optimizer_and_tiles_land_as_ordinary_tiles(app, fakes):
    app.drop_at(_wall_point(app))        # a hand-placed proposal to be replaced
    hand = app.tiles[0]
    n_hist = len(app._history)
    _key(app, "o")
    _key(app, "3")
    _key(app, "Return")
    assert app._prompt is None
    call = [c for c in fakes if c[0] == "optimize"][0]
    assert call[1] == 3 and call[2] == 0
    kw = call[3]
    assert kw["solver"] == "sa" and kw["rx_cgy"] == app.rx_cgy   # SA is the V2 default
    assert kw["fixed_tiles"] == [] and kw["eligible_faces"] is None
    assert kw["report"] is False
    from gtcore.planner import PLANNER_H_MM, PLANNER_N_SPINS
    assert kw["h_mm"] == PLANNER_H_MM == 4.0 and kw["n_spins"] == PLANNER_N_SPINS == 3
    assert "candidates at h 4 mm / 3 spins built in" in app._last_status
    assert len(app.tiles) == 3 and all(t.kind == "full" for t in app.tiles)
    assert set(app._tile_ids) == app._optimized_ids
    assert len(app._history) == n_hist + 1, "one undo step for the whole run"
    assert app.selected == 2
    s = app._last_status
    assert "optimized: 3 tiles placed by sa in" in s
    assert "V100" in s and "->" in s and "D90" in s and "+5 mm shell" in s
    assert "replaced 1 tile and" in s and "free detected seed" in s
    assert "3 optimizer tiles untouched" in s
    # section 4 V1: the planner never flags an optimizer output as overlapping
    app._refresh_overlaps()
    assert app._overlap_pairs == [] and "CAUTION" not in s
    assert app._last_optimize is not None
    # ordinary tiles: deletable and undoable
    app.selected = 0
    app._delete_selected()
    assert len(app.tiles) == 2
    app.undo()
    assert len(app.tiles) == 3
    app.undo()
    assert len(app.tiles) == 1 and app.tiles[0] is hand
    assert "optimizer tiles untouched" not in app._last_status


def test_empty_enter_uses_the_recommended_count(app, monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=2)
    _key(app, "o")
    _key(app, "Return")
    call = [c for c in calls if c[0] == "optimize"][0]
    assert call[1] == 2 and len(app.tiles) == 2


def test_zero_count_keeps_the_prompt_open(app, monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=2)
    _key(app, "o")
    _key(app, "0")
    _key(app, "Return")
    assert app._prompt is not None and "at least one tile" in app._last_status
    assert not any(c[0] == "optimize" for c in calls)
    _key(app, "Escape")


def test_optimized_tint_until_touched(app, fakes):
    app.run_optimize(2)
    assert len(app.tiles) == 2
    app.selected = -1
    for i in range(2):
        style = app._tile_style(i)
        assert style["color"] == _OPTIMIZED_COLOR["normal"]
        assert style["outline"] == _OPTIMIZED_OUTLINE
    app.selected = 0
    assert app._tile_style(0)["color"] == _OPTIMIZED_COLOR["selected"]
    assert app._tile_style(0)["outline"] == "white"
    # nudge -> the tile is the user's now
    app._translate_selected(1.0, 0.0)
    app.selected = -1
    assert app._tile_style(0)["color"] == _TILE_STYLE["normal"]["color"]
    assert app._tile_style(0)["outline"] is None
    assert app._tile_style(1)["color"] == _OPTIMIZED_COLOR["normal"]
    # rotate drops it too
    app.selected = 1
    app._rotate_selected(0.1)
    app.selected = -1
    assert app._tile_style(1)["color"] == _TILE_STYLE["normal"]["color"]
    assert app._optimized_ids == set()
    assert "optimizer tiles untouched" not in app._last_status


def test_drag_drops_the_tint(app, fakes, monkeypatch):
    app.run_optimize(1)
    tid = app._tile_ids[0]
    assert tid in app._optimized_ids
    # drive the drag path directly: the pick returns a wall point
    monkeypatch.setattr(app, "_pick_cavity_point", lambda x, y: _wall_point(app, 5))
    app._apply_drag(0, (10, 10))
    assert tid not in app._optimized_ids


def test_overlap_caution_still_wins_over_the_tint(app, fakes):
    app.run_optimize(2)
    app._overlap_pairs = [(0, 1)]
    app.selected = -1
    from gtcore.planner import _OVERLAP_COLOR
    assert app._tile_style(0)["color"] == _OVERLAP_COLOR["normal"]


def test_n_key_adds_exactly_one_tile(app, fakes):
    _key(app, "n")
    assert len(app.tiles) == 1
    calls = [c for c in fakes if c[0] == "suggest_next"]
    assert len(calls) == 1 and calls[0][1] == 0 and calls[0][2]["kind"] == "full"
    s = app._last_status
    assert "next tile suggested" in s and "gain +0.120" in s
    assert calls[0][2]["h_mm"] == 4.0 and calls[0][2]["n_spins"] == 3
    assert "V100 0.50 -> 0.62" in s
    assert app._tile_ids[0] in app._optimized_ids
    assert len(app._history) == 1
    app._toggle_kind()
    _key(app, "n")
    assert len(app.tiles) == 2 and app.tiles[1].kind == "half"
    assert [c for c in fakes if c[0] == "suggest_next"][1][1] == 1
    app._refresh_overlaps()
    assert app._overlap_pairs == []
    app.undo()
    assert len(app.tiles) == 1


def test_add_mode_keeps_everything_as_fixed(app, fakes):
    app.drop_at(_wall_point(app))
    app._opt_mode = "add"
    app.run_optimize(2)
    call = [c for c in fakes if c[0] == "optimize"][0]
    assert len(call[3]["fixed_tiles"]) == 1
    assert len(app.tiles) == 3
    assert app._tile_ids[0] not in app._optimized_ids
    assert "replaced" not in app._last_status


def test_fixed_tiles_force_greedy_and_status_names_solver_and_time(app, fakes):
    app.drop_at(_wall_point(app))
    app._opt_mode = "add"
    app._opt_solver = "sa"
    app.run_optimize(1)
    call = [c for c in fakes if c[0] == "optimize"][0]
    assert call[3]["solver"] == "greedy"
    assert "greedy used: 1 fixed tiles" in app._last_status
    assert "placed by greedy in" in app._last_status and " s" in app._last_status
    app._opt_mode = "replace"
    app.run_optimize(1, solver="continuous")
    call = [c for c in fakes if c[0] == "optimize"][-1]
    assert call[3]["solver"] == "continuous"
    assert "placed by continuous in" in app._last_status


def test_shift_o_cycles_the_solver(app, fakes):
    assert app._opt_solver == "sa"            # SA is the default after V2
    _key(app, "O", shift=1)
    assert app._opt_solver == "continuous" and app._prompt is None
    assert "optimizer solver: continuous" in app._last_status
    _key(app, "O", shift=1)
    assert app._opt_solver == "greedy"
    _key(app, "O", shift=1)
    assert app._opt_solver == "sa"
    assert tuple(OPTIMIZER_SOLVERS) == ("greedy", "sa", "continuous")
    # plain O (caps lock) still opens the prompt
    _key(app, "O", shift=0)
    assert app._prompt is not None
    _key(app, "Escape")


def test_solver_kwarg_reaches_the_planner(result):
    try:
        planner = _PlannerApp(result, off_screen=True, solver="sa")
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        assert planner._opt_solver == "sa"
        assert "optimizer: sa, replace mode" in planner._last_status
    finally:
        planner.close()


def test_not_implemented_is_reported_not_raised(app, monkeypatch):
    def stub(name):
        def fn(*a, **k):
            raise NotImplementedError("%s: implemented on branch plan/x" % name)
        return fn

    monkeypatch.setattr(plan, "recommend_tile_count", stub("recommend_tile_count"))
    monkeypatch.setattr(plan, "optimize", stub("optimize"))
    monkeypatch.setattr(plan, "suggest_next", stub("suggest_next"))
    _key(app, "o")
    assert app._prompt is not None and "no recommendation" in app._last_status
    assert "recommendation unavailable" in app._last_status
    _key(app, "2")
    _key(app, "Return")
    assert app._prompt is None
    assert "optimizer not available yet" in app._last_status
    assert "plan/x" in app._last_status and len(app.tiles) == 0 and not app._history
    _key(app, "n")
    assert "optimizer not available yet" in app._last_status and len(app.tiles) == 0

    monkeypatch.setattr(plan, "optimize", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("no candidate placement survived")))
    app.run_optimize(2)
    assert "optimize failed: no candidate placement survived" in app._last_status
    assert len(app.tiles) == 0


def test_legend_and_compact_help_mention_the_keys(app):
    assert "OPTIMIZE placement" in HELP_TEXT and "NEXT tile" in HELP_TEXT
    assert "Shift+O" in HELP_TEXT
    from gtcore.planner import HELP_TEXT_COMPACT
    assert "O optimize" in HELP_TEXT_COMPACT


# ------------------------------------------------------------ wall_mesh_for
def test_wall_mesh_for_matches_the_planner(result, app):
    mesh, label = wall_mesh_for(result)
    assert mesh is app.cavity and label == app._surface_label == "cavity wall"
    no_cavity = _shell_result(result)
    mesh, label = wall_mesh_for(no_cavity)
    assert mesh is no_cavity.meshes["body"] and label.startswith("phantom shell")
    bare = dataclasses.replace(result, meshes={})
    assert wall_mesh_for(bare) == (None, "cavity wall")


def test_phantom_shell_fallback_uses_visible_faces(result, monkeypatch, fakes):
    no_cavity = _shell_result(result)
    seen = {}

    def visible_faces(mesh, center):
        seen["center"] = np.asarray(center, dtype=float)
        return np.ones(len(mesh.faces), dtype=bool)

    monkeypatch.setattr(plan, "visible_faces", visible_faces)
    try:
        planner = _PlannerApp(no_cavity, off_screen=True)
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        mask = planner._eligible_faces()
        assert mask is not None and mask.all()
        expect = np.asarray(result.seeds.centers_ras).mean(axis=0)
        assert np.allclose(seen["center"], expect)
        assert planner._eligible_faces() is mask, "cached"
        planner.optimize_placement()
        rec = [c for c in fakes if c[0] == "recommend"][-1]
        assert rec[1] is False, "the recommendation is restricted to the eligible faces"
        assert fakes[-1][0] == "capacity"
        assert "shell faces eligible" in planner._eligible_note()
    finally:
        planner.close()


# --------------------------------------------------------------------- CLI
def test_cli_optimize_help_and_plan_optimizer_flag():
    from gtcore.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["optimize", "--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        main(["plan", "--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit):
        main(["optimize", "--solver", "magic"])
    with pytest.raises(SystemExit):
        main(["plan", "--optimizer", "magic"])


def test_cli_optimize_runs_on_the_phantom(result, monkeypatch, tmp_path, capsys):
    import gtcore.pipeline as pipeline_mod
    from gtcore.cli import main

    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=2)
    monkeypatch.setattr(pipeline_mod, "reconstruct", lambda vol, **kw: result)
    out = tmp_path / "opt"
    rc = main(["optimize", "--spacing", "1.0", "--out", str(out), "--no-report",
               "--solver", "sa", "--seed", "3", "--budget", "5"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "Recommended tiles: 2" in text
    assert "fits at this grid" in text
    assert "using the recommended count, 2" in text
    assert "2 tiles" in text and "wall: cavity wall" in text
    call = [c for c in calls if c[0] == "optimize"][0]
    assert call[1] == 2 and call[3]["solver"] == "sa" and call[3]["seed"] == 3
    assert call[3]["time_budget_s"] == 5.0 and call[3]["report"] is False
    for name in ("report.json", "report.csv", "plan_seeds.csv"):
        assert (out / name).exists(), name
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_tiles"] == 2 and len(rep["tiles"]) == 2
    seeds = (out / "plan_seeds.csv").read_text().splitlines()
    assert seeds[0].startswith("# rx_cgy=6000")
    assert sum(1 for line in seeds if line.startswith("placed,")) == 8
    assert sum(1 for line in seeds if line.startswith("detected,")) == len(result.seeds)

    # explicit --tiles wins; the shell fallback runs visible_faces
    no_cavity = _shell_result(result)
    monkeypatch.setattr(pipeline_mod, "reconstruct", lambda vol, **kw: no_cavity)
    monkeypatch.setattr(plan, "visible_faces",
                        lambda mesh, c: np.ones(len(mesh.faces), dtype=bool))
    out2 = tmp_path / "opt2"
    rc = main(["optimize", "--spacing", "1.0", "--out", str(out2), "--tiles", "1",
               "--no-report"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "phantom shell" in text and "eligible faces" in text
    call = [c for c in calls if c[0] == "optimize"][-1]
    assert call[1] == 1 and call[3]["eligible_faces"] is not None


def test_cli_uses_the_capacity_when_it_is_below_the_recommendation(result, monkeypatch,
                                                                     tmp_path, capsys):
    import gtcore.pipeline as pipeline_mod
    from gtcore.cli import main

    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=2, capacity=1)
    monkeypatch.setattr(pipeline_mod, "reconstruct", lambda vol, **kw: result)
    out = tmp_path / "cap"
    rc = main(["optimize", "--spacing", "1.0", "--out", str(out), "--no-report",
               "--h", "4", "--spins", "3"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "recommended 2 (4 cm^2 rule) / fits at this grid (h 4 mm, 3 spins): 1" in text
    assert "using the packing capacity, 1" in text
    assert [c for c in calls if c[0] == "optimize"][0][1] == 1
    cap = [c for c in calls if c[0] == "capacity"][0]
    assert cap[2] == 4.0 and cap[3] == 3


def test_cli_min_n_sweeps_without_forwarding_the_budget(result, monkeypatch, tmp_path, capsys):
    """``--min-n`` hands the eligible-wall target to ``sweep_n`` and never
    the continuous-solver budget (the discrete solvers reject it)."""
    import gtcore.pipeline as pipeline_mod
    from gtcore.cli import main
    from gtcore.plan import SweepResult

    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=3)
    monkeypatch.setattr(pipeline_mod, "reconstruct", lambda vol, **kw: result)
    seen = {}

    def sweep_n(mesh, target, n_max, rx_cgy=6000.0, solver="greedy", seed=0, **kw):
        seen.update(n_max=n_max, kw=dict(kw), target=target.name)
        rows = [{"N": n, "V100": 0.3 * n, "D90": 2000.0 * n, "V150": 0.0, "V200": 0.0,
                 "runtime_s": 0.1, "solver": solver} for n in range(1, n_max + 1)]
        return SweepResult(rows=rows, min_n={"D90>=rx": 3, "V100>=0.90": None})

    monkeypatch.setattr(plan, "sweep_n", sweep_n)
    out = tmp_path / "minn"
    rc = main(["optimize", "--spacing", "1.0", "--out", str(out), "--tiles", "4",
               "--min-n", "--no-report", "--budget", "7"])
    assert rc == 0
    # the sweep runs up to the packing capacity (fake: 10), not just --tiles 4
    assert seen["n_max"] == 10 and "time_budget_s" not in seen["kw"]
    assert seen["kw"]["h_mm"] == 2.5 and seen["target"].startswith("shell")
    call = [c for c in calls if c[0] == "optimize"][-1]
    assert call[1] == 3, "the smallest N with D90 >= rx is placed"
    text = capsys.readouterr().out
    assert "coverage vs N" in text and "D90>=rx -> 3" in text
    assert (out / "sweep.json").exists() and (out / "report.json").exists()


def test_cli_optimize_reports_stubs_and_failures(result, monkeypatch, tmp_path, capsys):
    import gtcore.pipeline as pipeline_mod
    from gtcore.cli import main

    monkeypatch.setattr(pipeline_mod, "reconstruct", lambda vol, **kw: result)
    monkeypatch.setattr(plan, "recommend_tile_count", lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("recommend_tile_count: implemented on branch plan/candidates")))
    rc = main(["optimize", "--spacing", "1.0", "--out", str(tmp_path / "a"), "--tiles", "2"])
    assert rc == 2 and "optimizer not available yet" in capsys.readouterr().out

    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=2)
    monkeypatch.setattr(plan, "optimize", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("greedy solver returned 1 tiles, 2 requested")))
    rc = main(["optimize", "--spacing", "1.0", "--out", str(tmp_path / "b"), "--tiles", "2"])
    assert rc == 1 and "optimize failed" in capsys.readouterr().out
    assert not os.path.exists(str(tmp_path / "b" / "report.json"))


# ------------------------------------------------- end-to-end (opt-in, slow)
@pytest.mark.skipif(os.environ.get("GT_E2E") != "1",
                    reason="real optimizer end-to-end run (~1-2 min); set GT_E2E=1")
def test_e2e_o_key_runs_the_real_optimizer(result):
    """The REAL gtcore.plan stack (no monkeypatches) behind the O key on the
    synthetic phantom at the planner grid (h 4 mm / 3 spins): prompt ->
    Enter with the recommended count -> tiles land, no overlap flagged
    (section 4 V1), before/after in the status line, undo removes them."""
    import time

    from gtcore.plan import api
    api.clear_cache()
    try:
        app = _PlannerApp(result, off_screen=True)
        app.pl.render()
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        t0 = time.perf_counter()
        _key(app, "o")
        assert app._prompt is not None and app._prompt.recommended is not None
        n = int(app._prompt.recommended)
        assert "recommended %d" % n in app._last_status
        _key(app, "Return")
        dt = time.perf_counter() - t0
        s = app._last_status
        assert "optimizer not available" not in s, s
        if "optimize failed" in s:
            # the manufacturer rule (area / 4 cm^2) ignores packing loss: on
            # this cavity more tiles are recommended than fit at h 4 mm / 3
            # spins.  Section 4 V8: fail loudly, never fewer tiles as success.
            assert "infeasible" in s and len(app.tiles) == 0 and not app._history, s
            print("E2E: recommended %d infeasible at h 4 / 3 spins (%.1f s): %s"
                  % (n, dt, s.splitlines()[-1]))
            n = 6
            t0 = time.perf_counter()
            _key(app, "o")
            _key(app, "6")
            _key(app, "Return")
            dt = time.perf_counter() - t0
            s = app._last_status
            assert "optimize failed" not in s, s
        assert len(app.tiles) == n, s
        assert set(app._tile_ids) == app._optimized_ids
        assert "optimized: %d tiles placed by sa in" % n in s
        assert "V100" in s and "->" in s and "D90" in s
        app._refresh_overlaps()
        assert app._overlap_pairs == [] and "CAUTION" not in s
        rep = app._last_optimize
        assert rep is not None and rep.overlaps == []
        assert rep.candidate_stats["h_mm"] == 4.0 and rep.candidate_stats["n_spins"] == 3
        print("E2E O key: %d tiles in %.1f s; %s" % (n, dt, s.splitlines()[-2:]))
        app.undo()
        assert len(app.tiles) == 0
        # N on the warm cache is quick and adds exactly one tile
        t1 = time.perf_counter()
        _key(app, "n")
        assert len(app.tiles) == 1 and "next tile suggested" in app._last_status
        print("E2E N key: %.1f s" % (time.perf_counter() - t1))
    finally:
        app.close()
