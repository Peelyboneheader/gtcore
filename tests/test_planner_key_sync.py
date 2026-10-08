"""T (suggest), O (optimize) and X (delete) share one board.

Three keys used to fight over the detected seeds:

- X on a suggested tile released its seeds, which reappeared as free gold
  seeds at the scan position (and kept counting in dose and export);
- T appended a second copy of the implant on top of the first (or on top
  of the tiles adopted from the scan at startup);
- O kept the suggested tiles as fixed obstacles and added more, while the
  optimizer's plan and the implant's free seeds were counted together.

Now: X takes the tile's seeds off the board; T re-infers the implant from
the scan (earlier scan tiles replaced, removed seeds back); O in replace
mode puts the plan on the board instead of the implant (tiles AND free
seeds), in add mode keeps everything fixed.  Each is one undo step.
Headless off-screen planner on the 2-tile phantom, optimizer faked as in
``test_planner_optimize.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pyvista")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_planner_optimize import _install_fakes, _key  # noqa: E402

from gtcore.phantom import make_head_phantom  # noqa: E402
from gtcore.pipeline import reconstruct  # noqa: E402
from gtcore.planner import _PlannerApp  # noqa: E402


@pytest.fixture(scope="module")
def result():
    vol, _truth = make_head_phantom(spacing=1.0, n_tiles=2, rng_seed=1)
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


@pytest.fixture()
def fakes(monkeypatch):
    calls = []
    _install_fakes(monkeypatch, calls, n_recommended=3, capacity=10)
    return calls


def _shown_det(app):
    return sorted(int(n[len("det_seed_"):]) for n in app.pl.actors
                  if n.startswith("det_seed_"))


def _suggest(app):
    placed = app.suggest_tiles()
    if len(placed) != 2:
        pytest.skip("suggest did not recover the 2 phantom tiles here")
    return placed


# -------------------------------------------------------------------- X
def test_delete_takes_the_tiles_seeds_off_the_board(app):
    n_det = len(app.result.seeds)
    _suggest(app)
    owned = app._owned_seed_indices()
    assert len(owned) == 8 and len(_shown_det(app)) == n_det - 8
    free_before = len(app._free_detected()[0])

    app.selected = 0
    gone = set(app._owned_seeds[app._tile_ids[0]])
    app._delete_selected()
    assert len(app.tiles) == 1
    # the seeds do NOT come back at the scan position ...
    assert len(_shown_det(app)) == n_det - 8
    assert not (gone & set(_shown_det(app)))
    assert app._removed_seeds == gone
    # ... and they are not counted in dose / export either
    assert len(app._free_detected()[0]) == free_before
    assert "tile deleted with its 4 seeds" in app._last_status

    # one undo step restores the tile with its seeds
    app.undo()
    assert len(app.tiles) == 2 and app._removed_seeds == set()
    assert len(_shown_det(app)) == n_det - 8
    assert len(app._free_detected()[0]) == free_before


def test_delete_then_second_tile_leaves_nothing_behind(app):
    n_det = len(app.result.seeds)
    _suggest(app)
    app.selected = 0
    app._delete_selected()
    app.selected = 0
    app._delete_selected()
    assert len(app.tiles) == 0
    assert len(app._removed_seeds) == 8
    assert len(_shown_det(app)) == n_det - 8
    assert len(app._free_detected()[0]) == n_det - 8
    # export sees the same board: no seed of a deleted tile
    assert len(app._free_detected()[0]) == n_det - 8


# -------------------------------------------------------------------- T
def test_t_twice_re_infers_instead_of_stacking(app):
    n_det = len(app.result.seeds)
    _suggest(app)
    first_ids = list(app._tile_ids)
    app.selected = 0
    moved_from = app.tiles[0].center_ras.copy()
    app._translate_selected(4.0, 0.0)
    assert not np.allclose(app.tiles[0].center_ras, moved_from)

    _suggest(app)
    assert len(app.tiles) == 2, "T must not stack a second copy"
    assert not (set(app._tile_ids) & set(first_ids))
    assert np.allclose(app.tiles[0].center_ras, moved_from, atol=1e-6) \
        or np.allclose(app.tiles[1].center_ras, moved_from, atol=1e-6)
    assert "re-inferred: 2 earlier scan tiles replaced" in app._last_status
    assert len(_shown_det(app)) == n_det - 8
    assert len(app._owned_seed_indices()) == 8

    # undo brings the moved tile back as it was
    app.undo()
    assert app._tile_ids == first_ids
    assert not np.allclose(app.tiles[0].center_ras, moved_from)


def test_t_after_delete_shows_the_whole_implant_again(app):
    n_det = len(app.result.seeds)
    _suggest(app)
    app.selected = 0
    app._delete_selected()
    assert len(app.tiles) == 1 and len(app._removed_seeds) == 4
    _suggest(app)
    assert len(app.tiles) == 2 and app._removed_seeds == set()
    assert len(app._owned_seed_indices()) == 8
    assert len(_shown_det(app)) == n_det - 8
    assert "1 earlier scan tile replaced" in app._last_status


def test_t_replaces_tiles_adopted_from_the_scan_at_startup():
    vol, _truth = make_head_phantom(spacing=1.0, n_tiles=2, rng_seed=1)
    res = reconstruct(vol, n_full_tiles=2, verbose=False)
    if res.tiles is None or not res.tiles.tiles:
        pytest.skip("tile fitting recovered nothing on this phantom")
    try:
        app = _PlannerApp(res, off_screen=True)
        app.pl.render()
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        n_adopted = len(app.tiles)
        assert n_adopted and app._adopted_ids == set(app._tile_ids)
        placed = app.suggest_tiles()
        assert len(app.tiles) == len(placed), \
            "T re-infers; the startup adoption is not kept underneath"
        assert "%d earlier scan tile" % n_adopted in app._last_status
        assert all(tid in app._adopted_ids for tid in app._tile_ids)
        app.undo()
        assert len(app.tiles) == n_adopted
    finally:
        app.close()


def test_t_keeps_hand_placed_tiles(app):
    _suggest(app)
    verts = np.asarray(app.cavity.vertices)
    app.drop_at(verts[0])
    hand_id = app._tile_ids[-1]
    hand = app.tiles[-1]
    assert len(app.tiles) == 3
    _suggest(app)
    assert len(app.tiles) == 3
    assert hand_id in app._tile_ids and hand in app.tiles
    assert "2 earlier scan tiles replaced" in app._last_status


# -------------------------------------------------------------------- O
def test_o_replace_puts_the_plan_instead_of_the_implant(app, fakes):
    n_det = len(app.result.seeds)
    _suggest(app)
    t_ids = list(app._tile_ids)
    n_free = len(app._free_detected()[0])
    shown_before = _shown_det(app)
    n_hist = len(app._history)

    _key(app, "o")
    assert "mode (replace)" in app._last_status
    cap = [c for c in fakes if c[0] == "capacity"][-1]
    assert cap[1] == 0, "replace mode: the suggested tiles are not obstacles"
    _key(app, "3")
    _key(app, "Return")
    call = [c for c in fakes if c[0] == "optimize"][-1]
    assert call[3]["fixed_tiles"] == [] and call[3]["solver"] == "sa"
    assert len(app.tiles) == 3 and set(app._tile_ids) == app._optimized_ids
    assert not (set(app._tile_ids) & set(t_ids))
    # the implant left the board with its seeds: nothing is drawn or
    # counted twice
    assert _shown_det(app) == []
    assert len(app._free_detected()[0]) == 0
    assert app._removed_seeds == set(range(n_det))
    s = app._last_status
    assert ("replaced 2 tiles and %d free detected seeds on the board" % n_free
            if n_free else "replaced 2 tiles on the board") in s
    assert len(app._history) == n_hist + 1

    # one undo step: the implant is back exactly as it was
    app.undo()
    assert app._tile_ids == t_ids and app._removed_seeds == set()
    assert _shown_det(app) == shown_before
    assert len(app._free_detected()[0]) == n_free


def test_o_replace_with_no_tiles_still_takes_the_free_seeds(app, fakes):
    n_det = len(app.result.seeds)
    assert len(_shown_det(app)) == n_det
    app.run_optimize(2)
    assert len(app.tiles) == 2 and _shown_det(app) == []
    assert "replaced %d free detected seeds on the board" % n_det in app._last_status
    app.undo()
    assert len(app.tiles) == 0 and len(_shown_det(app)) == n_det


def test_o_add_keeps_the_implant_fixed(app, fakes):
    n_det = len(app.result.seeds)
    _suggest(app)
    t_ids = list(app._tile_ids)
    app._opt_mode = "add"
    _key(app, "o")
    cap = [c for c in fakes if c[0] == "capacity"][-1]
    assert cap[1] == 2, "add mode: the suggested tiles are obstacles"
    _key(app, "1")
    _key(app, "Return")
    call = [c for c in fakes if c[0] == "optimize"][-1]
    assert len(call[3]["fixed_tiles"]) == 2 and call[3]["solver"] == "greedy"
    assert app._tile_ids[:2] == t_ids and len(app.tiles) == 3
    assert len(_shown_det(app)) == n_det - 8
    assert app._removed_seeds == set()
    assert "replaced" not in app._last_status


def test_t_after_o_keeps_the_plan_and_shows_the_implant_again(app, fakes):
    n_det = len(app.result.seeds)
    app.run_optimize(2)
    opt_ids = list(app._tile_ids)
    assert _shown_det(app) == []
    _suggest(app)
    assert len(app.tiles) == 4 and app._tile_ids[:2] == opt_ids
    assert app._removed_seeds == set()
    assert len(_shown_det(app)) == n_det - 8
    assert "earlier scan tile" not in app._last_status
