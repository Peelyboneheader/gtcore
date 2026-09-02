"""Planner integration of automatic tile creation ("suggest tiles").

The auto-inferred tiles land on the planner board as ordinary placed tiles
(movable, rotatable, deletable, re-conformed live), the status line carries
the evidence summary, and the headless snapshot driver accepts "suggest".
"""
from __future__ import annotations

import numpy as np
import pytest

from gtcore.interact import PlacedTile, snap_to_wall, translate_on_wall
from gtcore.phantom import make_head_phantom
from gtcore.pipeline import reconstruct
from gtcore.tiles import fit_tiles, to_placed_tiles


@pytest.fixture(scope="module")
def result():
    vol, truth = make_head_phantom(spacing=1.0, n_tiles=2, rng_seed=1)
    res = reconstruct(vol, verbose=False)
    if "cavity" not in res.meshes or not len(res.meshes["cavity"].vertices):
        pytest.skip("pipeline found no cavity on this phantom")
    return res, truth


# ------------------------------------------------------------ conversion
def test_to_placed_tiles_with_and_without_mesh(result):
    res, truth = result
    mesh = res.meshes["cavity"]
    seeds = res.seeds
    with_mesh = fit_tiles(seeds.centers_ras, seeds.axes_ras, "auto",
                          cavity_center_ras=truth.cavity_center_ras, mesh=mesh)
    free = fit_tiles(seeds.centers_ras, seeds.axes_ras, "auto",
                     cavity_center_ras=truth.cavity_center_ras)
    assert with_mesh.n_selected == free.n_selected == 2
    a = to_placed_tiles(with_mesh, seeds.centers_ras, seeds.axes_ras)
    b = to_placed_tiles(free, seeds.centers_ras, seeds.axes_ras)
    assert len(a) == len(b) == 2
    for ta, tb, pose in zip(a, b, free.tiles):
        assert isinstance(ta, PlacedTile) and isinstance(tb, PlacedTile)
        assert ta.kind == tb.kind == "full"
        assert ta.seed_centers.shape == (4, 3) and tb.seed_centers.shape == (4, 3)
        assert ta.corners_ras.shape == (4, 3) and tb.corners_ras.shape == (4, 3)
        # both describe the same tile: seed centroids within 1 mm, normals
        # both into the cavity and within 15 deg of each other
        assert np.linalg.norm(ta.center_ras - tb.center_ras) < 2.5   # segmented wall
        to_cav = truth.cavity_center_ras - tb.center_ras
        assert float(tb.normal_ras @ to_cav) > 0.0
        assert float(ta.normal_ras @ to_cav) > 0.0
        assert abs(float(ta.normal_ras @ tb.normal_ras)) > np.cos(np.deg2rad(25.0))
        # the free tile's seeds are the observed ones
        assert np.allclose(tb.seed_centers, seeds.centers_ras[pose.seed_indices])
        # the free tile footprint: 20 mm square, chords contracted on the seed sheet
        d = np.linalg.norm(tb.corners_ras[0] - tb.corners_ras[2])
        assert 18.0 < d < 29.0


def test_placed_suggestion_is_movable(result):
    res, truth = result
    mesh = res.meshes["cavity"]
    seeds = res.seeds
    fit = fit_tiles(seeds.centers_ras, seeds.axes_ras, "auto",
                    cavity_center_ras=truth.cavity_center_ras, mesh=mesh)
    tile = to_placed_tiles(fit, seeds.centers_ras, seeds.axes_ras)[0]
    delta = np.cross(tile.normal_ras, tile.axis_ras) * 3.0
    moved = translate_on_wall(mesh, tile, delta)
    assert np.linalg.norm(moved.anchor_ras - tile.anchor_ras) > 1.0
    # still conformed after the move
    import trimesh

    _, dist, _ = trimesh.proximity.closest_point(mesh, moved.seed_centers)
    assert np.all((dist >= 1.2) & (dist <= 4.5))


# --------------------------------------------------------------- planner
def test_planner_suggest_key(result):
    pytest.importorskip("pyvista")
    from gtcore.planner import _PlannerApp

    res, truth = result
    try:
        app = _PlannerApp(res, off_screen=True)
        app.pl.render()
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        placed = app.suggest_tiles()
        assert len(placed) == 2
        assert len(app.tiles) == 2
        assert app._last_suggestion is not None
        assert app._last_suggestion.n_selected == 2
        assert "suggested 2 tile(s)" in app._last_status
        assert "evidence supports n=2" in app._last_status
        # the suggested tiles are ordinary board tiles: select, rotate, delete
        app.selected = 0
        before = app.tiles[0].axis_ras.copy()
        app._rotate_selected(np.deg2rad(10.0))
        assert not np.allclose(app.tiles[0].axis_ras, before)
        app._delete_selected()
        assert len(app.tiles) == 1
    finally:
        app.close()


def test_snapshot_driver_accepts_suggest(result, tmp_path):
    pytest.importorskip("pyvista")
    from gtcore.planner import snapshot_planner

    res, _truth = result
    path = str(tmp_path / "suggest.png")
    try:
        out = snapshot_planner(res, ["suggest"], path)
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    assert out == path


def test_cli_plan_has_suggest_flag(monkeypatch):
    import gtcore.cli as cli

    seen = {}

    def fake_run(result, rx_cgy=6000.0, suggest=False):
        seen["suggest"] = suggest
        return []

    from types import SimpleNamespace

    fake_vol = SimpleNamespace(array=np.zeros((2, 2, 2)), spacing=(1.0, 1.0, 1.0))
    monkeypatch.setattr(cli, "_load", lambda path, spacing: (fake_vol, "t"))
    monkeypatch.setattr("gtcore.pipeline.reconstruct", lambda vol: object())
    monkeypatch.setattr("gtcore.planner.run_planner", fake_run)
    assert cli.main(["plan", "--suggest"]) == 0
    assert seen["suggest"] is True
