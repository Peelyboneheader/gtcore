"""Headless coverage for the planner's mouse interaction layer.

Drives the real VTK event pipeline with synthetic events
(``iren.SetEventPosition(...)`` + ``iren.InvokeEvent(...)``) against an
off-screen plotter built on the phantom scene:

- a right-click over the cavity places exactly ONE tile (the historical bug
  stacked several pick bindings, and on scans without a cavity bound nothing
  at all, so clicks died silently),
- a click that misses the wall places nothing and says so on screen,
- a scan with no cavity mesh reports that instead of staying silent,
- Ctrl+left-drag grabs a placed tile and slides it along the wall, and the
  relocated tile stays conformed (seeds 2.0-4.0 mm off the wall on the
  cavity side),
- overlap warnings are wired defensively through
  ``gtcore.interact.find_overlapping_tiles``.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyvista")
trimesh = pytest.importorskip("trimesh")

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import reconstruct
from gtcore.interact import snap_to_wall
from gtcore.planner import _PlannerApp


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


def _display_xy(planner, world):
    """Project a world point to integer display (pixel) coordinates."""
    ren = planner.pl.renderer
    ren.SetWorldPoint(float(world[0]), float(world[1]), float(world[2]), 1.0)
    ren.WorldToDisplay()
    x, y, _z = ren.GetDisplayPoint()
    return int(round(x)), int(round(y))


def _iren(planner):
    return planner.pl.iren.interactor


def _right_click(planner, x, y):
    iren = _iren(planner)
    iren.SetEventPosition(x, y)
    iren.InvokeEvent("RightButtonPressEvent")
    iren.InvokeEvent("RightButtonReleaseEvent")


def _cavity_aim(planner):
    """Screen position whose view ray safely crosses the cavity."""
    centroid = np.asarray(planner.cavity.vertices).mean(axis=0)
    return _display_xy(planner, centroid)


# --------------------------------------------------------------- placement
def test_right_click_places_exactly_one_tile(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    assert len(app.tiles) == 1, "one right-click must place exactly one tile"
    assert "tile placed" in app._last_status
    # the tile is anchored ON the cavity wall
    _surf, dist, _tid = trimesh.proximity.closest_point(
        app.cavity, [app.tiles[0].anchor_ras])
    assert dist[0] < 0.5


def test_miss_click_reports_and_places_nothing(app):
    _right_click(app, 3, 3)  # far corner of the window: no wall there
    assert len(app.tiles) == 0
    assert "missed the cavity wall" in app._last_status


def test_no_cavity_scan_reports_instead_of_silence(result):
    """A scan whose reconstruction found no cavity must still give feedback.

    This is the real-scan failure mode: the old code only bound its pick
    callbacks when a cavity mesh existed, so every click did nothing at all.
    """
    meshes = {k: v for k, v in result.meshes.items() if k != "cavity"}
    bare = dataclasses.replace(result, meshes=meshes)
    try:
        planner = _PlannerApp(bare, off_screen=True)
        planner.pl.render()
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    try:
        _right_click(planner, 640, 450)
        assert len(planner.tiles) == 0
        assert "no cavity surface" in planner._last_status
    finally:
        planner.close()


def test_p_key_places_via_key_pipeline(app):
    x, y = _cavity_aim(app)
    iren = _iren(app)
    iren.SetEventPosition(x, y)
    iren.SetKeyEventInformation(0, 0, "p", 0, "p")
    iren.InvokeEvent("KeyPressEvent")
    iren.InvokeEvent("CharEvent")
    assert len(app.tiles) == 1
    assert "tile placed" in app._last_status


# -------------------------------------------------------------------- drag
def test_ctrl_left_drag_relocates_and_stays_conformed(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    assert len(app.tiles) == 1
    start = app.tiles[0]
    gx, gy = _display_xy(app, start.anchor_ras)

    iren = _iren(app)
    iren.SetControlKey(1)
    iren.SetEventPosition(gx, gy)
    iren.InvokeEvent("LeftButtonPressEvent")
    assert app._drag_idx == 0, "Ctrl+left-press on the tile must grab it"

    # slide the cursor ~80 px across the cavity in small steps
    for frac in np.linspace(0.0, 1.0, 9):
        iren.SetEventPosition(int(gx + 80 * frac), int(gy + 25 * frac))
        iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)

    assert app._drag_idx == -1, "release must end the drag"
    assert len(app.tiles) == 1, "dragging must not change the tile count"
    moved = app.tiles[0]
    assert np.linalg.norm(moved.anchor_ras - start.anchor_ras) > 0.5, (
        "drag did not relocate the tile")

    # relocated tile is still conformed: every seed 2.0-4.0 mm off the wall,
    # on the cavity side of it
    _surf, dist, _tid = trimesh.proximity.closest_point(
        app.cavity, moved.seed_centers)
    for seed, wall_pt, d in zip(moved.seed_centers, _surf, dist):
        assert 2.0 <= d <= 4.0, "seed %.2f mm from wall after drag" % d
        _s, n_in = snap_to_wall(app.cavity, wall_pt)
        assert float(np.dot(seed - wall_pt, n_in)) > 0.0, (
            "seed ended up on the tissue side of the wall")


def test_ctrl_left_press_off_tile_does_not_grab(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    iren = _iren(app)
    iren.SetControlKey(1)
    iren.SetEventPosition(5, 5)  # nowhere near the tile
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)
    assert app._drag_idx == -1


# ---------------------------------------------------------------- overlaps
def test_overlap_warning_wiring(app, monkeypatch):
    """Wiring test: whatever pairs the detector returns get surfaced."""
    import gtcore.interact as interact_mod

    monkeypatch.setattr(interact_mod, "find_overlapping_tiles",
                        lambda tiles, threshold_mm=1.0: [(0, 1)],
                        raising=False)
    verts = np.asarray(app.cavity.vertices)
    app.drop_at(verts[0])
    app.drop_at(verts[len(verts) // 2])
    assert app._overlap_pairs == [(0, 1)]
    assert "WARNING: tiles 1 & 2 overlap" in app._last_status

    # resolved -> warning and tint state clear
    monkeypatch.setattr(interact_mod, "find_overlapping_tiles",
                        lambda tiles, threshold_mm=1.0: [],
                        raising=False)
    app._rotate_selected(0.1)
    assert app._overlap_pairs == []
    assert "WARNING" not in app._last_status


def test_overlap_detector_failure_is_silent(app, monkeypatch):
    """A broken/missing detector must never break tile placement."""
    import gtcore.interact as interact_mod

    def _boom(tiles, threshold_mm=1.0):
        raise AttributeError("not written yet")

    monkeypatch.setattr(interact_mod, "find_overlapping_tiles", _boom,
                        raising=False)
    verts = np.asarray(app.cavity.vertices)
    app.drop_at(verts[0])
    app.drop_at(verts[len(verts) // 2])
    assert len(app.tiles) == 2
    assert app._overlap_pairs == []
    assert "WARNING" not in app._last_status
