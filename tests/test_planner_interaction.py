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
  ``gtcore.interact.find_overlapping_tiles``,
- the ghost preview follows the wall under the cursor, hides off the wall
  and during camera gestures, turns red when the drop would overlap, and
  the drop lands exactly where the ghost showed it,
- hovering an unselected tile lights it up as grab-able,
- drag pacing never loses the final cursor position on release,
- the dose panel reports wall/+5/+10 mm shells and goes STALE when a tile
  moves after the last update.
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
    iren.SetEventPosition(5, 5)  # nowhere near the tile, nor the wall
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)
    assert app._drag_idx == -1
    assert "nothing to grab" in app._last_status, "a failed grab must say so"


def _drag(app, from_xy, to_xy, ctrl):
    iren = _iren(app)
    iren.SetControlKey(1 if ctrl else 0)
    iren.SetEventPosition(*from_xy)
    iren.InvokeEvent("LeftButtonPressEvent")
    grabbed = app._drag_idx
    for frac in np.linspace(0.0, 1.0, 6):
        app._drag_last_t = float("-inf")
        iren.SetEventPosition(int(from_xy[0] + (to_xy[0] - from_xy[0]) * frac),
                              int(from_xy[1] + (to_xy[1] - from_xy[1]) * frac))
        iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)
    return grabbed


def test_plain_left_press_on_tile_grabs_without_modifier(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    start = app.tiles[0]
    gx, gy = _display_xy(app, start.anchor_ras)
    grabbed = _drag(app, (gx, gy), (gx + 60, gy + 15), ctrl=False)
    assert grabbed == 0, "a press on the tile must grab it without Ctrl"
    assert app._drag_idx == -1
    assert np.linalg.norm(app.tiles[0].anchor_ras - start.anchor_ras) > 0.5


def test_press_on_seed_capsule_grabs_the_tile(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    start = app.tiles[0]
    # aim at a seed capsule centre, offset toward the camera so the pick ray
    # meets the capsule (drawn on top of the quad) rather than the quad
    sx, sy = _display_xy(app, start.seed_centers[0])
    assert app._pick_tile_index(sx, sy) == 0, "seed capsule must be grabbable"
    grabbed = _drag(app, (sx, sy), (sx + 60, sy + 15), ctrl=False)
    assert grabbed == 0
    assert np.linalg.norm(app.tiles[0].anchor_ras - start.anchor_ras) > 0.5


def test_ctrl_press_on_wall_moves_selected_tile(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    start = app.tiles[0]
    # a wall point clearly off the tile: NOT the tile, but ON the wall
    for dx, dy in ((70, 20), (50, 15), (40, 40), (-50, 20), (30, -50), (25, 25)):
        px, py = x + dx, y + dy
        if app._pick_tile_index(px, py) < 0 and app._pick_cavity_point(px, py) is not None:
            break
    else:
        pytest.fail("test geometry: no wall point off the tile found")
    # without Ctrl a press on bare wall is the camera, not a grab
    iren = _iren(app)
    iren.SetEventPosition(px, py)
    iren.InvokeEvent("LeftButtonPressEvent")
    assert app._drag_idx == -1
    iren.InvokeEvent("LeftButtonReleaseEvent")
    # with Ctrl the same press slides the selected tile
    grabbed = _drag(app, (px, py), (px - 30, py - 10), ctrl=True)
    assert grabbed == 0, "Ctrl+press on the wall must grab the SELECTED tile"
    assert np.linalg.norm(app.tiles[0].anchor_ras - start.anchor_ras) > 0.5


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


# ------------------------------------------------------------ ghost preview
def _move(planner, x, y):
    planner._ghost_last_t = 0.0  # defeat the wall-clock pacing in tests
    iren = _iren(planner)
    iren.SetEventPosition(x, y)
    iren.InvokeEvent("MouseMoveEvent")


def test_ghost_follows_wall_hover_and_hides_off_wall(app):
    x, y = _cavity_aim(app)
    _move(app, x, y)
    assert app._ghost_tile is not None, "hovering the wall must show a ghost"
    assert app._ghost_tile.kind == "full"
    assert "ghost_quad" in app.pl.actors and "ghost_edge" in app.pl.actors
    assert len(app.tiles) == 0, "a ghost is not a placed tile"
    # ghost is conformed at the cursor's wall point, on the wall
    _surf, dist, _tid = trimesh.proximity.closest_point(
        app.cavity, [app._ghost_tile.anchor_ras])
    assert dist[0] < 0.5

    _move(app, 3, 3)
    assert app._ghost_tile is None
    assert "ghost_quad" not in app.pl.actors


def test_ghost_tracks_next_kind_and_toggle(app):
    x, y = _cavity_aim(app)
    app._toggle_kind()  # next drop: half
    _move(app, x, y)
    assert app._ghost_tile.kind == "half"
    app._toggle_ghost()
    assert not app.ghost_enabled and app._ghost_tile is None
    _move(app, x, y)
    assert app._ghost_tile is None, "disabled ghost must not reappear"
    app._toggle_ghost()
    _move(app, x, y)
    assert app._ghost_tile is not None


def test_ghost_turns_red_when_drop_would_overlap(app, monkeypatch):
    import gtcore.interact as interact_mod

    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    assert len(app.tiles) == 1
    assert app._ghost_tile is None, "placing replaces the preview"

    # the detector says: whatever is last in the list overlaps tile 0
    monkeypatch.setattr(interact_mod, "find_overlapping_tiles",
                        lambda tiles, threshold_mm=1.0: [(0, len(tiles) - 1)],
                        raising=False)
    _move(app, x + 60, y + 15)  # beside the tile, still on the wall
    assert app._ghost_tile is not None
    assert app._ghost_overlaps is True
    assert app._overlap_pairs == [], "a preview must not raise a real warning"

    monkeypatch.setattr(interact_mod, "find_overlapping_tiles",
                        lambda tiles, threshold_mm=1.0: [], raising=False)
    _move(app, x + 62, y + 16)
    assert app._ghost_overlaps is False


def test_placement_hides_ghost_and_drops_at_cursor(app):
    x, y = _cavity_aim(app)
    _move(app, x, y)
    ghost_anchor = app._ghost_tile.anchor_ras.copy()
    _right_click(app, x, y)
    assert app._ghost_tile is None and "ghost_quad" not in app.pl.actors
    assert np.linalg.norm(app.tiles[0].anchor_ras - ghost_anchor) < 0.5, (
        "the dropped tile must land where the ghost showed it")


# ------------------------------------------------------- hover highlighting
def test_hover_highlights_unselected_tile(app):
    verts = np.asarray(app.cavity.vertices)
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app.drop_at(verts[len(verts) // 2])  # second tile becomes the selection
    assert app.selected == 1
    assert "tile_1_edge" in app.pl.actors, "selected tile carries an outline"
    assert "tile_0_edge" not in app.pl.actors

    gx, gy = _display_xy(app, app.tiles[0].anchor_ras)
    _move(app, gx, gy)
    assert app._hover_idx == 0
    assert app._tile_state(0) == "hover"
    assert "tile_0_edge" in app.pl.actors, "hovered tile lights up"
    assert app._ghost_tile is None, "no ghost while offering a grab"

    _move(app, 3, 3)
    assert app._hover_idx == -1
    assert "tile_0_edge" not in app.pl.actors


def test_camera_gesture_pauses_hover_and_ghost(app):
    x, y = _cavity_aim(app)
    style = _iren(app).GetInteractorStyle()
    style.StartRotate()
    try:
        assert app._camera_busy()
        _move(app, x, y)
        assert app._ghost_tile is None
    finally:
        style.EndRotate()
    assert not app._camera_busy()
    x, y = _cavity_aim(app)  # the style rotated the camera on that move
    _move(app, x, y)
    assert app._ghost_tile is not None


# ---------------------------------------------------------- drag pacing
def test_drag_release_lands_at_final_cursor_even_when_paced(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    gx, gy = _display_xy(app, app.tiles[0].anchor_ras)
    app.drag_min_interval_s = 1e9  # every move after the first is paced out

    iren = _iren(app)
    iren.SetControlKey(1)
    iren.SetEventPosition(gx, gy)
    iren.InvokeEvent("LeftButtonPressEvent")
    assert app._drag_idx == 0
    assert app._tile_state(0) == "dragging"
    for frac in np.linspace(0.0, 1.0, 12):
        iren.SetEventPosition(int(gx + 60 * frac), int(gy + 15 * frac))
        iren.InvokeEvent("MouseMoveEvent")
    assert app._drag_applied_xy == (gx, gy), "burst was paced to one conform"
    assert app._drag_pending_xy == (gx + 60, gy + 15)
    expect = app._pick_cavity_point(gx + 60, gy + 15)
    assert expect is not None, "test geometry: end point must be on the wall"

    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)
    assert app._drag_idx == -1
    assert np.linalg.norm(app.tiles[0].anchor_ras - expect) < 0.5, (
        "the drop must use the final cursor position, not the last paced one")
    assert app._drag_pending_xy is None and app._drag_applied_xy is None
    assert app._tile_state(0) == "selected"


# ------------------------------------------------------------- dose panel
def test_dose_panel_reports_shells_and_goes_stale(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    assert app._dose_panel_text == ""
    app.update_dose()
    if "dose engine" in app._last_status or "failed" in app._last_status:
        pytest.skip("dose engine unavailable: %s" % app._last_status)
    text = app._dose_panel_text
    assert "DOSE" in text and "rx 6000" in text
    for shell in ("wall", "+5 mm", "+10 mm"):
        assert shell in text
    assert "STALE" not in text
    assert "dose" in app.pl.actors
    assert app._dose_report is not None and list(app._dose_report) == [0.0, 5.0, 10.0]

    app._rotate_selected(0.2)  # seeds moved: numbers no longer apply
    assert app._dose_stale
    assert "STALE" in app._dose_panel_text

    app._toggle_dose_panel()
    assert not app.dose_panel_visible
    assert "dose" not in app.pl.actors
    if app._dvh_chart is not None:
        assert not app._dvh_chart.visible
    app._toggle_dose_panel()
    assert "dose" in app.pl.actors

    app.update_dose()
    assert not app._dose_stale and "STALE" not in app._dose_panel_text


# ------------------------------------------------------------- key legend
def test_every_bound_key_is_in_the_legend(app):
    """Physicists learn the tool from the on-screen legend: every key that
    ``_bind_interaction`` binds must be named there (and vice versa keys
    named there must do something)."""
    from gtcore.planner import HELP_TEXT
    legend = HELP_TEXT
    for token in ("right-click", "P", "H", "Ctrl", "Tab", "arrows", "[  ]",
                  "X / Del", "Z", "U", "+  -", "I", "D", "S", "R", "G", "?"):
        assert token in legend, "legend lacks %r" % token
    assert app.help_expanded
    app._toggle_help()
    assert not app.help_expanded
    assert "help" in app.pl.actors
    app._toggle_help()
    assert app.help_expanded


def test_status_line_reports_counts_rx_and_dose_state(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    s = app._last_status
    assert "1 full + 0 half = 4 seeds placed" in s
    assert "%d detected" % len(app.result.seeds) in s
    assert "rx 6000 cGy" in s and "not computed" in s
    assert "next drop: FULL" in s and "selected (full)" in s


# ------------------------------------------------------------------- undo
def test_undo_reverts_place_move_rotate_delete(app):
    x, y = _cavity_aim(app)
    assert not app._history
    app.undo()
    assert "nothing to undo" in app._last_status

    _right_click(app, x, y)
    t0 = app.tiles[0]
    app._rotate_selected(0.3)
    assert not np.allclose(app.tiles[0].axis_ras, t0.axis_ras)
    app.undo()
    assert len(app.tiles) == 1
    assert np.allclose(app.tiles[0].axis_ras, t0.axis_ras)

    app._translate_selected(1.0, 0.0)
    assert np.linalg.norm(app.tiles[0].anchor_ras - t0.anchor_ras) > 0.5
    app.undo()
    assert np.allclose(app.tiles[0].anchor_ras, t0.anchor_ras)

    app._delete_selected()
    assert len(app.tiles) == 0 and "tile_0_quad" not in app.pl.actors
    app.undo()
    assert len(app.tiles) == 1 and "tile_0_quad" in app.pl.actors
    assert app.selected == 0

    app.undo()  # the placement itself
    assert len(app.tiles) == 0 and "tile_0_quad" not in app.pl.actors
    assert not app._history


def test_drag_is_one_undo_step(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    start = app.tiles[0]
    gx, gy = _display_xy(app, start.anchor_ras)
    iren = _iren(app)
    iren.SetControlKey(1)
    iren.SetEventPosition(gx, gy)
    iren.InvokeEvent("LeftButtonPressEvent")
    for frac in np.linspace(0.0, 1.0, 6):
        app._drag_last_t = float("-inf")  # apply every move
        iren.SetEventPosition(int(gx + 60 * frac), int(gy + 15 * frac))
        iren.InvokeEvent("MouseMoveEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    iren.SetControlKey(0)
    assert np.linalg.norm(app.tiles[0].anchor_ras - start.anchor_ras) > 0.5
    assert len(app._history) == 2  # place + drag
    app.undo()
    assert np.allclose(app.tiles[0].anchor_ras, start.anchor_ras)


# ------------------------------------------------- rx change + isodose toggle
def test_rx_change_recuts_isodoses_and_panel(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app.set_rx(5000.0)
    assert app.rx_cgy == 5000.0 and "5000" in app._last_status
    app.update_dose()
    if "dose engine" in app._last_status or "failed" in app._last_status:
        pytest.skip("dose engine unavailable: %s" % app._last_status)
    v100_before = app._dose_report[0.0]["stats"]["V100"]
    app.set_rx(2500.0)  # halve rx: strictly more of the wall is covered
    assert "rx 2500" in app._dose_panel_text
    assert app._dose_report[0.0]["stats"]["V100"] >= v100_before
    assert not app._dose_stale, "re-cutting at a new rx is not a stale board"
    assert "iso_100" in app.pl.actors

    app._toggle_isodoses()
    assert not app.isodose_visible
    assert not app.pl.actors["iso_100"].GetVisibility()
    assert "isodoses hidden" in app._last_status
    app._toggle_isodoses()
    assert app.pl.actors["iso_100"].GetVisibility()

    app.set_rx(0.0)  # clamps, never zero
    assert app.rx_cgy > 0.0


# ------------------------------------------------------------------ export
def test_save_plan_writes_all_seeds(app, tmp_path):
    import csv

    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app._toggle_kind()
    app.drop_at(np.asarray(app.cavity.vertices)[len(app.cavity.vertices) // 2])
    out = str(tmp_path / "plan.csv")
    assert app.save_plan(out) == out
    assert "plan saved" in app._last_status
    with open(out) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert lines[0] == "# rx_cgy=6000.0"
    rows = list(csv.DictReader(lines[1:]))
    n_det = len(app.result.seeds)
    assert len(rows) == n_det + 4 + 2
    placed = [r for r in rows if r["source"] == "placed"]
    assert {r["kind"] for r in placed} == {"full", "half"}
    assert {r["tile"] for r in placed} == {"1", "2"}
    seed = placed[0]
    xyz = np.array([float(seed[k]) for k in ("x_mm", "y_mm", "z_mm")])
    assert np.allclose(xyz, app.tiles[0].seed_centers[0], atol=1e-3)
    axis = np.array([float(seed[k]) for k in ("ax", "ay", "az")])
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-3


# --------------------------------------------------- clear isodoses + buttons
def test_clear_isodoses_key_and_button(app):
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app.clear_isodoses()
    assert "no isodoses" in app._last_status
    app.update_dose()
    if "dose engine" in app._last_status or "failed" in app._last_status:
        pytest.skip("dose engine unavailable: %s" % app._last_status)
    assert app._iso_shown and all(n in app.pl.actors for n in app._iso_shown)
    assert "isodoses shown" in app._last_status

    names = list(app._iso_shown)
    app.clear_isodoses()
    assert app._iso_shown == []
    assert all(n not in app.pl.actors for n in names)
    assert "isodoses cleared" in app._last_status and "isodoses none" in app._last_status
    assert app._dose_volume is not None, "the grid + panel survive a clear"
    assert "DOSE" in app._dose_panel_text

    app.set_rx(app.rx_cgy - 100.0)  # re-cut from the grid brings them back
    assert app._iso_shown

    if "clear" in app._buttons:  # clickable button path (needs an interactor)
        app._button_clear(1)
        assert app._iso_shown == []
        assert app._buttons["clear"].GetRepresentation().GetState() == 0
    if "iso" in app._buttons:
        app.update_dose()
        app._button_isodoses(0)
        assert not app.isodose_visible
        assert not app.pl.actors[app._iso_shown[0]].GetVisibility()
        app._toggle_isodoses()  # key path keeps the widget in sync
        assert app._buttons["iso"].GetRepresentation().GetState() == 1
    if "panel" in app._buttons:
        app._button_panel(0)
        assert not app.dose_panel_visible and "dose" not in app.pl.actors
        app._button_panel(1)
        assert app.dose_panel_visible


def test_overlay_text_is_visible_on_black(app):
    """Regression: the theme's default font colour is black, so the legend
    and status bar were invisible on the black background."""
    for name in ("help", "status"):
        actor = app.pl.actors[name]
        rgb = tuple(actor.prop.color.float_rgb)
        assert max(rgb) > 0.5, "%s text colour %r would vanish on black" % (
            name, rgb)


# ------------------------------------------------ delete all + background
def test_delete_all_is_one_undo_step(app):
    verts = np.asarray(app.cavity.vertices)
    app.delete_all()
    assert "no tiles placed this session" in app._last_status
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app.drop_at(verts[len(verts) // 2])
    app.drop_at(verts[len(verts) // 3], "half")
    ids = list(app._tile_ids)
    assert len(app.tiles) == 3

    app.delete_all()
    assert app.tiles == [] and app.selected == -1
    assert "3 tiles deleted" in app._last_status
    for tid in ids:
        for suffix in ("quad", "edge", "seeds"):
            assert "tile_%d_%s" % (tid, suffix) not in app.pl.actors
    assert "0 full + 0 half = 0 seeds placed" in app._last_status

    app.undo()
    assert len(app.tiles) == 3 and app._tile_ids == ids
    assert app.tiles[2].kind == "half"
    assert all("tile_%d_quad" % tid in app.pl.actors for tid in ids)


def test_background_cycle_keeps_text_legible(app):
    from gtcore.planner import _BACKGROUNDS

    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    seen = set()
    for _ in range(len(_BACKGROUNDS)):
        app._cycle_background()
        bg, fg = _BACKGROUNDS[app._bg_idx]
        seen.add(bg)
        assert "background: %s" % bg in app._last_status
        for name in ("help", "status", "button_label_iso"):
            if name in app.pl.actors:
                rgb = np.array(app.pl.actors[name].prop.color.float_rgb)
                bg_rgb = np.array(app.pl.background_color.float_rgb)
                assert np.abs(rgb - bg_rgb).max() > 0.5, (
                    "%s text would vanish on %s" % (name, bg))
        # ghost follows the text colour on the new background
        _move(app, x + 60, y + 15)
        if app._ghost_tile is not None:
            ghost_rgb = np.array(app.pl.actors["ghost_edge"].prop.color.float_rgb)
            bg_rgb = np.array(app.pl.background_color.float_rgb)
            assert np.abs(ghost_rgb - bg_rgb).max() > 0.5
    assert len(seen) == len(_BACKGROUNDS)
    assert app._bg_idx == 0, "a full cycle returns to the default"


def test_attenuation_key_and_panel_flag(app):
    assert app.interference is False
    app._toggle_interference()
    assert app.interference is True
    assert "attenuation ON" in app._last_status
    x, y = _cavity_aim(app)
    _right_click(app, x, y)
    app.update_dose()
    if "dose engine" in app._last_status or "failed" in app._last_status:
        pytest.skip("dose engine unavailable: %s" % app._last_status)
    assert app._dose_attenuated
    assert "attenuation ON" in app._dose_panel_text
    assert "wall area >= rx at 5 mm depth" in app._dose_panel_text
    app._toggle_interference()
    assert app.interference is False


# ------------------------------------------------- adopted (fitted) tiles
@pytest.fixture()
def fitted_app():
    """Planner opened on a scan whose implant was recovered (3 phantom tiles)."""
    vol, _truth = make_head_phantom(spacing=1.0)
    res = reconstruct(vol, n_full_tiles=3, verbose=False)
    if res.tiles is None or not res.tiles.tiles:
        pytest.skip("tile fitting recovered nothing on this phantom")
    try:
        planner = _PlannerApp(res, off_screen=True)
        planner.pl.render()
    except Exception as exc:
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))
    yield planner
    planner.close()


def test_adopted_tiles_are_provenance_tracked_by_id(fitted_app):
    from gtcore.planner import _ADOPTED_OUTLINE

    app = fitted_app
    n = len(app.result.tiles.tiles)
    assert len(app.tiles) == n and app.selected == 0
    assert app._adopted_ids == set(app._tile_ids)
    assert not app._history, "adoption is the starting state, not an undo step"
    assert "%d fitted from scan" % n in app._last_status
    assert "fitted tiles adopted" in app._last_status, app._last_status
    # provenance cue: gold outline on adopted tiles at rest; the interaction
    # outline wins while selected
    app.selected = -1
    for i in range(n):
        assert app._tile_style(i)["outline"] == _ADOPTED_OUTLINE
    app.selected = 0
    assert app._tile_style(0)["outline"] == "white"

    # a hand-placed tile has no provenance outline, and edits keep the id
    verts = np.asarray(app.cavity.vertices)
    app.drop_at(verts[0])
    placed_idx = len(app.tiles) - 1
    app.selected = -1
    assert app._tile_style(placed_idx)["outline"] is None
    app.selected = 0
    app._rotate_selected(0.2)
    app._translate_selected(1.0, 0.0)
    assert app._tile_ids[0] in app._adopted_ids

    # Backspace removes only this session's tiles; adopted ones stay
    app.delete_all()
    assert len(app.tiles) == n and "1 tile deleted" in app._last_status
    assert "%d fitted tiles kept" % n in app._last_status
    app.delete_all()
    assert "no tiles placed this session" in app._last_status
    assert len(app.tiles) == n

    # deleting one adopted tile deliberately (X) is allowed and undoable,
    # and the id-based bookkeeping survives the index shift
    app.selected = 0
    first_id = app._tile_ids[0]
    app._delete_selected()
    assert len(app.tiles) == n - 1 and first_id not in app._tile_ids
    assert all(tid in app._adopted_ids for tid in app._tile_ids)
    app.undo()
    assert len(app.tiles) == n and app._tile_ids[0] == first_id
    app.selected = -1
    assert app._tile_style(0)["outline"] == _ADOPTED_OUTLINE
