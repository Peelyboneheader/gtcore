"""Interactive intraoperative tile planner (PyVista front-end).

Builds on the :mod:`gtcore.viz` scene style and adds tile drag-and-drop:
pick a point on the cavity wall to drop a conformed GammaTile, then slide,
spin, or delete it, and update TG-43 isodose surfaces over detected + placed
seeds on demand.

Feedback layers, all driven from the VTK event pipeline:

- **ghost tile** -- while the cursor hovers over the cavity wall a
  translucent conformed preview of the *next* drop follows it (red when that
  drop would overlap a placed tile), so the user sees size, orientation and
  draping before committing;
- **hover / selection / drag styling** -- the tile under the cursor lights
  up as grab-able, the selected tile carries a white outline, the tile being
  dragged turns cyan, overlapping tiles are tinted red;
- **dose panel** -- after 'u' a fixed-width table (upper right) reports
  D90/D50/Dmin/V100/V150 on the cavity wall and on +5/+10 mm tissue shells,
  plus a cumulative shell-DVH chart (lower right); it is flagged STALE as
  soon as any tile moves until the next 'u'.

Like ``viz.py``, this module is a rendering front-end and the only other
place in gtcore allowed to touch pyvista -- and it imports it lazily, inside
functions, so the algorithm core never depends on a rendering stack.  All
geometry lives in :mod:`gtcore.interact` (pure numpy/trimesh) and the DVH
maths in :mod:`gtcore.dose.dvh`.

Dose contract (implemented by :mod:`gtcore.dose`, loaded lazily on 'u')::

    compute_dose_grid(seed_centers, seed_axes, bounds_ras,
                      spacing_mm=2.0, sk_per_seed_u=3.5,
                      interference=None) -> Volume
    isodose_surfaces(dose_volume, levels_cgy) -> {level_cgy: trimesh.Trimesh}
    InterferenceModel.from_implant(seed_centers, seed_axes) -> model

``bounds_ras`` is a (2, 3) array of RAS min/max corners.  If the functions
are missing or fail, the planner shows a message and keeps running.

'A' toggles inter-seed attenuation (:mod:`gtcore.dose.interference`): the
seeds and tiles already on the board shadow one another, which plain TG-43
superposition cannot express.  It is off by default so what the planner
shows matches the formalism unless the user asks for the correction; tile
carriers stay out of the model entirely, since their contribution rests on
an unmeasured density (see docs/interference-notes.md).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .interact import (
    PlacedTile,
    conform_tile,
    rotate_on_wall,
    snap_to_wall,
    tiles_to_seed_arrays,
    translate_on_wall,
)
from .pipeline import PipelineResult

SEED_LENGTH_MM = 4.5
TRANSLATE_STEP_MM = 2.0
ROTATE_STEP_RAD = np.deg2rad(10.0)
DOSE_PAD_MM = 40.0
DOSE_SPACING_MM = 2.0
OVERLAP_THRESHOLD_MM = 1.0  # passed to interact.find_overlapping_tiles
SHELL_OFFSETS_MM = (0.0, 5.0, 10.0)  # dose panel: wall, +5 mm, +10 mm tissue
WALL_DEPTH_MM = 5.0         # GammaTile prescription depth (60 Gy at 5 mm)
DOSE_PANEL_POS = (0.62, 0.99)  # viewport anchor (top-left corner) of the panel

# Interaction pacing.  snap+conform costs ~13 ms and a tile redraw ~5 ms on
# the phantom cavity, so re-conforming on EVERY MouseMoveEvent stalls the
# event queue and the tile visibly lags the cursor; skipping every other
# event (the previous scheme) alternates between stall and skip and feels
# jittery.  Wall-clock pacing conforms at a steady rate however fast the
# mouse reports, and the final cursor position is always applied on release
# so a drop never lands one throttled move short of where the user let go.
DRAG_MIN_INTERVAL_S = 0.025   # ~40 conforms/s while dragging
GHOST_MIN_INTERVAL_S = 0.030  # ~33 preview updates/s while hovering

RX_STEP_CGY = 100.0
UNDO_DEPTH = 50

# Full key legend (monospace, upper left).  Every binding in
# ``_bind_interaction`` appears here, grouped the way a physicist works:
# place -> adjust -> evaluate -> export.  '?' collapses it to one line.
HELP_TEXT = """GAMMATILE PLANNER                       ?  hide/show this legend
PLACE    hover the blue wall : ghost preview of the next tile (red = overlap)
         gold outline        : tile fitted FROM THE SCAN (green = placed by hand)
         right-click or P    : drop tile there      H : next tile full/half
ADJUST   left-drag ON a tile : grab it (quad or seeds) and slide it along the wall
         Ctrl + left-drag    : slide the SELECTED tile from anywhere on the wall
         Tab                 : select next tile     arrows : nudge 2 mm
         [  ]                : rotate 10 deg        X / Del : delete tile
         Backspace           : delete ALL placed tiles
         Z                   : undo last change
DOSE     U                   : compute TG-43 dose, isodoses + dose panel
         A                   : inter-seed attenuation on/off (applies at next U)
         +  -                : prescription +/- 100 cGy (isodoses re-cut)
         I                   : isodoses on/off      C : clear isodoses
         D                   : dose panel on/off    (or click the buttons)
         isodose colours     : 100% red   50% orange   25% yellow
EXPORT   S                   : save plan (seed coordinates) to output/*.csv
VIEW     left-drag (off tiles) rotate   right-drag zoom   middle-drag pan   R reset
         G                   : ghost preview on/off      B : background colour"""
HELP_TEXT_COMPACT = ("? legend   right-click drop   Ctrl+drag move   "
                     "U dose   S save")

_ISO_STYLE = (  # (fraction of rx, actor name, color)
    (1.00, "iso_100", "red"),
    (0.50, "iso_50", "orange"),
    (0.25, "iso_25", "yellow"),
)

# Tile quad styling per interaction state; the overlap tint overrides colour.
# ``outline`` is the colour of a separate 4-edge loop actor (None = no loop).
_TILE_STYLE: Dict[str, Dict] = {
    "normal":   dict(color="green", opacity=0.45, outline=None, line_width=1),
    "hover":    dict(color="mediumseagreen", opacity=0.60,
                     outline="palegreen", line_width=2),
    "selected": dict(color="springgreen", opacity=0.75,
                     outline="white", line_width=3),
    "dragging": dict(color="cyan", opacity=0.85,
                     outline="white", line_width=4),
}
_OVERLAP_COLOR = {"normal": "red", "hover": "tomato",
                  "selected": "orangered", "dragging": "orangered"}
_ADOPTED_OUTLINE = "gold"  # provenance cue: tile recovered from the scan
_GHOST_OVERLAP_COLOR = "red"  # otherwise the ghost takes the text colour
# (background, text colour) pairs cycled by 'B'; text follows so the legend,
# status, panel and ghost stay legible on every one of them
_BACKGROUNDS = (("black", "white"), ("#1e2a38", "white"),
                ("dimgray", "white"), ("white", "black"))
_SHELL_COLOR = ("deepskyblue", "black", "gray")  # DVH chart lines


def _to_pv(pv, tm):
    faces = np.hstack(
        [np.full((len(tm.faces), 1), 3, dtype=np.int64), np.asarray(tm.faces)]
    ).ravel()
    return pv.PolyData(np.asarray(tm.vertices), faces)


def _quad_polydata(pv, tile):
    return pv.PolyData(
        np.asarray(tile.corners_ras, dtype=float),
        np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64),
    )


def _outline_polydata(pv, tile):
    """Closed 4-edge loop of the draped tile (show_edges would also draw
    the diagonal of the two-triangle quad)."""
    return pv.PolyData(np.asarray(tile.corners_ras, dtype=float),
                       lines=np.array([5, 0, 1, 2, 3, 0], dtype=np.int64))


def _seed_polydata(pv, tile):
    polys = None
    for c, a in zip(tile.seed_centers, tile.seed_axes):
        cyl = pv.Cylinder(center=c, direction=a, radius=0.6,
                          height=SEED_LENGTH_MM)
        polys = cyl if polys is None else polys + cyl
    return polys


class _PlannerApp:
    """State + rendering for the planner; drives one pyvista Plotter."""

    def __init__(self, result: PipelineResult, rx_cgy: float = 6000.0,
                 off_screen: bool = False, title: str = "GammaTile planner"):
        import pyvista as pv

        self.pv = pv
        self.result = result
        self.rx_cgy = float(rx_cgy)
        # interaction surface: the cavity wall, or -- for phantoms and other
        # scans without a segmented cavity -- the object/body shell, so tiles
        # can still be placed, selected and dragged on something real
        self.cavity = result.meshes.get("cavity")
        self._surface_label = "cavity wall"
        if self.cavity is None or not len(getattr(self.cavity, "vertices", ())):
            body = result.meshes.get("body")
            if body is not None and len(body.vertices):
                self.cavity = body
                self._surface_label = "phantom shell (no cavity segmented)"

        self.tiles: List[PlacedTile] = []
        self._tile_ids: List[int] = []
        # ids (not indices: deletes shift indices, edits keep ids) of tiles
        # recovered FROM THE SCAN -- the implant physically in the patient,
        # as opposed to proposals dropped by hand
        self._adopted_ids = set()
        self._next_id = 0
        self.selected = -1
        self.next_kind = "full"
        self.interference = False    # inter-seed attenuation for the next U
        self._bg_idx = 0
        self._fg = _BACKGROUNDS[0][1]  # text colour for the current background

        self._cavity_actor = None
        self._tile_actors = {}       # tile id -> quad vtkActor (for grabbing)
        self._seed_actors = {}       # tile id -> seed-capsule vtkActor (grabbable too)
        self._cavity_picker = None   # vtkCellPicker restricted to the cavity
        self._tile_picker = None     # vtkCellPicker restricted to tile quads
        self._overlap_pairs: List = []
        self._last_status = ""       # last status text (also for headless tests)

        # drag state
        self._drag_idx = -1          # index of the tile being dragged, or -1
        self._drag_last_t = float("-inf")  # first move conforms at once
        self._drag_pending_xy = None  # latest cursor pos seen while dragging
        self._drag_applied_xy = None  # cursor pos the tile was last conformed at
        self.drag_min_interval_s = DRAG_MIN_INTERVAL_S

        # hover + ghost preview state
        self._hover_idx = -1
        self.ghost_enabled = True
        self._ghost_tile: Optional[PlacedTile] = None
        self._ghost_overlaps = False
        self._ghost_last_t = 0.0
        self.ghost_min_interval_s = GHOST_MIN_INTERVAL_S

        self._history: List = []     # undo stack of (tiles, tile_ids)
        self.help_expanded = True
        self.isodose_visible = True
        self._iso_shown: List[str] = []  # isodose actor names on screen
        self._buttons = {}           # name -> vtkButtonWidget (clickable UI)

        # dose panel state
        self._dose_volume = None
        self._dose_report = None     # {offset_mm: {"stats", "curve_x", "curve_y"}}
        self._dose_n_seeds = 0
        self._dose_attenuated = False  # grid computed with the interference model
        self._dose_stale = False
        self._dose_panel_text = ""
        self.dose_panel_visible = True
        self._dvh_chart = None
        self._dvh_lines = {}

        self.pl = pv.Plotter(window_size=(1280, 900), title=title,
                             off_screen=off_screen)
        self._build_scene()
        self._bind_interaction()
        self._adopt_fitted_tiles()
        if not self._adopted_ids:
            self._update_status()

    # ------------------------------------------------------------- base scene
    def _build_scene(self):
        pv, pl = self.pv, self.pl
        pl.set_background(_BACKGROUNDS[self._bg_idx][0])
        styles = {
            "skull": dict(color="ivory", opacity=0.12),
            "brain": dict(color="rosybrown", opacity=0.25),
            "body": dict(color="lightsteelblue", opacity=0.25),  # phantom shell
            "cavity": dict(color="deepskyblue", opacity=0.55, specular=0.4),
        }
        for name, style in styles.items():
            mesh = self.result.meshes.get(name)
            if mesh is not None and len(mesh.vertices):
                actor = pl.add_mesh(_to_pv(pv, mesh), name=name, **style)
                if mesh is self.cavity:  # whichever mesh is the pick surface
                    self._cavity_actor = actor

        for c, a in zip(self.result.seeds.centers_ras,
                        self.result.seeds.axes_ras):
            pl.add_mesh(pv.Cylinder(center=c, direction=a, radius=0.6,
                                    height=SEED_LENGTH_MM),
                        color="gold", specular=0.8)

        self._draw_help()
        self._add_buttons()
        pl.add_axes(line_width=2)
        pl.camera_position = "yz"
        pl.camera.azimuth = 135
        pl.camera.elevation = 15

    def _add_buttons(self):
        """Clickable on-screen buttons above the DVH chart (lower right).

        Each mirrors a key so nothing depends on remembering the legend:
        isodoses on/off (I), clear isodoses (C), dose panel on/off (D).
        Widgets need a live interactor; any failure just leaves the keys.
        """
        pl = self.pl
        try:
            w, h = pl.window_size
        except Exception:
            w, h = 1280, 900
        size = 22
        y = int(0.32 * h)
        specs = (  # (name, label, initial state, callback(state))
            ("iso", "isodoses", True, self._button_isodoses),
            ("clear", "clear isodoses", False, self._button_clear),
            ("panel", "dose panel", True, self._button_panel),
        )
        x = int(0.69 * w)
        for name, label, state, cb in specs:
            try:
                widget = pl.add_checkbox_button_widget(
                    cb, value=state, position=(x, y), size=size,
                    border_size=2, color_on="springgreen", color_off="dimgray",
                    background_color="black")
                self._buttons[name] = widget
                pl.add_text(label, position=((x + size + 6) / w, (y + 4) / h),
                            viewport=True, font_size=8, color=self._fg,
                            name="button_label_%s" % name, render=False)
            except Exception:
                continue
            x += size + 12 + 8 * len(label)

    def _set_button(self, name, state):
        widget = self._buttons.get(name)
        if widget is None:
            return
        try:
            widget.GetRepresentation().SetState(1 if state else 0)
        except Exception:
            pass

    def _button_isodoses(self, state):
        if bool(state) != self.isodose_visible:
            self._toggle_isodoses()

    def _button_clear(self, _state):
        self.clear_isodoses()
        self._set_button("clear", False)  # momentary, not a toggle

    def _button_panel(self, state):
        if bool(state) != self.dose_panel_visible:
            self._toggle_dose_panel()

    def _bind_interaction(self):
        pl = self.pl
        self._bind_pick_observers()
        for keys, fn in (
            (("p", "P"), self._place_at_mouse),
            (("h", "H"), self._toggle_kind),
            (("g", "G"), self._toggle_ghost),
            (("d", "D"), self._toggle_dose_panel),
            (("i", "I"), self._toggle_isodoses),
            (("c", "C"), self.clear_isodoses),
            (("a", "A"), self._toggle_interference),
            (("z", "Z"), self.undo),
            (("s", "S"), self.save_plan),
            (("question", "?"), self._toggle_help),
            (("plus", "equal", "KP_Add"), lambda: self.set_rx(self.rx_cgy + RX_STEP_CGY)),
            (("minus", "KP_Subtract"), lambda: self.set_rx(self.rx_cgy - RX_STEP_CGY)),
            (("Tab",), self._cycle_selection),
            (("x", "X", "Delete"), self._delete_selected),
            (("BackSpace",), self.delete_all),
            (("b", "B"), self._cycle_background),
            (("u", "U"), self.update_dose),
            (("bracketleft", "["), lambda: self._rotate_selected(-ROTATE_STEP_RAD)),
            (("bracketright", "]"), lambda: self._rotate_selected(+ROTATE_STEP_RAD)),
            (("Up",), lambda: self._translate_selected(0.0, +1.0)),
            (("Down",), lambda: self._translate_selected(0.0, -1.0)),
            (("Left",), lambda: self._translate_selected(-1.0, 0.0)),
            (("Right",), lambda: self._translate_selected(+1.0, 0.0)),
        ):
            for key in keys:
                try:
                    pl.add_key_event(key, fn)
                except Exception:
                    pass

    def _bind_pick_observers(self):
        """Bind mouse gestures directly on the VTK interactor.

        Placement and drag both go through explicit ``vtkCellPicker``s with
        pick lists (cavity actor for placement/slide targets, tile quads for
        grabbing), so a click can never land on the translucent skull/brain
        in front of the cavity, and a miss is *known* to be a miss so it can
        be reported on screen.  (The previous implementation stacked
        ``enable_surface_point_picking`` -- which binds its own right-click
        handler -- on top of ``track_click_position`` -- whose
        ``vtkPointPicker`` picks any actor's nearest vertex: one click placed
        several tiles, centimetres from the aim point, and when no cavity
        mesh existed nothing was bound at all so clicks died silently.)

        Observers are registered at higher priority than the camera
        interactor style; a handler returning True aborts the event so the
        camera never fights a tile grab.  Bound even without a cavity mesh:
        placement attempts must always produce on-screen feedback.
        """
        try:
            import vtk
        except ImportError:  # rendering stack without separate vtk namespace
            return
        iren = getattr(self.pl.iren, "interactor", None) \
            if self.pl.iren is not None else None
        if iren is None:
            return

        self._cavity_picker = vtk.vtkCellPicker()
        self._cavity_picker.SetTolerance(0.0005)
        self._cavity_picker.PickFromListOn()
        if self._cavity_actor is not None:
            self._cavity_picker.AddPickList(self._cavity_actor)
        self._tile_picker = vtk.vtkCellPicker()
        self._tile_picker.SetTolerance(0.01)  # generous: a grab must not feel dead
        self._tile_picker.PickFromListOn()

        def _add(event, handler):
            holder = {}

            def _cb(_caller, _event, _handler=handler, _holder=holder):
                try:
                    handled = bool(_handler())
                except Exception:
                    handled = False
                cmd = _holder.get("cmd")
                if cmd is not None:
                    try:  # abort so the camera style never sees the event
                        cmd.SetAbortFlag(1 if handled else 0)
                    except Exception:
                        pass

            tag = iren.AddObserver(event, _cb, 30.0)
            try:
                holder["cmd"] = iren.GetCommand(tag)
            except Exception:
                holder["cmd"] = None

        _add("RightButtonPressEvent", self._on_right_press)
        _add("LeftButtonPressEvent", self._on_left_press)
        _add("MouseMoveEvent", self._on_mouse_move)
        _add("LeftButtonReleaseEvent", self._on_left_release)
        _add("LeaveEvent", self._on_leave)

        # Swallow VTK's BUILT-IN 'p' prop-pick (the interactor style's char
        # handler draws a red bounding box around whatever actor is under the
        # cursor -- users saw that instead of tile placement). Our own 'p'
        # binding rides KeyPressEvent, which still fires; only the CharEvent
        # that reaches the style is aborted.
        def _swallow_pick_char():
            try:
                return iren.GetKeySym() in ("p", "P")
            except Exception:
                return False

        _add("CharEvent", _swallow_pick_char)

    def _camera_busy(self):
        """True while the camera style owns a rotate/zoom/pan gesture.

        Hover and ghost updates pause then: they would fight the camera for
        CPU and flicker across the moving wall.  ``vtkInteractorStyle`` keeps
        its own state (VTKIS_NONE == 0 when idle), which is more reliable
        than tracking button events ourselves -- a release can reach the
        style without reaching a second observer.
        """
        try:
            style = self.pl.iren.interactor.GetInteractorStyle()
            return int(style.GetState()) != 0
        except Exception:
            return False

    # ----------------------------------------------------------- mouse picking
    def _mouse_xy(self):
        try:
            return self.pl.iren.interactor.GetEventPosition()
        except Exception:
            return None

    def _pick_cavity_point(self, x, y):
        """Cast the cursor ray against the CAVITY actor only; None on miss."""
        if self._cavity_picker is None or self._cavity_actor is None:
            return None
        try:
            self._cavity_picker.Pick(x, y, 0, self.pl.renderer)
            if self._cavity_picker.GetCellId() < 0:
                return None
            pt = np.asarray(self._cavity_picker.GetPickPosition(), dtype=float)
        except Exception:
            return None
        if pt.size != 3 or not np.all(np.isfinite(pt)):
            return None
        return pt

    def _pick_tile_index(self, x, y):
        """Index of the placed tile whose quad is under the cursor, or -1."""
        picker = self._tile_picker
        if picker is None:
            return -1
        try:
            picker.InitializePickList()
            order = []
            for i, tid in enumerate(self._tile_ids):
                for actor in (self._tile_actors.get(tid),
                              self._seed_actors.get(tid)):
                    if actor is not None:
                        picker.AddPickList(actor)
                        order.append((actor, i))
            if not order:
                return -1
            picker.Pick(x, y, 0, self.pl.renderer)
            if picker.GetCellId() < 0:
                return -1
            picked = picker.GetActor()
            if picked is None:
                return -1
            addr = picked.GetAddressAsString("vtkProp")
            for actor, i in order:
                if actor.GetAddressAsString("vtkProp") == addr:
                    return i
        except Exception:
            pass
        return -1

    # -------------------------------------------------------- mouse gestures
    def _on_right_press(self):
        """Right-click: drop a tile at the cursor (feedback on any outcome)."""
        xy = self._mouse_xy()
        if xy is not None:
            self.place_at_screen(xy[0], xy[1])
        return False  # never abort: right-drag zoom keeps working

    def _on_left_press(self):
        """Left-press ON a tile grabs it; Ctrl+press on the wall grabs the
        selected tile; anything else is the camera.

        No modifier is needed to grab: a press that lands on any part of a
        tile (its quad or its seed capsules) cannot mean "rotate the camera
        around this tile", so it takes the tile.  Ctrl is the forgiving
        mode -- with a tile selected, a press anywhere on the wall slides
        THAT tile, so a near miss never turns into a camera spin.  A Ctrl
        press that grabs nothing says so on screen instead of feeling dead.
        """
        iren = getattr(self.pl.iren, "interactor", None) \
            if self.pl.iren is not None else None
        xy = self._mouse_xy()
        if xy is None:
            return False
        ctrl = bool(iren is not None and iren.GetControlKey())
        idx = self._pick_tile_index(xy[0], xy[1])
        if idx < 0 and ctrl:
            if 0 <= self.selected < len(self.tiles) \
                    and self._pick_cavity_point(xy[0], xy[1]) is not None:
                idx = self.selected  # forgiving: slide the selection from here
            else:
                self._hide_ghost()
                self._update_status(
                    "nothing to grab: press on a tile, or hold Ctrl and press "
                    "on the wall with a tile selected"
                    if self.tiles else "no tiles on the board to grab")
                return False
        if idx < 0:
            self._hide_ghost()  # camera rotate starts: no preview meanwhile
            return False
        self._hide_ghost()
        self._set_hover(-1)
        self._push_history()  # one undo step per drag, however long
        self._drag_idx = idx
        self._drag_last_t = float("-inf")  # first move conforms at once
        self._drag_pending_xy = self._drag_applied_xy = None
        self.selected = idx
        self._redraw_tiles()
        self._update_status("dragging tile %d -- release to drop" % (idx + 1))
        return True  # abort so the camera style never sees this press

    def _on_mouse_move(self):
        """Dragging: paced re-conform.  Otherwise: hover + ghost feedback."""
        if 0 <= self._drag_idx < len(self.tiles):
            return self._drag_move()
        if self._camera_busy():
            return False  # camera gesture in progress: leave the scene alone
        xy = self._mouse_xy()
        if xy is None:
            return False
        self._set_hover(self._pick_tile_index(xy[0], xy[1]))
        if self._hover_idx >= 0:
            self._hide_ghost()  # a tile is under the cursor: offer the grab
        else:
            self._update_ghost(xy)
        return False

    def _on_left_release(self):
        if self._drag_idx < 0:
            return False
        idx, self._drag_idx = self._drag_idx, -1
        pending, applied = self._drag_pending_xy, self._drag_applied_xy
        if pending is not None and pending != applied:
            self._apply_drag(idx, pending)  # land exactly where the mouse is
        self._drag_pending_xy = self._drag_applied_xy = None
        self._after_change("tile %d dropped" % (idx + 1))
        return True

    def _on_leave(self):
        """Cursor left the window: no preview, no hover."""
        self._set_hover(-1)
        self._hide_ghost()
        return False

    # ----------------------------------------------------------------- drag
    def _drag_move(self):
        xy = self._mouse_xy()
        if xy is None:
            return True
        self._drag_pending_xy = xy
        now = time.perf_counter()
        if now - self._drag_last_t < self.drag_min_interval_s:
            return True  # paced: the pending position is applied on release
        self._apply_drag(self._drag_idx, xy, now)
        return True

    def _apply_drag(self, idx, xy, now=None):
        """Re-conform tile ``idx`` at the wall point under ``xy``.

        Only the dragged tile is redrawn unless the overlap tint of another
        tile changed -- a full redraw costs as much as the conform itself.
        """
        self._drag_last_t = time.perf_counter() if now is None else now
        self._drag_applied_xy = xy
        pt = self._pick_cavity_point(xy[0], xy[1])
        if pt is None:
            return False  # cursor slid off the wall: tile stays where it was
        tile = self.tiles[idx]
        surf, n_in = snap_to_wall(self.cavity, pt)
        self.tiles[idx] = conform_tile(self.cavity, surf, n_in, tile.axis_ras,
                                       kind=tile.kind)
        before = self._flagged()
        self._refresh_overlaps()
        if self._flagged() != before:
            self._redraw_tiles()
        else:
            self._redraw_tile(idx)
            self._render()
        return True

    # -------------------------------------------------------- hover + ghost
    def _set_hover(self, idx):
        if idx == self._hover_idx:
            return
        old, self._hover_idx = self._hover_idx, idx
        changed = False
        for i in (old, idx):
            if 0 <= i < len(self.tiles):
                self._redraw_tile(i)
                changed = True
        if changed:
            self._render()

    def _update_ghost(self, xy):
        """Follow the cursor with a conformed preview of the next drop."""
        if not self.ghost_enabled or self.cavity is None \
                or not len(self.cavity.vertices):
            self._hide_ghost()
            return
        now = time.perf_counter()
        if now - self._ghost_last_t < self.ghost_min_interval_s:
            return
        self._ghost_last_t = now
        pt = self._pick_cavity_point(xy[0], xy[1])
        if pt is None:
            self._hide_ghost()
            return
        surf, n_in = snap_to_wall(self.cavity, pt)
        ghost = conform_tile(self.cavity, surf, n_in, self._camera_right(),
                             kind=self.next_kind)
        self._ghost_tile = ghost
        self._ghost_overlaps = self._would_overlap(ghost)
        self._draw_ghost()

    def _would_overlap(self, tile):
        """Would ``tile`` overlap any placed tile?  Defensive like the rest."""
        if not self.tiles:
            return False
        try:
            from .interact import find_overlapping_tiles
            k = len(self.tiles)
            pairs = find_overlapping_tiles(self.tiles + [tile],
                                           threshold_mm=OVERLAP_THRESHOLD_MM)
            return any(k in (int(p[0]), int(p[1])) for p in pairs)
        except Exception:
            return False

    def _draw_ghost(self):
        pv, pl = self.pv, self.pl
        tile = self._ghost_tile
        if tile is None:
            return
        color = _GHOST_OVERLAP_COLOR if self._ghost_overlaps else self._fg
        pl.add_mesh(_quad_polydata(pv, tile), name="ghost_quad", color=color,
                    opacity=0.30, pickable=False, reset_camera=False)
        pl.add_mesh(_outline_polydata(pv, tile), name="ghost_edge", color=color,
                    line_width=2, pickable=False, reset_camera=False)
        seeds = _seed_polydata(pv, tile)
        if seeds is not None:
            pl.add_mesh(seeds, name="ghost_seeds", color=color, opacity=0.45,
                        pickable=False, reset_camera=False)
        self._render()

    def _hide_ghost(self):
        if self._ghost_tile is None:
            return
        self._ghost_tile = None
        self._ghost_overlaps = False
        for name in ("ghost_quad", "ghost_edge", "ghost_seeds"):
            try:
                self.pl.remove_actor(name, render=False)
            except Exception:
                pass
        self._render()

    def _toggle_ghost(self):
        self.ghost_enabled = not self.ghost_enabled
        if not self.ghost_enabled:
            self._hide_ghost()
        self._update_status("ghost preview %s"
                            % ("on" if self.ghost_enabled else "off"))

    # ------------------------------------------------------------ placement
    def _place_at_mouse(self):
        """'P'/'p' key: drop a tile at the current mouse position."""
        xy = self._mouse_xy()
        if xy is not None:
            self.place_at_screen(xy[0], xy[1])

    def place_at_screen(self, x, y):
        """Place a tile via a screen-space pick; ALWAYS reports the outcome."""
        if self.cavity is None or not len(self.cavity.vertices):
            self._update_status(
                "no cavity surface in this scan -- nothing to place on")
            return None
        pt = self._pick_cavity_point(x, y)
        if pt is None:
            self._update_status(
                "click missed the cavity wall (aim at the blue surface)")
            return None
        return self.drop_at(pt)

    # ------------------------------------------------------------- messaging
    def _render(self):
        try:
            self.pl.render()
        except Exception:
            pass

    def _draw_help(self):
        text = HELP_TEXT if self.help_expanded else HELP_TEXT_COMPACT
        try:
            self.pl.add_text(text, font_size=9, name="help",
                             position="upper_left", font="courier",
                             color=self._fg, render=False)
            self._render()
        except Exception:
            pass

    def _toggle_help(self):
        self.help_expanded = not self.help_expanded
        self._draw_help()

    def _dose_state(self):
        if self._dose_volume is None:
            return "not computed (press U)"
        return "STALE, press U" if self._dose_stale else "up to date"

    def _update_status(self, extra: str = ""):
        n_full = sum(1 for t in self.tiles if t.kind == "full")
        n_half = len(self.tiles) - n_full
        n_placed = sum(len(t.seed_centers) for t in self.tiles)
        n_det = len(self.result.seeds)
        if 0 <= self.selected < len(self.tiles):
            sel = "tile %d/%d selected (%s)" % (
                self.selected + 1, len(self.tiles),
                self.tiles[self.selected].kind)
        else:
            sel = "no tile selected"
        # no '|' separators: VTK's text renderer draws them as wide gaps
        n_adopted = sum(1 for tid in self._tile_ids if tid in self._adopted_ids)
        text = ("tiles: %d full + %d half = %d seeds placed, %d detected%s"
                "    next drop: %s    %s\n"
                "rx %.0f cGy    dose %s    isodoses %s%s" % (
                    n_full, n_half, n_placed, n_det,
                    " (%d fitted from scan)" % n_adopted if n_adopted else "",
                    self.next_kind.upper(), sel, self.rx_cgy,
                    self._dose_state(),
                    ("none" if not self._iso_shown else
                     "shown" if self.isodose_visible else "hidden"),
                    "    surface: " + self._surface_label
                    if self._surface_label != "cavity wall" else ""))
        if extra:
            text += "\n" + extra
        for i, j in self._overlap_pairs[:4]:  # tile numbers as displayed (1-based)
            text += "\nWARNING: tiles %d & %d overlap" % (i + 1, j + 1)
        self._last_status = text
        try:
            # explicit colour: the theme default is BLACK, invisible on the
            # black background (the legend was unreadable for that reason)
            self.pl.add_text(text, font_size=10, name="status",
                             position="lower_left", color=self._fg)
            self.pl.render()
        except Exception:
            pass

    # -------------------------------------------------------------- overlaps
    def _refresh_overlaps(self):
        """Recompute overlapping tile pairs, fully defensively.

        ``find_overlapping_tiles`` may not exist yet (it is developed against
        the contract ``find_overlapping_tiles(tiles, threshold_mm=1.0) ->
        [(i, j), ...]``); any import/call/shape failure silently means
        "no overlaps" so the planner never goes down over a warning feature.
        """
        pairs = []
        if len(self.tiles) >= 2:
            try:
                from .interact import find_overlapping_tiles
                pairs = list(find_overlapping_tiles(
                    self.tiles, threshold_mm=OVERLAP_THRESHOLD_MM))
            except Exception:
                pairs = []
        clean = []
        for pair in pairs:
            try:
                i, j = int(pair[0]), int(pair[1])
            except Exception:
                continue
            if 0 <= i < len(self.tiles) and 0 <= j < len(self.tiles) and i != j:
                clean.append((i, j))
        self._overlap_pairs = clean

    def _flagged(self):
        return frozenset(i for pair in self._overlap_pairs for i in pair)

    def _after_change(self, extra: str = ""):
        """One funnel for every tile mutation: overlaps -> redraw -> status."""
        self._refresh_overlaps()
        self._redraw_tiles()
        if self._dose_report is not None and not self._dose_stale:
            self._dose_stale = True  # seeds moved: the panel no longer applies
            self._draw_dose_panel()
        self._update_status(extra)

    # ------------------------------------------------------------------ undo
    def _push_history(self):
        """Snapshot the board before a mutation (tiles are immutable values)."""
        self._history.append((list(self.tiles), list(self._tile_ids),
                              self.selected))
        del self._history[:-UNDO_DEPTH]

    def undo(self):
        if not self._history:
            self._update_status("nothing to undo")
            return
        if self._drag_idx >= 0:  # never unwind under an active grab
            self._drag_idx = -1
            self._drag_pending_xy = self._drag_applied_xy = None
        tiles, ids, selected = self._history.pop()
        for tid in set(self._tile_ids) - set(ids):
            self._remove_tile_actors(tid)
        self.tiles, self._tile_ids = tiles, ids
        self.selected = selected if 0 <= selected < len(tiles) \
            else len(tiles) - 1
        self._hover_idx = -1
        self._after_change("undo (%d left)" % len(self._history))

    def _remove_tile_actors(self, tid):
        self._tile_actors.pop(tid, None)
        self._seed_actors.pop(tid, None)
        for suffix in ("quad", "edge", "seeds"):
            try:
                self.pl.remove_actor("tile_%d_%s" % (tid, suffix),
                                     render=False)
            except Exception:
                pass

    # ------------------------------------------------------------------ tiles
    def _adopt_fitted_tiles(self):
        """Import tiles RECOVERED FROM THE SCAN as selectable placed tiles.

        Without this, Tab/drag only touch tiles the user drops by hand and
        the fitted implant is display-only -- but adjusting the *actual*
        implant is the whole point of the planner. Each fitted pose is
        re-conformed onto the interaction surface at its own centre.

        Adoption is the board's starting state, not a user action: it is not
        on the undo stack, ``delete_all`` (Backspace) leaves adopted tiles
        alone, and they carry a gold outline so the surgeon can tell the
        algorithm's belief about the implant from their own proposals.  A
        single adopted tile can still be deleted with X (and restored by Z).
        """
        fit = getattr(self.result, "tiles", None)
        if fit is None or not getattr(fit, "tiles", None):
            return
        if self.cavity is None or not len(self.cavity.vertices):
            return
        adopted = 0
        for tp in fit.tiles:
            try:
                surf, n_in = snap_to_wall(self.cavity, tp.center_ras)
                tile = conform_tile(self.cavity, surf, n_in, tp.axis_ras,
                                    kind=tp.kind)
            except Exception:
                continue
            self.tiles.append(tile)
            self._tile_ids.append(self._next_id)
            self._adopted_ids.add(self._next_id)
            self._next_id += 1
            adopted += 1
        if adopted:
            self.selected = 0
            self._after_change(
                "%d fitted tiles adopted from the scan -- Tab selects, "
                "Ctrl+left-drag moves" % adopted)

    def drop_at(self, point_ras, kind: Optional[str] = None):
        """Snap ``point_ras`` to the cavity wall and drop a conformed tile."""
        if self.cavity is None or not len(self.cavity.vertices):
            self._update_status(
                "no cavity surface in this scan -- nothing to place on")
            return None
        kind = self.next_kind if kind is None else kind
        point_ras = np.asarray(point_ras, dtype=float).reshape(3)
        surf, n_in = snap_to_wall(self.cavity, point_ras)
        # default in-plane orientation: screen-right projected onto the wall,
        # falling back to any tangent if that is degenerate
        hint = self._camera_right()
        tile = conform_tile(self.cavity, surf, n_in, hint, kind=kind)
        self._hide_ghost()  # the real tile takes the preview's place
        self._push_history()
        self.tiles.append(tile)
        self._tile_ids.append(self._next_id)
        self._next_id += 1
        self.selected = len(self.tiles) - 1
        self._after_change("tile placed (%d on board)" % len(self.tiles))
        return tile

    def _toggle_kind(self):
        self.next_kind = "half" if self.next_kind == "full" else "full"
        self._hide_ghost()  # next hover rebuilds it at the new size
        self._update_status()

    def _cycle_selection(self):
        if not self.tiles:
            self.selected = -1
        else:
            self.selected = (self.selected + 1) % len(self.tiles)
        self._redraw_tiles()
        self._update_status()

    def _delete_selected(self):
        if not (0 <= self.selected < len(self.tiles)):
            return
        self._push_history()
        tid = self._tile_ids.pop(self.selected)
        self.tiles.pop(self.selected)
        if self._drag_idx == self.selected:
            self._drag_idx = -1
        self._hover_idx = -1  # indices shifted; the next move re-picks
        self._remove_tile_actors(tid)
        self.selected = min(self.selected, len(self.tiles) - 1)
        self._after_change("tile deleted")

    def delete_all(self):
        """Remove every tile placed this session (one undo step restores
        them).  Tiles adopted from the scan stay: delete those one at a time
        with X, deliberately."""
        keep = [(t, tid) for t, tid in zip(self.tiles, self._tile_ids)
                if tid in self._adopted_ids]
        n = len(self.tiles) - len(keep)
        if n == 0:
            self._update_status(
                "no tiles placed this session to delete" + (
                    " (%d fitted tiles kept; X deletes one)" % len(keep)
                    if keep else ""))
            return
        self._push_history()
        self._drag_idx = -1
        self._hover_idx = -1
        for tid in list(self._tile_ids):
            if tid not in self._adopted_ids:
                self._remove_tile_actors(tid)
        self.tiles = [t for t, _tid in keep]
        self._tile_ids = [tid for _t, tid in keep]
        self.selected = 0 if keep else -1
        self._after_change("%d tile%s deleted (Z restores them)%s" % (
            n, "" if n == 1 else "s",
            "; %d fitted tiles kept" % len(keep) if keep else ""))

    def _cycle_background(self):
        self._bg_idx = (self._bg_idx + 1) % len(_BACKGROUNDS)
        bg, self._fg = _BACKGROUNDS[self._bg_idx]
        try:
            self.pl.set_background(bg)
        except Exception:
            pass
        # every overlay re-draws in the new text colour
        self._draw_help()
        for name in ("iso", "clear", "panel"):
            actor = self.pl.actors.get("button_label_%s" % name)
            if actor is not None:
                try:
                    actor.prop.color = self._fg
                except Exception:
                    pass
        if self._dose_volume is not None:
            self._draw_dose_panel()
        if self._ghost_tile is not None:
            self._draw_ghost()
        self._update_status("background: %s" % bg)

    def _camera_right(self):
        try:
            cam = self.pl.camera
            view = np.asarray(cam.focal_point) - np.asarray(cam.position)
            up = np.asarray(cam.up)
            right = np.cross(view, up)
            n = np.linalg.norm(right)
            if n > 1e-9:
                return right / n
        except Exception:
            pass
        return np.array([1.0, 0.0, 0.0])

    def _camera_up(self):
        try:
            up = np.asarray(self.pl.camera.up, dtype=float)
            n = np.linalg.norm(up)
            if n > 1e-9:
                return up / n
        except Exception:
            pass
        return np.array([0.0, 0.0, 1.0])

    def _translate_selected(self, sx: float, sy: float):
        """Slide the selected tile ~2 mm; screen direction (sx, sy)."""
        if not (0 <= self.selected < len(self.tiles)):
            return
        tile = self.tiles[self.selected]
        move = sx * self._camera_right() + sy * self._camera_up()
        # project the camera-relative direction onto the tile's tangent plane
        n = tile.normal_ras
        move = move - float(move @ n) * n
        norm = np.linalg.norm(move)
        if norm < 1e-6:
            return
        delta = TRANSLATE_STEP_MM * move / norm
        self._push_history()
        self.tiles[self.selected] = translate_on_wall(self.cavity, tile, delta)
        self._after_change()

    def _rotate_selected(self, angle_rad: float):
        if not (0 <= self.selected < len(self.tiles)):
            return
        self._push_history()
        self.tiles[self.selected] = rotate_on_wall(
            self.cavity, self.tiles[self.selected], angle_rad)
        self._after_change()

    # -------------------------------------------------------------- drawing
    def _tile_state(self, i):
        if i == self._drag_idx:
            return "dragging"
        if i == self.selected:
            return "selected"
        if i == self._hover_idx:
            return "hover"
        return "normal"

    def _tile_style(self, i):
        state = self._tile_state(i)
        style = dict(_TILE_STYLE[state])
        if i in self._flagged():  # red-ish tint until the overlap is resolved
            style["color"] = _OVERLAP_COLOR[state]
            if style["outline"] is not None:
                style["outline"] = "white"
        # provenance: an adopted tile at rest keeps a gold outline; the
        # interaction outline (hover/selected/dragging) takes over while active
        if style["outline"] is None and self._tile_ids[i] in self._adopted_ids:
            style["outline"] = _ADOPTED_OUTLINE
            style["line_width"] = 2
        return style

    def _redraw_tile(self, i):
        """(Re)build the actors of tile ``i`` (quad, outline, seeds); no render."""
        pv, pl = self.pv, self.pl
        tile, tid = self.tiles[i], self._tile_ids[i]
        style = self._tile_style(i)
        self._tile_actors[tid] = pl.add_mesh(
            _quad_polydata(pv, tile), name="tile_%d_quad" % tid,
            color=style["color"], opacity=style["opacity"], reset_camera=False)
        edge_name = "tile_%d_edge" % tid
        if style["outline"] is not None:
            pl.add_mesh(_outline_polydata(pv, tile), name=edge_name,
                        color=style["outline"], line_width=style["line_width"],
                        pickable=False, reset_camera=False)
        else:
            try:
                pl.remove_actor(edge_name, render=False)
            except Exception:
                pass
        seeds = _seed_polydata(pv, tile)
        if seeds is not None:
            self._seed_actors[tid] = pl.add_mesh(
                seeds, name="tile_%d_seeds" % tid, color="gold",
                specular=0.8, reset_camera=False)

    def _redraw_tiles(self):
        for i in range(len(self.tiles)):
            self._redraw_tile(i)
        self._render()

    # ------------------------------------------------------------------- dose
    def _toggle_interference(self):
        """Flip inter-seed attenuation on/off for the next dose update.

        Deliberately does NOT recompute: a dose update takes seconds, and a
        surgeon tapping a key should not trigger one by surprise.
        """
        self.interference = not self.interference
        self._update_status(
            "inter-seed attenuation %s -- press 'U' to recompute"
            % ("ON" if self.interference else "OFF"))

    def update_dose(self):
        """Recompute + redraw isodose surfaces over detected + placed seeds."""
        try:
            from . import dose as dose_mod
            compute_dose_grid = dose_mod.compute_dose_grid
            isodose_surfaces = dose_mod.isodose_surfaces
        except (ImportError, AttributeError):
            self._update_status("dose engine not available yet")
            return

        placed_c, placed_a = tiles_to_seed_arrays(self.tiles)
        det_c = np.asarray(self.result.seeds.centers_ras, dtype=float).reshape(-1, 3)
        det_a = np.asarray(self.result.seeds.axes_ras, dtype=float).reshape(-1, 3)
        centers = np.vstack([det_c, placed_c])
        axes = np.vstack([det_a, placed_a])
        if centers.shape[0] == 0:
            self._update_status("no seeds on board -- nothing to compute")
            return

        bounds = np.vstack([centers.min(axis=0) - DOSE_PAD_MM,
                            centers.max(axis=0) + DOSE_PAD_MM])
        model = None
        try:
            if self.interference:
                # Capsules only: the carrier term rests on an unmeasured
                # density (docs/interference-notes.md), so the planner never
                # applies it.
                model = dose_mod.InterferenceModel.from_implant(centers, axes)
            dose_volume = compute_dose_grid(centers, axes, bounds,
                                            spacing_mm=DOSE_SPACING_MM,
                                            interference=model)
        except (ImportError, AttributeError):
            self._update_status("dose engine not available yet")
            return
        except Exception as exc:  # engine present but unhappy: stay alive
            self._update_status("dose update failed: %s" % exc)
            return
        self._dose_volume = dose_volume
        self._dose_n_seeds = int(centers.shape[0])
        self._dose_attenuated = model is not None
        self._dose_stale = False
        self._refresh_from_dose_volume()

    def set_rx(self, rx_cgy: float):
        """Change the prescription; isodoses and panel re-cut from the grid."""
        rx = max(RX_STEP_CGY, float(rx_cgy))
        if rx == self.rx_cgy:
            return
        self.rx_cgy = rx
        if self._dose_volume is None:
            self._update_status("prescription set to %.0f cGy" % rx)
            return
        self._refresh_from_dose_volume(
            "prescription %.0f cGy: isodoses re-cut" % rx)

    def _refresh_from_dose_volume(self, note: str = ""):
        """Isodose shells + panel from the stored grid at the current rx."""
        try:
            from .dose import isodose_surfaces
            levels = [frac * self.rx_cgy for frac, _n, _c in _ISO_STYLE]
            surfaces = isodose_surfaces(self._dose_volume, levels)
        except Exception as exc:
            self._update_status("isodose extraction failed: %s" % exc)
            return
        shown = self._draw_isodoses(surfaces)
        self._dose_report = self._compute_report(self._dose_volume)
        self._draw_dose_panel()
        self._update_status(note or (
            "isodose over %d seeds: %s of rx %.0f cGy%s"
            % (self._dose_n_seeds,
               "/".join(shown) if shown else "none in grid", self.rx_cgy,
               "  (inter-seed attenuation ON)" if self._dose_attenuated
               else "")))

    def _draw_isodoses(self, surfaces):
        shown = []
        self._iso_shown = []
        for frac, name, color in _ISO_STYLE:
            try:
                self.pl.remove_actor(name, render=False)
            except Exception:
                pass
            level = frac * self.rx_cgy
            surf = None
            if isinstance(surfaces, dict):
                surf = surfaces.get(level)
                if surf is None:  # tolerate float-key jitter from the engine
                    for k, v in surfaces.items():
                        try:
                            if abs(float(k) - level) < 1e-6 * max(1.0, level):
                                surf = v
                                break
                        except (TypeError, ValueError):
                            continue
            if surf is None or getattr(surf, "vertices", None) is None \
                    or len(surf.vertices) == 0 or len(surf.faces) == 0:
                continue
            actor = self.pl.add_mesh(_to_pv(self.pv, surf), name=name,
                                     color=color, opacity=0.35,
                                     reset_camera=False)
            try:
                actor.SetVisibility(bool(self.isodose_visible))
            except Exception:
                pass
            self._iso_shown.append(name)
            shown.append("%d%%" % round(100 * frac))
        return shown

    def _toggle_isodoses(self):
        self.isodose_visible = not self.isodose_visible
        for name in self._iso_shown:
            try:
                self.pl.actors[name].SetVisibility(bool(self.isodose_visible))
            except Exception:
                pass
        self._set_button("iso", self.isodose_visible)
        self._render()
        self._update_status("isodoses %s"
                            % ("shown" if self.isodose_visible else "hidden"))

    def clear_isodoses(self):
        """Remove the isodose shells from the scene (the dose grid and the
        panel stay valid; U recomputes, +/- re-cut them from the grid)."""
        if not self._iso_shown:
            self._update_status("no isodoses on screen")
            return
        for name in self._iso_shown:
            try:
                self.pl.remove_actor(name, render=False)
            except Exception:
                pass
        self._iso_shown = []
        self._render()
        self._update_status("isodoses cleared -- U recomputes them")

    # ----------------------------------------------------------------- export
    def save_plan(self, path: Optional[str] = None):
        """Write detected + placed seeds (RAS mm) to a CSV; returns the path."""
        import os
        import time as _time

        from .interact import export_plan_csv
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            out_dir = os.path.join(root, "output")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, _time.strftime("plan_%Y%m%d_%H%M%S.csv"))
        try:
            n = export_plan_csv(path, self.tiles,
                                self.result.seeds.centers_ras,
                                self.result.seeds.axes_ras, rx_cgy=self.rx_cgy)
        except Exception as exc:
            self._update_status("save failed: %s" % exc)
            return None
        self._update_status("plan saved: %d seeds -> %s" % (n, path))
        return path

    def _compute_report(self, dose_volume):
        """Shell DVH report over the cavity wall, or None when not possible."""
        if self.cavity is None or not len(self.cavity.vertices):
            return None
        try:
            from .dose.dvh import shell_report
            return shell_report(dose_volume, self.cavity, self.rx_cgy,
                                offsets_mm=SHELL_OFFSETS_MM)
        except Exception:
            return None

    # ------------------------------------------------------------ dose panel
    def _wall_coverage(self):
        """Area fraction of the cavity wall receiving >= rx at WALL_DEPTH_MM
        (GammaTile's prescription point); None when it cannot be scored.
        Advisory: any failure is swallowed so the panel never breaks."""
        cavity = self.cavity
        if self._dose_volume is None or cavity is None \
                or getattr(cavity, "faces", None) is None or not len(cavity.faces):
            return None
        try:
            from . import dose as dose_mod
            d = dose_mod.wall_dose(cavity, WALL_DEPTH_MM,
                                   dose_volume=self._dose_volume)
            frac = float(dose_mod.surface_coverage(cavity, d, self.rx_cgy))
            return frac if np.isfinite(frac) else None
        except Exception:
            return None

    def _dose_panel_lines(self):
        header = "DOSE  rx %.0f cGy   %d seeds   %g mm grid%s" % (
            self.rx_cgy, self._dose_n_seeds, DOSE_SPACING_MM,
            "   attenuation ON" if self._dose_attenuated else "")
        cov = self._wall_coverage()
        if cov is not None:
            header += "\nwall area >= rx at %g mm depth: %.0f%%" % (
                WALL_DEPTH_MM, 100.0 * cov)
        if self._dose_report is None:
            body = "no cavity wall to score"
        else:
            from .dose.dvh import format_report
            body = format_report(self._dose_report, self.rx_cgy)
        lines = [header] + body.splitlines()
        if self._dose_stale:
            lines.append("STALE -- tiles changed, press u")
        return "\n".join(lines)

    def _draw_dose_panel(self):
        """Text table (upper right) + shell-DVH chart (lower right)."""
        if self._dose_volume is None:
            return
        self._dose_panel_text = self._dose_panel_lines()
        try:
            if self.dose_panel_visible:
                # a positioned (viewport) text actor keeps the fixed-width
                # table left-justified; the corner annotation would
                # right-justify each line after trimming its padding
                actor = self.pl.add_text(
                    self._dose_panel_text, font_size=9, name="dose",
                    position=DOSE_PANEL_POS, viewport=True, font="courier",
                    color="gray" if self._dose_stale else self._fg,
                    render=False)
                try:
                    actor.prop.justification_horizontal = "left"
                    actor.prop.justification_vertical = "top"
                except Exception:
                    pass
            else:
                self.pl.remove_actor("dose", render=False)
        except Exception:
            pass
        self._draw_dvh_chart()
        self._render()

    def _draw_dvh_chart(self):
        """Cumulative DVH per shell; any chart failure is non-fatal."""
        report = self._dose_report
        pv = self.pv
        try:
            if report is None:
                if self._dvh_chart is not None:
                    self._dvh_chart.visible = False
                return
            if self._dvh_chart is None:
                chart = pv.Chart2D(size=(0.30, 0.28), loc=(0.69, 0.02),
                                   x_label="dose [% of rx]",
                                   y_label="shell fraction [%]")
                chart.x_range = [0.0, 300.0]
                chart.y_range = [0.0, 100.0]
                chart.background_color = (1.0, 1.0, 1.0, 0.85)
                chart.legend_visible = True
                self.pl.add_chart(chart)
                self._dvh_chart = chart
            chart = self._dvh_chart
            chart.title = "shell DVH%s" % (" (STALE)" if self._dose_stale else "")
            for k, (off, entry) in enumerate(report.items()):
                x = 100.0 * np.asarray(entry["curve_x"], dtype=float)
                y = 100.0 * np.asarray(entry["curve_y"], dtype=float)
                line = self._dvh_lines.get(off)
                if line is None:
                    label = "wall" if abs(off) < 1e-9 else "+%g mm" % off
                    line = chart.line(x, y, width=2, label=label,
                                      color=_SHELL_COLOR[k % len(_SHELL_COLOR)])
                    self._dvh_lines[off] = line
                else:
                    line.update(x, y)
            chart.visible = bool(self.dose_panel_visible)
        except Exception:
            self._dvh_chart = None
            self._dvh_lines = {}

    def _toggle_dose_panel(self):
        self.dose_panel_visible = not self.dose_panel_visible
        self._set_button("panel", self.dose_panel_visible)
        if self._dose_volume is None:
            self._update_status("dose panel %s (press u to compute dose)"
                                % ("on" if self.dose_panel_visible else "off"))
            return
        self._draw_dose_panel()
        self._update_status("dose panel %s"
                            % ("on" if self.dose_panel_visible else "off"))

    # -------------------------------------------------------------------- run
    def show(self):
        self.pl.show()

    def screenshot(self, path):
        self.pl.screenshot(path)
        return path

    def close(self):
        try:
            self.pl.close()
        except Exception:
            pass


def run_planner(result: PipelineResult, rx_cgy: float = 6000.0):
    """Open the interactive planner window (blocking)."""
    app = _PlannerApp(result, rx_cgy=rx_cgy, off_screen=False)
    app.show()
    return app.tiles


def snapshot_planner(result: PipelineResult, actions: Sequence, path: str,
                     rx_cgy: float = 6000.0):
    """Drive the planner programmatically and render to a PNG (off-screen).

    ``actions`` items may be:

    - a 3-vector: drop a full tile at that cavity point,
    - ``{"point": xyz, "kind": "full"|"half"}``: drop a tile of that kind,
    - ``"update"`` or ``{"update": True}``: run the isodose update (a no-op
      message if the dose engine is not importable yet),
    - ``"interference"`` or ``{"interference": True|False}``: toggle or set
      inter-seed attenuation for subsequent updates.
    """
    app = _PlannerApp(result, rx_cgy=rx_cgy, off_screen=True)
    try:
        for act in actions:
            if isinstance(act, str):
                if act == "update":
                    app.update_dose()
                elif act == "interference":
                    app._toggle_interference()
                continue
            if isinstance(act, dict):
                if "interference" in act:
                    app.interference = bool(act["interference"])
                elif act.get("update"):
                    app.update_dose()
                else:
                    app.drop_at(act["point"], act.get("kind", "full"))
                continue
            app.drop_at(act, "full")
        app.screenshot(path)
    finally:
        app.close()
    return path
