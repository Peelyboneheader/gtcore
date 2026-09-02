"""Interactive intraoperative tile planner (PyVista front-end).

Builds on the :mod:`gtcore.viz` scene style and adds tile drag-and-drop:
pick a point on the cavity wall to drop a conformed GammaTile, then slide,
spin, or delete it, and update TG-43 isodose surfaces over detected + placed
seeds on demand.

Like ``viz.py``, this module is a rendering front-end and the only other
place in gtcore allowed to touch pyvista -- and it imports it lazily, inside
functions, so the algorithm core never depends on a rendering stack.  All
geometry lives in :mod:`gtcore.interact` (pure numpy/trimesh).

Dose contract (implemented by :mod:`gtcore.dose`, loaded lazily on 'u')::

    compute_dose_grid(seed_centers, seed_axes, bounds_ras,
                      spacing_mm=2.0, sk_per_seed_u=3.5,
                      interference=None) -> Volume
    isodose_surfaces(dose_volume, levels_cgy) -> {level_cgy: trimesh.Trimesh}
    InterferenceModel.from_implant(seed_centers, seed_axes) -> model

``bounds_ras`` is a (2, 3) array of RAS min/max corners.  If the functions
are missing or fail, the planner shows a message and keeps running.

'i' toggles inter-seed attenuation (:mod:`gtcore.dose.interference`): the
seeds and tiles already on the board shadow one another, which plain TG-43
superposition cannot express.  It is off by default so what the planner
shows matches the formalism unless the user asks for the correction; tile
carriers stay out of the model entirely, since their contribution rests on
an unmeasured density (see docs/interference-notes.md).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

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
WALL_DEPTH_MM = 5.0         # GammaTile prescription depth (60 Gy at 5 mm)
DRAG_CONFORM_EVERY = 2      # re-conform every Nth MouseMoveEvent while dragging
OVERLAP_THRESHOLD_MM = 1.0  # passed to interact.find_overlapping_tiles

HELP_TEXT = (
    "right-click / P over the blue wall: drop tile   h: full/half for next drop\n"
    "Ctrl+left-drag a placed tile: grab it and slide it along the wall\n"
    "Tab: cycle selection   arrows: slide tile ~2 mm   [ ]: rotate 10 deg\n"
    "x: delete selected   u: update isodose (100/50/25% rx: red/orange/yellow)\n"
    "i: toggle inter-seed attenuation (takes effect on the next 'u')\n"
    "left-drag rotate, right-drag zoom, middle-drag pan, r: reset camera"
)

_ISO_STYLE = (  # (fraction of rx, actor name, color)
    (1.00, "iso_100", "red"),
    (0.50, "iso_50", "orange"),
    (0.25, "iso_25", "yellow"),
)


def _to_pv(pv, tm):
    faces = np.hstack(
        [np.full((len(tm.faces), 1), 3, dtype=np.int64), np.asarray(tm.faces)]
    ).ravel()
    return pv.PolyData(np.asarray(tm.vertices), faces)


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
        self._next_id = 0
        self.selected = -1
        self.next_kind = "full"
        self.interference = False

        self._cavity_actor = None
        self._tile_actors = {}       # tile id -> quad vtkActor (for grabbing)
        self._cavity_picker = None   # vtkCellPicker restricted to the cavity
        self._tile_picker = None     # vtkCellPicker restricted to tile quads
        self._drag_idx = -1          # index of the tile being dragged, or -1
        self._drag_moves = 0
        self._overlap_pairs: List = []
        self._last_status = ""       # last status text (also for headless tests)

        self.pl = pv.Plotter(window_size=(1280, 900), title=title,
                             off_screen=off_screen)
        self._build_scene()
        self._bind_interaction()
        self._adopt_fitted_tiles()
        self._update_status()

    # ------------------------------------------------------------- base scene
    def _build_scene(self):
        pv, pl = self.pv, self.pl
        pl.set_background("black")
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

        pl.add_text(HELP_TEXT, font_size=11, name="help", position="upper_left")
        pl.add_axes(line_width=2)
        pl.camera_position = "yz"
        pl.camera.azimuth = 135
        pl.camera.elevation = 15

    def _bind_interaction(self):
        pl = self.pl
        self._bind_pick_observers()
        for keys, fn in (
            (("p", "P"), self._place_at_mouse),
            (("h",), self._toggle_kind),
            (("Tab",), self._cycle_selection),
            (("x",), self._delete_selected),
            (("u",), self.update_dose),
            (("i",), self._toggle_interference),
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
        self._tile_picker.SetTolerance(0.005)
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
                actor = self._tile_actors.get(tid)
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
        """Ctrl+left-press on a placed tile grabs it for dragging."""
        iren = getattr(self.pl.iren, "interactor", None) \
            if self.pl.iren is not None else None
        if iren is None or not iren.GetControlKey():
            return False
        xy = self._mouse_xy()
        if xy is None:
            return False
        idx = self._pick_tile_index(xy[0], xy[1])
        if idx < 0:
            return False
        self._drag_idx = idx
        self._drag_moves = 0
        self.selected = idx
        self._redraw_tiles()
        self._update_status("dragging tile %d -- release to drop" % (idx + 1))
        return True  # abort so the camera style never sees this press

    def _on_mouse_move(self):
        """While dragging: re-place the tile at the wall point under the cursor."""
        if not (0 <= self._drag_idx < len(self.tiles)):
            return False
        self._drag_moves += 1
        if self._drag_moves % DRAG_CONFORM_EVERY:
            return True  # throttled (conform_tile costs ~ms); camera stays put
        xy = self._mouse_xy()
        pt = self._pick_cavity_point(xy[0], xy[1]) if xy is not None else None
        if pt is None:
            return True  # cursor slid off the wall: tile stays where it was
        tile = self.tiles[self._drag_idx]
        surf, n_in = snap_to_wall(self.cavity, pt)
        self.tiles[self._drag_idx] = conform_tile(
            self.cavity, surf, n_in, tile.axis_ras, kind=tile.kind)
        self._after_change("dragging tile %d -- release to drop"
                           % (self._drag_idx + 1))
        return True

    def _on_left_release(self):
        if self._drag_idx < 0:
            return False
        idx, self._drag_idx = self._drag_idx, -1
        self._after_change("tile %d dropped" % (idx + 1))
        return True

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
    def _update_status(self, extra: str = ""):
        n_full = sum(1 for t in self.tiles if t.kind == "full")
        n_half = len(self.tiles) - n_full
        sel = ("tile %d/%d selected" % (self.selected + 1, len(self.tiles))
               if 0 <= self.selected < len(self.tiles) else "no tile selected")
        text = "tiles: %d full + %d half | next drop: %s | %s" % (
            n_full, n_half, self.next_kind, sel)
        if extra:
            text += "\n" + extra
        for i, j in self._overlap_pairs[:4]:  # tile numbers as displayed (1-based)
            text += "\nWARNING: tiles %d & %d overlap" % (i + 1, j + 1)
        self._last_status = text
        try:
            self.pl.add_text(text, font_size=10, name="status",
                             position="lower_left")
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

    def _after_change(self, extra: str = ""):
        """One funnel for every tile mutation: overlaps -> redraw -> status."""
        self._refresh_overlaps()
        self._redraw_tiles()
        self._update_status(extra)

    # ------------------------------------------------------------------ tiles
    def _adopt_fitted_tiles(self):
        """Import tiles RECOVERED FROM THE SCAN as selectable placed tiles.

        Without this, Tab/drag only touch tiles the user drops by hand and
        the fitted implant is display-only -- but adjusting the *actual*
        implant is the whole point of the planner. Each fitted pose is
        re-conformed onto the interaction surface at its own centre.
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
        self.tiles.append(tile)
        self._tile_ids.append(self._next_id)
        self._next_id += 1
        self.selected = len(self.tiles) - 1
        self._after_change("tile placed (%d on board)" % len(self.tiles))
        return tile

    def _toggle_kind(self):
        self.next_kind = "half" if self.next_kind == "full" else "full"
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
        tid = self._tile_ids.pop(self.selected)
        self.tiles.pop(self.selected)
        self._tile_actors.pop(tid, None)
        if self._drag_idx == self.selected:
            self._drag_idx = -1
        for suffix in ("quad", "seeds"):
            try:
                self.pl.remove_actor("tile_%d_%s" % (tid, suffix))
            except Exception:
                pass
        self.selected = min(self.selected, len(self.tiles) - 1)
        self._after_change("tile deleted")

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
        self.tiles[self.selected] = translate_on_wall(self.cavity, tile, delta)
        self._after_change()

    def _rotate_selected(self, angle_rad: float):
        if not (0 <= self.selected < len(self.tiles)):
            return
        self.tiles[self.selected] = rotate_on_wall(
            self.cavity, self.tiles[self.selected], angle_rad)
        self._after_change()

    def _redraw_tiles(self):
        pv, pl = self.pv, self.pl
        flagged = set()
        for i, j in self._overlap_pairs:
            flagged.add(i)
            flagged.add(j)
        for i, (tile, tid) in enumerate(zip(self.tiles, self._tile_ids)):
            selected = i == self.selected
            if i in flagged:  # red-ish tint until the overlap is resolved
                color = "orangered" if selected else "red"
            else:
                color = "springgreen" if selected else "green"
            quad = pv.PolyData(
                np.asarray(tile.corners_ras, dtype=float),
                np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64),
            )
            self._tile_actors[tid] = pl.add_mesh(
                quad, name="tile_%d_quad" % tid,
                color=color,
                opacity=0.75 if selected else 0.45,
                show_edges=selected, edge_color="white",
            )
            seed_polys = None
            for c, a in zip(tile.seed_centers, tile.seed_axes):
                cyl = pv.Cylinder(center=c, direction=a, radius=0.6,
                                  height=SEED_LENGTH_MM)
                seed_polys = cyl if seed_polys is None else seed_polys + cyl
            if seed_polys is not None:
                pl.add_mesh(seed_polys, name="tile_%d_seeds" % tid,
                            color="gold", specular=0.8)
        try:
            pl.render()
        except Exception:
            pass

    # ------------------------------------------------------------------- dose
    def _toggle_interference(self):
        """Flip inter-seed attenuation on/off for the next dose update.

        Deliberately does NOT recompute: a dose update takes seconds, and a
        surgeon tapping a key should not trigger one by surprise.
        """
        self.interference = not self.interference
        self._update_status(
            "inter-seed attenuation %s -- press 'u' to recompute"
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
        levels = [frac * self.rx_cgy for frac, _n, _c in _ISO_STYLE]
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
            surfaces = isodose_surfaces(dose_volume, levels)
        except (ImportError, AttributeError):
            self._update_status("dose engine not available yet")
            return
        except Exception as exc:  # engine present but unhappy: stay alive
            self._update_status("dose update failed: %s" % exc)
            return

        shown = []
        for frac, name, color in _ISO_STYLE:
            try:
                self.pl.remove_actor(name)
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
            self.pl.add_mesh(_to_pv(self.pv, surf), name=name, color=color,
                             opacity=0.35)
            shown.append("%d%%" % round(100 * frac))
        # Wall coverage at the prescription depth (60 Gy at 5 mm is the
        # GammaTile prescription): area fraction of the cavity wall whose
        # point 5 mm deep in the tissue receives >= rx.
        coverage = ""
        cavity = getattr(self.result, "meshes", {}).get("cavity")
        if cavity is not None and getattr(cavity, "faces", None) is not None \
                and len(cavity.faces):
            try:
                d = dose_mod.wall_dose(cavity, WALL_DEPTH_MM,
                                       dose_volume=dose_volume)
                frac = dose_mod.surface_coverage(cavity, d, self.rx_cgy)
                if np.isfinite(frac):
                    coverage = "; wall @%g mm >= rx: %.0f%%" % (
                        WALL_DEPTH_MM, 100.0 * frac)
            except Exception:  # metrics are advisory: never break the update
                coverage = ""
        self._update_status(
            "isodose over %d seeds: %s of rx %.0f cGy%s%s"
            % (len(centers), "/".join(shown) if shown else "none in grid",
               self.rx_cgy, coverage,
               " | inter-seed attenuation ON" if model is not None else ""))

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
