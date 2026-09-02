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
                      spacing_mm=2.0, sk_per_seed_u=3.5) -> Volume
    isodose_surfaces(dose_volume, levels_cgy) -> {level_cgy: trimesh.Trimesh}

``bounds_ras`` is a (2, 3) array of RAS min/max corners.  If the functions
are missing or fail, the planner shows a message and keeps running.
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

HELP_TEXT = (
    "right-click / P over the blue wall: drop tile   h: full/half for next drop\n"
    "Tab: cycle selection   arrows: slide tile ~2 mm   [ ]: rotate 10 deg\n"
    "x: delete selected   u: update isodose (100/50/25% rx: red/orange/yellow)\n"
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
        self.cavity = result.meshes.get("cavity")

        self.tiles: List[PlacedTile] = []
        self._tile_ids: List[int] = []
        self._next_id = 0
        self.selected = -1
        self.next_kind = "full"

        self.pl = pv.Plotter(window_size=(1280, 900), title=title,
                             off_screen=off_screen)
        self._build_scene()
        self._bind_interaction()
        self._update_status()

    # ------------------------------------------------------------- base scene
    def _build_scene(self):
        pv, pl = self.pv, self.pl
        pl.set_background("black")
        styles = {
            "skull": dict(color="ivory", opacity=0.12),
            "brain": dict(color="rosybrown", opacity=0.25),
            "cavity": dict(color="deepskyblue", opacity=0.55, specular=0.4),
        }
        for name, style in styles.items():
            mesh = self.result.meshes.get(name)
            if mesh is not None and len(mesh.vertices):
                actor = pl.add_mesh(_to_pv(pv, mesh), name=name, **style)
                if name == "cavity":
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
        if self.cavity is not None and len(self.cavity.vertices):
            try:  # P-key picking on the surface under the mouse
                pl.enable_surface_point_picking(
                    callback=self._on_pick, show_message=False,
                    show_point=False, picker="cell",
                )
            except Exception:
                try:
                    pl.enable_surface_point_picking(
                        callback=self._on_pick, show_message=False,
                        show_point=False,
                    )
                except Exception:
                    pass
            try:  # right-click picking
                pl.track_click_position(callback=self._on_pick, side="right")
            except Exception:
                pass

        for key, fn in (
            ("h", self._toggle_kind),
            ("Tab", self._cycle_selection),
            ("x", self._delete_selected),
            ("u", self.update_dose),
        ):
            try:
                pl.add_key_event(key, fn)
            except Exception:
                pass
        for keys, fn in (
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
        try:
            self.pl.add_text(text, font_size=10, name="status",
                             position="lower_left")
            self.pl.render()
        except Exception:
            pass

    # ------------------------------------------------------------------ tiles
    def drop_at(self, point_ras, kind: Optional[str] = None):
        """Snap ``point_ras`` to the cavity wall and drop a conformed tile."""
        if self.cavity is None or not len(self.cavity.vertices):
            self._update_status("no cavity mesh -- cannot place tiles")
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
        self._redraw_tiles()
        self._update_status()
        return tile

    def _on_pick(self, *args):
        """Picking callback (surface pick or click-track)."""
        point = None
        for a in args:
            arr = np.asarray(a, dtype=float).ravel()
            if arr.size == 3 and np.all(np.isfinite(arr)):
                point = arr
                break
        if point is None or not np.any(point):
            return
        self.drop_at(point)

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
        for suffix in ("quad", "seeds"):
            try:
                self.pl.remove_actor("tile_%d_%s" % (tid, suffix))
            except Exception:
                pass
        self.selected = min(self.selected, len(self.tiles) - 1)
        self._redraw_tiles()
        self._update_status()

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
        self._redraw_tiles()
        self._update_status()

    def _rotate_selected(self, angle_rad: float):
        if not (0 <= self.selected < len(self.tiles)):
            return
        self.tiles[self.selected] = rotate_on_wall(
            self.cavity, self.tiles[self.selected], angle_rad)
        self._redraw_tiles()
        self._update_status()

    def _redraw_tiles(self):
        pv, pl = self.pv, self.pl
        for i, (tile, tid) in enumerate(zip(self.tiles, self._tile_ids)):
            selected = i == self.selected
            quad = pv.PolyData(
                np.asarray(tile.corners_ras, dtype=float),
                np.array([3, 0, 1, 2, 3, 0, 2, 3], dtype=np.int64),
            )
            pl.add_mesh(
                quad, name="tile_%d_quad" % tid,
                color="springgreen" if selected else "green",
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
        try:
            dose_volume = compute_dose_grid(centers, axes, bounds,
                                            spacing_mm=DOSE_SPACING_MM)
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
        self._update_status(
            "isodose over %d seeds: %s of rx %.0f cGy"
            % (len(centers), "/".join(shown) if shown else "none in grid",
               self.rx_cgy))

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
      message if the dose engine is not importable yet).
    """
    app = _PlannerApp(result, rx_cgy=rx_cgy, off_screen=True)
    try:
        for act in actions:
            if isinstance(act, str):
                if act == "update":
                    app.update_dose()
                continue
            if isinstance(act, dict):
                if act.get("update"):
                    app.update_dose()
                else:
                    app.drop_at(act["point"], act.get("kind", "full"))
                continue
            app.drop_at(act, "full")
        app.screenshot(path)
    finally:
        app.close()
    return path
