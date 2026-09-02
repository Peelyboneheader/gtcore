"""Optional 3D viewer front-end (PyVista).

This module is the ONE place in gtcore allowed to touch a rendering stack,
and it imports pyvista lazily so the algorithm core never depends on it.
The same scene will later host step-iv interaction (tile drag-and-drop with
snap-to-cavity-wall).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .pipeline import PipelineResult

SEED_LENGTH_MM = 4.5


def show_scene(result: PipelineResult, title: str = "IntraOp GammaTile",
               screenshot: Optional[str] = None):
    """Open an interactive 3D window (or render to ``screenshot`` off-screen)."""
    import pyvista as pv

    def to_pv(tm):
        faces = np.hstack(
            [np.full((len(tm.faces), 1), 3, dtype=np.int64), np.asarray(tm.faces)]
        ).ravel()
        return pv.PolyData(np.asarray(tm.vertices), faces)

    pl = pv.Plotter(window_size=(1280, 900), title=title,
                    off_screen=screenshot is not None)
    pl.set_background("black")

    styles = {
        "skull": dict(color="ivory", opacity=0.12),
        "brain": dict(color="rosybrown", opacity=0.25),
        "body": dict(color="lightsteelblue", opacity=0.25),  # phantom surface
        "cavity": dict(color="deepskyblue", opacity=0.55, specular=0.4),
    }
    if "brain" not in result.meshes and "skull" in result.meshes:
        # phantom / non-head scan: the "skull" is really the object shell and
        # is the only context there is -- make it clearly visible
        styles["skull"] = dict(color="lightsteelblue", opacity=0.30)
    for name, style in styles.items():
        mesh = result.meshes.get(name)
        if mesh is not None and len(mesh.vertices):
            pl.add_mesh(to_pv(mesh), name=name, **style)

    # seeds: uniform gold, or coloured per fitted tile when a fit exists;
    # when the implant assessment says NO implant, candidates render gray so
    # a pre-implant scan's stray dense-bone spots don't masquerade as seeds
    verdict = (result.implant or {}).get("verdict", "confirmed")
    no_implant = verdict == "absent"
    implant_uncertain = verdict == "uncertain"
    tile_of = {}
    if result.tiles is not None:
        for tp in result.tiles.tiles:
            for si in tp.seed_indices:
                tile_of[int(si)] = tp
    palette = ["gold", "orangered", "limegreen", "deepskyblue", "violet",
               "cyan", "salmon", "yellowgreen", "orange", "hotpink"]
    for i, (c, a) in enumerate(zip(result.seeds.centers_ras,
                                   result.seeds.axes_ras)):
        tp = tile_of.get(i)
        if no_implant:
            color = "dimgray"
        else:
            color = palette[tp.tile_id % len(palette)] if tp else \
                ("gold" if result.tiles is None else "gray")
        pl.add_mesh(pv.Cylinder(center=c, direction=a, radius=0.6,
                                height=SEED_LENGTH_MM),
                    color=color, specular=0.8)
    if result.tiles is not None and result.tiles.tiles:
        pts = np.array([tp.center_ras for tp in result.tiles.tiles])
        labels = ["T%d%s" % (tp.tile_id, " (deg)" if tp.degraded else "")
                  for tp in result.tiles.tiles]
        pl.add_point_labels(pts, labels, font_size=16, text_color="white",
                            point_size=1, shape_opacity=0.35,
                            always_visible=True)

    n_seeds = len(result.seeds)
    if "cavity" not in result.meshes:
        cavity_note = "no cavity found"
    elif n_seeds:
        cavity_note = "cavity wall (blue)"
    else:  # no seed prior: the low-HU fallback is ventricles/CSF, not a cavity
        cavity_note = "low-HU spaces, likely ventricles/CSF (blue; no seed prior)"
    if no_implant:
        seed_note = ("NO IMPLANT DETECTED (%d stray candidates, gray)"
                     % n_seeds) if n_seeds else "no seeds on board"
    elif implant_uncertain:
        seed_note = ("IMPLANT UNCERTAIN: %d candidates, tile-like but "
                     "ungrouped (calcifications?)" % n_seeds)
    else:
        seed_note = ("%d seeds (gold)" % n_seeds) if n_seeds else "no seeds on board"
    pl.add_text(
        "brain (rose) | skull (ivory) | %s | %s\n"
        "left-drag rotate, right-drag zoom, middle-drag pan, 'r' reset"
        % (cavity_note, seed_note),
        font_size=10,
    )
    pl.add_axes(line_width=2)
    pl.camera_position = "yz"
    pl.camera.azimuth = 135
    pl.camera.elevation = 15

    if screenshot is not None:
        pl.screenshot(screenshot)
        pl.close()
        return screenshot
    pl.show()
    return None
