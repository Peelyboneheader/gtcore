"""Interactive 3D view of the reconstructed scene.

Opens a desktop window with the brain (translucent), skull (faint), the
resection cavity wall, and the detected seeds drawn as oriented capsules.
This is the seed of the step-iv front-end: the same PyVista scene will later
host tile drag-and-drop with snap-to-wall.

Run:  python scripts/view_3d.py            (uses meshes in output/ if present,
                                            else regenerates the pipeline)
Controls: left-drag rotate | right-drag zoom | middle-drag pan | r reset
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def build_scene():
    import trimesh

    have = all(
        os.path.exists(os.path.join(OUT, n + ".ply"))
        for n in ("brain", "skull", "cavity")
    )
    from gtcore.phantom import make_head_phantom
    from gtcore.seeds import detect_seed_candidates

    vol, truth = make_head_phantom(spacing=0.7)
    cands = detect_seed_candidates(vol)

    if have:
        meshes = {
            n: trimesh.load(os.path.join(OUT, n + ".ply"))
            for n in ("brain", "skull", "cavity")
        }
    else:
        from gtcore.preprocess import inpaint_metal
        from gtcore.segment import mask_to_mesh, segment_cavity, segment_head

        clean = inpaint_metal(vol, cands.mask)
        masks = segment_head(clean, metal_mask=cands.mask)
        cavity = segment_cavity(
            clean, masks["cranial_interior"], masks["brain"], cands.centers_ras
        )
        meshes = {
            "brain": mask_to_mesh(masks["brain"], vol.affine),
            "skull": mask_to_mesh(masks["skull"], vol.affine),
            "cavity": mask_to_mesh(cavity, vol.affine),
        }
    return meshes, cands


def to_pv(tm):
    faces = np.hstack(
        [np.full((len(tm.faces), 1), 3, dtype=np.int64), tm.faces]
    ).ravel()
    return pv.PolyData(np.asarray(tm.vertices), faces)


def main():
    meshes, cands = build_scene()

    pl = pv.Plotter(window_size=(1280, 900), title="IntraOp GammaTile — reconstruction")
    pl.set_background("black")
    pl.add_mesh(to_pv(meshes["skull"]), color="ivory", opacity=0.12, name="skull")
    pl.add_mesh(to_pv(meshes["brain"]), color="rosybrown", opacity=0.25, name="brain")
    pl.add_mesh(
        to_pv(meshes["cavity"]),
        color="deepskyblue",
        opacity=0.55,
        specular=0.4,
        name="cavity",
    )

    # detected seeds as oriented capsules (4.5 x 0.8 mm, drawn slightly fat)
    for c, a in zip(cands.centers_ras, cands.axes_ras):
        pl.add_mesh(
            pv.Cylinder(center=c, direction=a, radius=0.6, height=4.5),
            color="gold",
            specular=0.8,
        )

    pl.add_axes(line_width=2)
    pl.add_text(
        "brain (rose) | skull (ivory) | cavity wall (blue) | detected seeds (gold)\n"
        "left-drag rotate, right-drag zoom, middle-drag pan, 'r' reset",
        font_size=10,
    )
    pl.camera_position = "yz"
    pl.camera.azimuth = 135
    pl.camera.elevation = 15
    pl.show()


if __name__ == "__main__":
    main()
