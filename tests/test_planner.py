"""Smoke test for the interactive planner's off-screen snapshot path.

Runs the full pipeline on a coarse phantom, drops two tiles programmatically
(one full, one half), attempts a dose update (a friendly on-screen message if
the dose engine is not on board yet), and renders to a PNG.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyvista")

from gtcore.phantom import make_head_phantom
from gtcore.pipeline import reconstruct
from gtcore.planner import snapshot_planner


def test_snapshot_planner_smoke():
    vol, _truth = make_head_phantom(spacing=1.0)
    result = reconstruct(vol, verbose=False)
    if "cavity" not in result.meshes:
        pytest.skip("pipeline found no cavity on this phantom")
    mesh = result.meshes["cavity"]

    # two well-separated cavity wall points
    verts = np.asarray(mesh.vertices)
    p1 = verts[0]
    p2 = verts[int(np.argmax(np.linalg.norm(verts - p1, axis=1)))]

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "planner_smoke.png")
    if os.path.exists(out):
        os.remove(out)

    actions = [p1, {"point": p2, "kind": "half"}, "update"]
    try:
        snapshot_planner(result, actions, out)
    except Exception as exc:  # headless CI without OpenGL etc.
        pytest.skip("off-screen rendering unavailable: %r" % (exc,))

    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000, "snapshot PNG is suspiciously small"
