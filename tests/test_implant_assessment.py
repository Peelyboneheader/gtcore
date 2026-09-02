"""Automatic implant-present/absent assessment (pipeline.assess_implant).

Real GammaTile implants form tile-spaced clusters; false positives that
survive the shape and vault filters (dense bone, clips, streaks) are
spatially scattered. The assessment turns that measured difference into an
explicit verdict so a pre-implant scan's stray candidates are not presented
as seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from gtcore.pipeline import assess_implant, reconstruct
from gtcore.phantom import make_head_phantom


def test_empty_and_tiny_inputs_are_not_implants():
    assert not assess_implant(np.zeros((0, 3)))["present"]
    assert not assess_implant(np.array([[0.0, 0, 0], [40, 0, 0], [0, 40, 0]]))["present"]


def test_scattered_candidates_are_not_an_implant():
    rng = np.random.default_rng(0)
    # 30 candidates spread through a head-sized volume with random axes --
    # even chance proximity must not read as manufactured tile geometry
    pts = rng.uniform(-70, 70, (30, 3))
    axes = rng.normal(size=(30, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    v = assess_implant(pts, axes)
    assert not v["present"]
    assert "pre-implant" in v["reason"]


def test_chained_bone_spots_are_not_an_implant():
    """Proximity chains (the DOE negative-control failure mode) must not count."""
    rng = np.random.default_rng(2)
    # a long chain of points ~12 mm apart along a curve, random axes:
    # single-linkage clusters them, but no 4 of them form a 10 mm quad
    t = np.linspace(0, 1, 25)
    chain = np.stack([120 * t, 40 * np.sin(6 * t), 25 * np.cos(5 * t)], axis=1)
    axes = rng.normal(size=(25, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    v = assess_implant(chain, axes)
    assert not v["present"]


def test_single_tile_quad_is_uncertain_not_confirmed():
    rng = np.random.default_rng(1)
    tile = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    t_axes = np.tile(np.array([1.0, 0, 0]), (4, 1))
    scatter = rng.uniform(60, 120, (10, 3))
    s_axes = rng.normal(size=(10, 3))
    s_axes /= np.linalg.norm(s_axes, axis=1, keepdims=True)
    v = assess_implant(np.vstack([tile, scatter]), np.vstack([t_axes, s_axes]))
    assert v["verdict"] == "uncertain"
    assert v["n_tile_evidence"] >= 1


def test_two_grouped_tiles_are_confirmed():
    t1 = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    t2 = t1 + np.array([22.0, 0, 0])
    axes = np.tile(np.array([1.0, 0, 0]), (8, 1))
    v = assess_implant(np.vstack([t1, t2]), axes)
    assert v["verdict"] == "confirmed" and v["present"]


def test_two_scattered_quads_stay_uncertain():
    """The calcification failure mode: tile-like quads far apart are NOT one
    implant (a real implant lines one cavity)."""
    t1 = np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
    t2 = t1 + np.array([90.0, 40, 0])  # way beyond one cavity
    axes = np.tile(np.array([1.0, 0, 0]), (8, 1))
    v = assess_implant(np.vstack([t1, t2]), axes)
    assert v["verdict"] == "uncertain"
    assert not v["present"]


def test_pipeline_verdicts_on_phantom():
    vol, _ = make_head_phantom(spacing=1.0, n_tiles=3)
    res = reconstruct(vol, verbose=False)
    assert res.implant is not None
    assert res.implant["verdict"] == "confirmed"
    assert res.implant["n_tile_evidence"] >= 2
