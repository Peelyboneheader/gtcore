"""Count-constrained completion of geometrically degraded (crumpled) tiles.

Motivated by the physical 8-tile printed phantom: one tile was folded during
placement, squeezing two of its sides to ~5 mm -- outside every geometry
gate -- while its four seeds remained compact and axis-coherent. With the
implant count as evidence, ``complete_degraded=True`` recovers it.
"""
from __future__ import annotations

import numpy as np

from gtcore.tiles import fit_tiles


def _good_tile(center, t1, t2, half=5.0):
    return [center + a * t1 + b * t2
            for a, b in ((-half, -half), (-half, half), (half, -half), (half, half))]


def _build_case():
    """Two well-formed tiles + one crumpled tile (measured phantom chords)."""
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 0.0])
    centers = []
    axes = []
    for c in (np.array([0.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0])):
        centers.extend(_good_tile(c, t1, t2))
        axes.extend([t1] * 4)
    # crumpled: sides ~5-6 mm, diagonals <11 mm (from the real phantom's tile)
    crumpled = np.array([
        [60.0, 0.0, 0.0],
        [60.8, 5.1, 2.8],
        [51.6, 0.5, 2.5],
        [53.2, 4.6, 4.5],
    ])
    centers.extend(list(crumpled))
    axes.extend([t1, t1, np.array([0.98, 0.2, 0.0]) / np.linalg.norm([0.98, 0.2, 0.0]), t1])
    return np.array(centers), np.array(axes)


def test_without_completion_crumpled_tile_is_missed():
    centers, axes = _build_case()
    fit = fit_tiles(centers, axes, 3)
    assert not fit.all_assigned
    assert len(fit.tiles) == 2
    assert len(fit.rejected_indices) == 4


def test_completion_recovers_crumpled_tile_and_flags_it():
    centers, axes = _build_case()
    fit = fit_tiles(centers, axes, 3, complete_degraded=True)
    assert fit.all_assigned
    assert len(fit.tiles) == 3
    degraded = [t for t in fit.tiles if t.degraded]
    assert len(degraded) == 1
    assert sorted(degraded[0].seed_indices) == [8, 9, 10, 11]
    # normal tiles unaffected and NOT flagged
    assert all(not t.degraded for t in fit.tiles if t is not degraded[0])


def test_completion_does_not_invent_tiles_from_scattered_decoys():
    centers, axes = _build_case()
    rng = np.random.default_rng(0)
    # replace the crumpled tile with 4 far-scattered decoys (>18 mm apart)
    centers[8:] = np.array([[60, 0, 0], [95, 40, 0], [60, -60, 30], [130, 0, -40]],
                           dtype=float)
    axes[8:] = rng.normal(size=(4, 3))
    axes[8:] /= np.linalg.norm(axes[8:], axis=1, keepdims=True)
    fit = fit_tiles(centers, axes, 3, complete_degraded=True)
    assert not fit.all_assigned
    assert len(fit.tiles) == 2
    assert len(fit.rejected_indices) == 4


def test_completion_prefers_real_crumpled_tile_over_clip_line():
    """A line of surgical clips is compact and perfectly axis-coherent but
    collinear -- completion must not fabricate a tile from it, and must still
    recover the true crumpled tile sharing the leftover pool."""
    centers, axes = _build_case()
    t1 = np.array([1.0, 0.0, 0.0])
    clip_line = [np.array([90.0, 0.0, 0.0]) + i * np.array([0.0, 4.0, 0.0])
                 for i in range(4)]
    centers = np.vstack([centers, clip_line])
    axes = np.vstack([axes, [t1] * 4])

    fit = fit_tiles(centers, axes, 3, complete_degraded=True)
    assert fit.all_assigned
    degraded = [t for t in fit.tiles if t.degraded]
    assert len(degraded) == 1
    assert sorted(degraded[0].seed_indices) == [8, 9, 10, 11]  # the crumpled tile
    assert sorted(fit.rejected_indices) == [12, 13, 14, 15]    # the clip line


def test_completion_never_accepts_collinear_group_alone():
    """Only a clip line in the leftovers: no tile may be invented from it."""
    centers, axes = _build_case()
    t1 = np.array([1.0, 0.0, 0.0])
    # remove the crumpled tile, keep 2 good tiles + a coherent clip line
    centers = np.vstack([centers[:8],
                         [np.array([60.0, 0.0, 0.0]) + i * np.array([0.0, 4.0, 0.0])
                          for i in range(4)]])
    axes = np.vstack([axes[:8], [t1] * 4])
    fit = fit_tiles(centers, axes, 3, complete_degraded=True)
    assert not fit.all_assigned
    assert len(fit.tiles) == 2
    assert sorted(fit.rejected_indices) == [8, 9, 10, 11]


def test_completion_stays_off_by_default():
    centers, axes = _build_case()
    fit = fit_tiles(centers, axes, 3)
    assert not any(getattr(t, "degraded", False) for t in fit.tiles)
