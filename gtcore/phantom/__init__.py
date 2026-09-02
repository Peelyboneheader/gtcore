"""Synthetic phantoms with exact ground truth (the pipeline's validation bed)."""
from .generate import (
    BRAIN_RADII,
    CAVITY_RADII,
    ENTRY_DIR_U0,
    PhantomTruth,
    SCALP_RADII,
    SKULL_INNER_RADII,
    SKULL_OUTER_RADII,
    SeedTruth,
    TileTruth,
    make_head_phantom,
)

__all__ = [
    "make_head_phantom",
    "SeedTruth",
    "TileTruth",
    "PhantomTruth",
    "SCALP_RADII",
    "SKULL_OUTER_RADII",
    "SKULL_INNER_RADII",
    "BRAIN_RADII",
    "CAVITY_RADII",
    "ENTRY_DIR_U0",
]
