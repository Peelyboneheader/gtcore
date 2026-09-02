"""Manufactured GammaTile geometry -- the single source of truth.

Every module that needs a tile dimension imports it from here so the model
of the device cannot drift between the phantom generator, the tile fitter,
the interactive conformer, and the dose engine.

Sources
-------
[1] Gessler DJ et al., "GammaTile: surgically targeted radiation therapy for
    glioblastomas" (Future Oncol 2020): 2 cm x 2 cm x 4 mm bioresorbable
    collagen tile carrying four Cs-131 seeds spaced 1 cm apart, ~3 mm
    seed-to-tissue offset.
[2] Brachman DG et al., GammaTile technical & clinical overview
    (J Neuro-Oncol 2022, PMC): 20 x 20 x 4 mm collagen tile, seeds
    embedded 3 mm from the tissue-facing surface, 3.5 U per seed on the
    day of implant.
[3] Ferreira C et al., GammaTile commissioning and clinical implementation
    (PMC): seed-plane depth measured 2.25-3.75 mm after hydration, i.e. the
    3 mm nominal offset with ~0.25 mm SD.
[4] Cs-131 collagen tile outcomes series (PMC): surgeons may cut tiles in
    half (20 x 10 mm, two seeds) to fit the resection bed.
"""
from __future__ import annotations

__all__ = [
    "TILE_SIZE_MM",
    "TILE_THICKNESS_MM",
    "TILE_HALF_SIZE_MM",
    "SEED_PITCH_MM",
    "SEED_EDGE_MARGIN_MM",
    "SEED_PLANE_OFFSET_MM",
    "SEED_PLANE_OFFSET_SD_MM",
    "SEED_PLANE_OFFSET_RANGE_MM",
    "SEED_LENGTH_MM",
    "SEED_DIAMETER_MM",
    "SEED_STRENGTH_U",
    "HALF_TILE_SIZE_MM",
]

# Collagen carrier: 20 x 20 mm square, 4 mm thick [1][2].
TILE_SIZE_MM = 20.0
TILE_THICKNESS_MM = 4.0
TILE_HALF_SIZE_MM = TILE_SIZE_MM / 2.0          # corners at +/- 10 mm
HALF_TILE_SIZE_MM = (TILE_SIZE_MM, TILE_SIZE_MM / 2.0)  # surgeon-cut 20 x 10 [4]

# Seed grid: 2 x 2 array on a 10 mm pitch, centred in the face, so each seed
# sits 5 mm from the two nearest tile edges [1][2].
SEED_PITCH_MM = 10.0
SEED_EDGE_MARGIN_MM = (TILE_SIZE_MM - SEED_PITCH_MM) / 2.0   # 5 mm

# Seed plane: 3.0 mm from the TISSUE-FACING surface (1 mm from the cavity-
# facing surface).  Hydrated tiles scatter 2.25-3.75 mm, ~N(3.0, 0.25^2)
# [2][3].  Seeds therefore sit ~3 mm off the cavity wall INTO the lumen.
SEED_PLANE_OFFSET_MM = 3.0
SEED_PLANE_OFFSET_SD_MM = 0.25
SEED_PLANE_OFFSET_RANGE_MM = (2.25, 3.75)

# Cs-131 CS-1 capsule (IsoRay Proxcelan): 4.5 mm x 0.8 mm; 3.5 U per seed on
# implant day [2].
SEED_LENGTH_MM = 4.5
SEED_DIAMETER_MM = 0.8
SEED_STRENGTH_U = 3.5
