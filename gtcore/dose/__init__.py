"""Dosimetry: TG-43U1 formalism for Cs-131 GammaTile seeds.

``DoseInterpolator`` is the verbatim v1 port (regression reference, frozen);
``TG43Engine`` / ``compute_dose_grid`` / ``dose_at_points`` /
``isodose_surfaces`` are the corrected, vectorized v2 engine (see
docs/tg43-port-notes.md); ``metrics`` adds DVH, cavity-rind and wall-at-depth
coverage statistics for the planner, and ``dvh`` the cheap cavity-shell
(wall / +5 / +10 mm) statistics behind the planner's on-screen dose panel;
``hrctv`` builds the 5 mm tissue rind as a voxel volume (the HR-CTV) and
scores it with the exact engine for volumetric D90 / V100.

``InterferenceModel`` adds the effect TG-43 superposition cannot express:
seeds and collagen tile carriers attenuating each other's primary fluence
(see docs/interference-notes.md).  It is opt-in -- pass one as
``compute_dose_grid(..., interference=model)`` -- so the bare formalism stays
the default and stays regression-pinned.
"""
from .engine import CLRP_V2, DATASETS, SeedDataset, TG43Engine, \
    TG43U1S2_CONSENSUS, compute_dose_grid, dose_at_points, isodose_surfaces
from .dvh import dvh_curve, dvh_stats, format_report, shell_report
from .engine import TG43Engine, compute_dose_grid, dose_at_points, \
    isodose_surfaces
from .interference import (
    InterferenceModel,
    SeedCapsule,
    TileCarrier,
    find_shadowing_tiles,
    interference_report,
    tile_prescription_points,
    tile_shadowing,
)
from .hrctv import HRCTV, HRCTV_DEPTH_MM, build_hrctv, hrctv_stats
from .metrics import DVH, dose_metrics, dvh, resample_mask_to, rind_mask, \
    surface_coverage, wall_dose
from .tg43 import DoseInterpolator

__all__ = [
    "DoseInterpolator",
    "HRCTV",
    "HRCTV_DEPTH_MM",
    "build_hrctv",
    "hrctv_stats",
    "SeedDataset",
    "DATASETS",
    "TG43U1S2_CONSENSUS",
    "CLRP_V2",
    "TG43Engine",
    "compute_dose_grid",
    "dose_at_points",
    "isodose_surfaces",
    "InterferenceModel",
    "SeedCapsule",
    "TileCarrier",
    "interference_report",
    "tile_shadowing",
    "find_shadowing_tiles",
    "tile_prescription_points",
    "DVH",
    "dvh",
    "dose_metrics",
    "resample_mask_to",
    "rind_mask",
    "wall_dose",
    "surface_coverage",
    "dvh_curve",
    "dvh_stats",
    "format_report",
    "shell_report",
]
