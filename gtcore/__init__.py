"""IntraOp GammaTile algorithm core.

Standalone and host-agnostic: no slicer/vtk/qt imports anywhere in this
package. Front-ends (desktop 3D viewer, web) consume it as a library.
"""
from .volume import Volume, apply_affine

__version__ = "0.1.0"
__all__ = ["Volume", "apply_affine", "__version__"]
