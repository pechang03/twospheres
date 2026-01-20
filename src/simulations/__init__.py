# Optical simulations module (pyoptools wrappers)
from .ray_tracing import *
from .wavefront import *
from .fiber_optics import *

# Explicit exports for PDMS lens support
from .ray_tracing import PDMSLens, MATERIAL_LIBRARY, create_pdms_air_lens, get_material

# Explicit exports for fiber optics
from .fiber_optics import (
    FiberSpec, LensSpec, FourFSystem, CouplingResult,
    FIBER_TYPES, compute_spot_size, design_4f_system
)
