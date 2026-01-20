"""Optics simulation modules for LOC photonics.

F₀-F₂ level: Core physics computations.
"""

from .ray_tracing import (
    RayTracer,
    PDMSLens,
    MATERIAL_LIBRARY,
    MATERIAL_LIBRARY_NIR,
    get_material,
    get_material_nir,
    create_pdms_air_lens,
    trace_parallel_beam,
    trace_point_source,
)

from .fiber_optics import (
    FiberSpec,
    LensSpec,
    MeniscusLens,
    FourFSystem,
    CouplingResult,
    FIBER_TYPES,
    PDMS_NIR_INDEX,
    design_4f_system,
    design_meniscus_lens,
    compute_spot_size,
)

from .wavefront import (
    WavefrontAnalyzer,
)

__all__ = [
    'RayTracer', 'PDMSLens', 'MATERIAL_LIBRARY', 'MATERIAL_LIBRARY_NIR',
    'get_material', 'get_material_nir', 'create_pdms_air_lens',
    'trace_parallel_beam', 'trace_point_source',
    'FiberSpec', 'LensSpec', 'MeniscusLens', 'FourFSystem', 'CouplingResult',
    'FIBER_TYPES', 'PDMS_NIR_INDEX', 'design_4f_system', 'design_meniscus_lens',
    'compute_spot_size', 'WavefrontAnalyzer',
]
