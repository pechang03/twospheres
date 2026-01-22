"""Simulation modules for microfluidics and glymphatic flow."""

from .glymphatic_flow import (
    GlymphaticFlowSimulator,
    PerivascularSpace,
    MicrofluidicChannel,
    CSF_VISCOSITY,
    CSF_DENSITY,
)
from .clearance_network import (
    ClearanceNetworkAnalyzer,
    disc_dimension_clearance_model,
)
from .glymphatic_fmri_integration import (
    GlymphaticFMRIIntegrator,
    BrainState,
    RegionalClearance,
    AmyloidAccumulation,
    GlymphaticAnalysisResult,
    analyze_brain_clearance,
)
from .cfd_3d import (
    CFD3DSimulator,
    StokesSolver3D,
    Geometry3D,
    GeometryType,
    FlowField3D,
    BoundaryConditions,
)
from .brain_chip_designer import (
    BrainChipDesigner,
    FreeCADExporter,
    ChipDesign,
    design_brain_chip,
)

__all__ = [
    # Flow simulation
    "GlymphaticFlowSimulator",
    "PerivascularSpace",
    "MicrofluidicChannel",
    "CSF_VISCOSITY",
    "CSF_DENSITY",
    # Network analysis
    "ClearanceNetworkAnalyzer",
    "disc_dimension_clearance_model",
    # fMRI integration
    "GlymphaticFMRIIntegrator",
    "BrainState",
    "RegionalClearance",
    "AmyloidAccumulation",
    "GlymphaticAnalysisResult",
    "analyze_brain_clearance",
    # 3D CFD
    "CFD3DSimulator",
    "StokesSolver3D",
    "Geometry3D",
    "GeometryType",
    "FlowField3D",
    "BoundaryConditions",
    # Brain chip designer
    "BrainChipDesigner",
    "FreeCADExporter",
    "ChipDesign",
    "design_brain_chip",
]
