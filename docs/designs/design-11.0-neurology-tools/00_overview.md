# Design 11.0: Neurology Tools MCP Extension

**Version**: 1.0.0  
**Date**: 2026-01-21  
**Status**: Specification

## Summary

Add neurology-specific MCP tools to twosphere-mcp for brain surface modeling, glymphatic simulation, MRI analysis, and neuroimaging format export.

## Tool Categories

| Category | Tools | Priority |
|----------|-------|----------|
| Geometry | `generate_fractal_surface`, `embed_graph_on_manifold` | P0 |
| Simulation | `solve_shallow_water_on_surface`, `simulate_sleep_transition` | P2 |
| MRI Analysis | `preprocess_4d_mri`, `extract_phase_coherence`, `extract_pvs_from_mri` | P1 |
| Knowledge | `search_glymphatic_literature`, `lookup_brain_atlas` | P3 |
| Export | `export_to_nifti`, `export_to_gifti`, `export_to_vtk`, `export_graph` | P0 |

## Spec Documents

| File | Description |
|------|-------------|
| `01_geometry_tools.md` | Fractal surface and graph embedding |
| `02_simulation_tools.md` | PDE solvers and state transitions |
| `03_mri_analysis_tools.md` | Preprocessing and feature extraction |
| `04_knowledge_tools.md` | Literature search and atlas lookup |
| `05_export_tools.md` | NIfTI, GIFTI, VTK, graph formats |

## Architecture

```
bin/twosphere_mcp.py
    │
    ├── @server.call_tool("generate_fractal_surface")
    │       └── src/mri_analysis/fractal_surface.py
    │
    ├── @server.call_tool("embed_graph_on_manifold")
    │       └── src/mri_analysis/graph_embedding.py
    │
    ├── @server.call_tool("solve_shallow_water_on_surface")
    │       └── src/simulations/shallow_water.py
    │
    ├── @server.call_tool("extract_phase_coherence")
    │       └── src/mri_analysis/phase_analysis.py
    │
    ├── @server.call_tool("export_to_gifti")
    │       └── src/export/gifti_exporter.py
    │
    └── ... (other tools)
```

## Dependencies

```
# Core
numpy>=1.24
scipy>=1.11
networkx>=3.0

# MRI/Neuroimaging
nibabel>=5.0
nilearn>=0.10

# Visualization/Export
pyvista>=0.42
vtk>=9.2

# Performance
numba>=0.58

# Optional
petsc4py  # For large-scale PDE solving
antspyx   # For registration
```

## Implementation Order

1. **Phase 1 (P0)**: Geometry + Export
   - `generate_fractal_surface`
   - `export_to_gifti`
   - `export_to_vtk`

2. **Phase 2 (P1)**: MRI Analysis
   - `extract_pvs_from_mri`
   - `extract_phase_coherence`
   - `embed_graph_on_manifold`

3. **Phase 3 (P2)**: Simulation
   - `solve_shallow_water_on_surface`
   - `simulate_sleep_transition`

4. **Phase 4 (P3)**: Knowledge
   - `lookup_brain_atlas`
   - `search_glymphatic_literature`
