# Glymphatic Extension Design Overview

**Version**: 0.1.0  
**Date**: 2026-01-21  
**Status**: Draft

## Purpose

Extend twosphere-mcp to support fractal cortical surface modeling and glymphatic flow simulation, with export capabilities for integration with external neurology tools and MRI analysis pipelines.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Server Layer                         │
│  twosphere_mcp.py — exposes tools via stdio/HTTP                │
├─────────────────────────────────────────────────────────────────┤
│                        Export Layer                             │
│  export_nifti.py │ export_gifti.py │ export_vtk.py │ export_graph.py
├─────────────────────────────────────────────────────────────────┤
│                      Simulation Layer                           │
│  shallow_water_solver.py │ sleep_state.py │ neural_coupling.py  │
├─────────────────────────────────────────────────────────────────┤
│                       Geometry Layer                            │
│  fractal_surface.py │ glymphatic_network.py │ two_sphere.py     │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  mri_feature_extractor.py │ fft_correlation.py                  │
└─────────────────────────────────────────────────────────────────┘
```

## Design Documents

| Document | Description |
|----------|-------------|
| `01_fractal_surface.md` | Julia/L-system surface generation |
| `02_glymphatic_network.md` | G_pvs graph construction and hydraulics |
| `03_export_tools.md` | NIfTI, GIFTI, VTK, GraphML export |
| `04_mcp_tools_enhancement.md` | New MCP tools for improved research queries |
| `05_simulation_engine.md` | Time-stepping solver architecture |

## Key Equations

### Fractal Surface
```
R(θ,φ) = R_WM + ε·f(θ,φ)
f(θ,φ) = f_ℂ(tan(θ/2)·e^{iφ})  [Julia potential]
```

### Perivascular Flow
```
q(s,t) = –(δ³/12μ) ∂p/∂s
```

### CSF Shallow Water (on manifold M)
```
∂h/∂t + ∇_s·(h u) = S_v
u = –(h²/3μ) ∇_s p + u_forcing
```

### Neural Coupling
```
∑_{e ∈ incident(v)} q_e = α m_v
```

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Core numerics | ≥1.24 |
| scipy | Sparse solvers, interpolation | ≥1.11 |
| nibabel | NIfTI/GIFTI I/O | ≥5.0 |
| pyvista | VTK mesh visualization | ≥0.42 |
| networkx | Graph algorithms | ≥3.0 |
| numba | JIT for Julia iteration | ≥0.58 |

## Milestones

1. **M1**: Fractal surface generation with ε-safety bounds
2. **M2**: PVS network extraction and hydraulic graph
3. **M3**: Export tools (NIfTI, GIFTI, VTK)
4. **M4**: Shallow-water solver with IMEX stepping
5. **M5**: Sleep-wake state transitions
6. **M6**: MRI feature extraction pipeline
7. **M7**: New MCP tools for neurology queries
