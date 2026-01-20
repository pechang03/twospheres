# Design: Physics Architecture - TwoSphere-MCP Backend

**Task ID**: PH-1  
**Category**: Physics / Optics / MRI  
**Priority**: High  
**Status**: Implemented

---

## Overview

This design documents the TwoSphere-MCP physics backend architecture (`./src/backend/`) which provides:
- **Optical simulations** using pyoptools for LOC (Lab-on-Chip) photonics
- **MRI spherical geometry** analysis for brain region modeling
- **Visualization** services for ray trace diagrams and resonator plots
- **Integration** with ernie2_swarm domain expert via MCP tools

---

## Design Documents

| Document | Description | Status |
|----------|-------------|--------|
| `photonic_loc_elements.md` | Catalog of LOC optical elements | ✅ Complete |
| `optical_resonators.md` | Ring resonator and sensing specs | ✅ Complete |
| `design-ph.1-physics_architecture.md` | This architecture document | ✅ Complete |

---

## Current Implementation Status

### Implementation Status: 100% (Core Modules)

**Implemented Structure**:
```
src/backend/
├── __init__.py              ✅ Functor-level module exports
├── optics/                  ✅ F₀-F₂ Core Physics
│   ├── __init__.py
│   ├── ray_tracing.py       ✅ RayTracer, PDMSLens, materials
│   ├── fiber_optics.py      ✅ FiberSpec, MeniscusLens, 4F system
│   └── wavefront.py         ✅ WavefrontAnalyzer, Zernike polynomials
│
├── mri/                     ✅ F₀-F₂ MRI Geometry
│   ├── __init__.py
│   ├── two_sphere.py        ✅ TwoSphereModel class
│   ├── vortex_ring.py       ✅ VortexRing, Frenet-Serret frames
│   └── fft_correlation.py   ✅ FFT-based signal correlation
│
├── services/                ⏳ F₃-F₄ Composed Services (placeholder)
│   └── __init__.py
│
└── visualize/               ✅ F₁-F₂ Plotting
    ├── __init__.py
    └── ray_plot.py          ✅ plot_phooc_system, plot_ring_resonator
```

**Legacy Compatibility** (retained for backward compatibility):
```
src/simulations/             ✅ Original optics modules
src/mri_analysis/            ✅ Original MRI modules
```

---

## Functor Level Organization

The architecture follows categorical functor levels for composability:

### F₀: Primitive Operations
- Material refractive indices (`MATERIAL_LIBRARY`, `MATERIAL_LIBRARY_NIR`)
- Fiber specifications (`FiberSpec`, `FIBER_TYPES`)
- Physical constants (`PDMS_NIR_INDEX`)

### F₁: Basic Components
- `PDMSLens` - Single optical element
- `MeniscusLens` - SA-optimized lens design
- `TwoSphereModel` - Paired brain region geometry

### F₂: Component Composition
- `RayTracer` - Compose lenses into optical systems
- `WavefrontAnalyzer` - Zernike analysis over apertures
- `VortexRing` - Trefoil knot pathway modeling
- `FourFSystem` - Fiber coupling telescope

### F₃-F₄: Services (Planned)
- `LOCSimulator` - Full chip simulation
- `SensingService` - Biomarker detection pipelines
- Integration with ernie2_swarm for physics queries

### F₅: Meta/Planning
- MCP tool exposure for domain expert integration
- ernie2_swarm coordination for cross-domain physics

---

## Key Features

### 1. Optical Simulations (pyoptools)

**Ray Tracing**:
- Sequential ray tracing through PDMS/air/glass interfaces
- Material library with NIR-specific indices (800-1000nm)
- Plano-convex and meniscus lens designs
- Graceful degradation if pyoptools unavailable

**Fiber Coupling**:
- 4F telescope design for mode matching
- Fiber specifications (SM600, SM800, SMF-28, MMF)
- Spot size calculation via Gaussian optics
- SA-optimized meniscus design (79% SA reduction)

**Wavefront Analysis**:
- Zernike polynomial decomposition
- Aberration analysis (spherical, coma, astigmatism)
- Strehl ratio estimation

### 2. MRI Spherical Geometry

**Two-Sphere Model**:
- Mesh generation for paired brain regions
- Distance and overlap calculations
- Integration with MRISpheres/twospheres research

**Vortex Ring**:
- Trefoil knot neural pathway modeling
- Frenet-Serret frame computation
- Spline interpolation via scipy

**FFT Correlation**:
- Frequency-domain signal correlation
- Coherence analysis
- Phase correlation between signals

### 3. Visualization

**Ray Plot Module**:
- `plot_phooc_system()` - PhOOC 4F fiber coupling diagrams
- `plot_ring_resonator()` - Ring resonator schematics
- `draw_plano_convex_lens()` - Curved surface rendering
- PNG output for documentation

---

## MCP Tools Exposed

The following tools are exposed via the MCP server (`bin/twosphere_mcp.py`):

1. **`two_sphere_model`** - Create/visualize paired brain region spheres
2. **`vortex_ring`** - Generate trefoil knot vortex structures
3. **`fft_correlation`** - Frequency-domain signal correlation
4. **`ray_trace`** - Optical ray tracing through systems
5. **`wavefront_analysis`** - Zernike wavefront analysis
6. **`list_twosphere_files`** - List MRISpheres research files

---

## Integration Points

### With Merge2Docs Ecosystem
- **Task tracking**: Uses yada-work.db with PHY task-ids (e.g., `PHY.1.1`, `PH-1`)
- **Pattern similarity**: Follows same functor-level organization as economics subsystem
- **MCP coordination**: ernie2_swarm for physics domain queries

### With External Tools
- **pyoptools** >= 0.3.7: Core optical simulations
- **scipy**: Spline interpolation, signal processing
- **matplotlib**: Visualization output
- **MRISpheres data**: Linked via `data/twospheres` symlink

---

## Gap Analysis

### G1: Service Layer Incomplete
The `src/backend/services/` layer is a placeholder. Planned services:
- LOC device simulator (combining optics modules)
- Biomarker sensing pipeline
- MRI analysis orchestration

### G2: MCP Tool Coverage
Only 6 MCP tools exposed. Additional tools needed:
- `design_fiber_coupling` - Design 4F systems interactively
- `optimize_resonator` - Ring resonator parameter optimization
- `simulate_loc_chip` - Full chip simulation

### G3: Testing Infrastructure
Limited test coverage for new backend modules. Need:
- Unit tests for each module
- Integration tests for optical pipelines
- Validation against published optical data

### G4: Documentation
Missing:
- API documentation with examples
- Jupyter notebooks for interactive exploration
- Fabrication constraint reference

---

## Validation Criteria

### Completeness Check
- [x] Core optics modules implemented
- [x] MRI geometry modules implemented
- [x] Visualization module implemented
- [ ] Service composition layer
- [ ] Complete MCP tool coverage

### Quality Check
- [x] Functor-level organization documented
- [x] Material libraries validated
- [x] Ray diagrams generated correctly
- [ ] Full test suite

### Technical Feasibility
- [x] pyoptools integration verified
- [x] Graceful degradation without pyoptools
- [x] MRISpheres data linkage works

---

## Recommendations

### RB-Domination Analysis

#### Left Partition - Design Requirements
- **High-Utility**: SA-optimized meniscus lens (79% improvement)
- **High-Utility**: NIR sensing parameters (800-1000nm band)
- **Cross-level**: Functor composition F₀→F₂

#### Right Partition - Implementation Tasks
- Service layer implementation
- Additional MCP tools
- Test infrastructure
- Documentation

#### Domination Relationships
- SA optimization **Dominates** basic plano-convex design
- Functor organization **Dominates** flat module structure
- MCP integration **Dominates** CLI-only access

### Implementation Priority Matrix
1. **High Priority**: Service layer for LOC simulation
2. **Medium Priority**: Additional MCP tools
3. **Low Priority**: Extended documentation

---

## Implementation Roadmap

### Phase 1: Core Architecture ✅ Complete
- [x] Directory structure reorganization
- [x] Optics modules migration
- [x] MRI modules migration
- [x] Visualization module creation
- [x] Import verification

### Phase 2: Service Layer (Next)
- [ ] LOCSimulator service
- [ ] Sensing service pipeline
- [ ] MRI orchestration service

### Phase 3: MCP Expansion
- [ ] Additional MCP tools
- [ ] ernie2_swarm integration
- [ ] Interactive design tools

### Phase 4: Testing & Documentation
- [ ] Unit test suite
- [ ] Integration tests
- [ ] API documentation
- [ ] Example notebooks

---

## References

- **pyoptools**: https://github.com/cihologramas/pyoptools
- **MRISpheres**: Linked research project for brain geometry
- **Implementation**: `src/backend/`
- **MCP Server**: `bin/twosphere_mcp.py`
- **HTTP Server**: `bin/twosphere_http_server.py`
- **Optical Elements**: `docs/photonic_loc_elements.md`
- **Resonator Specs**: `docs/optical_resonators.md`

---

## Version History
- **1.0** (2025-01-20): Initial architecture design document
- Changes tracked in version control
