# Glymphatic Simulation & Expert Integration Milestone

**Date**: 2026-01-22
**Status**: Milestone achieved - PH-7 Phase 1 complete

## What Was Accomplished

### 1. ernie2_swarm Integration Fixed
- Fixed yada-services-secure path bug (SRC_DIR in sys.path broke imports)
- Installed missing dependencies (neo4j, pydantic-ai/pydantic_graph)
- Created `bin/ernie2_swarm_mcp_e.py` symlink and `bin/ernie2_query.py` wrapper
- **Task closed**: twosphere-mcp-3sg

### 2. Expert Query Integration (Phase 1)
Added `expert_query` parameter to 3 MCP tools:
- `interferometric_sensing` → neuroscience_MRI, physics_optics
- `lock_in_detection` → physics_optics, statistics
- `absorption_spectroscopy` → bioengineering_LOC, physics_optics

Example: "What refractive index sensitivity for GABA detection?" → ~1×10⁻⁷ RIU

### 3. Glymphatic Simulation Module
Created `src/backend/simulation/`:

**glymphatic_flow.py**:
- `GlymphaticFlowSimulator`: Stokes flow (Re << 1) in perivascular spaces
- `PerivascularSpace`: Annular gap geometry (3-50 µm typical)
- `MicrofluidicChannel`: Rectangular LOC channel geometry
- Sleep vs awake clearance coefficients (60% increase in sleep)

**clearance_network.py**:
- `disc_dimension_clearance_model`: η = η_max × exp(-β × (disc - disc_opt)²)
- `ClearanceNetworkAnalyzer`: K₅/K₃,₃ obstruction detection
- Links Paul et al. 2023 theory to clearance prediction

### 4. New MCP Tools
| Tool | Purpose |
|------|---------|
| `simulate_perivascular_flow` | CSF flow in brain PVS |
| `analyze_clearance_network` | Topology → clearance efficiency |
| `design_brain_chip_channel` | LOC design matching brain conditions |

## Key Insight Validated

**Same Stokes flow physics** (Re << 1) governs:
- PHLoC microfluidic channels (10-100 µm)
- Brain perivascular spaces (3-50 µm)

This enables direct translation between brain-on-chip experiments and in-vivo predictions.

## What's Next

### Immediate (Priority 1)

1. **twosphere-mcp-2u7**: Integrate CFD simulation for microfluidics
   - Add OpenFOAM or FEniCS for more complex geometries
   - Velocity fields, pressure distribution, mixing dynamics
   - Would enhance `simulate_perivascular_flow` with 2D/3D capability

2. **twosphere-mcp-ph7a**: Continue PH-7 integration
   - Connect simulation outputs to actual fMRI data
   - Validate clearance model against Xie et al. 2013 sleep data
   - Add amyloid-β accumulation dynamics

3. **twosphere-mcp-3ge**: Port MRISpheres signal processing
   - FFT correlation, signal alignment
   - Would complement network analysis with time-series features

### Medium Term (Priority 2)

4. **twosphere-mcp-6ez**: Adaptive feedback control for resonators
   - PID control for biosensing stability
   - Could use expert_query for control parameter tuning

5. **twosphere-mcp-i8c**: Overlay connectivity graphs on two-sphere
   - Visualize clearance network on brain geometry
   - Quaternion rotation from overlay_graph.py

### Deferred

- **twosphere-mcp-qyv**: FreeCAD MCP integration (user prefers not now)
- **optimize_resonator** tool: Not yet implemented (in backlog)

## Files Created/Modified

```
src/backend/simulation/
├── __init__.py
├── glymphatic_flow.py      # 350 lines - flow simulation
└── clearance_network.py    # 300 lines - network analysis

src/backend/services/
└── ernie2_integration.py   # Added YadaServicesMCPClient

bin/
├── ernie2_swarm_mcp_e.py   # Symlink to merge2docs
├── ernie2_query.py         # Wrapper with fallback
└── twosphere_mcp.py        # +200 lines - new tools

docs/beads/
└── 2026-01-22-q-model-accuracy-validation.md  # +5% Spearman from obstruction features
```

## Test Commands

```bash
# Test perivascular flow
python -c "
from backend.simulation.glymphatic_flow import *
pvs = PerivascularSpace(50, 20, 5)
sim = GlymphaticFlowSimulator('sleep')
print(sim.compute_steady_flow(pvs, 10))
"

# Test clearance network
python -c "
from backend.simulation.clearance_network import *
networks = create_test_networks()
analyzer = ClearanceNetworkAnalyzer()
for name, adj in networks.items():
    r = analyzer.analyze_network(adj)
    print(f'{name}: disc={r[\"clearance_prediction\"][\"disc_dimension\"]:.2f}')
"

# Test expert query
python bin/ernie2_swarm_mcp_e.py \
  --question "What flow rates for brain-on-chip perivascular simulation?" \
  --collection docs_library_bioengineering_LOC \
  --collection docs_library_neuroscience_MRI \
  --num-minions 2
```

## Related Issues

- twosphere-mcp-3sg: ✅ Closed (ernie2_swarm integration)
- twosphere-mcp-ph7a: Open (PH-7 main task)
- twosphere-mcp-2u7: Open (CFD simulation)
- twosphere-mcp-af4: ✅ Closed (Paul et al. 2023 documentation)
