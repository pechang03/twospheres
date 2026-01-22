# TwoSphere MCP Server

A cross-domain physics research platform integrating **fMRI neuroscience**, **optical simulations**, **graph theory**, and **topological structures**. Built as an MCP server with tools for brain network analysis, optical system design, and quantum-inspired algorithms.

## Research Domains

### 1. Neuroimaging (fMRI/MRI Analysis)

**Two-Sphere Brain Model** - Geometric framework for functional connectivity:
- Paired brain region geometry with distance/overlap metrics
- FFT correlation for frequency-domain coupling analysis
- Disc dimension estimation for network complexity
- Integration with MRISpheres research data

**Key modules**: `src/backend/mri/`
- `two_sphere.py` - Paired region geometry
- `fft_correlation.py` - Frequency-domain cross-correlation
- `disc_dimension_analysis.py` - Graph embedding dimension
- `fast_obstruction_detection.py` - K₅/K₃,₃ planarity testing
- `tripartite_multiplex_analysis.py` - Multi-layer brain networks

### 2. Optical Physics & Photonics

**Optical Simulation Stack** - pyoptools integration with advanced analysis:
- Ray tracing and wavefront analysis (Zernike decomposition)
- Fiber optics mode matching and coupling design
- Optical resonator optimization
- Adaptive feedback control systems

**Key modules**: `src/backend/optics/`
- `ray_tracing.py` - pyoptools wrapper
- `wavefront.py` - Zernike polynomial decomposition
- `fiber_optics.py` - Mode matching, coupling efficiency
- `feedback_control.py` - Adaptive resonator control
- `alignment_sensitivity.py` - Tolerance analysis

**Services**: `src/backend/services/`
- `resonator_optimizer.py` - Optical cavity design
- `fiber_coupling_designer.py` - Fiber-to-chip coupling
- `loc_simulator.py` - Lab-on-chip photonic circuits
- `sensing_service.py` - Interferometric biosensing

### 3. Topology & Geometric Structures

**Vortex Ring / Knot Theory** - Neural pathway modeling via topology:
- Trefoil knot parametric curves
- Frenet-Serret frame computation for tube surfaces
- Topological invariants for connectivity patterns

**Key module**: `src/backend/mri/vortex_ring.py`
```python
from src.backend.mri import VortexRing

# Create trefoil knot for neural pathway visualization
vortex = VortexRing(n_turns=3)  # Trefoil
x, y, z = vortex.compute_curve(num_points=1000)
surface = vortex.compute_tube_surface(tube_radius=0.2)
```

**Applications**:
- Resonator cavity shapes (toroidal, helical, Mobius geometries)
- White matter tract topology
- Optical vortex beam generation

### 4. Graph Theory & Obstruction Detection

**Structural Graph Analysis** - Based on Robertson-Seymour theory:
- Kuratowski obstruction detection (K₅, K₃,₃)
- Disc dimension estimation via planarity
- PAC-accelerated k-common neighbor queries

**Theoretical Foundation**: Paul, Protopapas, Thilikos (2023) - "Graph Parameters, Universal Obstructions, and WQO" (arXiv:2304.03688)

```python
from src.backend.mri.fast_obstruction_detection import disc_dimension_via_obstructions

result = disc_dimension_via_obstructions(brain_graph, use_pac=True)
# Returns: disc_dim_estimate (2=planar, 3+=non-planar)
```

### 5. Quantum-Inspired Algorithms

**Quantum Network Operators** - Spectral methods for brain networks:
- Quantum walk-based connectivity analysis
- Tensor network routing (QEC integration)
- Eigenvalue-based graph characterization

**Key modules**:
- `src/backend/mri/quantum_network_operators.py`
- `src/backend/services/qec_tensor_service.py`

## Cross-Domain Connections

| Domain | Shared Tools | Application |
|--------|--------------|-------------|
| Neuro + Graph | Obstruction detection | Brain network complexity bounds |
| Optics + Topology | Vortex structures | Resonator cavity design |
| Neuro + Optics | Wavefront analysis | Neural signal decomposition |
| Graph + Quantum | Spectral methods | Connectivity eigenvalues |

## Quick Start

### Installation

```bash
conda create -n twosphere python=3.11
conda activate twosphere
pip install -r requirements.txt

# Link to MRISpheres research data
ln -s ~/MRISpheres/twospheres data/twospheres
```

### Configuration

Create `.env_twosphere`:
```bash
TWOSPHERES_PATH=~/MRISpheres/twospheres
TWOSPHERE_PORT=8006
```

### Run MCP Server

```bash
# stdio mode (for Claude Desktop, etc.)
python bin/twosphere_mcp.py

# HTTP/SSE mode
python bin/twosphere_http_server.py --port 8006
```

## Project Structure

```
twosphere-mcp/
├── bin/
│   ├── twosphere_mcp.py           # MCP server (stdio)
│   └── twosphere_http_server.py   # HTTP/SSE server
├── src/
│   ├── backend/
│   │   ├── mri/                   # Brain network analysis
│   │   │   ├── two_sphere.py
│   │   │   ├── vortex_ring.py
│   │   │   ├── fft_correlation.py
│   │   │   ├── fast_obstruction_detection.py
│   │   │   ├── disc_dimension_analysis.py
│   │   │   └── quantum_network_operators.py
│   │   ├── optics/                # Optical simulations
│   │   │   ├── ray_tracing.py
│   │   │   ├── wavefront.py
│   │   │   ├── fiber_optics.py
│   │   │   └── feedback_control.py
│   │   ├── services/              # High-level APIs
│   │   │   ├── resonator_optimizer.py
│   │   │   ├── fiber_coupling_designer.py
│   │   │   ├── loc_simulator.py
│   │   │   └── sensing_service.py
│   │   └── visualization/         # 3D rendering
│   ├── atlases/                   # Brain atlas integration
│   └── simulations/               # Legacy pyoptools wrappers
├── docs/
│   ├── papers/                    # Research references
│   └── beads/                     # Session notes
└── tests/
```

## Research Context

**Primary Paper**: "Integrating Correlation and Distance Analysis in Alzheimer's Disease"

The two-sphere model provides a geometric framework unifying:
- Functional connectivity via frequency-domain correlation
- Structural complexity via graph obstruction theory
- Topological pathway analysis via knot invariants

**Key References** (see `docs/papers/`):
- Paul et al. 2023 - Universal obstructions for graph parameters
- Robertson-Seymour Graph Minor Theorem
- Kuratowski's planarity criterion

## Dependencies

- **pyoptools** >= 0.3.7 - Optical simulation
- **networkx** - Graph algorithms
- **numpy/scipy** - Numerical computing
- **matplotlib** - Visualization
- **mcp** - Model Context Protocol

## Task Tracking

Uses `bd` (beads) for issue tracking with PHY prefix:
- `PHY.1.x` - MRI analysis tasks
- `PHY.2.x` - Optical simulations
- `PHY.3.x` - Signal processing
- `PHY.4.x` - Topology/vortex

```bash
bd list          # Show all tasks
bd ready         # Show unblocked work
bd show PHY.2.1  # Task details
```

## License

Research code - see MRISpheres project for licensing.
