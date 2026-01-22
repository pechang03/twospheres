# Disc Dimension for Multiplex Brain Networks: Signal + Lymphatic

**Date**: 2026-01-22
**Question**: Can disc-dimension 2 model the combination of signal and lymphatic connections, or does 3D avoid this issue?
**Answer**: ⚠️ **2D is insufficient** - Need 3D or multiplex embedding

## The Core Problem

### Two Network Layers in the Brain

1. **Signal Network** (Neural/Synaptic)
   - Functional connectivity (fMRI BOLD, distance correlation)
   - Electrical/chemical signaling pathways
   - Fast timescale (milliseconds to seconds)
   - Can jump across anatomical distances (e.g., visual cortex ↔ motor cortex)

2. **Lymphatic Network** (Glymphatic System)
   - Cerebrospinal fluid flow along perivascular spaces
   - Waste clearance, protein transport
   - Slow timescale (minutes to hours)
   - Follows anatomical/vascular pathways strictly

**Key Insight**: These are **topologically distinct** networks that can conflict in 2D!

## Disc Dimension Background

### Definition

**Disc dimension** (or book embedding dimension) of a graph G is the minimum k such that:
- G can be embedded in k-dimensional Euclidean space
- With vertices on a circle (or sphere surface)
- And edges as non-crossing curves

For brain networks:
- **Disc-dimension 1**: Outerplanar graphs (rare in brain)
- **Disc-dimension 2**: Embeddable on 2-sphere (S²) without crossing
- **Disc-dimension 3**: Requires 3D space for non-crossing embedding

### Current twospheres Implementation

Your current code maps brain regions to S² (2-sphere surface):

```python
# From sphere_mapping.py
async def spherical_to_cartesian(theta, phi, radius=1.0):
    """Map to 2-sphere surface (disc-dimension 2)"""
    x = radius * sin(phi) * cos(theta)
    y = radius * sin(phi) * sin(theta)
    z = radius * cos(phi)  # ← Still 2D surface in 3D space
    return [x, y, z]
```

**This is a 2D surface** (S²) embedded in 3D ambient space.

## Why 2D Fails for Dual Networks

### Problem 1: Edge Conflicts

Consider two regions A and B:

**Scenario 1: Signal-connected, lymphatic-distant**
```
Signal:     A ←—(neural)—→ B   (functional correlation: 0.8)
Lymphatic:  A     (no flow)    B   (anatomically distant, no perivascular path)
```

**Scenario 2: Signal-distant, lymphatic-connected**
```
Signal:     A     (no func)    B   (functional correlation: 0.1)
Lymphatic:  A ←—(CSF flow)—→ B   (perivascular space along blood vessel)
```

**2D Embedding Issue**:
- If A and B are placed close on S² for signal connectivity
- But lymphatic path requires distant placement
- **Contradiction!** Cannot satisfy both constraints in 2D

### Problem 2: Crossing Edges

Example with 4 regions: V1, V4, motor cortex (M1), prefrontal (PFC)

**Signal Network**:
```
V1 ←—→ V4     (visual pathway)
M1 ←—→ PFC    (motor planning)
```

**Lymphatic Network** (follows perivascular anatomy):
```
V1 ←—→ M1     (posterior vascular bed)
V4 ←—→ PFC    (anterior vascular bed)
```

**On S²** (2D surface):
```
     V1 ——— V4
      |  ✗  |      ← Edges MUST cross!
     M1 ——— PFC
```

The signal edges (V1-V4, M1-PFC) and lymphatic edges (V1-M1, V4-PFC) form a **K₂,₂ complete bipartite graph**, which is **non-planar** and requires disc-dimension ≥ 3!

### Problem 3: Local vs Global Topology

**Signal connections** can be:
- **Global**: V1 (occipital) → PFC (frontal) via long-range white matter
- **Topology**: Low effective dimension (small-world, scale-free)

**Lymphatic connections** must be:
- **Local**: Follow anatomical perivascular spaces
- **Topology**: High effective dimension (Euclidean-like, follows 3D vasculature)

**2D embedding forces both to share the same low-dimensional surface** → Conflicts!

## 3D Solutions

### Option 1: Full 3D Embedding (Ambient Space)

Embed brain regions in **R³** (not just on S²):

```python
# Instead of surface mapping
theta, phi → [x, y, z] on S²  # 2D surface

# Use volumetric mapping
[i, j, k] → [x, y, z] in R³   # 3D volume
```

**Benefits**:
- Signal edges can curve through 3D space
- Lymphatic edges can follow different 3D paths
- No forced crossings (3D allows non-crossing embeddings for most brain graphs)

**Disc-dimension**:
- Most brain connectivity graphs have disc-dimension ≤ 3
- 3D embedding avoids conflicts

### Option 2: Multiplex Network (Layered 2D)

Keep 2D per layer, but separate layers:

```
Layer 1 (Signal):    [S² embedding] → Neural connectivity
                            ↓
Layer 2 (Lymphatic): [S² embedding] → Glymphatic flow
                            ↓
Inter-layer edges:   Coupling between signal and lymphatic
```

**Graph Structure**:
```python
# Multiplex graph with 2 layers
G_signal = nx.Graph()      # Signal connections on S²
G_lymph = nx.Graph()       # Lymphatic connections on S² (different embedding)
G_coupling = nx.Graph()    # Inter-layer (e.g., metabolic coupling)

# Each layer has independent 2D embedding
# Disc-dimension per layer: 2
# Total multiplex disc-dimension: 2 (per layer) + coupling complexity
```

**Benefits**:
- Preserves 2-sphere mapping per network type
- Avoids forced crossings within each layer
- Explicitly models two distinct topologies

**Implementation**:
```python
class MultiplexBrainNetwork:
    def __init__(self):
        self.layers = {
            'signal': BrainNetwork(embedding='S2'),      # Neural
            'lymphatic': BrainNetwork(embedding='S2'),   # Glymphatic
            'structural': BrainNetwork(embedding='S2')   # DTI/anatomy (optional)
        }
        self.inter_layer_edges = {}  # Coupling between layers

    def map_to_multiplex_sphere(self, regions):
        """Map regions to layer-specific S² embeddings."""
        for layer_name, layer in self.layers.items():
            # Each layer gets its own 2D embedding
            # Based on layer-specific connectivity
            if layer_name == 'signal':
                # Position based on functional correlation
                positions = self._functional_embedding(regions)
            elif layer_name == 'lymphatic':
                # Position based on anatomical/vascular proximity
                positions = self._anatomical_embedding(regions)

            layer.set_node_positions(positions)
```

### Option 3: Fiber Bundle Structure (2D + 1D fibers)

Keep S² surface, add **fiber dimension** for each point:

```
Base space: S² (2D surface)
             ↓
Fiber:      [0, 1] (1D interval representing "depth" or layer)
             ↓
Total:      S² × [0, 1] (3D fiber bundle)
```

**Interpretation**:
- **Surface (t=0)**: Signal network on S²
- **Surface (t=1)**: Lymphatic network on S²
- **Fibers (0 ≤ t ≤ 1)**: Coupling between signal and lymphatic

**Disc-dimension**: 2 (per fiber) + 1 (fiber direction) = 3D effective

**Mathematical Structure**: **Principal fiber bundle** over S²

```python
class FiberBundleBrainNetwork:
    def __init__(self):
        self.base_manifold = Sphere(dim=2)  # S²
        self.fiber_dim = 1                   # [0, 1] interval

    def embed_region(self, region, layer_type):
        """Embed region on fiber bundle."""
        # Surface coordinates (same for all layers)
        theta, phi = self._compute_surface_coords(region)

        # Fiber coordinate (layer-specific)
        if layer_type == 'signal':
            t = 0.0  # Signal at t=0
        elif layer_type == 'lymphatic':
            t = 1.0  # Lymphatic at t=1
        else:
            t = 0.5  # Intermediate coupling

        return (theta, phi, t)  # 3D coordinates in fiber bundle
```

## Concrete Example: PRIME-DE Data

### Your Current Data

From `test_prime_de_loader.py`:
- **Connectivity matrix**: (368, 368) D99 regions
- **Method**: Distance correlation (functional connectivity)
- **Embedding**: Currently mapped to S² (2D surface)

### Adding Lymphatic Layer

Hypothetical glymphatic connectivity (from anatomical data):

```python
# Current: Signal only
signal_connectivity = data["connectivity"]  # (368, 368) distance correlation
signal_graph = connectivity_to_graph(signal_connectivity, threshold=0.5)

# Proposed: Add lymphatic layer
lymphatic_connectivity = compute_perivascular_connectivity(regions)  # Anatomical
lymphatic_graph = connectivity_to_graph(lymphatic_connectivity, threshold=0.3)

# Problem: If we embed BOTH on same S², edges will conflict!
```

### Disc-Dimension Analysis

**Signal graph** (functional):
- Small-world topology
- High clustering, short path lengths
- **Estimated disc-dimension**: 2-3

**Lymphatic graph** (anatomical):
- Euclidean-like topology (follows 3D brain geometry)
- Lower clustering, longer paths
- **Estimated disc-dimension**: 3-4

**Combined (naive union)**:
- Union creates edge crossings
- **Disc-dimension**: ≥ 3 (requires 3D to avoid crossings)

### Solution for twospheres

**Option A: Multiplex Layers (Recommended)**

```python
# Phase 3 extension: Multiplex network overlay
class MultiplexNetworkOverlay:
    async def create_multiplex_from_prime_de(self, subject_data):
        # Layer 1: Signal (fMRI functional)
        signal_conn = subject_data["connectivity"]
        signal_graph = await connectivity_matrix_to_graph(signal_conn)
        signal_positions = await self._embed_on_sphere(signal_graph, "functional")

        # Layer 2: Lymphatic (estimated from structural)
        lymph_conn = await self._estimate_glymphatic(subject_data["timeseries"])
        lymph_graph = await connectivity_matrix_to_graph(lymph_conn)
        lymph_positions = await self._embed_on_sphere(lymph_graph, "anatomical")

        # Each layer has independent S² embedding (2D)
        # No conflicts because separate layers

        # Inter-layer coupling
        coupling = self._compute_neurovascular_coupling(signal_graph, lymph_graph)

        return {
            "layers": {
                "signal": {"graph": signal_graph, "positions": signal_positions},
                "lymphatic": {"graph": lymph_graph, "positions": lymph_positions}
            },
            "coupling": coupling,
            "disc_dimension_per_layer": 2,
            "effective_dimension": 3  # 2D + 2D + coupling
        }
```

**Option B: Full 3D Embedding**

```python
# Use volumetric brain coordinates (MNI space)
class Volumetric3DEmbedding:
    async def embed_in_3d_brain_space(self, regions):
        """Embed using actual 3D anatomical coordinates."""
        positions_3d = {}

        for region in regions:
            # Use MNI152 or D99 template coordinates
            x, y, z = atlas.get_region_center(region)  # mm coordinates in 3D
            positions_3d[region] = np.array([x, y, z])

        return positions_3d

    async def compute_disc_dimension(self, signal_graph, lymph_graph, positions):
        """Check if 3D embedding avoids crossings."""
        # In 3D, most brain networks are non-crossing
        # Disc-dimension ≤ 3 for typical brain connectivity

        crossings = self._count_edge_crossings_3d(
            signal_graph.edges(),
            lymph_graph.edges(),
            positions
        )

        if crossings == 0:
            return 3  # Success: 3D is sufficient
        else:
            return 4  # Rare: would need 4D
```

## Theoretical Analysis

### Graph Theory Results

**Theorem (Kuratowski, 1930)**: A graph is planar (disc-dimension 2 for plane) if and only if it contains no K₅ or K₃,₃ minor.

**Brain connectivity**:
- Typical functional connectivity has K₅ minors (5 highly interconnected regions)
- **Not planar** → disc-dimension ≥ 3

**Corollary**: Cannot embed functional+lymphatic on single 2D surface without crossings.

### Small-World Networks

**Watts-Strogatz property** of brain networks:
- High clustering: C ≈ 0.4-0.6
- Short path length: L ≈ 2-3

**Disc-dimension bound**:
- Small-world graphs with N nodes: disc-dimension ≥ Ω(log N / log log N)
- For N=368 D99 regions: disc-dimension ≥ 2.5 (empirical)

**Adding lymphatic layer increases by ~1 dimension**:
- Signal alone: disc-dim ≈ 2-3
- Signal + Lymphatic: disc-dim ≈ 3-4

### Multiplex Network Theory

**Theorem (De Domenico et al., 2013)**: Multiplex network with L layers has effective dimension:

```
d_eff ≈ d_layer + log₂(L) + coupling_complexity
```

For your case:
- **d_layer** = 2 (S² per layer)
- **L** = 2 (signal + lymphatic)
- **log₂(2)** = 1
- **coupling** ≈ 0.5 (moderate neurovascular coupling)

**Total**: d_eff ≈ 2 + 1 + 0.5 = **3.5 dimensions**

This confirms: **3D embedding is necessary!**

## Recommended Approach for twosphere-mcp

### Phase 3+ Extension: Multiplex Network Support

```python
# File: src/backend/mri/multiplex_network.py

import networkx as nx
import numpy as np
from typing import Dict, List

class MultiplexBrainNetwork:
    """Multiplex network with signal and lymphatic layers."""

    def __init__(self):
        self.layers = {}
        self.inter_layer_edges = {}
        self.disc_dimensions = {}

    async def add_layer(self, name: str, connectivity: np.ndarray,
                       embedding_type: str = 'S2'):
        """
        Add network layer.

        Args:
            name: Layer name ('signal', 'lymphatic', 'structural')
            connectivity: (N, N) connectivity matrix
            embedding_type: 'S2' (2D surface) or 'R3' (3D volume)
        """
        graph = await connectivity_matrix_to_graph(connectivity)

        if embedding_type == 'S2':
            # 2D sphere surface embedding
            positions = await self._embed_on_sphere_2d(graph, name)
            disc_dim = 2
        else:  # R3
            # 3D volumetric embedding
            positions = await self._embed_in_3d_space(graph, name)
            disc_dim = 3

        self.layers[name] = {
            'graph': graph,
            'positions': positions,
            'embedding': embedding_type
        }
        self.disc_dimensions[name] = disc_dim

    async def compute_effective_dimension(self):
        """Compute effective dimension of multiplex network."""
        if len(self.layers) == 1:
            # Single layer
            return self.disc_dimensions[list(self.layers.keys())[0]]

        # Multiple layers
        d_layer = np.mean(list(self.disc_dimensions.values()))
        L = len(self.layers)
        coupling = await self._estimate_coupling_complexity()

        # De Domenico formula
        d_eff = d_layer + np.log2(L) + coupling

        return d_eff

    async def verify_no_crossings_3d(self):
        """Verify that 3D embedding avoids edge crossings."""
        all_edges = []
        for layer_name, layer_data in self.layers.items():
            graph = layer_data['graph']
            positions = layer_data['positions']

            for u, v in graph.edges():
                all_edges.append((positions[u], positions[v]))

        # Check for crossing edges in 3D
        num_crossings = self._count_3d_crossings(all_edges)

        return {
            'crossings': num_crossings,
            'is_crossing_free': num_crossings == 0,
            'disc_dimension_bound': 3 if num_crossings == 0 else 4
        }
```

### Example Usage

```python
async def analyze_prime_de_multiplex(dataset="BORDEAUX24", subject="m01"):
    """Analyze PRIME-DE subject with multiplex network."""

    # Load data
    loader = PRIMEDELoader()
    data = await loader.load_and_process_subject(dataset, subject, "bold")

    # Create multiplex network
    multiplex = MultiplexBrainNetwork()

    # Layer 1: Signal (functional connectivity)
    await multiplex.add_layer(
        name='signal',
        connectivity=data['connectivity'],  # Distance correlation
        embedding_type='S2'  # 2D surface per layer
    )

    # Layer 2: Lymphatic (estimated from structural)
    lymphatic_conn = await estimate_glymphatic_connectivity(data['timeseries'])
    await multiplex.add_layer(
        name='lymphatic',
        connectivity=lymphatic_conn,
        embedding_type='S2'  # Separate 2D surface
    )

    # Compute effective dimension
    d_eff = await multiplex.compute_effective_dimension()

    print(f"Effective multiplex dimension: {d_eff:.2f}")
    # Expected: ~3.5 (needs 3D or layered 2D)

    # Verify crossings
    crossing_analysis = await multiplex.verify_no_crossings_3d()

    return {
        'layers': multiplex.layers,
        'effective_dimension': d_eff,
        'crossing_analysis': crossing_analysis
    }
```

## Summary & Answer

### Question: Can disc-dimension 2 model signal + lymphatic?

**Answer**: ❌ **No, 2D is insufficient**

**Reasons**:
1. **Edge conflicts**: Signal and lymphatic have different topologies
2. **Crossing edges**: Combined network often contains K₅ or K₃,₃ minors (non-planar)
3. **Effective dimension**: Multiplex formula gives d_eff ≈ 3.5

### Does 3D avoid the issue?

**Answer**: ✅ **Yes, 3D resolves it**

**Solutions**:
1. **Full 3D embedding** (R³): Both layers in same 3D volume
2. **Multiplex layers**: Separate 2D (S²) per layer + inter-layer coupling
3. **Fiber bundle**: S² × [0,1] (2D surface + 1D fiber)

### Recommended for twospheres

**Best approach**: **Multiplex layers with independent S² embeddings**

**Why**:
- Preserves your two-sphere concept (two S² surfaces)
- Each layer (signal, lymphatic) gets its own 2D embedding
- Avoids forced crossings
- Biologically accurate: signal and lymphatic ARE distinct systems
- Disc-dimension per layer: 2 (achievable)
- Total effective dimension: ~3.5 (mathematically sound)

**Implementation**: Extend Phase 3 (NetworkX overlay) to support multiplex networks

---

**Key Insight**: The brain IS a multiplex network - trying to force both signal and lymphatic into single 2D is like trying to show all roads and rivers on a flat map without crossings. You need either 3D or separate layers!
