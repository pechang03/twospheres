# Graph Types and Example Patterns

## Overview

The two-sphere visualization now supports **5 graph types**, including the newly added **Erdős-Rényi random graph** model. These graphs can be mapped onto sphere surfaces using quaternion rotation.

## Supported Graph Types

### 1. Random Geometric Graph (`random_geometric`)
**Generator**: `nx.random_geometric_graph(n, 0.15, seed=seed)`

**Description**: Nodes placed randomly in 2D unit square, edges connect nodes within distance threshold.

**Characteristics**:
- Spatially-embedded network
- Good spatial clustering
- Distance-based connectivity

**Use cases**:
- Cortical connectivity modeling
- Spatial neural networks
- Retinal cell networks

**Example**:
```python
create_two_sphere_graph_visualization(
    graph_type="random_geometric",
    n_nodes=100,
    rotation_x=30, rotation_y=45
)
```

### 2. Erdős-Rényi Random Graph (`erdos_renyi`) ✨ NEW
**Generator**: `nx.erdos_renyi_graph(n, p=0.1, seed=seed)`

**Description**: Classic G(n,p) model where each edge exists independently with probability p.

**Characteristics**:
- Uniform random connectivity
- Low clustering coefficient (~p)
- Poisson degree distribution
- No spatial structure

**Use cases**:
- Baseline for network comparisons
- Null model for statistical tests
- Random network theory studies

**Example**:
```python
create_two_sphere_graph_visualization(
    graph_type="erdos_renyi",
    n_nodes=100,
    rotation_x=25, rotation_y=40
)
```

**Typical metrics** (n=100, p=0.1):
- Average degree: ~10
- Clustering: ~0.1
- Edges: ~500

### 3. Small-World Graph (`small_world`)
**Generator**: `nx.watts_strogatz_graph(n, k=6, p=0.3, seed=seed)`

**Description**: Watts-Strogatz model with local clustering and short path lengths.

**Characteristics**:
- High clustering coefficient
- Short average path length
- "Six degrees of separation"

**Use cases**:
- Brain functional networks
- Social networks
- Biological neural networks

**Example**:
```python
create_two_sphere_graph_visualization(
    graph_type="small_world",
    n_nodes=80,
    rotation_x=20, rotation_y=60
)
```

### 4. Scale-Free Graph (`scale_free`)
**Generator**: `nx.barabasi_albert_graph(n, m=3, seed=seed)`

**Description**: Barabási-Albert model with preferential attachment.

**Characteristics**:
- Power-law degree distribution
- Hub nodes with high connectivity
- "Rich get richer" dynamics

**Use cases**:
- Hub-based network architectures
- Internet/web topology
- Citation networks

**Example**:
```python
create_two_sphere_graph_visualization(
    graph_type="scale_free",
    n_nodes=70,
    rotation_x=15, rotation_y=30
)
```

### 5. Grid Graph (`grid`)
**Generator**: `nx.grid_2d_graph(side, side)`

**Description**: Regular 2D lattice structure.

**Characteristics**:
- Regular structure
- Each node has 2-4 neighbors
- High local clustering

**Use cases**:
- Regular cortical maps
- Retinotopic organization
- Testing rotations (clear structure)

**Example**:
```python
create_two_sphere_graph_visualization(
    graph_type="grid",
    n_nodes=49,  # 7x7 grid
    rotation_x=30, rotation_y=45
)
```

## Patterns from BrainstormingForPathways

While exploring the BrainstormingForPathways codebase, I found interesting patterns that could be adapted:

### Node Labeling with "Theta" Attributes
From `networkx_adapter/subgraph_action.py`:

```python
# Label nodes with semantic types
for u in G.nodes():
    theta = random.choice(["text", "vignette", "data", "code"])
    G.nodes[u]["theta"] = theta

# Color by type
node_colors = []
for u, attrs in G.nodes(data=True):
    if attrs["theta"] == "text":
        node_colors.append("blue")
    elif attrs["theta"] == "vignette":
        node_colors.append("yellow")
    elif attrs["theta"] == "data":
        node_colors.append("green")
    elif attrs["theta"] == "code":
        node_colors.append("orange")
```

**Application for brain networks**:
- Label nodes by brain region type (cortical, subcortical, cerebellar, etc.)
- Color-code by functional system (motor, visual, auditory, etc.)
- Represent cell types (excitatory, inhibitory, modulatory)

### Minimum Spanning Tree + Vertex Cover
From `subgraph_action.py`:

```python
# Create graph with specific seed for reproducibility
G = nx.random_geometric_graph(120, 0.16, seed=651151)

# Extract minimum spanning tree
T = nx.minimum_spanning_tree(G)

# Select vertex cover
cover = random.sample(G.nodes(), 4)

# Build subtree connecting cover nodes
sub_tree = subtree(G, T, cover)
```

**Application for sphere visualization**:
- Use MST to find principal connectivity pathways
- Visualize sparse backbone of dense network
- Highlight critical nodes via vertex cover

### Random Walk on Graph Overlay
From `subgraph_action.py`:

```python
# Random walk starting from leaf node
leaf_nodes = [node for node in T.nodes() if T.degree(node) == 1]
prev_node = random.choice(leaf_nodes)

random_walk_path = []
random_walk_path.append(prev_node)
for _ in range(epsilon):
    w = random.choice(list(G.neighbors(prev_node)))
    random_walk_path.append(w)
    prev_node = w
```

**Application for sphere visualization**:
- Visualize dynamic pathways on sphere surface
- Animate signal propagation
- Study geodesics vs graph paths

## Future Graph Types to Consider

Based on NetworkX documentation:

1. **Stochastic Block Model** (`stochastic_block_model`)
   - Community structure
   - Modular brain networks
   - Functional modules

2. **Hexagonal Lattice** (`hexagonal_lattice_graph`)
   - Retinal photoreceptor organization
   - Cortical columns
   - Regular tiling

3. **Caveman Graph** (`caveman_graph`)
   - Clustered structure
   - Cortical columns with inter-column connections
   - Hierarchical organization

4. **LFR Benchmark** (`LFR_benchmark_graph`)
   - Realistic community structure
   - Power-law degree distribution + communities
   - Benchmark for community detection

## Testing and Validation

All graph types tested with:
- n=50-100 nodes
- Quaternion rotation (x=25°, y=40°)
- Inter-sphere connectivity
- Network metrics (degree, clustering, path length)

## Usage

### Via MCP Tool
```json
{
    "tool": "two_sphere_graph_mapping",
    "arguments": {
        "graph_type": "erdos_renyi",
        "n_nodes": 100,
        "rotation_x": 25.0,
        "rotation_y": 40.0,
        "show_inter_edges": false,
        "save_plot": "output.png"
    }
}
```

### Via Python API
```python
from backend.visualization.graph_on_sphere import create_two_sphere_graph_visualization

result = create_two_sphere_graph_visualization(
    graph_type="erdos_renyi",
    n_nodes=100,
    rotation_x=25.0,
    rotation_y=40.0
)
```

### Via Demo Script
```bash
# Run Erdős-Rényi demo
python examples/demo_two_sphere_graphs.py --demo erdos_renyi

# Run all demos
python examples/demo_two_sphere_graphs.py --demo all

# Custom graph
python examples/demo_two_sphere_graphs.py --demo basic \
    --graph-type erdos_renyi \
    --n-nodes 150 \
    --show-inter-edges
```

## References

- **NetworkX Generators**: https://networkx.org/documentation/stable/reference/generators.html
- **Erdős-Rényi Model**: Erdős, P., & Rényi, A. (1959). "On random graphs"
- **BrainstormingForPathways patterns**: `networkx_adapter/subgraph_action.py`
- **Quaternion rotations**: `~/MRISpheres/twospheres/`

---

**Status**: ✅ All 5 graph types implemented and tested
**Last updated**: 2026-01-21
