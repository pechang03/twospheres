# Two-Sphere Graph Mapping with Quaternion Rotation

## Overview

Maps planar graphs onto two sphere surfaces using quaternion-based rotation. This tool visualizes paired brain regions (like left/right hemispheres) with network connectivity patterns.

**Built on**: `~/MRISpheres/twospheres` quaternion work
**Key advantage**: Quaternions avoid gimbal lock and allow complex rotations for later analysis

## Features

### Graph Types Supported
- **Random Geometric**: Spatially-embedded networks (good for modeling cortical connectivity)
- **Erdős-Rényi**: Classic random graph model G(n,p) - each edge exists independently with probability p
- **Small-World**: Watts-Strogatz model (high clustering + short path lengths)
- **Scale-Free**: Barabási-Albert model (power-law degree distribution)
- **Grid**: Regular lattice structure

### Quaternion Rotation
Uses `numpy-quaternion` for smooth, composable 3D rotations:
```python
# Create rotation quaternion from Euler angles
q_x = quaternion.from_float_array([cos(θ/2), sin(θ/2), 0, 0])
q_y = quaternion.from_float_array([cos(φ/2), 0, sin(φ/2), 0])
q_z = quaternion.from_float_array([cos(ψ/2), 0, 0, sin(ψ/2)])

# Combined rotation: Rz * Ry * Rx
q_rot = qz * qy * qx

# Rotate point: q_rot * point * q_rot.conj()
```

**Why quaternions?**
1. No gimbal lock (unlike Euler angles)
2. Smooth interpolation (SLERP)
3. Composable rotations
4. Enables later complex analysis (geodesics, parallel transport)

### Coordinate Mapping
Planar graphs (u,v) ∈ [0,1]² → Sphere surface:
1. **Planar → Spherical**: (u,v) → (θ, φ) where θ=2πu, φ=πv
2. **Spherical → Cartesian**: (θ,φ,r) → (x,y,z)
3. **Quaternion Rotation**: q * point * q†
4. **Translation**: Add sphere center

## Usage

### MCP Tool

```python
# Via MCP
{
    "tool": "two_sphere_graph_mapping",
    "arguments": {
        "graph_type": "random_geometric",  # or small_world, scale_free, grid
        "n_nodes": 100,
        "radius": 1.0,
        "rotation_x": 30.0,  # degrees
        "rotation_y": 45.0,
        "rotation_z": 0.0,
        "show_inter_edges": false,  # Show corpus callosum-like connections
        "save_plot": "spheres.png"  # Optional
    }
}
```

### Direct Python API

```python
from backend.visualization.graph_on_sphere import (
    create_two_sphere_graph_visualization,
    SphereGraphConfig
)

# Simple usage
result = create_two_sphere_graph_visualization(
    graph_type="random_geometric",
    n_nodes=100,
    rotation_x=30.0,
    rotation_y=45.0,
    show_inter_edges=True
)

print(f"Clustering: {result['clustering_sphere1']:.3f}")
print(f"Average degree: {result['avg_degree_sphere1']:.2f}")
```

### Advanced Usage with Custom Config

```python
from backend.visualization.graph_on_sphere import (
    SphereGraphConfig,
    visualize_two_spheres_with_graphs,
    generate_graph
)

# Custom configuration
config = SphereGraphConfig(
    radius=1.0,
    center1=[0, 1.2, 0],  # More separated
    center2=[0, -1.2, 0],
    graph_type="small_world",
    n_nodes=100,
    rotation_x_deg=25.0,
    rotation_y_deg=35.0,
    rotation_z_deg=10.0,  # 3-axis rotation
    edge_color="darkblue",
    node_color="red",
    sphere1_color="lightcyan",
    sphere2_color="lightpink",
    sphere_alpha=0.2,
    show_inter_sphere_edges=True
)

# Generate graphs
G1 = generate_graph("small_world", 100, seed=42)
G2 = generate_graph("small_world", 100, seed=43)

# Inter-hemisphere connections
inter_edges = [(i, i) for i in range(min(len(G1), len(G2)))]

# Visualize
fig = visualize_two_spheres_with_graphs(
    config, G1, G2, inter_edges,
    save_path="publication_figure.png",
    figsize=(14, 12)
)
```

## Demo Script

Run the demonstration:
```bash
cd /Users/petershaw/code/aider/twosphere-mcp

# All demos
python examples/demo_two_sphere_graphs.py

# Specific demo
python examples/demo_two_sphere_graphs.py --demo small_world

# Custom parameters
python examples/demo_two_sphere_graphs.py --demo basic \
    --graph-type random_geometric \
    --n-nodes 150 \
    --show-inter-edges
```

## Applications

### Neuroscience
- **Bilateral brain connectivity**: Left/right hemisphere functional networks
- **fMRI analysis**: Map functional connectivity onto anatomical spheres
- **Corpus callosum modeling**: Inter-sphere edges represent callosal fibers
- **Network metrics**: Clustering, path length, modularity on spherical geometry

### Graph Theory
- **Spherical graph embedding**: Preserve network structure on manifolds
- **Geodesic analysis**: Shortest paths on sphere surface
- **Curvature effects**: How spherical geometry affects network properties

### Visualization
- **Publication figures**: High-quality paired sphere visualizations
- **Interactive rotation**: Quaternion-based smooth animations
- **Multi-scale analysis**: From local connectivity to global topology

## Technical Details

### Spherical Coordinate System
- **Azimuthal angle** θ ∈ [0, 2π]: Longitude-like
- **Polar angle** φ ∈ [0, π]: Latitude-like (0 at north pole)
- **Radius** r: Distance from center

### Cartesian Conversion
```
x = r sin(φ) cos(θ)
y = r sin(φ) sin(θ)
z = r cos(φ)
```

### Quaternion Representation
A quaternion q = w + xi + yj + zk represents rotation:
- **Scalar part** w = cos(θ/2): Rotation magnitude
- **Vector part** (x,y,z): Rotation axis (normalized)
- **Rotation**: v' = q v q† where v is pure quaternion (w=0)

### Network Statistics Returned
```python
{
    'graph_type': 'random_geometric',
    'n_nodes_sphere1': 100,
    'n_edges_sphere1': 423,
    'n_nodes_sphere2': 100,
    'n_edges_sphere2': 418,
    'avg_degree_sphere1': 8.46,
    'avg_degree_sphere2': 8.36,
    'clustering_sphere1': 0.234,
    'clustering_sphere2': 0.227,
    'inter_sphere_edges': 100,
    'quaternion_rotation': {'x_degrees': 30, 'y_degrees': 45, 'z_degrees': 0}
}
```

## Connection to Existing Work

### Builds on ~/MRISpheres/twospheres

**From your existing code**:
```python
# Your spherical_mapping.py
def spherical_coordinates(u, v):
    r = 1
    theta = u * 2 * np.pi
    phi = v * np.pi
    return theta, phi, r

# Your quaternion rotation
qx = quaternion.from_float_array([np.cos(30*π/180), 0, 0, np.sin(30*π/180)])
qy = quaternion.from_float_array([np.cos(45*π/180), 0, np.sin(45*π/180), 0])
q_rot = qx * qy

# Rotate point
q1 = quaternion.quaternion(x, y, z, 0.0)
rotated_point = q_rot * q1 * q_rot.conj()
```

**Now enhanced with**:
- Multiple graph types (not just random geometric)
- Configurable rotation (3-axis)
- Inter-sphere connectivity
- Network statistics
- MCP tool integration
- Publication-quality rendering

### Future Extensions (Enabled by Quaternions)

1. **Geodesic computation**: Use quaternions for parallel transport along geodesics
2. **Smooth animation**: SLERP between rotation states
3. **Alignment optimization**: Find optimal rotation to minimize edge crossings
4. **Curvature analysis**: Gaussian curvature effects on network metrics
5. **Multi-sphere systems**: Extend to >2 spheres (multiple brain regions)

## Examples Gallery

### Random Geometric Graph
```python
create_two_sphere_graph_visualization(
    graph_type="random_geometric",
    n_nodes=100,
    rotation_x=30, rotation_y=45
)
```
**Use case**: Spatially-embedded cortical networks

### Erdős-Rényi Random Graph
```python
create_two_sphere_graph_visualization(
    graph_type="erdos_renyi",
    n_nodes=100,
    rotation_x=25, rotation_y=40
)
```
**Use case**: Classic random graph model - baseline for network comparison studies

### Small-World Graph
```python
create_two_sphere_graph_visualization(
    graph_type="small_world",
    n_nodes=80,
    rotation_x=20, rotation_y=60
)
```
**Use case**: Balanced local/global connectivity (typical of brain networks)

### Scale-Free Graph
```python
create_two_sphere_graph_visualization(
    graph_type="scale_free",
    n_nodes=70,
    rotation_x=15, rotation_y=30
)
```
**Use case**: Hub-based network architecture

### With Inter-Hemisphere Connections
```python
create_two_sphere_graph_visualization(
    graph_type="random_geometric",
    n_nodes=50,
    show_inter_edges=True
)
```
**Use case**: Corpus callosum fiber modeling

## References

- **Quaternions**: `numpy-quaternion` package (Boyle 2016)
- **Graph generation**: NetworkX library
- **Visualization**: matplotlib with mpl_toolkits.mplot3d
- **Original work**: `~/MRISpheres/twospheres/overlay_graph.py`

## See Also

- `src/backend/visualization/graph_on_sphere.py` - Main implementation
- `examples/demo_two_sphere_graphs.py` - Demonstration script
- `~/MRISpheres/twospheres/` - Original quaternion rotation work
- Phase 2 docs: Quaternion rotation and two-sphere geodesics

---

**Status**: ✅ Production ready
**Integration**: MCP tool `two_sphere_graph_mapping`
**Dependencies**: numpy, networkx, matplotlib, numpy-quaternion
