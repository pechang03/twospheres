# Beads: Brain Hemisphere Routing Enhancements

## 🔴 BEAD-BHR-1: Geodesic Edge Routing (CRITICAL)
**Priority**: CRITICAL (blocking correctness)
**Dependencies**: Current two-sphere implementation
**Estimated Complexity**: Medium
**Status**: ✅ COMPLETED (2026-01-21)

### Problem
**Current behavior**: Edges drawn as straight lines in 3D space, which pass through sphere volume.

**Why this is wrong**:
1. **Biological implausibility**: Cortical connections follow brain surface, not straight lines through white matter
2. **Distance metrics incorrect**: Euclidean distance ≠ geodesic (surface) distance
3. **Graph algorithms invalid**: Shortest paths computed on wrong metric
4. **Visualization misleading**: Shows impossible connections

### Objective
Implement geodesic (great circle) edge routing where all edges follow sphere surface.

### Implementation Tasks

#### 1. Geodesic Distance Computation
**File**: `src/backend/visualization/graph_on_sphere.py`

```python
def geodesic_distance_on_sphere(p1: np.ndarray, p2: np.ndarray, radius: float) -> float:
    """Compute geodesic (great circle) distance between two points on sphere.

    Args:
        p1, p2: 3D points on sphere surface (x, y, z)
        radius: Sphere radius

    Returns:
        Arc length along sphere surface
    """
    # Normalize to unit vectors
    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)

    # Angle between vectors (spherical distance)
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    # Arc length = radius * angle
    return radius * angle
```

**Mathematical background**:
- Two points on sphere define unique great circle
- Shortest path on surface is arc of this circle
- Arc length: `s = r·θ` where θ is central angle

#### 2. Great Circle Arc Computation
**File**: `src/backend/visualization/graph_on_sphere.py`

```python
def compute_great_circle_arc(
    p1: np.ndarray,
    p2: np.ndarray,
    n_points: int = 20
) -> np.ndarray:
    """Compute points along great circle arc between p1 and p2.

    Uses Spherical Linear Interpolation (SLERP) to generate smooth arc.

    Args:
        p1, p2: 3D points on sphere (x, y, z)
        n_points: Number of intermediate points

    Returns:
        Array of shape (n_points, 3) with arc coordinates
    """
    radius = np.linalg.norm(p1)

    # Normalize to unit sphere
    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)

    # Angle between vectors
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    sin_angle = np.sin(angle)

    # Handle degenerate case (points very close)
    if sin_angle < 1e-6:
        return np.linspace(p1, p2, n_points)

    # Slerp formula: p(t) = [sin((1-t)θ)/sin(θ)]·v1 + [sin(t·θ)/sin(θ)]·v2
    t = np.linspace(0, 1, n_points)
    a = np.sin((1 - t) * angle) / sin_angle
    b = np.sin(t * angle) / sin_angle

    # Interpolate on unit sphere
    arc_points = np.outer(a, v1) + np.outer(b, v2)

    # Scale back to original sphere radius
    return arc_points * radius
```

**SLERP Properties**:
- Constant angular velocity
- Preserves sphere radius
- Smooth interpolation
- Used in quaternion rotation (similar math)

#### 3. Update Visualization to Use Geodesic Arcs
**File**: `src/backend/visualization/graph_on_sphere.py`

**Current code** (in `visualize_two_spheres_with_graphs()`):
```python
# WRONG: Straight line through volume
for u, v in edges1:
    p1 = pos1_3d[u]
    p2 = pos1_3d[v]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            c=config.edge_color, alpha=0.6, linewidth=1)
```

**New code**:
```python
# CORRECT: Geodesic arc on surface
for u, v in edges1:
    p1 = np.array(pos1_3d[u])
    p2 = np.array(pos1_3d[v])

    # Compute arc points
    arc = compute_great_circle_arc(p1, p2, n_points=20)

    # Plot arc
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
            c=config.edge_color, alpha=0.6, linewidth=1)
```

#### 4. Graph Construction with Geodesic Metric
**File**: `src/backend/visualization/graph_on_sphere.py`

**Problem**: Random geometric graphs use Euclidean distance threshold
**Solution**: Use geodesic distance threshold

```python
def generate_graph_geodesic(
    graph_type: str,
    n_nodes: int,
    radius: float,
    threshold_distance: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """Generate graph using geodesic distance metric on sphere.

    Args:
        graph_type: Currently only 'random_geometric' supported
        n_nodes: Number of nodes
        radius: Sphere radius
        threshold_distance: Maximum geodesic distance for edge
        seed: Random seed
    """
    if graph_type == "random_geometric":
        # Generate random positions on sphere
        G = nx.Graph()

        # Random spherical coordinates
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, 2*np.pi, n_nodes)
        phi = rng.uniform(0, np.pi, n_nodes)

        # Convert to Cartesian and store
        positions_3d = []
        for i in range(n_nodes):
            x = radius * np.sin(phi[i]) * np.cos(theta[i])
            y = radius * np.sin(phi[i]) * np.sin(theta[i])
            z = radius * np.cos(phi[i])
            positions_3d.append(np.array([x, y, z]))

            # Store 2D coords for compatibility
            u = theta[i] / (2 * np.pi)
            v = phi[i] / np.pi
            G.add_node(i, pos=(u, v), pos_3d=(x, y, z))

        # Add edges based on geodesic distance
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = geodesic_distance_on_sphere(
                    positions_3d[i],
                    positions_3d[j],
                    radius
                )
                if dist <= threshold_distance:
                    G.add_edge(i, j, weight=dist)

        return G

    else:
        # For non-geometric graphs, use existing method
        # then compute geodesic weights
        G = generate_graph(graph_type, n_nodes, seed)

        # Add geodesic weights to edges
        # ... (implementation)

        return G
```

#### 5. Update Edge Weighting Throughout
**Files to modify**:
- `src/backend/visualization/graph_on_sphere.py`
- Any graph analysis functions

**Changes**:
- Replace Euclidean distance with geodesic distance
- Update `shortest_path` to use edge weights
- Update treewidth computation (if using distances)

```python
# Example: Weighted shortest path
path = nx.shortest_path(G, source, target, weight='weight')
path_length = nx.shortest_path_length(G, source, target, weight='weight')
```

### Test Coverage

**Test file**: `tests/test_geodesic_routing.py`

```python
def test_geodesic_distance():
    """Test geodesic distance computation."""
    # North and south poles
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 0, -1])

    dist = geodesic_distance_on_sphere(p1, p2, radius=1.0)
    expected = np.pi  # Half circumference
    assert np.isclose(dist, expected, rtol=1e-5)

def test_great_circle_arc():
    """Test great circle arc computation."""
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])

    arc = compute_great_circle_arc(p1, p2, n_points=100)

    # All points should be on sphere
    radii = np.linalg.norm(arc, axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5)

    # Arc should be smooth (no kinks)
    # ... test angular differences

def test_geodesic_graph_edges():
    """Test that graph edges use geodesic distances."""
    G = generate_graph_geodesic("random_geometric", n_nodes=50,
                                radius=1.0, threshold_distance=0.5)

    for u, v in G.edges():
        # Edge weight should equal geodesic distance
        pos_u = G.nodes[u]['pos_3d']
        pos_v = G.nodes[v]['pos_3d']

        computed_dist = geodesic_distance_on_sphere(pos_u, pos_v, 1.0)
        stored_weight = G[u][v]['weight']

        assert np.isclose(computed_dist, stored_weight, rtol=1e-5)
```

### Integration Points
- **Phase 1 (current)**: Visualization uses geodesic arcs
- **Shortest paths**: Use weighted paths on geodesic metric
- **Treewidth**: May need adjustment for weighted graphs
- **Hub detection**: Still based on degree (unaffected)

### Success Criteria
- ✅ All edges follow sphere surface (no volume penetration)
- ✅ Geodesic distances used for edge weights
- ✅ Shortest paths computed using geodesic metric
- ✅ Visual inspection: edges appear as curves on surface
- ✅ Test coverage: >90% for new functions

### Mathematical Validation
Compare geodesic vs. Euclidean distances:

| Points | Euclidean | Geodesic | Ratio |
|--------|-----------|----------|-------|
| Adjacent (θ=10°) | 0.174r | 0.175r | 1.00 |
| Quarter sphere (θ=90°) | 1.414r | 1.571r | 1.11 |
| Opposite poles (θ=180°) | 2.000r | 3.142r | 1.57 |

**Key insight**: Large angular separations have significant differences!

### ✅ Implementation Completed (2026-01-21)

**Files Modified**:
- `src/backend/visualization/graph_on_sphere.py`:
  - Added `geodesic_distance_on_sphere()` function
  - Added `compute_great_circle_arc()` function with SLERP
  - Updated `visualize_two_spheres_with_graphs()` to draw geodesic arcs
  - Special handling for antipodal (opposite) points

**Test Coverage**:
- Created `tests/test_geodesic_routing.py` with 13 tests:
  - 5 tests for geodesic distance computation
  - 6 tests for great circle arc generation
  - 2 integration tests
  - **All tests pass ✅**

**Validation Results**:
- North/South poles: geodesic = π·r ✓
- Quarter sphere: geodesic = (π/2)·r ✓
- All arc points lie exactly on sphere surface ✓
- SLERP produces constant angular velocity ✓
- Geodesic/Euclidean ratios match expected values ✓

**Demonstration**:
- `examples/demo_geodesic_routing.py` generates visualization
- Output: `geodesic_brain_architecture.png` (1.0 MB)
- Shows edges as curved arcs on sphere surfaces
- No volume penetration ✅

**Impact**:
- ✅ All edges now follow sphere surface correctly
- ✅ Distance metrics are mathematically correct
- ✅ Shortest path computations will use proper geodesic distances
- ✅ Visualization is biologically plausible

**Next Steps**: Ready for BEAD-BHR-2 (Inter-Sphere Geodesic Routing)

---

## 🔵 BEAD-BHR-2: Inter-Sphere Geodesic Routing
**Priority**: High
**Dependencies**: BEAD-BHR-1
**Estimated Complexity**: Medium
**Status**: 🚧 Not Started

### Problem
Corpus callosum connection between sphere hubs currently drawn as straight line through space. Should follow realistic fiber bundle path.

### Objective
Model corpus callosum as Bézier curve or spline connecting hemisphere hubs.

### Implementation Ideas
```python
def compute_corpus_callosum_curve(hub1_pos, hub2_pos, control_points=2):
    """Compute smooth curve for corpus callosum connection.

    Uses quadratic or cubic Bézier curve with control points
    positioned to avoid penetrating either sphere.
    """
    # Control points positioned "outward" from direct line
    # To ensure curve stays in inter-hemisphere space
```

**Biological motivation**: Corpus callosum fibers arc through space, not straight lines.

---

## 🔵 BEAD-BHR-3: Geodesic-Based Hub Placement
**Priority**: Medium
**Dependencies**: BEAD-BHR-1
**Estimated Complexity**: Low
**Status**: 🚧 Not Started

### Objective
When selecting hub placement, optimize for minimal **average geodesic distance** to all other nodes, not just degree.

### Rationale
Current hub selection: highest degree node
Better metric: node minimizing `Σ geodesic_distance(hub, v)` for all v

### Implementation
```python
def find_optimal_hub_geodesic(G: nx.Graph, pos_3d: List, radius: float):
    """Find node minimizing total geodesic distance to all others."""
    min_total_dist = float('inf')
    best_hub = None

    for node in G.nodes():
        total_dist = 0
        for other in G.nodes():
            if node != other:
                dist = geodesic_distance_on_sphere(
                    pos_3d[node], pos_3d[other], radius
                )
                total_dist += dist

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_hub = node

    return best_hub
```

**Complexity**: O(n²) but only run once per hemisphere

---

## 🟢 BEAD-BHR-4: Geodesic Path Visualization
**Priority**: Low
**Dependencies**: BEAD-BHR-1
**Estimated Complexity**: Medium
**Status**: 🚧 Not Started

### Objective
Visualize shortest paths as highlighted geodesic arcs on sphere surface.

### Features
- Color-code path edges (e.g., red for active path)
- Animate "signal propagation" along geodesic arcs
- Show multiple paths simultaneously with different colors
- Display path length (geodesic distance) as annotation

---

## 🟢 BEAD-BHR-5: Geodesic Voronoi Tessellation
**Priority**: Low (research)
**Dependencies**: BEAD-BHR-1
**Estimated Complexity**: High
**Status**: 🚧 Not Started

### Objective
Partition sphere surface into Voronoi cells based on geodesic distance to hub nodes.

### Applications
- Visualize "catchment areas" for each hub
- Assign nodes to nearest hub (geodesic routing)
- Biological analogy: cortical areas served by major hubs

### Algorithm
Compute geodesic Voronoi diagram on sphere using iterative Lloyd's algorithm with spherical distances.

---

## Priority Summary

**CRITICAL (must do before release)**:
- 🔴 BEAD-BHR-1: Geodesic Edge Routing

**High (needed for correctness)**:
- 🔵 BEAD-BHR-2: Inter-Sphere Geodesic Routing
- 🔵 BEAD-BHR-3: Geodesic-Based Hub Placement

**Low (enhancements)**:
- 🟢 BEAD-BHR-4: Geodesic Path Visualization
- 🟢 BEAD-BHR-5: Geodesic Voronoi Tessellation

**Recommended order**: BHR-1 → BHR-2 → BHR-3 → BHR-4 → BHR-5

---

## References

### Great Circle Mathematics
- Slerp (Spherical Linear Interpolation): Shoemake, K. (1985). "Animating rotation with quaternion curves"
- Geodesic distance: Haversine formula, spherical trigonometry
- Great circles: Any plane through sphere center intersects surface in great circle

### Neuroanatomy
- Corpus callosum fiber paths: Glasser, M. F. et al. (2016). "The Human Connectome Project"
- Cortical surface geometry: Fischl, B. (2012). "FreeSurfer"

### Graph Theory on Manifolds
- Geodesic graphs: Milnor, J. (1963). "Morse Theory"
- Shortest paths on surfaces: Mitchell, J. S. B. et al. (1987). "The discrete geodesic problem"

---

**Next Session**: Start with BEAD-BHR-1 (geodesic routing) - this is blocking correctness of all metrics!
