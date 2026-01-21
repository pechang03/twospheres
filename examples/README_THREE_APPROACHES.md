# Three Approaches to Two-Sphere Graph Visualization

## Overview

This project demonstrates three different approaches for mapping graphs onto sphere surfaces, progressing from simple to realistic:

1. **Simple Spheres + Straight Edges** - Basic visualization
2. **Simple Spheres + Geodesic Arcs** - Mathematically correct
3. **Folded Spheres + Mesh Geodesics** - Biologically realistic

---

## Approach 1: Simple Spheres + Straight Edges

**Script**: `demo_simple_spheres_straight.py`
**Output**: `simple_spheres_straight.png`

### Description
The most basic approach - draws graphs on smooth spheres with straight line edges through 3D space.

### Features
- ✓ Simple and fast
- ✓ Easy to understand
- ✗ Edges pass through sphere volume
- ✗ Not geometrically correct

### Use Case
Quick prototyping, understanding basic structure

### Mathematical Properties
- Edge length: Euclidean distance `‖p₁ - p₂‖`
- Rendering: 2 points per edge

---

## Approach 2: Simple Spheres + Geodesic Arcs

**Script**: `demo_simple_spheres_geodesic.py`
**Output**: `simple_spheres_geodesic.png`

### Description
Smooth spheres with edges following great circle arcs (geodesics) on the sphere surface.

### Features
- ✓ Mathematically correct geodesic distances
- ✓ No volume penetration
- ✓ SLERP interpolation for smooth arcs
- ✓ Hub-based corpus callosum routing
- ✗ Spheres are smooth (not folded)

### Use Case
Scientifically valid analysis, correct distance metrics

### Mathematical Properties
- Edge length: Geodesic distance `s = r·θ`
- Rendering: SLERP with 20 interpolation points
- Formula: `p(t) = [sin((1-t)θ)/sin(θ)]·v₁ + [sin(t·θ)/sin(θ)]·v₂`

### Implementation
Uses `compute_great_circle_arc()` from `graph_on_sphere.py`:
```python
arc = compute_great_circle_arc(p1, p2, n_points=20)
```

---

## Approach 3: Folded Spheres + Mesh Geodesics

**Script**: `demo_folded_spheres_mesh_geodesic.py`
**Output**: `folded_spheres_mesh_geodesic.png`

### Description
Fractal-folded cortical surfaces with edges following the actual mesh topology using Dijkstra's algorithm.

### Features
- ✓ Realistic cortical folding (Julia sets)
- ✓ Edges follow mesh surface topology
- ✓ Proper geodesic distances on curved manifold
- ✓ Biologically plausible
- ✓ Fractal dimension D ≈ 2.0-2.3

### Use Case
Realistic brain modeling, neuroscience simulations

### Mathematical Properties
- Surface: `R(θ,φ) = radius × (1 + ε·f(θ,φ))`
- Fractal function: Julia set via stereographic projection
- Edge length: Dijkstra shortest path on mesh
- Rendering: Path interpolation along mesh edges

### Implementation
Uses `compute_mesh_geodesic_path()` from `fractal_surface.py`:
```python
path = compute_mesh_geodesic_path(
    vertices, faces,
    start_vertex_idx, end_vertex_idx,
    n_interpolation_points=15
)
```

### Algorithm
1. Build mesh adjacency graph with edge lengths
2. Run Dijkstra's algorithm to find shortest path
3. Interpolate uniformly along path for smooth visualization

---

## Comparison

| Feature | Straight Edges | Geodesic Arcs | Mesh Geodesics |
|---------|---------------|---------------|----------------|
| **Speed** | Fast ⚡⚡⚡ | Medium ⚡⚡ | Slow ⚡ |
| **Correctness** | ❌ Wrong | ✅ Correct | ✅ Correct |
| **Realism** | Low | Medium | High |
| **Surface Type** | Smooth | Smooth | Folded |
| **Edge Computation** | 2 points | SLERP | Dijkstra |
| **Use Case** | Prototyping | Analysis | Simulation |

---

## Visual Comparison

### Straight Edges (Approach 1)
- Blue edges are straight lines
- Many edges appear to cut through sphere volume
- Fast to render, but geometrically incorrect

### Geodesic Arcs (Approach 2)
- Blue edges are smooth curves on sphere surface
- All edges follow great circles
- Mathematically correct, scientifically valid

### Mesh Geodesics (Approach 3)
- Cyan/magenta wireframe shows folded cortical surface
- Blue edges follow the bumpy mesh topology
- Most realistic representation of brain connectivity

---

## Running the Demos

```bash
# Approach 1: Straight edges (fastest)
python examples/demo_simple_spheres_straight.py

# Approach 2: Geodesic arcs (recommended)
python examples/demo_simple_spheres_geodesic.py

# Approach 3: Mesh geodesics (most realistic)
python examples/demo_folded_spheres_mesh_geodesic.py
```

---

## Technical Details

### Geodesic Distance on Smooth Sphere
```
s = r·θ
where θ = arccos(v₁·v₂)
```

### Mesh Geodesic Distance
```
d(u,v) = min Σ ‖vᵢ - vᵢ₊₁‖
         path
```
Computed via Dijkstra's algorithm on mesh graph.

### Fractal Surface Perturbation
```
R(θ,φ) = R_base × (1 + ε·f(θ,φ))

where f(θ,φ) is Julia set potential via:
z = tan(θ/2) e^(iφ)  (stereographic projection)
zₙ₊₁ = zₙ² + c       (Julia iteration)
```

---

## Test Results

### Approach 1: Straight Edges
- ✅ Runs successfully
- ⚠️  Some edges visibly penetrate spheres

### Approach 2: Geodesic Arcs
- ✅ All 13 geodesic routing tests pass
- ✅ Edges follow sphere surfaces correctly
- ✅ SLERP produces smooth interpolation

### Approach 3: Mesh Geodesics
- ✅ All 14 fractal surface tests pass
- ✅ Dijkstra paths follow mesh topology
- ✅ Edges match folded surface correctly

---

## Which to Use?

### Use **Approach 1** when:
- Quick prototyping
- Performance is critical
- Visual accuracy doesn't matter

### Use **Approach 2** when:
- Need correct distance metrics
- Scientific analysis/publication
- Smooth sphere assumption is valid

### Use **Approach 3** when:
- Modeling realistic brain structure
- Studying cortical folding effects
- Neuroscience simulations
- Publication-quality visualizations

---

## Future Extensions

### KISS Improvements
- [ ] Adaptive mesh resolution
- [ ] Parallel geodesic computation
- [ ] Caching of common paths

### Advanced (if needed)
- [ ] SVG/Beamer rendering (high memory)
- [ ] Interactive rotation
- [ ] Animation of signal propagation
- [ ] Multi-scale mesh hierarchy

---

## References

### Geodesic Mathematics
- SLERP: Shoemake (1985) "Animating rotation with quaternion curves"
- Dijkstra: Dijkstra (1959) "A note on two problems in connexion with graphs"

### Fractal Cortex
- Julia sets: Douady & Hubbard (1985)
- Stereographic projection: Complex analysis textbooks

### Brain Architecture
- Corpus callosum: Glasser et al. (2016) "Human Connectome Project"
- Cortical folding: Fischl (2012) "FreeSurfer"

---

**Status**: ✅ All three approaches implemented and tested
**Date**: 2026-01-21
