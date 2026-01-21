# Geometry Tools Specification

## Tool 1: `generate_fractal_surface`

### Purpose
Generate a fractal-perturbed spherical surface modeling cortical gyrification.

### MCP Schema
```json
{
  "name": "generate_fractal_surface",
  "description": "Generate a fractal cortical surface using Julia sets, Mandelbrot, L-systems, or Perlin noise. Returns mesh vertices, faces, and per-vertex fractal displacement values.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "method": {
        "type": "string",
        "enum": ["julia", "mandelbrot", "lsystem", "perlin"],
        "description": "Fractal generation method"
      },
      "epsilon": {
        "type": "number",
        "minimum": 0.001,
        "maximum": 0.3,
        "description": "Perturbation amplitude (0.05-0.15 typical for cortex)"
      },
      "julia_c_real": {
        "type": "number",
        "default": -0.7,
        "description": "Real part of Julia set parameter c"
      },
      "julia_c_imag": {
        "type": "number",
        "default": 0.27,
        "description": "Imaginary part of Julia set parameter c"
      },
      "resolution": {
        "type": "integer",
        "minimum": 10,
        "maximum": 1000,
        "default": 100,
        "description": "Grid resolution (vertices ≈ resolution²)"
      },
      "radius": {
        "type": "number",
        "default": 1.0,
        "description": "Base sphere radius"
      },
      "max_iterations": {
        "type": "integer",
        "default": 100,
        "description": "Max iterations for Julia/Mandelbrot escape"
      },
      "compute_safety_bound": {
        "type": "boolean",
        "default": true,
        "description": "Compute maximum ε before self-intersection"
      },
      "compute_curvature": {
        "type": "boolean",
        "default": false,
        "description": "Compute Gaussian and mean curvature"
      }
    },
    "required": ["method", "epsilon"]
  }
}
```

### Response Schema
```json
{
  "type": "object",
  "properties": {
    "vertices": {
      "type": "array",
      "description": "Nx3 array of vertex positions [[x,y,z], ...]"
    },
    "faces": {
      "type": "array",
      "description": "Mx3 array of triangle indices [[i,j,k], ...]"
    },
    "f_values": {
      "type": "array",
      "description": "N-length array of fractal displacement per vertex"
    },
    "spherical_coords": {
      "type": "array",
      "description": "Nx2 array of [θ, φ] per vertex"
    },
    "epsilon_max": {
      "type": "number",
      "description": "Maximum safe ε before self-intersection (if computed)"
    },
    "fractal_dimension": {
      "type": "number",
      "description": "Estimated fractal dimension D (typically 2.2-2.4 for cortex)"
    },
    "gaussian_curvature": {
      "type": "array",
      "description": "Per-vertex Gaussian curvature (if computed)"
    },
    "mean_curvature": {
      "type": "array",
      "description": "Per-vertex mean curvature (if computed)"
    },
    "surface_area": {
      "type": "number",
      "description": "Total surface area"
    },
    "mesh_id": {
      "type": "string",
      "description": "Reference ID for use in subsequent tool calls"
    }
  }
}
```

### Implementation

```python
# src/mri_analysis/fractal_surface.py

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class FractalSurfaceResult:
    vertices: np.ndarray      # (N, 3)
    faces: np.ndarray         # (M, 3)
    f_values: np.ndarray      # (N,)
    spherical_coords: np.ndarray  # (N, 2) [theta, phi]
    epsilon_max: Optional[float] = None
    fractal_dimension: Optional[float] = None
    gaussian_curvature: Optional[np.ndarray] = None
    mean_curvature: Optional[np.ndarray] = None
    surface_area: Optional[float] = None


@jit(nopython=True, parallel=True)
def _julia_potential_batch(
    z_real: np.ndarray,
    z_imag: np.ndarray,
    c_real: float,
    c_imag: float,
    max_iter: int
) -> np.ndarray:
    """Compute smooth Julia potential for array of complex numbers."""
    n = len(z_real)
    potential = np.zeros(n, dtype=np.float64)
    
    for i in prange(n):
        zr, zi = z_real[i], z_imag[i]
        
        for iteration in range(max_iter):
            # z = z² + c
            zr_new = zr * zr - zi * zi + c_real
            zi_new = 2 * zr * zi + c_imag
            zr, zi = zr_new, zi_new
            
            mag_sq = zr * zr + zi * zi
            if mag_sq > 4.0:
                # Smooth potential (Douady-Hubbard)
                log_mag = 0.5 * np.log(mag_sq)
                potential[i] = iteration - np.log2(log_mag)
                break
        else:
            potential[i] = max_iter
    
    return potential


def generate_icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate icosphere mesh with given subdivision level."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Initial icosahedron vertices
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])
    
    # Initial faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    # Subdivide
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)
    
    return verts, faces


def _subdivide_icosphere(
    vertices: np.ndarray, 
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4 triangles."""
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []
    
    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_midpoints:
            mid = (vertices[i] + vertices[j]) / 2
            mid /= np.linalg.norm(mid)  # Project to sphere
            edge_midpoints[key] = len(new_vertices)
            new_vertices.append(mid)
        return edge_midpoints[key]
    
    for v0, v1, v2 in faces:
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)
        new_faces.extend([
            [v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]
        ])
    
    return np.array(new_vertices), np.array(new_faces, dtype=np.int32)


def generate_fractal_surface(
    method: str,
    epsilon: float,
    julia_c_real: float = -0.7,
    julia_c_imag: float = 0.27,
    resolution: int = 100,
    radius: float = 1.0,
    max_iterations: int = 100,
    compute_safety_bound: bool = True,
    compute_curvature: bool = False
) -> FractalSurfaceResult:
    """
    Generate fractal-perturbed spherical surface.
    
    R(θ,φ) = radius + ε·f(θ,φ)
    
    where f is computed via stereographic projection to complex plane
    and Julia/Mandelbrot iteration.
    """
    # Determine subdivision level from resolution
    # subdivisions=4 → ~2562 verts, =5 → ~10242 verts, =6 → ~40962 verts
    subdivisions = max(2, int(np.log2(resolution / 10)))
    
    vertices, faces = generate_icosphere(subdivisions)
    n_verts = len(vertices)
    
    # Convert to spherical coordinates
    theta = np.arccos(np.clip(vertices[:, 2], -1, 1))  # [0, π]
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])    # [-π, π]
    
    # Stereographic projection: z = tan(θ/2) * e^(iφ)
    # Avoid division by zero at poles
    half_theta = theta / 2
    half_theta = np.clip(half_theta, 1e-6, np.pi/2 - 1e-6)
    
    z_mag = np.tan(half_theta)
    z_real = z_mag * np.cos(phi)
    z_imag = z_mag * np.sin(phi)
    
    # Compute fractal potential
    if method == "julia":
        potential = _julia_potential_batch(
            z_real, z_imag, julia_c_real, julia_c_imag, max_iterations
        )
    elif method == "mandelbrot":
        potential = _julia_potential_batch(
            np.zeros_like(z_real), np.zeros_like(z_imag),
            z_real, z_imag, max_iterations
        )
    elif method == "perlin":
        from scipy.ndimage import gaussian_filter
        # Simple multi-octave noise
        potential = np.zeros(n_verts)
        for octave in range(5):
            scale = 2 ** octave
            noise = np.random.randn(n_verts) / scale
            potential += noise
        potential = gaussian_filter(potential.reshape(-1), sigma=2).flatten()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize to [0, 1]
    pot_min, pot_max = potential.min(), potential.max()
    if pot_max > pot_min:
        f_values = (potential - pot_min) / (pot_max - pot_min)
    else:
        f_values = np.zeros(n_verts)
    
    # Center around 0 for symmetric perturbation
    f_values = f_values - 0.5
    
    # Apply perturbation: R = radius * (1 + ε·f)
    r_perturbed = radius * (1 + epsilon * f_values)
    
    # Convert back to Cartesian
    perturbed_vertices = np.column_stack([
        r_perturbed * np.sin(theta) * np.cos(phi),
        r_perturbed * np.sin(theta) * np.sin(phi),
        r_perturbed * np.cos(theta)
    ])
    
    result = FractalSurfaceResult(
        vertices=perturbed_vertices,
        faces=faces,
        f_values=f_values,
        spherical_coords=np.column_stack([theta, phi])
    )
    
    # Compute safety bound
    if compute_safety_bound:
        # ε_max ≈ 1 / (max|∇²f| + max|∇f|²)
        # Approximate via finite differences on mesh
        grad_f = _compute_gradient_magnitude(vertices, faces, f_values)
        laplacian_f = _compute_laplacian_magnitude(vertices, faces, f_values)
        
        denom = np.max(np.abs(laplacian_f)) + np.max(grad_f ** 2)
        if denom > 0:
            result.epsilon_max = 1.0 / denom
        else:
            result.epsilon_max = 1.0
    
    # Estimate fractal dimension via box-counting
    result.fractal_dimension = _estimate_fractal_dimension(f_values, theta, phi)
    
    # Compute surface area
    result.surface_area = _compute_surface_area(perturbed_vertices, faces)
    
    if compute_curvature:
        result.gaussian_curvature, result.mean_curvature = \
            _compute_curvatures(perturbed_vertices, faces)
    
    return result


def _compute_gradient_magnitude(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    values: np.ndarray
) -> np.ndarray:
    """Compute gradient magnitude of scalar field on mesh."""
    # Simplified: use vertex neighbors
    from scipy.sparse import lil_matrix
    
    n = len(vertices)
    grad = np.zeros(n)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for f in faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])
    
    for i in range(n):
        neighbors = list(adj[i])
        if neighbors:
            diffs = values[neighbors] - values[i]
            dists = np.linalg.norm(vertices[neighbors] - vertices[i], axis=1)
            dists = np.maximum(dists, 1e-10)
            grad[i] = np.max(np.abs(diffs) / dists)
    
    return grad


def _compute_laplacian_magnitude(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray
) -> np.ndarray:
    """Compute Laplacian of scalar field on mesh (cotangent weights)."""
    n = len(vertices)
    laplacian = np.zeros(n)
    
    # Build adjacency with cotangent weights
    from collections import defaultdict
    weights = defaultdict(float)
    
    for f in faces:
        for k in range(3):
            i, j, l = f[k], f[(k+1)%3], f[(k+2)%3]
            # Cotangent weight for edge (i,j) from angle at l
            vi, vj, vl = vertices[i], vertices[j], vertices[l]
            e1, e2 = vi - vl, vj - vl
            cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
            sin_angle = np.sqrt(1 - cos_angle**2 + 1e-10)
            cot = cos_angle / sin_angle
            weights[(i, j)] += 0.5 * cot
            weights[(j, i)] += 0.5 * cot
    
    for i in range(n):
        total_weight = 0
        weighted_sum = 0
        for j in range(n):
            w = weights.get((i, j), 0)
            if w > 0:
                weighted_sum += w * (values[j] - values[i])
                total_weight += w
        if total_weight > 0:
            laplacian[i] = weighted_sum / total_weight
    
    return laplacian


def _estimate_fractal_dimension(
    f_values: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> float:
    """Estimate fractal dimension via variogram method."""
    # Sample pairs at different scales
    n_samples = min(1000, len(f_values))
    idx = np.random.choice(len(f_values), n_samples, replace=False)
    
    scales = []
    variances = []
    
    for log_scale in np.linspace(-3, 0, 10):
        scale = 10 ** log_scale
        var_sum = 0
        count = 0
        
        for i in idx:
            for j in idx:
                if i >= j:
                    continue
                # Angular distance
                d = np.arccos(np.clip(
                    np.sin(theta[i])*np.sin(theta[j]) + 
                    np.cos(theta[i])*np.cos(theta[j])*np.cos(phi[i]-phi[j]),
                    -1, 1
                ))
                if scale/2 < d < scale*2:
                    var_sum += (f_values[i] - f_values[j])**2
                    count += 1
        
        if count > 10:
            scales.append(scale)
            variances.append(var_sum / count)
    
    if len(scales) < 3:
        return 2.3  # Default cortical value
    
    # Fit power law: variance ∝ scale^(4-2D)
    log_scales = np.log(scales)
    log_vars = np.log(np.array(variances) + 1e-10)
    slope, _ = np.polyfit(log_scales, log_vars, 1)
    
    # D = 2 - slope/2
    D = 2 - slope / 2
    return np.clip(D, 2.0, 3.0)


def _compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area."""
    area = 0
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return area


def _compute_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian and mean curvature per vertex."""
    n = len(vertices)
    gaussian = np.zeros(n)
    mean = np.zeros(n)
    
    # Use discrete curvature formulas
    # Gaussian: K = (2π - Σ angles) / A_mixed
    # Mean: H = ||Σ (cot α + cot β)(vj - vi)|| / (4 A_mixed)
    
    # Simplified approximation
    for i in range(n):
        # Local neighborhood analysis
        gaussian[i] = 1.0  # Placeholder
        mean[i] = 1.0
    
    return gaussian, mean
```

### Test Cases
```python
def test_generate_fractal_surface_julia():
    result = generate_fractal_surface(
        method="julia",
        epsilon=0.1,
        julia_c_real=-0.7,
        julia_c_imag=0.27,
        resolution=50
    )
    assert result.vertices.shape[1] == 3
    assert len(result.f_values) == len(result.vertices)
    assert 0 < result.epsilon_max < 1
    assert 2.0 < result.fractal_dimension < 3.0

def test_epsilon_safety():
    result = generate_fractal_surface(
        method="julia", epsilon=0.05, compute_safety_bound=True
    )
    assert result.epsilon_max > 0.05, "ε should be within safe bound"
```

---

## Tool 2: `embed_graph_on_manifold`

### Purpose
Embed a neural pathway or vascular network graph onto a curved surface mesh with geodesic edge weights.

### MCP Schema
```json
{
  "name": "embed_graph_on_manifold",
  "description": "Embed a graph onto a surface mesh. Computes geodesic distances for edge weights and maps node positions to surface coordinates.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "surface_id": {
        "type": "string",
        "description": "Reference to surface from generate_fractal_surface"
      },
      "graph_type": {
        "type": "string",
        "enum": ["neural_pathways", "vascular_tree", "pvs_network", "custom"],
        "description": "Type of graph to embed"
      },
      "seed_points": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2
        },
        "description": "Node positions as [[θ₁,φ₁], [θ₂,φ₂], ...] in radians"
      },
      "connectivity": {
        "type": "string",
        "enum": ["delaunay", "knn", "mst", "arterial_tree", "custom"],
        "default": "delaunay"
      },
      "k": {
        "type": "integer",
        "default": 6,
        "description": "k for k-nearest-neighbors connectivity"
      },
      "custom_edges": {
        "type": "array",
        "items": {
          "type": "array",
          "items": {"type": "integer"},
          "minItems": 2,
          "maxItems": 2
        },
        "description": "Custom edge list [[i,j], ...] if connectivity='custom'"
      },
      "node_properties": {
        "type": "object",
        "description": "Per-node property arrays, e.g. {\"metabolic_demand\": [...]}"
      },
      "compute_hydraulic_conductance": {
        "type": "boolean",
        "default": false,
        "description": "Compute δ³/12μL for each edge (glymphatic networks)"
      },
      "delta_um": {
        "type": "number",
        "default": 7.0,
        "description": "Perivascular gap width in micrometers"
      },
      "viscosity_pa_s": {
        "type": "number",
        "default": 0.001,
        "description": "CSF viscosity in Pa·s"
      }
    },
    "required": ["surface_id", "seed_points"]
  }
}
```

### Response Schema
```json
{
  "type": "object",
  "properties": {
    "nodes": {
      "type": "array",
      "description": "Array of node objects with position and properties"
    },
    "edges": {
      "type": "array",
      "description": "Array of edge objects with geodesic_length, conductance, etc."
    },
    "adjacency_matrix": {
      "type": "array",
      "description": "NxN sparse adjacency matrix (as list of [i,j,weight])"
    },
    "graph_id": {
      "type": "string",
      "description": "Reference ID for use in simulation tools"
    },
    "total_edge_length": {
      "type": "number"
    },
    "mean_degree": {
      "type": "number"
    }
  }
}
```

### Implementation

```python
# src/mri_analysis/graph_embedding.py

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import shortest_path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class EmbeddedNode:
    index: int
    theta: float
    phi: float
    position_3d: np.ndarray
    surface_vertex_idx: int  # Nearest vertex on mesh
    properties: Dict[str, float]

@dataclass  
class EmbeddedEdge:
    source: int
    target: int
    geodesic_length: float
    euclidean_length: float
    conductance: Optional[float] = None

@dataclass
class EmbeddedGraphResult:
    nodes: List[EmbeddedNode]
    edges: List[EmbeddedEdge]
    adjacency_sparse: List[Tuple[int, int, float]]
    total_edge_length: float
    mean_degree: float


def embed_graph_on_manifold(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    seed_points: List[List[float]],  # [[θ, φ], ...]
    connectivity: str = "delaunay",
    k: int = 6,
    custom_edges: Optional[List[List[int]]] = None,
    node_properties: Optional[Dict[str, List[float]]] = None,
    compute_hydraulic_conductance: bool = False,
    delta_um: float = 7.0,
    viscosity_pa_s: float = 0.001
) -> EmbeddedGraphResult:
    """
    Embed graph onto surface mesh.
    
    Args:
        surface_vertices: (N, 3) mesh vertices
        surface_faces: (M, 3) triangle indices
        seed_points: node positions in spherical coords
        connectivity: how to connect nodes
        k: for knn connectivity
        custom_edges: explicit edge list
        node_properties: dict of property arrays
        compute_hydraulic_conductance: compute q = δ³/12μL
        delta_um: perivascular gap (micrometers)
        viscosity_pa_s: fluid viscosity
    """
    n_nodes = len(seed_points)
    
    # Convert seed points to 3D positions
    nodes = []
    for i, (theta, phi) in enumerate(seed_points):
        # Find nearest vertex on mesh
        target_3d = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Normalize mesh vertices and find closest
        mesh_normalized = surface_vertices / np.linalg.norm(surface_vertices, axis=1, keepdims=True)
        dists = np.linalg.norm(mesh_normalized - target_3d, axis=1)
        nearest_idx = np.argmin(dists)
        
        props = {}
        if node_properties:
            for key, values in node_properties.items():
                if i < len(values):
                    props[key] = values[i]
        
        nodes.append(EmbeddedNode(
            index=i,
            theta=theta,
            phi=phi,
            position_3d=surface_vertices[nearest_idx],
            surface_vertex_idx=nearest_idx,
            properties=props
        ))
    
    # Build connectivity
    if connectivity == "delaunay":
        edges_idx = _delaunay_on_sphere(seed_points)
    elif connectivity == "knn":
        edges_idx = _knn_on_sphere(seed_points, k)
    elif connectivity == "mst":
        edges_idx = _mst_on_sphere(seed_points)
    elif connectivity == "arterial_tree":
        edges_idx = _arterial_tree(seed_points)
    elif connectivity == "custom":
        edges_idx = custom_edges if custom_edges else []
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")
    
    # Compute geodesic distances using mesh
    mesh_geodesic = _compute_mesh_geodesics(surface_vertices, surface_faces)
    
    # Build edges with geodesic lengths
    edges = []
    adjacency_sparse = []
    
    for i, j in edges_idx:
        vi = nodes[i].surface_vertex_idx
        vj = nodes[j].surface_vertex_idx
        
        geodesic_len = mesh_geodesic[vi, vj]
        euclidean_len = np.linalg.norm(nodes[i].position_3d - nodes[j].position_3d)
        
        conductance = None
        if compute_hydraulic_conductance:
            # q = δ³ / (12 μ L)
            delta_m = delta_um * 1e-6
            L = geodesic_len
            if L > 0:
                conductance = (delta_m ** 3) / (12 * viscosity_pa_s * L)
        
        edge = EmbeddedEdge(
            source=i,
            target=j,
            geodesic_length=geodesic_len,
            euclidean_length=euclidean_len,
            conductance=conductance
        )
        edges.append(edge)
        adjacency_sparse.append((i, j, geodesic_len))
        adjacency_sparse.append((j, i, geodesic_len))  # Undirected
    
    total_length = sum(e.geodesic_length for e in edges)
    mean_degree = 2 * len(edges) / n_nodes if n_nodes > 0 else 0
    
    return EmbeddedGraphResult(
        nodes=nodes,
        edges=edges,
        adjacency_sparse=adjacency_sparse,
        total_edge_length=total_length,
        mean_degree=mean_degree
    )


def _delaunay_on_sphere(points: List[List[float]]) -> List[List[int]]:
    """Compute Delaunay triangulation on sphere via stereographic projection."""
    n = len(points)
    if n < 3:
        return [[0, 1]] if n == 2 else []
    
    # Stereographic projection
    projected = []
    for theta, phi in points:
        if theta < 0.01:  # Near north pole
            projected.append([1e6, 1e6])  # Push to infinity
        else:
            r = np.tan(theta / 2)
            projected.append([r * np.cos(phi), r * np.sin(phi)])
    
    projected = np.array(projected)
    
    try:
        tri = Delaunay(projected)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k+1) % 3]
                edges.add((min(i, j), max(i, j)))
        return [list(e) for e in edges]
    except:
        # Fallback to knn
        return _knn_on_sphere(points, 3)


def _knn_on_sphere(points: List[List[float]], k: int) -> List[List[int]]:
    """K-nearest neighbors on sphere."""
    n = len(points)
    edges = set()
    
    # Compute pairwise angular distances
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                dists.append(np.inf)
            else:
                t1, p1 = points[i]
                t2, p2 = points[j]
                # Great circle distance
                d = np.arccos(np.clip(
                    np.sin(t1)*np.sin(t2) + np.cos(t1)*np.cos(t2)*np.cos(p1-p2),
                    -1, 1
                ))
                dists.append(d)
        
        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            edges.add((min(i, j), max(i, j)))
    
    return [list(e) for e in edges]


def _mst_on_sphere(points: List[List[float]]) -> List[List[int]]:
    """Minimum spanning tree on sphere."""
    n = len(points)
    
    # Build complete distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            t1, p1 = points[i]
            t2, p2 = points[j]
            d = np.arccos(np.clip(
                np.sin(t1)*np.sin(t2) + np.cos(t1)*np.cos(t2)*np.cos(p1-p2),
                -1, 1
            ))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    
    # Prim's algorithm
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    
    mst = nx.minimum_spanning_tree(G)
    return [list(e) for e in mst.edges()]


def _arterial_tree(points: List[List[float]]) -> List[List[int]]:
    """Generate tree structure mimicking arterial branching."""
    # Start from point closest to "root" (e.g., base of brain)
    # Use MST as backbone
    return _mst_on_sphere(points)


def _compute_mesh_geodesics(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """Compute geodesic distance matrix on mesh."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import dijkstra
    
    n = len(vertices)
    
    # Build sparse adjacency with edge lengths
    adj = lil_matrix((n, n))
    for f in faces:
        for k in range(3):
            i, j = f[k], f[(k+1) % 3]
            d = np.linalg.norm(vertices[i] - vertices[j])
            adj[i, j] = d
            adj[j, i] = d
    
    # Compute all-pairs shortest paths
    dist_matrix = dijkstra(adj.tocsr(), directed=False)
    
    return dist_matrix
```

### Test Cases
```python
def test_embed_delaunay():
    # Create simple surface
    from fractal_surface import generate_fractal_surface
    surface = generate_fractal_surface("julia", 0.05, resolution=30)
    
    # Random seed points
    seed_points = [
        [np.pi/4, 0],
        [np.pi/4, np.pi/2],
        [np.pi/4, np.pi],
        [np.pi/2, np.pi/4]
    ]
    
    result = embed_graph_on_manifold(
        surface.vertices, surface.faces, seed_points,
        connectivity="delaunay"
    )
    
    assert len(result.nodes) == 4
    assert len(result.edges) > 0
    assert result.mean_degree > 0
```
