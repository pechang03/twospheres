"""Fractal cortical surface generation using Julia sets and stereographic projection.

Implements R(θ,φ) = radius + ε·f(θ,φ) perturbation model for realistic brain gyrification.
Based on design-11.0-neurology-tools specification.
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FractalSurfaceResult:
    """Result from fractal surface generation."""
    vertices: np.ndarray           # (N, 3) 3D vertex positions
    faces: np.ndarray             # (M, 3) triangle indices
    f_values: np.ndarray          # (N,) fractal displacement per vertex
    spherical_coords: np.ndarray  # (N, 2) [theta, phi] per vertex
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
    """Compute smooth Julia potential for array of complex numbers.

    Uses Douady-Hubbard potential: ν(z) = lim(log|z_n|/2^n) as n→∞
    For points that escape, uses smooth iteration count.

    Args:
        z_real: Real parts of initial z values
        z_imag: Imaginary parts of initial z values
        c_real: Real part of Julia set parameter c
        c_imag: Imaginary part of Julia set parameter c
        max_iter: Maximum iterations before assuming point in set

    Returns:
        Array of potential values (smooth, normalized)
    """
    n = len(z_real)
    potential = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        zr, zi = z_real[i], z_imag[i]

        for iteration in range(max_iter):
            # Julia iteration: z_{n+1} = z_n² + c
            zr_new = zr * zr - zi * zi + c_real
            zi_new = 2 * zr * zi + c_imag
            zr, zi = zr_new, zi_new

            mag_sq = zr * zr + zi * zi
            if mag_sq > 4.0:
                # Escaped - compute smooth potential
                log_mag = 0.5 * np.log(mag_sq)
                potential[i] = iteration - np.log2(log_mag)
                break
        else:
            # Didn't escape - in Julia set
            potential[i] = max_iter

    return potential


def generate_icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate icosphere mesh with given subdivision level.

    Starts with regular icosahedron (12 vertices, 20 faces) and subdivides
    each triangle into 4 smaller triangles, projecting new vertices onto sphere.

    Args:
        subdivisions: Number of subdivision iterations
            0: 12 vertices, 20 faces
            1: 42 vertices, 80 faces
            2: 162 vertices, 320 faces
            3: 642 vertices, 1280 faces
            4: 2562 vertices, 5120 faces (default)
            5: 10242 vertices, 20480 faces

    Returns:
        vertices: (N, 3) normalized vertices on unit sphere
        faces: (M, 3) triangle indices
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Initial icosahedron vertices (normalized to unit sphere)
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)

    # Normalize to unit sphere
    verts /= np.linalg.norm(verts[0])

    # Initial icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    # Subdivide iteratively
    for _ in range(subdivisions):
        verts, faces = _subdivide_icosphere(verts, faces)

    return verts, faces


def _subdivide_icosphere(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4 triangles.

    For each edge, create midpoint and project to sphere.
    Each original triangle becomes 4 triangles.

    Args:
        vertices: (N, 3) current vertices
        faces: (M, 3) current faces

    Returns:
        new_vertices: (N', 3) with N' > N
        new_faces: (M', 3) with M' = 4*M
    """
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []

    def get_midpoint(i: int, j: int) -> int:
        """Get or create midpoint of edge (i,j)."""
        key = (min(i, j), max(i, j))
        if key not in edge_midpoints:
            mid = (vertices[i] + vertices[j]) / 2
            mid /= np.linalg.norm(mid)  # Project to sphere
            edge_midpoints[key] = len(new_vertices)
            new_vertices.append(mid)
        return edge_midpoints[key]

    # Subdivide each face
    for v0, v1, v2 in faces:
        # Get midpoints of three edges
        a = get_midpoint(v0, v1)
        b = get_midpoint(v1, v2)
        c = get_midpoint(v2, v0)

        # Create 4 new triangles
        new_faces.extend([
            [v0, a, c],  # Corner at v0
            [v1, b, a],  # Corner at v1
            [v2, c, b],  # Corner at v2
            [a, b, c]    # Center triangle
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
    """Generate fractal-perturbed spherical surface modeling cortical gyrification.

    Uses formula: R(θ,φ) = radius * (1 + ε·f(θ,φ))

    where f(θ,φ) is computed via:
    1. Stereographic projection: (θ,φ) → z = tan(θ/2) e^(iφ)
    2. Julia set iteration on complex plane
    3. Smooth potential function gives f ∈ [-0.5, 0.5]

    Args:
        method: Fractal generation method ("julia", "mandelbrot", "perlin")
        epsilon: Perturbation amplitude (0.05-0.15 typical for cortex)
        julia_c_real: Real part of Julia set parameter c
        julia_c_imag: Imaginary part of Julia set parameter c
        resolution: Approximate target resolution (vertices ≈ resolution²)
        radius: Base sphere radius
        max_iterations: Max iterations for Julia/Mandelbrot escape
        compute_safety_bound: Compute maximum safe ε before self-intersection
        compute_curvature: Compute Gaussian and mean curvature

    Returns:
        FractalSurfaceResult with vertices, faces, and analysis

    Example:
        >>> result = generate_fractal_surface("julia", epsilon=0.1, resolution=50)
        >>> print(f"Generated {len(result.vertices)} vertices")
        >>> print(f"Fractal dimension: {result.fractal_dimension:.2f}")
        >>> print(f"Safe epsilon < {result.epsilon_max:.2f}")
    """
    # Determine subdivision level from resolution
    # subdivisions=3 → ~642 verts, =4 → ~2562 verts, =5 → ~10242 verts
    subdivisions = max(2, int(np.log2(resolution / 10)))

    print(f"Generating icosphere with subdivision level {subdivisions}...")
    vertices, faces = generate_icosphere(subdivisions)
    n_verts = len(vertices)
    print(f"Created mesh with {n_verts} vertices, {len(faces)} faces")

    # Convert to spherical coordinates
    theta = np.arccos(np.clip(vertices[:, 2], -1, 1))  # [0, π]
    phi = np.arctan2(vertices[:, 1], vertices[:, 0])    # [-π, π]

    # Stereographic projection: z = tan(θ/2) * e^(iφ)
    # Avoid singularity at poles
    half_theta = theta / 2
    half_theta = np.clip(half_theta, 1e-6, np.pi/2 - 1e-6)

    z_mag = np.tan(half_theta)
    z_real = z_mag * np.cos(phi)
    z_imag = z_mag * np.sin(phi)

    print(f"Computing {method} fractal potential...")

    # Compute fractal potential
    if method == "julia":
        potential = _julia_potential_batch(
            z_real, z_imag, julia_c_real, julia_c_imag, max_iterations
        )
    elif method == "mandelbrot":
        # Mandelbrot: z₀=0, c varies
        potential = _julia_potential_batch(
            np.zeros_like(z_real), np.zeros_like(z_imag),
            z_real, z_imag, max_iterations
        )
    elif method == "perlin":
        # Simple multi-octave noise (placeholder)
        potential = np.zeros(n_verts)
        for octave in range(5):
            scale = 2 ** octave
            noise = np.random.randn(n_verts) / scale
            potential += noise
    else:
        raise ValueError(f"Unknown method: {method}. Use 'julia', 'mandelbrot', or 'perlin'")

    # Normalize to [0, 1]
    pot_min, pot_max = potential.min(), potential.max()
    if pot_max > pot_min:
        f_values = (potential - pot_min) / (pot_max - pot_min)
    else:
        f_values = np.zeros(n_verts)

    # Center around 0 for symmetric perturbation: f ∈ [-0.5, 0.5]
    f_values = f_values - 0.5

    print(f"Applying perturbation with ε={epsilon}...")

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
        print("Computing safety bound...")
        # ε_max ≈ 1 / (max|∇²f| + max|∇f|²)
        grad_f = _compute_gradient_magnitude(vertices, faces, f_values)
        laplacian_f = _compute_laplacian_magnitude(vertices, faces, f_values)

        denom = np.max(np.abs(laplacian_f)) + np.max(grad_f ** 2)
        if denom > 0:
            result.epsilon_max = 1.0 / denom
        else:
            result.epsilon_max = 1.0

        print(f"Maximum safe ε = {result.epsilon_max:.3f}")

    # Estimate fractal dimension
    print("Estimating fractal dimension...")
    result.fractal_dimension = _estimate_fractal_dimension(f_values, theta, phi)
    print(f"Fractal dimension D ≈ {result.fractal_dimension:.2f}")

    # Compute surface area
    result.surface_area = _compute_surface_area(perturbed_vertices, faces)
    print(f"Surface area: {result.surface_area:.2f} (vs {4*np.pi*radius**2:.2f} for smooth sphere)")

    if compute_curvature:
        print("Computing curvatures...")
        result.gaussian_curvature, result.mean_curvature = \
            _compute_curvatures(perturbed_vertices, faces)

    return result


def _compute_gradient_magnitude(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray
) -> np.ndarray:
    """Compute gradient magnitude of scalar field on mesh."""
    n = len(vertices)
    grad = np.zeros(n)

    # Build adjacency lists
    adj = [set() for _ in range(n)]
    for f in faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])

    # Estimate gradient at each vertex
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
    """Compute Laplacian of scalar field on mesh (simplified)."""
    n = len(vertices)
    laplacian = np.zeros(n)

    # Build adjacency for uniform Laplacian
    from collections import defaultdict
    neighbors = defaultdict(list)

    for f in faces:
        for k in range(3):
            i, j = f[k], f[(k+1) % 3]
            neighbors[i].append(j)
            neighbors[j].append(i)

    # Uniform Laplacian: Δf(i) = (1/|N(i)|) Σ(f(j) - f(i))
    for i in range(n):
        nbrs = list(set(neighbors[i]))  # Remove duplicates
        if nbrs:
            laplacian[i] = np.mean(values[nbrs] - values[i])

    return laplacian


def _estimate_fractal_dimension(
    f_values: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray
) -> float:
    """Estimate fractal dimension via variogram method.

    Fits power law: variance(scale) ∝ scale^(4-2D)
    Real cortex has D ≈ 2.2-2.4
    """
    n_samples = min(500, len(f_values))
    idx = np.random.choice(len(f_values), n_samples, replace=False)

    scales = []
    variances = []

    for log_scale in np.linspace(-2.5, -0.5, 8):
        scale = 10 ** log_scale
        var_sum = 0
        count = 0

        for i in idx[:100]:  # Sample subset
            for j in idx[:100]:
                if i >= j:
                    continue
                # Angular distance on sphere
                d = np.arccos(np.clip(
                    np.sin(theta[i])*np.sin(theta[j]) +
                    np.cos(theta[i])*np.cos(theta[j])*np.cos(phi[i]-phi[j]),
                    -1, 1
                ))
                if scale/2 < d < scale*2:
                    var_sum += (f_values[i] - f_values[j])**2
                    count += 1

        if count > 5:
            scales.append(scale)
            variances.append(var_sum / count)

    if len(scales) < 3:
        return 2.3  # Default cortical value

    # Fit power law: log(variance) = slope * log(scale) + intercept
    log_scales = np.log(scales)
    log_vars = np.log(np.array(variances) + 1e-10)
    slope, _ = np.polyfit(log_scales, log_vars, 1)

    # D = 2 - slope/2
    D = 2 - slope / 2
    return np.clip(D, 2.0, 3.0)


def _compute_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of mesh."""
    area = 0.0
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        # Area of triangle = 0.5 * |cross product|
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return area


def _compute_curvatures(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian and mean curvature per vertex (simplified)."""
    n = len(vertices)

    # Placeholder: return zeros
    # Full implementation would use discrete differential geometry
    gaussian = np.zeros(n)
    mean = np.zeros(n)

    return gaussian, mean


def compute_mesh_geodesic_path(
    vertices: np.ndarray,
    faces: np.ndarray,
    start_idx: int,
    end_idx: int,
    n_interpolation_points: int = 20
) -> np.ndarray:
    """Compute geodesic path on triangular mesh between two vertices.

    Uses Dijkstra's algorithm on mesh adjacency graph.

    Args:
        vertices: (N, 3) array of mesh vertices
        faces: (M, 3) array of triangle indices
        start_idx: Index of starting vertex
        end_idx: Index of ending vertex
        n_interpolation_points: Number of points to interpolate along path

    Returns:
        (n_interpolation_points, 3) array of points along geodesic path
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import dijkstra
    from collections import deque

    n = len(vertices)

    # Build adjacency matrix with edge lengths
    adj = lil_matrix((n, n))

    for face in faces:
        # Add edges for all three sides of triangle
        for k in range(3):
            i, j = face[k], face[(k+1) % 3]
            if adj[i, j] == 0:  # Only add if not already present
                edge_length = np.linalg.norm(vertices[i] - vertices[j])
                adj[i, j] = edge_length
                adj[j, i] = edge_length

    # Convert to CSR for Dijkstra
    adj_csr = adj.tocsr()

    # Compute shortest path using Dijkstra
    dist_matrix, predecessors = dijkstra(
        adj_csr,
        directed=False,
        indices=start_idx,
        return_predecessors=True
    )

    # Reconstruct path
    path_indices = []
    current = end_idx
    path_indices.append(current)

    while current != start_idx:
        if predecessors[current] == -9999:  # No path exists
            # Fallback: return straight line
            return np.linspace(vertices[start_idx], vertices[end_idx], n_interpolation_points)

        current = predecessors[current]
        path_indices.append(current)

    path_indices.reverse()

    # Get 3D coordinates of path vertices
    path_vertices = vertices[path_indices]

    # Interpolate along path edges to get smooth curve
    if len(path_vertices) == 1:
        # Start and end are same
        return np.tile(path_vertices[0], (n_interpolation_points, 1))

    # Compute cumulative distance along path
    segment_lengths = np.linalg.norm(np.diff(path_vertices, axis=0), axis=1)
    cumulative_distances = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_distance = cumulative_distances[-1]

    if total_distance < 1e-10:
        return np.tile(path_vertices[0], (n_interpolation_points, 1))

    # Interpolate uniformly along cumulative distance
    target_distances = np.linspace(0, total_distance, n_interpolation_points)
    interpolated_path = np.zeros((n_interpolation_points, 3))

    for i, target_dist in enumerate(target_distances):
        # Find which segment this falls in
        seg_idx = np.searchsorted(cumulative_distances, target_dist) - 1
        seg_idx = max(0, min(seg_idx, len(path_vertices) - 2))

        # Linear interpolation within segment
        seg_start_dist = cumulative_distances[seg_idx]
        seg_end_dist = cumulative_distances[seg_idx + 1]
        seg_length = seg_end_dist - seg_start_dist

        if seg_length < 1e-10:
            t = 0
        else:
            t = (target_dist - seg_start_dist) / seg_length

        interpolated_path[i] = (
            (1 - t) * path_vertices[seg_idx] +
            t * path_vertices[seg_idx + 1]
        )

    return interpolated_path


def compute_mesh_geodesic_distance_matrix(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """Compute all-pairs geodesic distances on mesh.

    Args:
        vertices: (N, 3) array of mesh vertices
        faces: (M, 3) array of triangle indices

    Returns:
        (N, N) array of geodesic distances between all vertex pairs
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(vertices)

    # Build adjacency matrix
    adj = lil_matrix((n, n))

    for face in faces:
        for k in range(3):
            i, j = face[k], face[(k+1) % 3]
            if adj[i, j] == 0:
                edge_length = np.linalg.norm(vertices[i] - vertices[j])
                adj[i, j] = edge_length
                adj[j, i] = edge_length

    # Compute all-pairs shortest paths
    dist_matrix = dijkstra(adj.tocsr(), directed=False)

    return dist_matrix
