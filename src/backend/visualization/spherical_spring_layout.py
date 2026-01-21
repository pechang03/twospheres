"""Spherical spring embedding for graph layout on sphere surface.

Based on ../merge2docs/docs/spring_sphere.md
Adapts Fruchterman-Reingold force-directed layout to spherical geometry.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional


def spherical_spring_layout(
    G: nx.Graph,
    iterations: int = 100,
    k: float = 0.1,          # Spring constant
    K: float = 0.01,         # Repulsion constant
    d0: float = 0.3,         # Rest length in radians (~17 degrees)
    eta0: float = 0.1,       # Initial learning rate
    seed: Optional[int] = None
) -> Dict[int, Tuple[float, float, float]]:
    """Compute spherical spring embedding for graph.

    Args:
        G: NetworkX graph
        iterations: Number of iterations
        k: Spring constant (attractive force)
        K: Repulsion constant
        d0: Desired edge rest-length in radians
        eta0: Initial learning rate (annealed)
        seed: Random seed

    Returns:
        Dictionary mapping node -> (x,y,z) position on unit sphere
    """
    if seed is not None:
        np.random.seed(seed)

    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Initialize: random points on unit sphere
    positions = {}
    for node in nodes:
        # Random point on sphere via normal distribution
        p = np.random.randn(3)
        p = p / np.linalg.norm(p)
        positions[node] = p

    # Spring embedding iterations
    for t in range(iterations):
        # Annealed learning rate
        eta = eta0 * (1 - t / iterations)

        # Compute forces
        forces = {node: np.zeros(3) for node in nodes}

        for i, v in enumerate(nodes):
            p_v = positions[v]

            for u in nodes:
                if u == v:
                    continue

                p_u = positions[u]

                # Geodesic distance
                dot = np.clip(np.dot(p_v, p_u), -1.0, 1.0)
                d = np.arccos(dot)

                if d < 1e-9:
                    continue

                # Unit tangent vector pointing from v to u
                tangent = p_u - dot * p_v
                tangent_norm = np.linalg.norm(tangent)

                if tangent_norm < 1e-9:
                    continue

                u_vu = tangent / tangent_norm

                # Spring force (attractive for edges)
                if G.has_edge(v, u):
                    F_spring = k * (d - d0) * u_vu
                else:
                    F_spring = np.zeros(3)

                # Repulsive force (for all pairs)
                F_rep = -K / (d * d) * u_vu

                forces[v] += F_spring + F_rep

        # Move nodes using exponential map
        for node in nodes:
            F = forces[node]
            F_norm = np.linalg.norm(F)

            if F_norm > 1e-9:
                p = positions[node]

                # Exponential map: exp_p(eta * F)
                step_size = eta * F_norm

                # Avoid numerical issues
                if step_size < 1e-6:
                    positions[node] = p
                else:
                    F_unit = F / F_norm
                    p_new = np.cos(step_size) * p + np.sin(step_size) * F_unit

                    # Renormalize (should be automatic but ensure)
                    p_new = p_new / np.linalg.norm(p_new)
                    positions[node] = p_new

    # Convert to tuples
    return {node: tuple(pos) for node, pos in positions.items()}


def spherical_spring_layout_with_scale(
    G: nx.Graph,
    radius: float = 1.0,
    center: np.ndarray = None,
    iterations: int = 100,
    k: float = 0.1,
    K: float = 0.01,
    d0: float = 0.3,
    eta0: float = 0.1,
    seed: Optional[int] = None
) -> Dict[int, Tuple[float, float, float]]:
    """Spherical spring layout scaled to given radius and center.

    Args:
        G: NetworkX graph
        radius: Sphere radius
        center: Sphere center [x, y, z] (default: origin)
        iterations: Number of layout iterations
        k: Spring constant
        K: Repulsion constant
        d0: Rest length in radians
        eta0: Initial learning rate
        seed: Random seed

    Returns:
        Dictionary mapping node -> (x,y,z) world position
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center)

    # Compute layout on unit sphere
    unit_positions = spherical_spring_layout(
        G, iterations=iterations, k=k, K=K, d0=d0, eta0=eta0, seed=seed
    )

    # Scale and translate
    world_positions = {}
    for node, pos in unit_positions.items():
        pos_array = np.array(pos)
        pos_scaled = pos_array * radius + center
        world_positions[node] = tuple(pos_scaled)

    return world_positions


def should_use_straight_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    threshold_factor: float = 2.0
) -> bool:
    """Determine if straight line is acceptable for edge.

    Straight line is OK if it doesn't penetrate sphere volume significantly.

    Args:
        p1: First point
        p2: Second point
        sphere_center: Sphere center
        sphere_radius: Sphere radius
        threshold_factor: Multiplier for radius check

    Returns:
        True if straight line is acceptable
    """
    # Convert to local coordinates
    p1_local = p1 - sphere_center
    p2_local = p2 - sphere_center

    # Euclidean distance
    euclidean_dist = np.linalg.norm(p1_local - p2_local)

    # If edge is short (< 2*radius), straight line is fine
    if euclidean_dist < threshold_factor * sphere_radius:
        return True

    # Check if midpoint penetrates sphere
    midpoint_local = (p1_local + p2_local) / 2
    midpoint_radius = np.linalg.norm(midpoint_local)

    # If midpoint is close to surface, straight line is OK
    penetration = sphere_radius - midpoint_radius

    # Allow small penetration (5% of radius)
    return penetration < 0.05 * sphere_radius
