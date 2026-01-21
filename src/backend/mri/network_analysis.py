"""Network analysis and connectivity graph overlay on sphere surfaces.

Integrates NetworkX with sphere geometry for brain connectivity visualization.
Implements connectivity matrix → graph → sphere surface overlay pipeline.
"""

import asyncio
from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from .sphere_mapping import (
    spherical_to_cartesian,
    compute_geodesic_distance,
)


async def connectivity_matrix_to_graph(
    connectivity_matrix: NDArray,
    threshold: float = 0.0,
    node_labels: Optional[List[str]] = None
) -> nx.Graph:
    """
    Convert connectivity matrix to NetworkX graph.

    Args:
        connectivity_matrix: NxN symmetric matrix (e.g., distance correlation)
        threshold: Minimum edge weight to include (default: 0 = all edges)
        node_labels: Optional labels for nodes (default: 0, 1, 2, ...)

    Returns:
        NetworkX Graph with weighted edges

    Example:
        >>> # Functional connectivity matrix (dCor values)
        >>> conn = np.array([[1.0, 0.8, 0.3],
        ...                  [0.8, 1.0, 0.6],
        ...                  [0.3, 0.6, 1.0]])
        >>> G = await connectivity_matrix_to_graph(conn, threshold=0.5)
        >>> # Returns graph with 2 edges (0-1, 1-2) above threshold
    """
    def _convert():
        n_nodes = connectivity_matrix.shape[0]

        # Create graph
        G = nx.Graph()

        # Add nodes
        if node_labels is None:
            nodes = list(range(n_nodes))
        else:
            nodes = node_labels

        G.add_nodes_from(nodes)

        # Add edges above threshold
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                weight = connectivity_matrix[i, j]
                if weight > threshold:
                    G.add_edge(nodes[i], nodes[j], weight=weight)

        return G

    return await asyncio.to_thread(_convert)


async def map_nodes_to_sphere(
    node_locations: Dict[str, Dict[str, float]],
    sphere_center: Optional[NDArray] = None,
    radius: float = 1.0
) -> Dict[str, NDArray]:
    """
    Map graph nodes to Cartesian coordinates on sphere surface.

    Args:
        node_locations: Dict mapping node labels to spherical coords
                       {"node1": {"theta": θ, "phi": φ}, ...}
        sphere_center: Sphere center [x, y, z] (default: origin)
        radius: Sphere radius

    Returns:
        Dict mapping node labels to Cartesian coords
        {"node1": [x, y, z], "node2": [x, y, z], ...}

    Example:
        >>> locations = {
        ...     "V1": {"theta": 0, "phi": np.pi/4},
        ...     "V4": {"theta": np.pi/6, "phi": np.pi/3}
        ... }
        >>> positions = await map_nodes_to_sphere(locations)
    """
    positions = {}

    for node, location in node_locations.items():
        theta = location["theta"]
        phi = location["phi"]
        cart = await spherical_to_cartesian(theta, phi, radius, sphere_center)
        positions[node] = cart

    return positions


async def identify_interhemispheric_edges(
    graph: nx.Graph,
    node_positions: Dict[str, NDArray],
    hemisphere_boundary: float = 0.0,
    axis: int = 1
) -> List[Tuple[any, any]]:
    """
    Identify edges crossing between hemispheres.

    For two-sphere brain model, interhemispheric edges cross the
    mid-sagittal plane (y=0, representing corpus callosum).

    Args:
        graph: NetworkX graph
        node_positions: Dict mapping nodes to Cartesian [x, y, z]
        hemisphere_boundary: y-coordinate of mid-sagittal plane (default: 0)
        axis: Axis index for hemisphere boundary (0=x, 1=y, 2=z)

    Returns:
        List of edges (node1, node2) crossing hemisphere boundary

    Example:
        >>> # Graph with nodes on both hemispheres
        >>> positions = {
        ...     "right_V1": np.array([0, 1, 0]),   # y > 0
        ...     "left_V1": np.array([0, -1, 0])    # y < 0
        ... }
        >>> interhemispheric = await identify_interhemispheric_edges(G, positions)
        >>> # Returns edges crossing y=0 (corpus callosum connections)
    """
    def _identify():
        interhemispheric = []

        for edge in graph.edges():
            node1, node2 = edge
            pos1 = node_positions[node1]
            pos2 = node_positions[node2]

            # Check if nodes are on opposite sides of boundary
            coord1 = pos1[axis]
            coord2 = pos2[axis]

            if (coord1 - hemisphere_boundary) * (coord2 - hemisphere_boundary) < 0:
                # Different signs → crossing boundary
                interhemispheric.append(edge)

        return interhemispheric

    return await asyncio.to_thread(_identify)


async def compute_edge_geodesic_lengths(
    graph: nx.Graph,
    node_locations: Dict[str, Dict[str, float]],
    radius: float = 1.0
) -> Dict[Tuple[any, any], float]:
    """
    Compute geodesic arc lengths for all edges in graph.

    Args:
        graph: NetworkX graph
        node_locations: Dict mapping nodes to spherical coords
        radius: Sphere radius

    Returns:
        Dict mapping edges to geodesic distances
        {(node1, node2): distance, ...}

    Example:
        >>> locations = {
        ...     "V1": {"theta": 0, "phi": np.pi/2},
        ...     "V4": {"theta": np.pi/2, "phi": np.pi/2}
        ... }
        >>> edge_lengths = await compute_edge_geodesic_lengths(G, locations)
        >>> # Returns {("V1", "V4"): π/2, ...}
    """
    edge_lengths = {}

    for edge in graph.edges():
        node1, node2 = edge
        loc1 = node_locations[node1]
        loc2 = node_locations[node2]

        distance = await compute_geodesic_distance(loc1, loc2, radius)
        edge_lengths[edge] = distance

    return edge_lengths


async def compute_network_metrics(
    graph: nx.Graph,
    node_positions: Optional[Dict[str, NDArray]] = None
) -> Dict[str, any]:
    """
    Compute network topology metrics.

    Args:
        graph: NetworkX graph
        node_positions: Optional node positions for spatial metrics

    Returns:
        Dict with network metrics:
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - density: Edge density
        - avg_clustering: Average clustering coefficient
        - connected_components: Number of connected components
        - avg_degree: Average node degree
        - interhemispheric_fraction: Fraction of interhemispheric edges (if positions provided)

    Example:
        >>> metrics = await compute_network_metrics(G, positions)
        >>> print(f"Network density: {metrics['density']:.3f}")
    """
    def _compute():
        metrics = {}

        # Basic metrics
        metrics["n_nodes"] = graph.number_of_nodes()
        metrics["n_edges"] = graph.number_of_edges()
        metrics["density"] = nx.density(graph)
        metrics["avg_degree"] = sum(dict(graph.degree()).values()) / metrics["n_nodes"] \
            if metrics["n_nodes"] > 0 else 0

        # Clustering coefficient
        if metrics["n_nodes"] > 0:
            clustering = nx.clustering(graph)
            metrics["avg_clustering"] = sum(clustering.values()) / len(clustering)
        else:
            metrics["avg_clustering"] = 0

        # Connected components
        metrics["connected_components"] = nx.number_connected_components(graph)

        return metrics

    return await asyncio.to_thread(_compute)


async def overlay_network_on_sphere(
    connectivity_matrix: NDArray,
    node_locations: Dict[str, Dict[str, float]],
    node_labels: Optional[List[str]] = None,
    threshold: float = 0.0,
    radius: float = 1.0,
    sphere_center: Optional[NDArray] = None
) -> Dict[str, any]:
    """
    Complete pipeline: connectivity matrix → graph → sphere overlay.

    Args:
        connectivity_matrix: NxN functional connectivity matrix
        node_locations: Spherical coordinates for each node
        node_labels: Optional node labels (default: use dict keys)
        threshold: Minimum connectivity to include as edge
        radius: Sphere radius
        sphere_center: Sphere center position

    Returns:
        Dict containing:
        - graph: NetworkX graph
        - node_positions: Cartesian positions on sphere
        - edge_lengths: Geodesic lengths for each edge
        - interhemispheric_edges: Edges crossing hemispheres
        - metrics: Network topology metrics

    Example:
        >>> conn_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        >>> locations = {
        ...     "V1": {"theta": 0, "phi": np.pi/2},
        ...     "V4": {"theta": np.pi/2, "phi": np.pi/2}
        ... }
        >>> overlay = await overlay_network_on_sphere(
        ...     conn_matrix, locations, threshold=0.5
        ... )
        >>> print(f"Network has {overlay['metrics']['n_edges']} edges")
    """
    # Step 1: Convert to graph
    if node_labels is None:
        node_labels = list(node_locations.keys())

    graph = await connectivity_matrix_to_graph(
        connectivity_matrix, threshold, node_labels
    )

    # Step 2: Map to sphere
    node_positions = await map_nodes_to_sphere(
        node_locations, sphere_center, radius
    )

    # Step 3: Compute geodesic edge lengths
    edge_lengths = await compute_edge_geodesic_lengths(
        graph, node_locations, radius
    )

    # Step 4: Identify interhemispheric edges
    interhemispheric = await identify_interhemispheric_edges(
        graph, node_positions
    )

    # Step 5: Compute network metrics
    metrics = await compute_network_metrics(graph, node_positions)
    metrics["interhemispheric_edges"] = len(interhemispheric)
    metrics["interhemispheric_fraction"] = (
        len(interhemispheric) / metrics["n_edges"]
        if metrics["n_edges"] > 0 else 0
    )

    return {
        "graph": graph,
        "node_positions": node_positions,
        "edge_lengths": edge_lengths,
        "interhemispheric_edges": interhemispheric,
        "metrics": metrics
    }


async def filter_by_geodesic_distance(
    graph: nx.Graph,
    node_locations: Dict[str, Dict[str, float]],
    min_distance: float = 0.0,
    max_distance: float = np.inf,
    radius: float = 1.0
) -> nx.Graph:
    """
    Filter graph edges by geodesic distance range.

    Useful for separating local vs. long-range connections.

    Args:
        graph: NetworkX graph
        node_locations: Spherical coordinates for nodes
        min_distance: Minimum geodesic distance to keep edge
        max_distance: Maximum geodesic distance to keep edge
        radius: Sphere radius

    Returns:
        Filtered graph with only edges in distance range

    Example:
        >>> # Keep only local connections (< 1.0 cm)
        >>> local_graph = await filter_by_geodesic_distance(
        ...     G, locations, max_distance=1.0
        ... )
        >>> # Keep only long-range connections (> 2.0 cm)
        >>> longrange_graph = await filter_by_geodesic_distance(
        ...     G, locations, min_distance=2.0
        ... )
    """
    def _filter():
        filtered_graph = graph.copy()

        edges_to_remove = []
        for edge in filtered_graph.edges():
            node1, node2 = edge
            loc1 = node_locations[node1]
            loc2 = node_locations[node2]

            # Compute geodesic distance synchronously (we're in thread)
            theta1, phi1 = loc1["theta"], loc1["phi"]
            theta2, phi2 = loc2["theta"], loc2["phi"]

            delta_theta = theta2 - theta1
            cos_angle = (np.cos(phi1) * np.cos(phi2) +
                        np.sin(phi1) * np.sin(phi2) * np.cos(delta_theta))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            distance = radius * angle

            if distance < min_distance or distance > max_distance:
                edges_to_remove.append(edge)

        filtered_graph.remove_edges_from(edges_to_remove)

        return filtered_graph

    return await asyncio.to_thread(_filter)
