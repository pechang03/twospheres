"""Map planar graphs onto sphere surfaces for brain connectivity visualization.

Based on ~/MRISpheres/twospheres work on paired brain region visualization.
Maps planar graphs (random geometric, small-world, scale-free) onto two spheres
representing paired brain regions (e.g., left/right hemispheres).

Uses quaternion rotation for sphere orientation and NetworkX for graph generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import quaternion  # numpy-quaternion package


@dataclass
class SphereGraphConfig:
    """Configuration for graph-on-sphere visualization.

    Attributes:
        radius: Sphere radius (deprecated, use radius1/radius2)
        radius1: Radius of first sphere
        radius2: Radius of second sphere
        center1: Center of first sphere [x, y, z]
        center2: Center of second sphere [x, y, z]
        graph_type: Type of graph ('random_geometric', 'small_world', 'scale_free', 'custom')
        n_nodes: Number of nodes in graph
        rotation_x_deg: Rotation around x-axis in degrees
        rotation_y_deg: Rotation around y-axis in degrees
        rotation_z_deg: Rotation around z-axis in degrees
        edge_color: Color for graph edges
        node_color: Color for graph nodes
        sphere1_color: Color for first sphere
        sphere2_color: Color for second sphere
        sphere_alpha: Transparency of spheres (0-1)
        show_inter_sphere_edges: Whether to show edges between spheres
        ensure_connected: Whether to ensure graph connectivity by bridging components
    """
    radius: float = 1.0
    radius1: Optional[float] = None
    radius2: Optional[float] = None
    center1: List[float] = None
    center2: List[float] = None
    graph_type: str = "random_geometric"
    n_nodes: int = 100
    rotation_x_deg: float = 30.0
    rotation_y_deg: float = 45.0
    rotation_z_deg: float = 0.0
    edge_color: str = "blue"
    node_color: str = "red"
    sphere1_color: str = "cyan"
    sphere2_color: str = "magenta"
    sphere_alpha: float = 0.3
    show_inter_sphere_edges: bool = False
    ensure_connected: bool = True

    def __post_init__(self):
        """Initialize default centers and radii if not provided."""
        # Set individual radii if not provided
        if self.radius1 is None:
            self.radius1 = self.radius
        if self.radius2 is None:
            self.radius2 = self.radius

        if self.center1 is None:
            self.center1 = [0, self.radius1, 0]
        if self.center2 is None:
            self.center2 = [0, -self.radius2, 0]


def spherical_coordinates(u: float, v: float, r: float = 1.0) -> Tuple[float, float, float]:
    """Map 2D coordinates (u,v) ∈ [0,1]² to spherical coordinates.

    Args:
        u: Horizontal coordinate [0, 1]
        v: Vertical coordinate [0, 1]
        r: Sphere radius

    Returns:
        (theta, phi, r) in spherical coordinates
        theta ∈ [0, 2π]: azimuthal angle
        phi ∈ [0, π]: polar angle
    """
    theta = u * 2 * np.pi
    phi = v * np.pi
    return theta, phi, r


def spherical_to_cartesian(theta: float, phi: float, r: float = 1.0) -> Tuple[float, float, float]:
    """Convert spherical coordinates to Cartesian coordinates.

    Args:
        theta: Azimuthal angle [0, 2π]
        phi: Polar angle [0, π]
        r: Radius

    Returns:
        (x, y, z) in Cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def geodesic_distance_on_sphere(p1: np.ndarray, p2: np.ndarray, radius: float) -> float:
    """Compute geodesic (great circle) distance between two points on sphere surface.

    The geodesic distance is the arc length along the sphere surface between two points,
    following the shortest path (great circle). This is the correct distance metric for
    manifold-embedded graphs, as opposed to Euclidean distance through the sphere volume.

    Args:
        p1: First point on sphere (x, y, z) - must be on sphere surface
        p2: Second point on sphere (x, y, z) - must be on sphere surface
        radius: Sphere radius

    Returns:
        Arc length along sphere surface (geodesic distance)

    Mathematical background:
        - Two points on a sphere define a unique great circle
        - The shortest path on the surface follows this great circle
        - Arc length: s = r·θ where θ is the central angle between vectors
    """
    # Normalize to unit vectors (ensures points are exactly on sphere surface)
    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)

    # Compute angle between vectors (spherical distance)
    cos_angle = np.dot(v1, v2)
    # Clip to avoid numerical errors in arccos
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Arc length = radius * angle
    return radius * angle


def compute_great_circle_arc(
    p1: np.ndarray,
    p2: np.ndarray,
    n_points: int = 20
) -> np.ndarray:
    """Compute points along great circle arc between p1 and p2 using SLERP.

    Uses Spherical Linear Interpolation (SLERP) to generate a smooth geodesic arc
    on the sphere surface. All intermediate points will lie exactly on the sphere.

    Args:
        p1: Starting point on sphere (x, y, z)
        p2: Ending point on sphere (x, y, z)
        n_points: Number of interpolation points (default: 20)

    Returns:
        Array of shape (n_points, 3) with arc coordinates

    SLERP Properties:
        - Constant angular velocity along the arc
        - Preserves sphere radius (all points exactly on surface)
        - Smooth interpolation without artifacts
        - Same mathematical foundation as quaternion rotation

    Mathematical formula:
        p(t) = [sin((1-t)θ)/sin(θ)]·v1 + [sin(t·θ)/sin(θ)]·v2
        where θ is the angle between v1 and v2, t ∈ [0,1]
    """
    # Get radius from first point
    radius = np.linalg.norm(p1)

    # Normalize to unit sphere
    v1 = p1 / np.linalg.norm(p1)
    v2 = p2 / np.linalg.norm(p2)

    # Compute angle between vectors
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    sin_angle = np.sin(angle)

    # Handle degenerate cases
    if sin_angle < 1e-6:
        # Points are nearly identical or antipodal
        if angle < np.pi / 2:
            # Nearly identical - return linearly interpolated points on surface
            return np.linspace(p1, p2, n_points)
        else:
            # Antipodal (opposite) points - choose arbitrary great circle
            # Use perpendicular vector to define the arc plane
            # Find a vector perpendicular to v1
            if abs(v1[0]) < 0.9:
                perp = np.cross(v1, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(v1, np.array([0.0, 1.0, 0.0]))
            perp = perp / np.linalg.norm(perp)

            # Create arc using the perpendicular as intermediate point
            t = np.linspace(0, 1, n_points)
            arc_points = np.outer(np.cos(t * np.pi), v1) + np.outer(np.sin(t * np.pi), perp)
            return arc_points * radius

    # SLERP interpolation parameter
    t = np.linspace(0, 1, n_points)

    # SLERP coefficients
    a = np.sin((1 - t) * angle) / sin_angle  # Weight for v1
    b = np.sin(t * angle) / sin_angle        # Weight for v2

    # Interpolate on unit sphere: weighted combination of v1 and v2
    arc_points = np.outer(a, v1) + np.outer(b, v2)

    # Scale back to original sphere radius
    return arc_points * radius


def create_rotation_quaternion(
    x_deg: float = 0.0,
    y_deg: float = 0.0,
    z_deg: float = 0.0
) -> quaternion.quaternion:
    """Create a rotation quaternion from Euler angles.

    Args:
        x_deg: Rotation around x-axis in degrees
        y_deg: Rotation around y-axis in degrees
        z_deg: Rotation around z-axis in degrees

    Returns:
        Quaternion representing the combined rotation
    """
    # Convert to radians and create quaternions for each axis
    qx = quaternion.from_float_array([
        np.cos(x_deg * np.pi / 360),  # Half angle for quaternion
        np.sin(x_deg * np.pi / 360),
        0,
        0
    ])

    qy = quaternion.from_float_array([
        np.cos(y_deg * np.pi / 360),
        0,
        np.sin(y_deg * np.pi / 360),
        0
    ])

    qz = quaternion.from_float_array([
        np.cos(z_deg * np.pi / 360),
        0,
        0,
        np.sin(z_deg * np.pi / 360)
    ])

    # Combined rotation: Rz * Ry * Rx
    return qz * qy * qx


def rotate_point_quaternion(
    point: Tuple[float, float, float],
    q_rot: quaternion.quaternion
) -> Tuple[float, float, float]:
    """Rotate a 3D point using quaternion rotation.

    Args:
        point: (x, y, z) coordinates
        q_rot: Rotation quaternion

    Returns:
        Rotated (x, y, z) coordinates
    """
    x, y, z = point
    # Create pure quaternion (scalar part = 0)
    q_point = quaternion.quaternion(0.0, x, y, z)
    # Rotate: q_rot * q_point * q_rot.conj()
    q_rotated = q_rot * q_point * q_rot.conj()
    return q_rotated.x, q_rotated.y, q_rotated.z


def quaternion_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> quaternion.quaternion:
    """Create quaternion that rotates v_from to v_to.

    Args:
        v_from: Source vector (will be normalized)
        v_to: Target vector (will be normalized)

    Returns:
        Quaternion that rotates v_from to v_to
    """
    # Normalize vectors
    v1 = v_from / np.linalg.norm(v_from)
    v2 = v_to / np.linalg.norm(v_to)

    # Compute rotation axis and angle
    dot = np.dot(v1, v2)

    # Handle special cases
    if dot > 0.9999:
        # Vectors are already aligned
        return quaternion.quaternion(1, 0, 0, 0)
    elif dot < -0.9999:
        # Vectors are opposite - rotate 180° around any perpendicular axis
        # Find perpendicular axis
        if abs(v1[0]) < 0.9:
            axis = np.cross(v1, np.array([1, 0, 0]))
        else:
            axis = np.cross(v1, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        # 180° rotation
        return quaternion.quaternion(0, axis[0], axis[1], axis[2])
    else:
        # Normal case
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))

        # Create quaternion from axis-angle
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = np.sin(half_angle) * axis

        return quaternion.quaternion(w, xyz[0], xyz[1], xyz[2])


def rotate_sphere_to_place_hub_at_touching_point(
    pos_3d: List[Tuple[float, float, float]],
    hub_idx: int,
    sphere_center: np.ndarray,
    touching_point: np.ndarray
) -> List[Tuple[float, float, float]]:
    """Rotate all positions on sphere so hub node is at touching point.

    Uses quaternion rotation to move hub to the exact touching point
    between two spheres.

    Args:
        pos_3d: List of 3D positions on sphere surface
        hub_idx: Index of hub node
        sphere_center: Center of this sphere [x, y, z]
        touching_point: World coordinates where spheres touch

    Returns:
        List of rotated 3D positions
    """
    # Get hub position in local coordinates (relative to sphere center)
    hub_pos_world = np.array(pos_3d[hub_idx])
    hub_pos_local = hub_pos_world - sphere_center

    # Target position in local coordinates
    target_pos_local = touching_point - sphere_center

    # Normalize to get directions
    hub_dir = hub_pos_local / np.linalg.norm(hub_pos_local)
    target_dir = target_pos_local / np.linalg.norm(target_pos_local)

    # Create quaternion to rotate hub to target
    q_rotation = quaternion_from_vectors(hub_dir, target_dir)

    # Apply rotation to all positions
    rotated_pos = []
    for pos in pos_3d:
        pos_local = np.array(pos) - sphere_center
        pos_rotated = rotate_point_quaternion(tuple(pos_local), q_rotation)
        pos_world = np.array(pos_rotated) + sphere_center
        rotated_pos.append(tuple(pos_world))

    return rotated_pos


def generate_graph(graph_type: str, n_nodes: int, seed: Optional[int] = None) -> nx.Graph:
    """Generate a planar graph of specified type.

    Args:
        graph_type: Type of graph to generate
        n_nodes: Number of nodes
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph with 'pos' node attribute containing (u,v) coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    if graph_type == "random_geometric":
        # Random geometric graph in unit square
        G = nx.random_geometric_graph(n_nodes, 0.15, seed=seed)
        # pos is automatically stored as node attribute

    elif graph_type == "small_world":
        # Watts-Strogatz small-world graph
        k = 6  # Each node connected to k nearest neighbors
        p = 0.3  # Rewiring probability
        G = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)
        # Generate 2D layout
        pos = nx.spring_layout(G, dim=2, seed=seed)
        # Normalize to [0, 1]
        pos_array = np.array(list(pos.values()))
        pos_min = pos_array.min(axis=0)
        pos_max = pos_array.max(axis=0)
        pos_normalized = {
            node: ((coords - pos_min) / (pos_max - pos_min)).tolist()
            for node, coords in pos.items()
        }
        nx.set_node_attributes(G, pos_normalized, 'pos')

    elif graph_type == "scale_free":
        # Barabási-Albert scale-free graph
        m = 3  # Number of edges to attach from new node
        G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        # Generate 2D layout
        pos = nx.spring_layout(G, dim=2, seed=seed)
        # Normalize to [0, 1]
        pos_array = np.array(list(pos.values()))
        pos_min = pos_array.min(axis=0)
        pos_max = pos_array.max(axis=0)
        pos_normalized = {
            node: ((coords - pos_min) / (pos_max - pos_min)).tolist()
            for node, coords in pos.items()
        }
        nx.set_node_attributes(G, pos_normalized, 'pos')

    elif graph_type == "erdos_renyi":
        # Erdős-Rényi random graph (G(n,p) model)
        p = 0.1  # Edge probability (produces ~n*p edges per node on average)
        G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
        # Generate 2D layout
        pos = nx.spring_layout(G, dim=2, seed=seed)
        # Normalize to [0, 1]
        pos_array = np.array(list(pos.values()))
        pos_min = pos_array.min(axis=0)
        pos_max = pos_array.max(axis=0)
        pos_normalized = {
            node: ((coords - pos_min) / (pos_max - pos_min)).tolist()
            for node, coords in pos.items()
        }
        nx.set_node_attributes(G, pos_normalized, 'pos')

    elif graph_type == "grid":
        # 2D grid graph
        side = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(side, side)
        # Convert node labels to integers and create positions
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        pos = {
            i: [node[0] / (side - 1), node[1] / (side - 1)]
            for node, i in mapping.items()
        }
        nx.set_node_attributes(G, pos, 'pos')

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return G


def ensure_graph_connectivity(G: nx.Graph) -> nx.Graph:
    """Ensure graph is connected by adding bridges between components.

    Uses merge2docs wisdom: find connected components and bridge them
    with minimum additional edges.

    Args:
        G: NetworkX graph (may be disconnected)

    Returns:
        Connected graph with bridging edges added
    """
    if nx.is_connected(G):
        return G

    # Split into connected components
    components = list(nx.connected_components(G))

    if len(components) == 1:
        return G

    print(f"Graph has {len(components)} disconnected components. Adding {len(components)-1} bridging edges...")

    # Create a copy to modify
    G_connected = G.copy()

    # Bridge components: connect each component to the next
    # Choose nodes with highest degree as bridge points (hubs)
    for i in range(len(components) - 1):
        comp1 = list(components[i])
        comp2 = list(components[i + 1])

        # Find highest degree nodes in each component as bridge points
        node1 = max(comp1, key=lambda n: G_connected.degree(n))
        node2 = max(comp2, key=lambda n: G_connected.degree(n))

        # Add bridging edge
        G_connected.add_edge(node1, node2)

    return G_connected


def find_shortest_path_on_graph(G: nx.Graph, source: int, target: int) -> Optional[List[int]]:
    """Find shortest path between two nodes in graph.

    Args:
        G: NetworkX graph
        source: Source node
        target: Target node

    Returns:
        List of nodes in shortest path, or None if no path exists
    """
    try:
        return nx.shortest_path(G, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def compute_treewidth_approximation(G: nx.Graph) -> int:
    """Compute treewidth approximation for a graph.

    Uses simple degree-based heuristic for speed.
    Treewidth measures how "tree-like" a graph is:
    - Low treewidth (1-3): Very tree-like, sparse
    - Medium treewidth (4-10): Moderately connected
    - High treewidth (>10): Dense, highly interconnected

    Args:
        G: NetworkX graph

    Returns:
        Approximate treewidth (upper bound)
    """
    if G.number_of_nodes() == 0:
        return 0

    # Simple approximation: min vertex cover bound
    # Treewidth ≤ vertex cover number ≤ max degree
    degrees = {node: deg for node, deg in G.degree()}
    if not degrees:
        return 0

    max_degree = max(degrees.values())

    # Better approximation: minimum degree ordering
    # This gives a tighter upper bound
    G_copy = G.copy()
    treewidth_estimate = 0

    while G_copy.number_of_nodes() > 0:
        # Find vertex with minimum degree
        degrees = {node: deg for node, deg in G_copy.degree()}
        if not degrees:
            break

        min_deg_node = min(degrees.keys(), key=lambda n: degrees[n])
        min_deg = degrees[min_deg_node]

        # Treewidth is at least the minimum degree at this step
        treewidth_estimate = max(treewidth_estimate, min_deg)

        # Remove this vertex
        G_copy.remove_node(min_deg_node)

    return int(min(treewidth_estimate, max_degree))


def find_hub_nodes(G: nx.Graph, min_degree: int = 5) -> List[int]:
    """Find hub nodes with degree >= min_degree.

    Args:
        G: NetworkX graph
        min_degree: Minimum degree for hub nodes (default: 5)

    Returns:
        List of node IDs sorted by degree (highest first)
    """
    hubs = [(node, degree) for node, degree in G.degree() if degree >= min_degree]
    # Sort by degree descending
    hubs.sort(key=lambda x: x[1], reverse=True)
    return [node for node, degree in hubs]


def rotate_graph_to_place_node_at_pole(G: nx.Graph, target_node: int, pole: str = "south") -> None:
    """Rotate graph positions so target node is at sphere pole.

    Modifies the 'pos' node attribute in place.

    Args:
        G: NetworkX graph with 'pos' attribute
        target_node: Node to place at pole
        pole: 'south' or 'north' pole
    """
    pos = nx.get_node_attributes(G, 'pos')
    target_pos = pos[target_node]

    # Target positions: south pole = (0.5, 1.0), north pole = (0.5, 0.0)
    if pole == "south":
        target_uv = (0.5, 1.0)  # Bottom of sphere (south pole in spherical coords)
    else:  # north
        target_uv = (0.5, 0.0)  # Top of sphere (north pole)

    # Calculate translation
    du = target_uv[0] - target_pos[0]
    dv = target_uv[1] - target_pos[1]

    # Apply translation with wrapping
    new_pos = {}
    for node, (u, v) in pos.items():
        new_u = (u + du) % 1.0
        new_v = max(0.0, min(1.0, v + dv))  # Clamp v to [0, 1]
        new_pos[node] = (new_u, new_v)

    # Update graph
    nx.set_node_attributes(G, new_pos, 'pos')


def map_graph_to_sphere(
    G: nx.Graph,
    config: SphereGraphConfig,
    sphere_center: List[float],
    sphere_radius: float = None
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """Map a planar graph onto a sphere surface.

    Args:
        G: NetworkX graph with 'pos' node attribute
        config: Sphere configuration
        sphere_center: Center of sphere [x, y, z]
        sphere_radius: Radius of this specific sphere (overrides config.radius)

    Returns:
        (node_positions_3d, edges)
        node_positions_3d: List of (x, y, z) coordinates on sphere
        edges: List of (u, v) edge tuples
    """
    # Get 2D positions
    pos = nx.get_node_attributes(G, 'pos')

    # Use specified radius or fall back to config
    r = sphere_radius if sphere_radius is not None else config.radius

    # Create rotation quaternion
    q_rot = create_rotation_quaternion(
        config.rotation_x_deg,
        config.rotation_y_deg,
        config.rotation_z_deg
    )

    # Map each node to sphere surface
    node_positions_3d = []
    for node in G.nodes():
        u, v = pos[node]

        # Map to spherical coordinates
        theta, phi, radius = spherical_coordinates(u, v, r)

        # Convert to Cartesian (centered at origin)
        x, y, z = spherical_to_cartesian(theta, phi, r)

        # Rotate using quaternion
        x_rot, y_rot, z_rot = rotate_point_quaternion((x, y, z), q_rot)

        # Translate to sphere center
        x_final = x_rot + sphere_center[0]
        y_final = y_rot + sphere_center[1]
        z_final = z_rot + sphere_center[2]

        node_positions_3d.append((x_final, y_final, z_final))

    # Get edges
    edges = list(G.edges())

    return node_positions_3d, edges


def create_sphere_mesh(
    center: List[float],
    radius: float,
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a sphere mesh for visualization.

    Args:
        center: Sphere center [x, y, z]
        radius: Sphere radius
        resolution: Number of points in mesh

    Returns:
        (X, Y, Z) mesh arrays for plot_surface
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    U, V = np.meshgrid(u, v)

    X = center[0] + radius * np.sin(V) * np.cos(U)
    Y = center[1] + radius * np.sin(V) * np.sin(U)
    Z = center[2] + radius * np.cos(V)

    return X, Y, Z


def visualize_two_spheres_with_graphs(
    config: SphereGraphConfig,
    G1: Optional[nx.Graph] = None,
    G2: Optional[nx.Graph] = None,
    inter_sphere_edges: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Visualize two spheres with planar graphs mapped onto their surfaces.

    Args:
        config: Configuration for visualization
        G1: Graph for first sphere (generated if None)
        G2: Graph for second sphere (generated if None)
        inter_sphere_edges: List of (node1, node2) connecting spheres
        save_path: Path to save figure
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    # Generate graphs if not provided
    if G1 is None:
        G1 = generate_graph(config.graph_type, config.n_nodes, seed=42)
    if G2 is None:
        G2 = generate_graph(config.graph_type, config.n_nodes, seed=43)

    # Ensure connectivity if requested
    if config.ensure_connected:
        G1 = ensure_graph_connectivity(G1)
        G2 = ensure_graph_connectivity(G2)

    # Find hub nodes (corpus callosum connection points)
    hubs1 = find_hub_nodes(G1, min_degree=5)
    hubs2 = find_hub_nodes(G2, min_degree=5)

    # Rotate graphs so hub nodes are at touching point
    if hubs1 and hubs2:
        hub1_node = hubs1[0]  # Highest degree hub on sphere 1
        hub2_node = hubs2[0]  # Highest degree hub on sphere 2

        print(f"Corpus callosum hubs: Sphere1 node {hub1_node} (degree={G1.degree(hub1_node)}), " +
              f"Sphere2 node {hub2_node} (degree={G2.degree(hub2_node)})")

        # Rotate G1 so hub is at south pole (touching point)
        rotate_graph_to_place_node_at_pole(G1, hub1_node, pole="south")
        # Rotate G2 so hub is at north pole (touching point)
        rotate_graph_to_place_node_at_pole(G2, hub2_node, pole="north")

        # Create inter-sphere edges ONLY through hub nodes
        inter_sphere_edges = [(hub1_node, hub2_node)]
        print(f"All inter-hemisphere communication goes through hub connection")
    else:
        print("Warning: No hub nodes with degree >= 5 found, using default inter-sphere edges")

    # Map graphs to spheres with individual radii
    pos1_3d, edges1 = map_graph_to_sphere(G1, config, config.center1, sphere_radius=config.radius1)
    pos2_3d, edges2 = map_graph_to_sphere(G2, config, config.center2, sphere_radius=config.radius2)

    # Rotate spheres so hubs are at touching point (y=0)
    if hubs1 and hubs2:
        touching_point = np.array([0.0, 0.0, 0.0])  # Origin is where spheres touch

        print(f"Rotating left sphere to place hub at touching point...")
        pos1_3d = rotate_sphere_to_place_hub_at_touching_point(
            pos1_3d, hub1_node, np.array(config.center1), touching_point
        )

        print(f"Rotating right sphere to place hub at touching point...")
        pos2_3d = rotate_sphere_to_place_hub_at_touching_point(
            pos2_3d, hub2_node, np.array(config.center2), touching_point
        )

        # Verify hubs are now at touching point
        hub1_final = np.array(pos1_3d[hub1_node])
        hub2_final = np.array(pos2_3d[hub2_node])
        print(f"Hub1 final position: [{hub1_final[0]:.3f}, {hub1_final[1]:.3f}, {hub1_final[2]:.3f}]")
        print(f"Hub2 final position: [{hub2_final[0]:.3f}, {hub2_final[1]:.3f}, {hub2_final[2]:.3f}]")
        print(f"Distance between hubs: {np.linalg.norm(hub1_final - hub2_final):.6f}")

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Draw spheres with individual radii
    X1, Y1, Z1 = create_sphere_mesh(config.center1, config.radius1)
    X2, Y2, Z2 = create_sphere_mesh(config.center2, config.radius2)

    ax.plot_surface(X1, Y1, Z1, color=config.sphere1_color, alpha=config.sphere_alpha)
    ax.plot_surface(X2, Y2, Z2, color=config.sphere2_color, alpha=config.sphere_alpha)

    # Identify hub nodes for highlighting
    hub1_idx = hubs1[0] if hubs1 and hubs2 else None
    hub2_idx = hubs2[0] if hubs1 and hubs2 else None

    # Draw graph 1 nodes and edges
    for i, (x, y, z) in enumerate(pos1_3d):
        if i == hub1_idx:
            # Corpus callosum hub - larger, bright yellow
            ax.scatter(x, y, z, c='gold', s=150, marker='*', edgecolors='black', linewidths=2, zorder=100)
        else:
            ax.scatter(x, y, z, c=config.node_color, s=30, marker='o', edgecolors='black', linewidths=0.5)

    # Draw edges as geodesic arcs on sphere surface
    for u, v in edges1:
        p1 = np.array(pos1_3d[u])
        p2 = np.array(pos1_3d[v])

        # Compute geodesic arc on sphere surface
        arc = compute_great_circle_arc(p1, p2, n_points=20)

        # Plot geodesic arc
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                c=config.edge_color, alpha=0.6, linewidth=1)

    # Draw graph 2 nodes and edges
    for i, (x, y, z) in enumerate(pos2_3d):
        if i == hub2_idx:
            # Corpus callosum hub - larger, bright yellow
            ax.scatter(x, y, z, c='gold', s=150, marker='*', edgecolors='black', linewidths=2, zorder=100)
        else:
            ax.scatter(x, y, z, c=config.node_color, s=30, marker='o', edgecolors='black', linewidths=0.5)

    # Draw edges as geodesic arcs on sphere surface
    for u, v in edges2:
        p1 = np.array(pos2_3d[u])
        p2 = np.array(pos2_3d[v])

        # Compute geodesic arc on sphere surface
        arc = compute_great_circle_arc(p1, p2, n_points=20)

        # Plot geodesic arc
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                c=config.edge_color, alpha=0.6, linewidth=1)

    # Draw inter-sphere edges (corpus callosum) if requested
    if config.show_inter_sphere_edges and inter_sphere_edges:
        for u, v in inter_sphere_edges:
            if u < len(pos1_3d) and v < len(pos2_3d):
                p1 = pos1_3d[u]
                p2 = pos2_3d[v]
                # Corpus callosum connection - thicker, bright yellow
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        c='gold', alpha=0.9, linewidth=4, linestyle='-', zorder=99)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = config.radius * 3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_title(f'Two-Sphere Graph Visualization ({config.graph_type})', fontsize=14, fontweight='bold')

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# Convenience function for MCP tool
def create_two_sphere_graph_visualization(
    graph_type: str = "random_geometric",
    n_nodes: int = 100,
    radius: float = 1.0,
    rotation_x: float = 30.0,
    rotation_y: float = 45.0,
    show_inter_edges: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create two-sphere graph visualization (MCP tool interface).

    Args:
        graph_type: Type of graph ('random_geometric', 'small_world', 'scale_free', 'grid')
        n_nodes: Number of nodes in each graph
        radius: Sphere radius
        rotation_x: Rotation around x-axis (degrees)
        rotation_y: Rotation around y-axis (degrees)
        show_inter_edges: Show edges between corresponding nodes on spheres
        save_path: Path to save visualization PNG

    Returns:
        Dict with visualization info and statistics
    """
    # Create configuration
    config = SphereGraphConfig(
        radius=radius,
        graph_type=graph_type,
        n_nodes=n_nodes,
        rotation_x_deg=rotation_x,
        rotation_y_deg=rotation_y,
        show_inter_sphere_edges=show_inter_edges
    )

    # Generate graphs
    G1 = generate_graph(graph_type, n_nodes, seed=42)
    G2 = generate_graph(graph_type, n_nodes, seed=43)

    # Create inter-sphere edges (correspondence between paired regions)
    inter_edges = None
    if show_inter_edges:
        # Connect corresponding nodes (same index on each sphere)
        inter_edges = [(i, i) for i in range(min(len(G1), len(G2)))]

    # Create visualization
    fig = visualize_two_spheres_with_graphs(
        config, G1, G2, inter_edges, save_path
    )

    # Compute statistics
    stats = {
        'graph_type': graph_type,
        'n_nodes_sphere1': G1.number_of_nodes(),
        'n_edges_sphere1': G1.number_of_edges(),
        'n_nodes_sphere2': G2.number_of_nodes(),
        'n_edges_sphere2': G2.number_of_edges(),
        'avg_degree_sphere1': float(np.mean([G1.degree(n) for n in G1.nodes()])),
        'avg_degree_sphere2': float(np.mean([G2.degree(n) for n in G2.nodes()])),
        'clustering_sphere1': float(nx.average_clustering(G1)),
        'clustering_sphere2': float(nx.average_clustering(G2)),
        'inter_sphere_edges': len(inter_edges) if inter_edges else 0,
        'visualization_saved': save_path is not None,
        'save_path': save_path
    }

    return stats
