"""Hierarchical brain model with clustered thinking hubs.

Implements multi-scale brain architecture:
1. Generate clustered synthetic data (make_blobs)
2. Build similarity graphs from Euclidean distances
3. Detect communities using cluster editing algorithms
4. Contract clusters for hierarchical visualization
5. Map to sphere surfaces with cluster regions

This captures different "thinking hubs" in brain architecture.
"""

import numpy as np
import networkx as nx
from sklearn.datasets import make_blobs
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


def generate_clustered_brain_nodes(
    n_samples: int = 200,
    n_clusters: int = 10,
    n_features: int = 3,
    cluster_std: float = 0.3,
    center_box: Tuple[float, float] = (-2.0, 2.0),
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic brain nodes organized into clusters.

    Uses make_blobs to create naturally clustered data representing
    different "thinking hubs" or functional regions in the brain.

    Args:
        n_samples: Total number of nodes
        n_clusters: Number of clusters (thinking hubs)
        n_features: Dimensionality (3 for 3D space)
        cluster_std: Standard deviation of clusters (lower = tighter)
        center_box: Bounding box for cluster centers (min, max)
        random_state: Random seed for reproducibility

    Returns:
        positions: (n_samples, n_features) array of node positions
        labels: (n_samples,) array of ground-truth cluster labels

    Example:
        >>> positions, labels = generate_clustered_brain_nodes(
        ...     n_samples=200, n_clusters=10, random_state=42
        ... )
        >>> print(f"Generated {len(positions)} nodes in {n_clusters} clusters")
    """
    positions, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        center_box=center_box,
        random_state=random_state
    )

    logger.info(
        f"Generated {n_samples} nodes in {n_clusters} clusters "
        f"(std={cluster_std:.2f})"
    )

    return positions, labels


def build_knn_graph(
    positions: np.ndarray,
    k: int = 5,
    distance_metric: str = "euclidean"
) -> nx.Graph:
    """Build k-nearest neighbor graph from node positions.

    Creates edges between each node and its k nearest neighbors
    based on Euclidean distance. This creates a graph with natural
    cluster structure reflecting spatial proximity.

    Args:
        positions: (n_samples, n_features) array of positions
        k: Number of nearest neighbors
        distance_metric: Distance metric (default: euclidean)

    Returns:
        NetworkX graph with edges between k-nearest neighbors

    Example:
        >>> G = build_knn_graph(positions, k=5)
        >>> print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    """
    n_samples = positions.shape[0]
    G = nx.Graph()

    # Add nodes with position attributes
    for i in range(n_samples):
        G.add_node(i, pos=positions[i])

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(positions, positions, metric=distance_metric)

    # For each node, connect to k nearest neighbors
    for i in range(n_samples):
        # Get indices of k+1 nearest neighbors (includes self)
        nearest_indices = np.argsort(dist_matrix[i])[:k+1]

        for j in nearest_indices:
            if i != j:
                distance = dist_matrix[i, j]
                G.add_edge(i, j, weight=distance)

    logger.info(
        f"Built k-NN graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges (k={k})"
    )

    return G


def build_threshold_graph(
    positions: np.ndarray,
    threshold: float = 1.0,
    distance_metric: str = "euclidean"
) -> nx.Graph:
    """Build graph by connecting nodes within distance threshold.

    Creates edges between all pairs of nodes within a distance threshold.
    This produces a graph where edge density reflects cluster structure.

    Args:
        positions: (n_samples, n_features) array of positions
        threshold: Distance threshold for edge creation
        distance_metric: Distance metric (default: euclidean)

    Returns:
        NetworkX graph with edges for close node pairs

    Example:
        >>> G = build_threshold_graph(positions, threshold=1.5)
    """
    n_samples = positions.shape[0]
    G = nx.Graph()

    # Add nodes with position attributes
    for i in range(n_samples):
        G.add_node(i, pos=positions[i])

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(positions, positions, metric=distance_metric)

    # Connect nodes within threshold
    edges_added = 0
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if dist_matrix[i, j] <= threshold:
                G.add_edge(i, j, weight=dist_matrix[i, j])
                edges_added += 1

    logger.info(
        f"Built threshold graph: {G.number_of_nodes()} nodes, "
        f"{edges_added} edges (threshold={threshold:.2f})"
    )

    return G


def compute_cluster_statistics(
    labels: np.ndarray,
    positions: np.ndarray
) -> Dict[int, Dict[str, any]]:
    """Compute statistics for each cluster.

    Args:
        labels: Ground-truth cluster labels
        positions: Node positions

    Returns:
        Dictionary mapping cluster_id -> statistics dict containing:
            - size: Number of nodes
            - center: Centroid position
            - radius: Average distance from centroid
            - std: Standard deviation
    """
    unique_labels = np.unique(labels)
    stats = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_positions = positions[mask]

        size = len(cluster_positions)
        center = np.mean(cluster_positions, axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(cluster_positions - center, axis=1)
        radius = np.mean(distances)
        std = np.std(cluster_positions, axis=0)

        stats[int(cluster_id)] = {
            'size': size,
            'center': center,
            'radius': float(radius),
            'std': std
        }

    return stats


# Convenience function for complete pipeline
def generate_hierarchical_brain_graph(
    n_samples: int = 200,
    n_clusters: int = 10,
    connection_type: str = "knn",
    k: int = 5,
    threshold: float = 1.0,
    random_state: Optional[int] = None
) -> Tuple[nx.Graph, np.ndarray, np.ndarray, Dict]:
    """Generate complete hierarchical brain graph in one call.

    Args:
        n_samples: Number of nodes
        n_clusters: Number of clusters
        connection_type: "knn" or "threshold"
        k: For knn - number of neighbors
        threshold: For threshold - distance threshold
        random_state: Random seed

    Returns:
        graph: NetworkX graph
        positions: Node positions
        labels: Ground-truth cluster labels
        stats: Cluster statistics

    Example:
        >>> G, pos, labels, stats = generate_hierarchical_brain_graph(
        ...     n_samples=200, n_clusters=10, connection_type="knn", k=5
        ... )
    """
    # Generate clustered data
    positions, labels = generate_clustered_brain_nodes(
        n_samples=n_samples,
        n_clusters=n_clusters,
        random_state=random_state
    )

    # Build graph
    if connection_type == "knn":
        G = build_knn_graph(positions, k=k)
    elif connection_type == "threshold":
        G = build_threshold_graph(positions, threshold=threshold)
    else:
        raise ValueError(f"Unknown connection_type: {connection_type}")

    # Compute statistics
    stats = compute_cluster_statistics(labels, positions)

    logger.info(f"Generated hierarchical brain graph with {n_clusters} clusters")
    for cluster_id, cluster_stats in stats.items():
        logger.info(
            f"  Cluster {cluster_id}: {cluster_stats['size']} nodes, "
            f"radius={cluster_stats['radius']:.2f}"
        )

    return G, positions, labels, stats


# Phase 3: Community detection using cluster editing


async def detect_communities_cluster_editing(
    G: nx.Graph,
    method: str = "cluster_editing_vs",
    k: Optional[int] = None,
    use_gpu: bool = True,
    **kwargs
) -> Optional[Dict[int, int]]:
    """Detect communities using cluster editing algorithms from merge2docs.

    Uses advanced FPT algorithms (cluster editing, cluster editing with vertex selection)
    to find optimal or near-optimal community structure.

    Args:
        G: NetworkX graph
        method: Algorithm to use:
            - "cluster_editing": Standard cluster editing
            - "cluster_editing_vs": Cluster editing with vertex selection
        k: Parameter for cluster editing (max edits)
            If None, will be estimated as 10% of edges
        use_gpu: Use GPU-accelerated version if available
        **kwargs: Additional algorithm parameters

    Returns:
        Dictionary mapping node_id -> cluster_id, or None if failed

    Example:
        >>> communities = await detect_communities_cluster_editing(G, k=20)
        >>> print(f"Found {len(set(communities.values()))} communities")
    """
    try:
        from backend.integration.merge2docs_bridge import call_algorithm_service
    except ImportError:
        logger.error("Cannot import merge2docs bridge")
        return None

    # Estimate k if not provided
    if k is None:
        k = max(1, G.number_of_edges() // 10)
        logger.info(f"Estimated k={k} ({G.number_of_edges()} edges)")

    # Convert graph to format expected by merge2docs
    graph_data = {
        'nodes': list(G.nodes()),
        'edges': list(G.edges())
    }

    # Select algorithm
    if use_gpu:
        algorithm_name = f"{method}_gpu"
    else:
        algorithm_name = method

    logger.info(f"Running {algorithm_name} with k={k}...")

    # Call merge2docs algorithm service
    try:
        result = await call_algorithm_service(
            algorithm_name=algorithm_name,
            graph_data=graph_data,
            k=k,
            **kwargs
        )

        if result is None:
            logger.error(f"Algorithm {algorithm_name} returned None")
            return None

        # Extract cluster assignments from result
        # Result is an AlgorithmResult object with attributes
        logger.info(f"Algorithm result type: {type(result)}")

        # Try to access as object attributes first
        if hasattr(result, 'clusters'):
            return _parse_cluster_editing_solution(G, result.clusters)
        elif hasattr(result, 'solution'):
            return _parse_cluster_editing_solution(G, result.solution)
        # Fallback to dict access
        elif isinstance(result, dict):
            if 'clusters' in result:
                return _parse_cluster_editing_solution(G, result['clusters'])
            elif 'solution' in result:
                return _parse_cluster_editing_solution(G, result['solution'])

        # Try to extract from AlgorithmResult result_data attribute
        if hasattr(result, 'result_data'):
            logger.info(f"Extracting from result.result_data: {type(result.result_data)}")
            result_data = result.result_data

            # result_data could be the solution directly, or a dict containing it
            if isinstance(result_data, dict):
                if 'clusters' in result_data:
                    return _parse_cluster_editing_solution(G, result_data['clusters'])
                elif 'solution' in result_data:
                    return _parse_cluster_editing_solution(G, result_data['solution'])
            else:
                # result_data is the solution itself (list of clusters)
                return _parse_cluster_editing_solution(G, result_data)

        logger.error(f"Unexpected result format: {type(result)}, attrs={dir(result) if hasattr(result, '__dict__') else 'N/A'}")
        return None

    except Exception as e:
        logger.error(f"Failed to run {algorithm_name}: {e}")
        return None


def _parse_cluster_editing_solution(
    G: nx.Graph,
    solution: any
) -> Dict[int, int]:
    """Parse cluster editing solution into node -> cluster mapping.

    Args:
        G: Original graph
        solution: Solution from cluster editing algorithm

    Returns:
        Dictionary mapping node_id -> cluster_id
    """
    # If solution is already a dict, verify it has all nodes
    if isinstance(solution, dict):
        # Check if all nodes are present
        missing_nodes = set(G.nodes()) - set(solution.keys())
        if missing_nodes:
            logger.warning(f"Solution dict missing {len(missing_nodes)} nodes, using connected components")
            return _fallback_to_connected_components(G)
        return solution

    # If solution is a list of clusters
    if isinstance(solution, list):
        node_to_cluster = {}
        for cluster_id, cluster_nodes in enumerate(solution):
            if isinstance(cluster_nodes, (list, set)):
                for node in cluster_nodes:
                    node_to_cluster[node] = cluster_id
            else:
                # Single node cluster
                node_to_cluster[cluster_nodes] = cluster_id

        # Check if all nodes are assigned
        missing_nodes = set(G.nodes()) - set(node_to_cluster.keys())
        if missing_nodes:
            logger.warning(f"Solution list missing {len(missing_nodes)} nodes, using connected components")
            return _fallback_to_connected_components(G)

        return node_to_cluster

    # Fallback: use connected components
    logger.warning(f"Unknown solution format: {type(solution)}, using connected components")
    return _fallback_to_connected_components(G)


def _fallback_to_connected_components(G: nx.Graph) -> Dict[int, int]:
    """Fallback to connected components for clustering.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary mapping node_id -> cluster_id
    """
    communities = nx.connected_components(G)
    node_to_cluster = {}
    for cluster_id, component in enumerate(communities):
        for node in component:
            node_to_cluster[node] = cluster_id
    return node_to_cluster


def detect_communities_simple(
    G: nx.Graph,
    method: str = "louvain"
) -> Dict[int, int]:
    """Detect communities using simple NetworkX algorithms.

    Fallback method when merge2docs is not available.

    Args:
        G: NetworkX graph
        method: Algorithm to use:
            - "louvain": Louvain community detection
            - "connected_components": Simple connected components

    Returns:
        Dictionary mapping node_id -> cluster_id
    """
    if method == "louvain":
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            return partition
        except ImportError:
            logger.warning("python-louvain not available, using connected components")
            method = "connected_components"

    if method == "connected_components":
        communities = nx.connected_components(G)
        node_to_cluster = {}
        for cluster_id, component in enumerate(communities):
            for node in component:
                node_to_cluster[node] = cluster_id
        return node_to_cluster

    raise ValueError(f"Unknown method: {method}")


async def detect_communities_auto(
    G: nx.Graph,
    target_clusters: Optional[int] = None,
    method: str = "cluster_editing_vs",
    use_gpu: bool = True
) -> Optional[Dict[int, int]]:
    """Automatically detect communities with adaptive parameter selection.

    Tries multiple k values and selects the best clustering based on
    modularity and optionally target cluster count.

    Args:
        G: NetworkX graph
        target_clusters: Optional target number of clusters (from ground truth)
        method: Clustering algorithm
        use_gpu: Use GPU acceleration

    Returns:
        Dictionary mapping node_id -> cluster_id

    Example:
        >>> communities = await detect_communities_auto(G, target_clusters=10)
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    logger.info(f"Auto-tuning clustering for {n_nodes} nodes, {n_edges} edges")

    # Try multiple k values (budget for edits)
    k_candidates = [
        n_edges // 20,   # 5% of edges
        n_edges // 10,   # 10%
        n_edges // 5,    # 20%
        n_edges // 2,    # 50%
        n_edges,         # 100%
        n_edges * 2      # 200%
    ]

    best_communities = None
    best_score = -float('inf')
    best_k = None

    for k in k_candidates:
        if k < 1:
            continue

        logger.info(f"  Trying k={k}...")

        communities = await detect_communities_cluster_editing(
            G, method=method, k=k, use_gpu=use_gpu
        )

        if communities is None:
            continue

        quality = evaluate_clustering_quality(G, communities)
        n_clusters = quality['n_clusters']
        modularity = quality['modularity']

        # Scoring function
        score = modularity

        # Bonus for matching target cluster count
        if target_clusters is not None:
            cluster_diff = abs(n_clusters - target_clusters)
            # Penalize deviation from target
            cluster_penalty = cluster_diff / target_clusters
            score = modularity * (1.0 - 0.5 * cluster_penalty)

        logger.info(f"    Found {n_clusters} clusters, modularity={modularity:.3f}, score={score:.3f}")

        if score > best_score:
            best_score = score
            best_communities = communities
            best_k = k

    if best_communities is not None:
        quality = evaluate_clustering_quality(G, best_communities)
        logger.info(f"✅ Auto-selected k={best_k}: {quality['n_clusters']} clusters, modularity={quality['modularity']:.3f}")

    return best_communities


def evaluate_clustering_quality(
    G: nx.Graph,
    communities: Dict[int, int]
) -> Dict[str, float]:
    """Evaluate quality of community detection.

    Args:
        G: NetworkX graph
        communities: Node -> cluster mapping

    Returns:
        Dictionary with quality metrics:
            - modularity: Graph modularity
            - n_clusters: Number of clusters
            - avg_cluster_size: Average cluster size
            - internal_edges: Fraction of edges within clusters
    """
    n_clusters = len(set(communities.values()))
    cluster_sizes = {}
    for node, cluster_id in communities.items():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    avg_cluster_size = np.mean(list(cluster_sizes.values()))

    # Count internal vs external edges
    internal_edges = 0
    external_edges = 0
    for u, v in G.edges():
        if communities[u] == communities[v]:
            internal_edges += 1
        else:
            external_edges += 1

    total_edges = internal_edges + external_edges
    internal_fraction = internal_edges / total_edges if total_edges > 0 else 0

    # Compute modularity
    # Create partition format for networkx
    partition_sets = {}
    for node, cluster_id in communities.items():
        if cluster_id not in partition_sets:
            partition_sets[cluster_id] = set()
        partition_sets[cluster_id].add(node)

    partition = list(partition_sets.values())
    modularity = nx.community.modularity(G, partition)

    return {
        'modularity': modularity,
        'n_clusters': n_clusters,
        'avg_cluster_size': avg_cluster_size,
        'internal_edges': internal_fraction
    }


# Phase 4: Cluster contraction


def contract_clusters_to_supernodes(
    G: nx.Graph,
    communities: Dict[int, int],
    positions: np.ndarray
) -> Tuple[nx.Graph, Dict[int, np.ndarray], Dict[int, List[int]]]:
    """Contract clusters into super-nodes for hierarchical visualization.

    Each community becomes a single node in the contracted graph.
    Edges between communities are preserved.

    Args:
        G: Original graph
        communities: Node -> cluster mapping
        positions: Original node positions (n_nodes, 3)

    Returns:
        contracted_graph: Graph where each node is a cluster
        cluster_positions: Super-node positions (cluster centroids)
        cluster_members: Mapping cluster_id -> list of original node IDs

    Example:
        >>> G_contracted, pos, members = contract_clusters_to_supernodes(G, communities, positions)
        >>> print(f"Contracted from {G.number_of_nodes()} to {G_contracted.number_of_nodes()} nodes")
    """
    # Create mapping of cluster_id -> member nodes
    cluster_members = {}
    for node, cluster_id in communities.items():
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(node)

    # Create contracted graph
    G_contracted = nx.Graph()

    # Add super-nodes with cluster statistics
    cluster_positions = {}
    for cluster_id, members in cluster_members.items():
        # Compute cluster centroid
        member_positions = [positions[node] for node in members]
        centroid = np.mean(member_positions, axis=0)
        cluster_positions[cluster_id] = centroid

        # Add super-node with attributes
        G_contracted.add_node(
            cluster_id,
            size=len(members),
            members=members,
            pos=centroid
        )

    # Add edges between super-nodes
    # Edge exists if any nodes in the clusters are connected
    edge_weights = {}  # (cluster_i, cluster_j) -> edge count
    for u, v in G.edges():
        cluster_u = communities[u]
        cluster_v = communities[v]

        if cluster_u != cluster_v:
            edge_key = tuple(sorted([cluster_u, cluster_v]))
            edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1

    # Add weighted edges to contracted graph
    for (cluster_u, cluster_v), weight in edge_weights.items():
        G_contracted.add_edge(
            cluster_u, cluster_v,
            weight=weight,
            edge_count=weight
        )

    logger.info(
        f"Contracted graph: {G.number_of_nodes()} -> {G_contracted.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} -> {G_contracted.number_of_edges()} edges"
    )

    return G_contracted, cluster_positions, cluster_members


def compute_cluster_regions(
    positions: np.ndarray,
    communities: Dict[int, int],
    expansion_factor: float = 1.3
) -> Dict[int, Dict[str, any]]:
    """Compute bounding regions for each cluster for visualization.

    Creates spherical regions around each cluster for transparent overlay.

    Args:
        positions: Node positions (n_nodes, 3)
        communities: Node -> cluster mapping
        expansion_factor: How much to expand region beyond nodes (default 1.3)

    Returns:
        Dictionary mapping cluster_id -> region dict with:
            - center: Region centroid
            - radius: Region radius
            - members: List of node IDs
            - color: Suggested color (RGB tuple)

    Example:
        >>> regions = compute_cluster_regions(positions, communities)
        >>> for cluster_id, region in regions.items():
        ...     print(f"Cluster {cluster_id}: {len(region['members'])} nodes, radius={region['radius']:.2f}")
    """
    cluster_members = {}
    for node, cluster_id in communities.items():
        if cluster_id not in cluster_members:
            cluster_members[cluster_id] = []
        cluster_members[cluster_id].append(node)

    # Generate distinct colors for clusters
    n_clusters = len(cluster_members)
    colors = _generate_distinct_colors(n_clusters)

    regions = {}
    for idx, (cluster_id, members) in enumerate(cluster_members.items()):
        # Get member positions
        member_positions = np.array([positions[node] for node in members])

        # Compute centroid
        centroid = np.mean(member_positions, axis=0)

        # Compute radius (max distance from centroid + expansion)
        distances = np.linalg.norm(member_positions - centroid, axis=1)
        radius = np.max(distances) * expansion_factor

        regions[cluster_id] = {
            'center': centroid,
            'radius': float(radius),
            'members': members,
            'color': colors[idx]
        }

    return regions


def _generate_distinct_colors(n_colors: int) -> List[Tuple[float, float, float]]:
    """Generate visually distinct colors for cluster visualization.

    Args:
        n_colors: Number of colors to generate

    Returns:
        List of RGB tuples (values in [0, 1])
    """
    import colorsys

    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        # Use high saturation and medium lightness for distinct colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)

    return colors
