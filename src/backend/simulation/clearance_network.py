"""
Clearance Network Analyzer - Disc Dimension and Network Topology Analysis

Links brain network topology (disc dimension) to glymphatic clearance efficiency.

Key insight from PH-7: Low disc dimension networks (≤3) provide more efficient
waste clearance routing due to their planar/near-planar structure.

Based on Paul et al. 2023 obstruction theory:
- pobs(tw) = {K₅, K₃,₃} characterizes treewidth-class parameters
- Planar networks have disc dimension 2
- Non-planar networks have disc dimension ≥3
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Clearance Efficiency Model
# =============================================================================

def disc_dimension_clearance_model(
    disc_dimension: float,
    disc_optimal: float = 2.5,
    eta_max: float = 0.95,
    beta: float = 0.3
) -> float:
    """Compute clearance efficiency from disc dimension.

    Model: η = η_max × exp(-β × (disc - disc_opt)²)

    Hypothesis: Networks with disc dimension close to optimal (2-3) provide
    the most efficient waste clearance due to:
    - Low disc dimension (planar) = efficient routing without crossings
    - Too low (tree-like) = limited connectivity
    - Too high (hyperbolic) = inefficient tangled paths

    Args:
        disc_dimension: Network disc dimension (1-∞, typically 2-5 for brain)
        disc_optimal: Optimal disc dimension for clearance (default: 2.5)
        eta_max: Maximum efficiency (0-1)
        beta: Width parameter (lower = broader optimum)

    Returns:
        Clearance efficiency (0-1)
    """
    return eta_max * np.exp(-beta * (disc_dimension - disc_optimal)**2)


# =============================================================================
# Network Analysis
# =============================================================================

@dataclass
class NetworkMetrics:
    """Metrics for a brain connectivity network."""
    num_nodes: int
    num_edges: int
    average_degree: float
    clustering_coefficient: float
    is_planar: bool
    has_k5: bool
    has_k33: bool
    disc_dimension_estimate: float
    treewidth_estimate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'average_degree': self.average_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'is_planar': self.is_planar,
            'has_k5_obstruction': self.has_k5,
            'has_k33_obstruction': self.has_k33,
            'disc_dimension_estimate': self.disc_dimension_estimate,
            'treewidth_estimate': self.treewidth_estimate
        }


class ClearanceNetworkAnalyzer:
    """Analyzer for brain network topology and clearance efficiency.

    Connects network structure (disc dimension, planarity) to
    predicted waste clearance efficiency.
    """

    def __init__(self, optimal_disc_dim: float = 2.5, beta: float = 0.3):
        """Initialize analyzer.

        Args:
            optimal_disc_dim: Optimal disc dimension for clearance
            beta: Sensitivity parameter for efficiency model
        """
        self.optimal_disc_dim = optimal_disc_dim
        self.beta = beta

    def analyze_network(self,
                       adjacency_matrix: np.ndarray,
                       node_positions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze network topology for clearance prediction.

        Args:
            adjacency_matrix: N×N symmetric adjacency matrix (0/1 or weighted)
            node_positions: Optional N×3 array of node positions

        Returns:
            Analysis results including disc dimension and clearance prediction
        """
        n = adjacency_matrix.shape[0]

        # Basic metrics
        adj_binary = (adjacency_matrix > 0).astype(int)
        np.fill_diagonal(adj_binary, 0)

        num_edges = np.sum(adj_binary) // 2
        degrees = np.sum(adj_binary, axis=1)
        avg_degree = np.mean(degrees)

        # Clustering coefficient (local transitivity)
        clustering = self._compute_clustering(adj_binary)

        # Planarity check via obstruction detection
        has_k5, has_k33, obstruction_strength = self._check_obstructions(adj_binary)
        is_planar = not (has_k5 or has_k33)

        # Estimate disc dimension
        disc_dim = self._estimate_disc_dimension(
            adj_binary, is_planar, has_k5, has_k33, obstruction_strength
        )

        # Compute clearance efficiency
        efficiency = disc_dimension_clearance_model(
            disc_dim,
            self.optimal_disc_dim,
            beta=self.beta
        )

        metrics = NetworkMetrics(
            num_nodes=n,
            num_edges=num_edges,
            average_degree=float(avg_degree),
            clustering_coefficient=float(clustering),
            is_planar=is_planar,
            has_k5=has_k5,
            has_k33=has_k33,
            disc_dimension_estimate=float(disc_dim)
        )

        return {
            'network_metrics': metrics.to_dict(),
            'clearance_prediction': {
                'disc_dimension': float(disc_dim),
                'optimal_disc_dimension': self.optimal_disc_dim,
                'clearance_efficiency': float(efficiency),
                'efficiency_interpretation': self._interpret_efficiency(efficiency),
                'planarity_status': 'planar' if is_planar else 'non-planar'
            },
            'obstruction_analysis': {
                'has_k5': has_k5,
                'has_k33': has_k33,
                'obstruction_strength': float(obstruction_strength),
                'theoretical_basis': 'Paul et al. 2023: pobs(tw) = {K₅, K₃,₃}'
            }
        }

    def compare_networks(self,
                        networks: List[Tuple[str, np.ndarray]]) -> Dict[str, Any]:
        """Compare multiple networks for clearance efficiency.

        Args:
            networks: List of (name, adjacency_matrix) tuples

        Returns:
            Comparison results sorted by efficiency
        """
        results = []
        for name, adj in networks:
            analysis = self.analyze_network(adj)
            results.append({
                'name': name,
                'disc_dimension': analysis['clearance_prediction']['disc_dimension'],
                'clearance_efficiency': analysis['clearance_prediction']['clearance_efficiency'],
                'is_planar': analysis['network_metrics']['is_planar'],
                'num_nodes': analysis['network_metrics']['num_nodes'],
                'num_edges': analysis['network_metrics']['num_edges']
            })

        # Sort by efficiency
        results.sort(key=lambda x: x['clearance_efficiency'], reverse=True)

        return {
            'networks': results,
            'best_network': results[0]['name'] if results else None,
            'efficiency_range': {
                'min': min(r['clearance_efficiency'] for r in results) if results else 0,
                'max': max(r['clearance_efficiency'] for r in results) if results else 0
            },
            'optimal_disc_dimension': self.optimal_disc_dim
        }

    def _compute_clustering(self, adj: np.ndarray) -> float:
        """Compute global clustering coefficient."""
        n = adj.shape[0]
        if n < 3:
            return 0.0

        triangles = 0
        triplets = 0

        for i in range(n):
            neighbors = np.where(adj[i] > 0)[0]
            k = len(neighbors)
            if k >= 2:
                # Count triangles through node i
                for j_idx, j in enumerate(neighbors):
                    for k_node in neighbors[j_idx+1:]:
                        if adj[j, k_node] > 0:
                            triangles += 1
                triplets += k * (k - 1) // 2

        return triangles / triplets if triplets > 0 else 0.0

    def _check_obstructions(self, adj: np.ndarray) -> Tuple[bool, bool, float]:
        """Check for K₅ and K₃,₃ obstructions using PAC-inspired heuristic.

        Returns:
            (has_k5, has_k33, obstruction_strength)
        """
        n = adj.shape[0]
        degrees = np.sum(adj, axis=1)

        # K₅ detection: look for 5-clique or high-degree dense subgraph
        has_k5 = False
        k5_score = 0.0

        # Check if average degree high enough for K₅
        if np.mean(degrees) >= 4:
            # Look for dense 5-node subgraphs
            high_degree_nodes = np.where(degrees >= 4)[0]
            if len(high_degree_nodes) >= 5:
                # Sample and check density
                for _ in range(min(100, len(high_degree_nodes))):
                    sample = np.random.choice(high_degree_nodes, min(5, len(high_degree_nodes)), replace=False)
                    if len(sample) == 5:
                        subgraph = adj[np.ix_(sample, sample)]
                        density = np.sum(subgraph) / 20  # K₅ has 10 edges, counted twice
                        if density > 0.8:
                            has_k5 = True
                            k5_score = max(k5_score, density)

        # K₃,₃ detection: look for bipartite-like structure
        has_k33 = False
        k33_score = 0.0

        if n >= 6:
            # Simple heuristic: check for nodes that share many common neighbors
            for i in range(min(n, 50)):
                neighbors_i = set(np.where(adj[i] > 0)[0])
                if len(neighbors_i) >= 3:
                    for j in range(i + 1, min(n, 50)):
                        neighbors_j = set(np.where(adj[j] > 0)[0])
                        common = neighbors_i & neighbors_j
                        if len(common) >= 3:
                            # Potential K₃,₃ structure
                            k33_score = max(k33_score, len(common) / max(len(neighbors_i), len(neighbors_j)))
                            if k33_score > 0.5:
                                has_k33 = True

        obstruction_strength = max(k5_score, k33_score)
        return has_k5, has_k33, obstruction_strength

    def _estimate_disc_dimension(self,
                                adj: np.ndarray,
                                is_planar: bool,
                                has_k5: bool,
                                has_k33: bool,
                                obstruction_strength: float) -> float:
        """Estimate disc dimension from network structure.

        Uses heuristics based on:
        - Planarity (disc dim = 2 for planar graphs)
        - Obstruction presence and strength
        - Network density and structure
        """
        if is_planar:
            # Planar graphs have disc dimension 2
            # But sparse trees have disc dimension ~1
            n = adj.shape[0]
            m = np.sum(adj) // 2
            if n > 0:
                density = 2 * m / (n * (n - 1)) if n > 1 else 0
                # Tree-like (sparse) → disc_dim closer to 1
                # Dense planar → disc_dim = 2
                return 1.5 + 0.5 * min(density * 10, 1)
            return 2.0

        # Non-planar: disc dimension ≥ 3
        base_dim = 3.0

        # Increase based on obstruction strength
        if has_k5 and has_k33:
            base_dim = 3.5
        elif has_k5:
            base_dim = 3.2
        elif has_k33:
            base_dim = 3.1

        # Further increase for very dense graphs
        n = adj.shape[0]
        m = np.sum(adj) // 2
        if n > 1:
            density = 2 * m / (n * (n - 1))
            base_dim += density * 0.5

        return min(base_dim + obstruction_strength * 0.5, 5.0)

    def _interpret_efficiency(self, efficiency: float) -> str:
        """Interpret clearance efficiency value."""
        if efficiency >= 0.9:
            return "Excellent - optimal network topology for waste clearance"
        elif efficiency >= 0.7:
            return "Good - efficient clearance expected"
        elif efficiency >= 0.5:
            return "Moderate - some routing inefficiency"
        elif efficiency >= 0.3:
            return "Reduced - significant topological barriers"
        else:
            return "Poor - highly inefficient network structure"


# =============================================================================
# Utility Functions
# =============================================================================

def create_test_networks() -> Dict[str, np.ndarray]:
    """Create test networks with different topologies.

    Returns:
        Dictionary of network name → adjacency matrix
    """
    networks = {}

    # 1. Tree (disc dimension ~1)
    n = 15
    tree = np.zeros((n, n))
    for i in range(1, n):
        parent = (i - 1) // 2
        tree[i, parent] = tree[parent, i] = 1
    networks['tree'] = tree

    # 2. Grid (planar, disc dimension = 2)
    grid_size = 4
    n = grid_size ** 2
    grid = np.zeros((n, n))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if j + 1 < grid_size:
                grid[idx, idx + 1] = grid[idx + 1, idx] = 1
            if i + 1 < grid_size:
                grid[idx, idx + grid_size] = grid[idx + grid_size, idx] = 1
    networks['grid_4x4'] = grid

    # 3. K₅ (non-planar, disc dimension > 2)
    k5 = np.ones((5, 5)) - np.eye(5)
    networks['K5'] = k5

    # 4. Random planar-ish (low density)
    n = 20
    random_sparse = np.zeros((n, n))
    for i in range(n - 1):
        random_sparse[i, i + 1] = random_sparse[i + 1, i] = 1
    # Add some random edges but keep sparse
    for _ in range(n // 2):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            random_sparse[i, j] = random_sparse[j, i] = 1
    networks['random_sparse'] = random_sparse

    # 5. Dense random (likely non-planar)
    n = 15
    random_dense = np.random.rand(n, n)
    random_dense = (random_dense + random_dense.T) / 2
    random_dense = (random_dense > 0.6).astype(float)
    np.fill_diagonal(random_dense, 0)
    networks['random_dense'] = random_dense

    return networks
