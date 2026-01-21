"""Whole-brain network analysis for functional connectivity.

Orchestrates complete pipeline from fMRI signals to network metrics:
1. Signal preprocessing and alignment
2. Pairwise connectivity (distance correlation, PLV, FFT correlation)
3. Network construction and thresholding
4. Graph-theoretic analysis (centrality, modularity, small-world metrics)
5. Sphere-based visualization with geodesic edges
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from .mri_signal_processing import (
    compute_pairwise_correlation_fft,
    compute_distance_correlation,
    compute_phase_locking_value
)
from .network_analysis import (
    connectivity_matrix_to_graph,
    map_nodes_to_sphere,
    identify_interhemispheric_edges
)


class WholeBrainNetworkAnalyzer:
    """Orchestrator for whole-brain functional connectivity analysis.

    Combines MRI signal processing, network construction, and graph metrics
    into a unified pipeline for brain connectivity studies.
    """

    def __init__(
        self,
        region_labels: List[str],
        sampling_rate_hz: float = 0.5,  # Typical fMRI TR = 2s
        connectivity_method: str = "distance_correlation"
    ):
        """Initialize whole-brain network analyzer.

        Args:
            region_labels: List of brain region names (e.g., ["V1_L", "V1_R", ...])
            sampling_rate_hz: fMRI sampling rate in Hz (1/TR)
            connectivity_method: Method for connectivity estimation
                Options: "distance_correlation", "fft_correlation", "phase_locking"
        """
        self.region_labels = region_labels
        self.n_regions = len(region_labels)
        self.sampling_rate_hz = sampling_rate_hz
        self.connectivity_method = connectivity_method

    async def compute_connectivity_matrix(
        self,
        time_series: List[NDArray]
    ) -> NDArray:
        """Compute full pairwise connectivity matrix from time-series.

        Args:
            time_series: List of time-series arrays, one per brain region
                        Each array shape: (n_timepoints,) or (n_voxels, n_timepoints)

        Returns:
            NxN connectivity matrix (symmetric, values depend on method)
        """
        n_regions = len(time_series)
        connectivity_matrix = np.zeros((n_regions, n_regions))

        if self.connectivity_method == "fft_correlation":
            # Frequency-domain correlation (fast, vectorized)
            connectivity_matrix = await compute_pairwise_correlation_fft(time_series)

        elif self.connectivity_method == "distance_correlation":
            # Distance correlation (multivariate, detects non-linear dependencies)
            for i in range(n_regions):
                connectivity_matrix[i, i] = 1.0  # Self-correlation
                for j in range(i + 1, n_regions):
                    dCor = await compute_distance_correlation(
                        time_series[i],
                        time_series[j]
                    )
                    connectivity_matrix[i, j] = dCor
                    connectivity_matrix[j, i] = dCor

        elif self.connectivity_method == "phase_locking":
            # Phase-locking value (PLV) - phase synchronization
            for i in range(n_regions):
                connectivity_matrix[i, i] = 1.0
                for j in range(i + 1, n_regions):
                    # Ensure 1D signals for PLV
                    sig_i = time_series[i].flatten() if time_series[i].ndim > 1 else time_series[i]
                    sig_j = time_series[j].flatten() if time_series[j].ndim > 1 else time_series[j]
                    plv = await compute_phase_locking_value(sig_i, sig_j)
                    connectivity_matrix[i, j] = plv
                    connectivity_matrix[j, i] = plv

        else:
            raise ValueError(f"Unknown connectivity method: {self.connectivity_method}")

        return connectivity_matrix

    async def construct_network(
        self,
        connectivity_matrix: NDArray,
        threshold: Optional[float] = None,
        density: Optional[float] = None
    ) -> nx.Graph:
        """Construct brain network graph from connectivity matrix.

        Args:
            connectivity_matrix: NxN symmetric connectivity matrix
            threshold: Minimum edge weight to include (absolute threshold)
            density: Target network density (0-1). If provided, computes
                    adaptive threshold to achieve specified edge density

        Returns:
            NetworkX graph with weighted edges
        """
        # Determine threshold
        if density is not None:
            # Adaptive thresholding for target density
            # Density = 2*E / (N*(N-1)) for undirected graph
            n_nodes = connectivity_matrix.shape[0]
            n_possible_edges = n_nodes * (n_nodes - 1) // 2
            n_target_edges = int(density * n_possible_edges)

            # Get upper triangular values (excluding diagonal)
            upper_tri = connectivity_matrix[np.triu_indices(n_nodes, k=1)]
            sorted_weights = np.sort(upper_tri)[::-1]  # Descending
            threshold = float(sorted_weights[n_target_edges - 1]) if n_target_edges > 0 else 0.0
        elif threshold is None:
            # Default: mean + 1 std dev
            threshold = float(np.mean(connectivity_matrix) + np.std(connectivity_matrix))

        # Build graph
        graph = await connectivity_matrix_to_graph(
            connectivity_matrix,
            threshold=threshold,
            node_labels=self.region_labels
        )

        return graph

    async def compute_network_metrics(
        self,
        graph: nx.Graph
    ) -> Dict[str, Any]:
        """Compute comprehensive graph-theoretic network metrics.

        Args:
            graph: NetworkX graph of brain connectivity

        Returns:
            Dict with network metrics:
                - n_nodes, n_edges: Basic counts
                - density: Edge density
                - clustering_coefficient: Global clustering coefficient
                - avg_path_length: Average shortest path length
                - small_world_sigma: Small-worldness coefficient
                - modularity: Modularity (Louvain community detection)
                - degree_centrality: Dict of degree centrality per node
                - betweenness_centrality: Dict of betweenness centrality per node
                - eigenvector_centrality: Dict of eigenvector centrality per node
                - hub_nodes: Top 10% nodes by degree
                - communities: Community assignments (dict: node -> community_id)
        """
        def _compute():
            # Basic metrics
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            density = nx.density(graph)

            # Clustering and path length
            clustering = nx.average_clustering(graph, weight='weight')

            # Average path length (only for connected graphs)
            if nx.is_connected(graph):
                avg_path_length = nx.average_shortest_path_length(graph, weight='weight')
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')

            # Small-worldness: σ = (C/C_rand) / (L/L_rand)
            # Compare to random graph with same degree distribution
            # Approximation: C_rand ≈ <k>/N, L_rand ≈ ln(N)/ln(<k>)
            avg_degree = 2 * n_edges / n_nodes
            C_rand = avg_degree / n_nodes if n_nodes > 0 else 1.0
            L_rand = np.log(n_nodes) / np.log(avg_degree) if avg_degree > 1 else 1.0

            # Avoid division by zero
            if C_rand > 0 and L_rand > 0:
                small_world_sigma = (clustering / C_rand) / (avg_path_length / L_rand)
            else:
                small_world_sigma = 1.0

            # Centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                # Fallback to degree centrality if eigenvector fails
                eigenvector_centrality = degree_centrality

            # Hub identification (top 10% by degree)
            sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            n_hubs = max(1, n_nodes // 10)
            hub_nodes = [node for node, _ in sorted_nodes[:n_hubs]]

            # Community detection (Louvain algorithm)
            communities_dict = nx.community.louvain_communities(graph, weight='weight', seed=42)
            # Convert to node -> community_id mapping
            node_to_community = {}
            for comm_id, community in enumerate(communities_dict):
                for node in community:
                    node_to_community[node] = comm_id

            # Modularity (handle degenerate cases)
            try:
                modularity = nx.community.modularity(graph, communities_dict, weight='weight')
            except ZeroDivisionError:
                # Can occur with very sparse or disconnected graphs
                modularity = 0.0

            return {
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": float(density),
                "clustering_coefficient": float(clustering),
                "avg_path_length": float(avg_path_length),
                "small_world_sigma": float(small_world_sigma),
                "modularity": float(modularity),
                "degree_centrality": {k: float(v) for k, v in degree_centrality.items()},
                "betweenness_centrality": {k: float(v) for k, v in betweenness_centrality.items()},
                "eigenvector_centrality": {k: float(v) for k, v in eigenvector_centrality.items()},
                "hub_nodes": hub_nodes,
                "communities": node_to_community,
                "n_communities": len(communities_dict)
            }

        return await asyncio.to_thread(_compute)

    async def analyze_interhemispheric_connectivity(
        self,
        graph: nx.Graph,
        node_positions: Dict[str, NDArray]
    ) -> Dict[str, Any]:
        """Analyze interhemispheric (corpus callosum) connectivity patterns.

        Args:
            graph: Brain network graph
            node_positions: Dict mapping node labels to [x,y,z] coordinates

        Returns:
            Dict with:
                - interhemispheric_edges: List of edges crossing hemispheres
                - n_interhemispheric: Count of interhemispheric connections
                - fraction_interhemispheric: Fraction of all edges
                - avg_interhemispheric_weight: Mean weight of crossing edges
                - avg_intrahemispheric_weight: Mean weight of within-hemisphere edges
        """
        # Identify interhemispheric edges (crossing y=0 plane)
        interhemispheric = await identify_interhemispheric_edges(
            graph,
            node_positions,
            hemisphere_boundary=0.0,
            axis=1  # y-axis separates hemispheres
        )

        n_inter = len(interhemispheric)
        n_total = graph.number_of_edges()
        fraction_inter = n_inter / n_total if n_total > 0 else 0.0

        # Compute average weights
        inter_weights = [graph[u][v]['weight'] for u, v in interhemispheric]
        avg_inter_weight = np.mean(inter_weights) if inter_weights else 0.0

        # Intrahemispheric edges (complement)
        all_edges = set(graph.edges())
        intrahemispheric = all_edges - set(interhemispheric)
        intra_weights = [graph[u][v]['weight'] for u, v in intrahemispheric]
        avg_intra_weight = np.mean(intra_weights) if intra_weights else 0.0

        return {
            "interhemispheric_edges": [(str(u), str(v)) for u, v in interhemispheric],
            "n_interhemispheric": n_inter,
            "fraction_interhemispheric": float(fraction_inter),
            "avg_interhemispheric_weight": float(avg_inter_weight),
            "avg_intrahemispheric_weight": float(avg_intra_weight),
            "interhemispheric_ratio": float(avg_inter_weight / avg_intra_weight) if avg_intra_weight > 0 else 0.0
        }

    async def run_complete_analysis(
        self,
        time_series: List[NDArray],
        node_locations: Optional[Dict[str, Dict[str, float]]] = None,
        network_density: float = 0.15
    ) -> Dict[str, Any]:
        """Run complete whole-brain network analysis pipeline.

        Args:
            time_series: List of fMRI time-series (one per region)
            node_locations: Optional spherical coordinates for 3D visualization
                           {"region_name": {"theta": θ, "phi": φ}, ...}
            network_density: Target network edge density (0-1)

        Returns:
            Complete analysis results with connectivity matrix, graph metrics,
            interhemispheric analysis, and optional 3D coordinates
        """
        # Step 1: Compute connectivity matrix
        connectivity_matrix = await self.compute_connectivity_matrix(time_series)

        # Step 2: Construct network graph
        graph = await self.construct_network(
            connectivity_matrix,
            density=network_density
        )

        # Step 3: Compute network metrics
        network_metrics = await self.compute_network_metrics(graph)

        # Step 4: Spatial analysis (if node locations provided)
        spatial_analysis = None
        if node_locations is not None:
            # Map nodes to sphere surface
            node_positions = await map_nodes_to_sphere(
                node_locations,
                sphere_center=np.array([0, 0, 0]),
                radius=1.0
            )

            # Interhemispheric analysis
            spatial_analysis = await self.analyze_interhemispheric_connectivity(
                graph,
                node_positions
            )
            spatial_analysis["node_positions_3d"] = {
                k: v.tolist() for k, v in node_positions.items()
            }

        return {
            "connectivity_matrix": connectivity_matrix.tolist(),
            "connectivity_method": self.connectivity_method,
            "network_threshold": float(np.min([
                graph[u][v]['weight'] for u, v in graph.edges()
            ])) if graph.number_of_edges() > 0 else 0.0,
            "network_metrics": network_metrics,
            "spatial_analysis": spatial_analysis,
            "region_labels": self.region_labels
        }
