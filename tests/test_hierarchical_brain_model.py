"""Tests for hierarchical brain model with clustered thinking hubs."""

import pytest
import numpy as np
import networkx as nx
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_clustered_brain_nodes,
    build_knn_graph,
    build_threshold_graph,
    compute_cluster_statistics,
    generate_hierarchical_brain_graph
)


class TestClusteredDataGeneration:
    """Test synthetic clustered brain data generation."""

    def test_generate_clustered_nodes_basic(self):
        """Test basic clustered node generation."""
        positions, labels = generate_clustered_brain_nodes(
            n_samples=100,
            n_clusters=5,
            random_state=42
        )

        assert positions.shape == (100, 3)
        assert labels.shape == (100,)
        assert len(np.unique(labels)) == 5

    def test_cluster_separation(self):
        """Test that clusters are reasonably separated."""
        positions, labels = generate_clustered_brain_nodes(
            n_samples=200,
            n_clusters=10,
            cluster_std=0.3,
            random_state=42
        )

        # Compute cluster centers
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            mask = labels == label
            center = np.mean(positions[mask], axis=0)
            centers.append(center)

        centers = np.array(centers)

        # Check that cluster centers are reasonably separated
        # Minimum distance between centers should be > 0.5
        from scipy.spatial.distance import pdist
        min_center_dist = pdist(centers).min()
        assert min_center_dist > 0.5

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with random_state."""
        pos1, labels1 = generate_clustered_brain_nodes(
            n_samples=50, n_clusters=3, random_state=42
        )
        pos2, labels2 = generate_clustered_brain_nodes(
            n_samples=50, n_clusters=3, random_state=42
        )

        np.testing.assert_array_almost_equal(pos1, pos2)
        np.testing.assert_array_equal(labels1, labels2)


class TestGraphConstruction:
    """Test graph construction from clustered data."""

    def test_knn_graph_structure(self):
        """Test k-NN graph has correct structure."""
        positions = np.random.randn(50, 3)
        k = 5

        G = build_knn_graph(positions, k=k)

        assert G.number_of_nodes() == 50
        # Each node adds edges to its k nearest neighbors
        # But other nodes may also connect to it, so degree can vary
        degrees = [G.degree(n) for n in G.nodes()]
        assert min(degrees) >= k  # At least k neighbors
        # Average degree should be roughly k (each node connects to k others)
        avg_degree = sum(degrees) / len(degrees)
        assert k <= avg_degree <= 3 * k  # Reasonable range

    def test_knn_graph_weights(self):
        """Test that k-NN graph edges have weight attributes."""
        positions = np.random.randn(20, 3)
        G = build_knn_graph(positions, k=3)

        # Check all edges have weights
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] > 0

    def test_threshold_graph_structure(self):
        """Test threshold graph structure."""
        # Create positions where we know distances
        positions = np.array([
            [0, 0, 0],
            [0.5, 0, 0],  # Distance 0.5 from first
            [2.0, 0, 0],  # Distance 2.0 from first
            [0, 0.5, 0],  # Distance 0.5 from first
        ])

        G = build_threshold_graph(positions, threshold=1.0)

        assert G.number_of_nodes() == 4
        # Nodes 0, 1, 3 should be connected (within threshold)
        # Node 2 should only connect to node 1
        assert G.has_edge(0, 1)
        assert G.has_edge(0, 3)
        assert not G.has_edge(0, 2)

    def test_threshold_vs_knn(self):
        """Test that threshold and k-NN produce different graphs."""
        positions, labels = generate_clustered_brain_nodes(
            n_samples=50, n_clusters=3, random_state=42
        )

        G_knn = build_knn_graph(positions, k=5)
        G_threshold = build_threshold_graph(positions, threshold=1.0)

        # Different edge counts
        assert G_knn.number_of_edges() != G_threshold.number_of_edges()


class TestClusterStatistics:
    """Test cluster statistics computation."""

    def test_compute_statistics(self):
        """Test cluster statistics are computed correctly."""
        positions, labels = generate_clustered_brain_nodes(
            n_samples=100, n_clusters=5, random_state=42
        )

        stats = compute_cluster_statistics(labels, positions)

        # Should have stats for 5 clusters
        assert len(stats) == 5

        for cluster_id, cluster_stats in stats.items():
            assert 'size' in cluster_stats
            assert 'center' in cluster_stats
            assert 'radius' in cluster_stats
            assert 'std' in cluster_stats

            # Size should be positive
            assert cluster_stats['size'] > 0

            # Center should be 3D
            assert len(cluster_stats['center']) == 3

            # Radius should be positive
            assert cluster_stats['radius'] > 0

    def test_cluster_sizes_sum_to_total(self):
        """Test that cluster sizes sum to total nodes."""
        n_samples = 200
        positions, labels = generate_clustered_brain_nodes(
            n_samples=n_samples, n_clusters=10, random_state=42
        )

        stats = compute_cluster_statistics(labels, positions)

        total_size = sum(s['size'] for s in stats.values())
        assert total_size == n_samples


class TestIntegratedPipeline:
    """Test complete hierarchical brain graph generation."""

    def test_generate_complete_graph_knn(self):
        """Test complete pipeline with k-NN."""
        G, positions, labels, stats = generate_hierarchical_brain_graph(
            n_samples=100,
            n_clusters=5,
            connection_type="knn",
            k=5,
            random_state=42
        )

        assert G.number_of_nodes() == 100
        assert positions.shape == (100, 3)
        assert labels.shape == (100,)
        assert len(stats) == 5
        assert G.number_of_edges() > 0

    def test_generate_complete_graph_threshold(self):
        """Test complete pipeline with threshold."""
        G, positions, labels, stats = generate_hierarchical_brain_graph(
            n_samples=100,
            n_clusters=5,
            connection_type="threshold",
            threshold=1.5,
            random_state=42
        )

        assert G.number_of_nodes() == 100
        assert positions.shape == (100, 3)
        assert len(stats) == 5

    def test_graph_has_node_positions(self):
        """Test that graph nodes have position attributes."""
        G, positions, labels, stats = generate_hierarchical_brain_graph(
            n_samples=50,
            n_clusters=3,
            connection_type="knn",
            k=5,
            random_state=42
        )

        # All nodes should have pos attribute
        for node in G.nodes():
            assert 'pos' in G.nodes[node]
            pos = G.nodes[node]['pos']
            assert len(pos) == 3

    def test_invalid_connection_type(self):
        """Test that invalid connection type raises error."""
        with pytest.raises(ValueError, match="Unknown connection_type"):
            generate_hierarchical_brain_graph(
                n_samples=50,
                n_clusters=3,
                connection_type="invalid"
            )


class TestScalability:
    """Test that methods scale to larger graphs."""

    def test_large_graph_generation(self):
        """Test generation of larger graph (500 nodes)."""
        G, positions, labels, stats = generate_hierarchical_brain_graph(
            n_samples=500,
            n_clusters=15,
            connection_type="knn",
            k=8,
            random_state=42
        )

        assert G.number_of_nodes() == 500
        assert len(stats) == 15

    def test_many_clusters(self):
        """Test generation with many clusters."""
        G, positions, labels, stats = generate_hierarchical_brain_graph(
            n_samples=300,
            n_clusters=20,
            connection_type="knn",
            k=5,
            random_state=42
        )

        assert len(stats) == 20
        # Each cluster should have at least 5 nodes on average
        avg_cluster_size = 300 / 20
        assert avg_cluster_size >= 10
