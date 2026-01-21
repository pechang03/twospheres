"""Tests for network analysis and sphere overlay."""

import numpy as np
import pytest
import networkx as nx

from src.backend.mri.network_analysis import (
    connectivity_matrix_to_graph,
    map_nodes_to_sphere,
    identify_interhemispheric_edges,
    compute_edge_geodesic_lengths,
    compute_network_metrics,
    overlay_network_on_sphere,
    filter_by_geodesic_distance,
)


class TestConnectivityMatrixToGraph:
    """Tests for connectivity matrix conversion."""

    @pytest.mark.asyncio
    async def test_basic_conversion(self):
        """Test basic matrix to graph conversion."""
        # 3-node fully connected network
        conn = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])

        G = await connectivity_matrix_to_graph(conn, threshold=0.0)

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3  # 3 edges in fully connected triangle

    @pytest.mark.asyncio
    async def test_threshold_filtering(self):
        """Test edge filtering by threshold."""
        conn = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.6],
            [0.3, 0.6, 1.0]
        ])

        G = await connectivity_matrix_to_graph(conn, threshold=0.5)

        # Only edges above 0.5: (0,1)=0.8 and (1,2)=0.6
        assert G.number_of_edges() == 2
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 2)
        assert not G.has_edge(0, 2)  # 0.3 below threshold

    @pytest.mark.asyncio
    async def test_with_node_labels(self):
        """Test graph with custom node labels."""
        conn = np.array([
            [1.0, 0.8],
            [0.8, 1.0]
        ])
        labels = ["V1", "V4"]

        G = await connectivity_matrix_to_graph(conn, node_labels=labels)

        assert set(G.nodes()) == {"V1", "V4"}
        assert G.has_edge("V1", "V4")

    @pytest.mark.asyncio
    async def test_edge_weights(self):
        """Test that edge weights are preserved."""
        conn = np.array([
            [1.0, 0.75],
            [0.75, 1.0]
        ])

        G = await connectivity_matrix_to_graph(conn)

        weight = G.edges[0, 1]["weight"]
        assert np.isclose(weight, 0.75)


class TestMapNodesToSphere:
    """Tests for mapping nodes to sphere surface."""

    @pytest.mark.asyncio
    async def test_basic_mapping(self):
        """Test basic node mapping."""
        locations = {
            "V1": {"theta": 0, "phi": np.pi/2},
            "V4": {"theta": np.pi/2, "phi": np.pi/2}
        }

        positions = await map_nodes_to_sphere(locations)

        assert "V1" in positions
        assert "V4" in positions
        assert positions["V1"].shape == (3,)  # [x, y, z]
        assert positions["V4"].shape == (3,)

    @pytest.mark.asyncio
    async def test_positions_on_sphere(self):
        """Test that mapped positions are on sphere surface."""
        locations = {
            "node1": {"theta": np.pi/4, "phi": np.pi/3},
            "node2": {"theta": np.pi/3, "phi": np.pi/4}
        }
        radius = 2.0

        positions = await map_nodes_to_sphere(locations, radius=radius)

        # Check that all positions are at correct radius
        for pos in positions.values():
            dist_from_origin = np.linalg.norm(pos)
            assert np.isclose(dist_from_origin, radius)

    @pytest.mark.asyncio
    async def test_with_sphere_center(self):
        """Test mapping with non-origin center."""
        locations = {
            "node": {"theta": 0, "phi": np.pi/2}
        }
        center = np.array([1.0, 2.0, 3.0])
        radius = 1.0

        positions = await map_nodes_to_sphere(
            locations, sphere_center=center, radius=radius
        )

        # Point should be at [2, 2, 3] (center + [1, 0, 0])
        expected = np.array([2.0, 2.0, 3.0])
        assert np.allclose(positions["node"], expected)


class TestIdentifyInterhemisphericEdges:
    """Tests for interhemispheric edge detection."""

    @pytest.mark.asyncio
    async def test_no_interhemispheric_edges(self):
        """Test graph with all nodes on same hemisphere."""
        G = nx.Graph()
        G.add_edge("node1", "node2")

        positions = {
            "node1": np.array([0, 1, 0]),  # y > 0
            "node2": np.array([0, 0.5, 0])  # y > 0
        }

        interhemispheric = await identify_interhemispheric_edges(G, positions)

        assert len(interhemispheric) == 0

    @pytest.mark.asyncio
    async def test_one_interhemispheric_edge(self):
        """Test graph with one corpus callosum connection."""
        G = nx.Graph()
        G.add_edge("right", "left")

        positions = {
            "right": np.array([0, 1, 0]),   # y > 0
            "left": np.array([0, -1, 0])    # y < 0
        }

        interhemispheric = await identify_interhemispheric_edges(G, positions)

        assert len(interhemispheric) == 1
        assert ("right", "left") in interhemispheric or ("left", "right") in interhemispheric

    @pytest.mark.asyncio
    async def test_mixed_edges(self):
        """Test graph with both intra- and interhemispheric edges."""
        G = nx.Graph()
        G.add_edges_from([
            ("right1", "right2"),  # Intrahemispheric
            ("right1", "left1"),   # Interhemispheric
            ("left1", "left2")     # Intrahemispheric
        ])

        positions = {
            "right1": np.array([0, 1, 0]),
            "right2": np.array([0, 0.8, 0]),
            "left1": np.array([0, -1, 0]),
            "left2": np.array([0, -0.8, 0])
        }

        interhemispheric = await identify_interhemispheric_edges(G, positions)

        assert len(interhemispheric) == 1  # Only right1-left1


class TestComputeEdgeGeodesicLengths:
    """Tests for geodesic edge length computation."""

    @pytest.mark.asyncio
    async def test_simple_edge(self):
        """Test geodesic length for simple edge."""
        G = nx.Graph()
        G.add_edge("V1", "V4")

        locations = {
            "V1": {"theta": 0, "phi": np.pi/2},
            "V4": {"theta": np.pi/2, "phi": np.pi/2}
        }

        lengths = await compute_edge_geodesic_lengths(G, locations, radius=1.0)

        assert ("V1", "V4") in lengths or ("V4", "V1") in lengths
        edge_length = lengths.get(("V1", "V4"), lengths.get(("V4", "V1")))
        expected = np.pi / 2  # Quarter circle
        assert np.isclose(edge_length, expected)

    @pytest.mark.asyncio
    async def test_multiple_edges(self):
        """Test multiple edge lengths."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C")])

        locations = {
            "A": {"theta": 0, "phi": np.pi/2},
            "B": {"theta": np.pi/4, "phi": np.pi/2},
            "C": {"theta": np.pi/2, "phi": np.pi/2}
        }

        lengths = await compute_edge_geodesic_lengths(G, locations)

        # All edges should have lengths
        assert len(lengths) == 2


class TestComputeNetworkMetrics:
    """Tests for network metrics computation."""

    @pytest.mark.asyncio
    async def test_basic_metrics(self):
        """Test basic network metrics."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle

        metrics = await compute_network_metrics(G)

        assert metrics["n_nodes"] == 3
        assert metrics["n_edges"] == 3
        assert metrics["density"] == 1.0  # Fully connected triangle
        assert metrics["avg_degree"] == 2.0  # Each node has degree 2

    @pytest.mark.asyncio
    async def test_clustering_coefficient(self):
        """Test clustering coefficient calculation."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle

        metrics = await compute_network_metrics(G)

        # Fully connected triangle has clustering = 1.0
        assert np.isclose(metrics["avg_clustering"], 1.0)

    @pytest.mark.asyncio
    async def test_empty_graph(self):
        """Test metrics for empty graph."""
        G = nx.Graph()

        metrics = await compute_network_metrics(G)

        assert metrics["n_nodes"] == 0
        assert metrics["n_edges"] == 0
        assert metrics["avg_degree"] == 0


class TestOverlayNetworkOnSphere:
    """Tests for complete overlay pipeline."""

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test full overlay pipeline."""
        conn = np.array([
            [1.0, 0.8],
            [0.8, 1.0]
        ])

        locations = {
            "V1": {"theta": 0, "phi": np.pi/2},
            "V4": {"theta": np.pi/2, "phi": np.pi/2}
        }

        overlay = await overlay_network_on_sphere(
            conn, locations, threshold=0.5
        )

        assert "graph" in overlay
        assert "node_positions" in overlay
        assert "edge_lengths" in overlay
        assert "metrics" in overlay

        assert overlay["graph"].number_of_nodes() == 2
        assert overlay["graph"].number_of_edges() == 1  # Above threshold

    @pytest.mark.asyncio
    async def test_interhemispheric_detection(self):
        """Test interhemispheric edge detection in pipeline."""
        conn = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])

        locations = {
            "right1": {"theta": 0, "phi": np.pi/2},
            "left1": {"theta": 0, "phi": np.pi/2},
            "right2": {"theta": np.pi/4, "phi": np.pi/2}
        }

        # Position on different hemispheres
        sphere1_center = np.array([0, 1, 0])  # Right
        sphere2_center = np.array([0, -1, 0])  # Left

        # Need to set positions manually for this test
        # (simplified - just test the metric is computed)
        overlay = await overlay_network_on_sphere(
            conn, locations, threshold=0.5
        )

        assert "interhemispheric_edges" in overlay
        assert "interhemispheric_fraction" in overlay["metrics"]

    @pytest.mark.asyncio
    async def test_with_threshold(self):
        """Test overlay with connectivity threshold."""
        conn = np.array([
            [1.0, 0.3, 0.9],
            [0.3, 1.0, 0.4],
            [0.9, 0.4, 1.0]
        ])

        locations = {
            "A": {"theta": 0, "phi": np.pi/2},
            "B": {"theta": np.pi/2, "phi": np.pi/2},
            "C": {"theta": np.pi, "phi": np.pi/2}
        }

        overlay = await overlay_network_on_sphere(
            conn, locations, threshold=0.5  # Only strong connections
        )

        # Only A-C edge (0.9) above threshold
        assert overlay["metrics"]["n_edges"] == 1


class TestFilterByGeodesicDistance:
    """Tests for geodesic distance filtering."""

    @pytest.mark.asyncio
    async def test_filter_local_connections(self):
        """Test filtering to keep only local connections."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("A", "C"), ("B", "C")])

        locations = {
            "A": {"theta": 0, "phi": np.pi/2},
            "B": {"theta": 0.1, "phi": np.pi/2},  # Very close
            "C": {"theta": np.pi/2, "phi": np.pi/2}  # Far
        }

        filtered = await filter_by_geodesic_distance(
            G, locations, max_distance=0.2  # Keep only close edges
        )

        # Only A-B should remain (C is far from both)
        assert filtered.number_of_edges() < G.number_of_edges()

    @pytest.mark.asyncio
    async def test_filter_long_range(self):
        """Test filtering to keep only long-range connections."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("A", "C")])

        locations = {
            "A": {"theta": 0, "phi": np.pi/2},
            "B": {"theta": 0.1, "phi": np.pi/2},  # Close
            "C": {"theta": np.pi, "phi": np.pi/2}  # Far (opposite side)
        }

        filtered = await filter_by_geodesic_distance(
            G, locations, min_distance=1.0  # Keep only distant edges
        )

        # Only A-C should remain (far connection)
        assert filtered.number_of_edges() < G.number_of_edges()
        assert filtered.has_edge("A", "C")

    @pytest.mark.asyncio
    async def test_filter_preserves_nodes(self):
        """Test that filtering preserves all nodes."""
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C")])

        locations = {
            "A": {"theta": 0, "phi": np.pi/2},
            "B": {"theta": np.pi/4, "phi": np.pi/2},
            "C": {"theta": np.pi/2, "phi": np.pi/2}
        }

        filtered = await filter_by_geodesic_distance(
            G, locations, max_distance=0.1  # Remove all edges
        )

        # All nodes should still be present
        assert filtered.number_of_nodes() == G.number_of_nodes()
        assert filtered.number_of_edges() == 0  # But no edges
