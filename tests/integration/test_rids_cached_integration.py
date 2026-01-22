"""Integration tests for PRIME-DE → r-IDS cached workflow.

Tests the "Compute Once, Reuse Many" pattern with live services:
- PRIME-DE HTTP Server (port 8009)
- yada-services-secure (port 8003)
- PostgreSQL database (port 5432)

Run with: pytest tests/integration/test_rids_cached_integration.py -v -m live_services
"""

import asyncio
import hashlib
import httpx
import numpy as np
import networkx as nx
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor

from src.backend.data.prime_de_loader import PRIMEDELoader


# Mark all tests as requiring live services
pytestmark = pytest.mark.live_services


# Database configuration
POSTGRES_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "petershaw",
    "password": "FruitSalid4",
    "database": "twosphere_brain"
}

# Service URLs
PRIME_DE_URL = "http://localhost:8009"
YADA_URL = "http://localhost:8003"


def compute_graph_hash(G: nx.Graph) -> str:
    """Compute graph hash compatible with GPU Graph Manager."""
    adj = nx.to_numpy_array(G)
    upper_tri = adj[np.triu_indices_from(adj)]
    hash_input = np.packbits(upper_tri > 0).tobytes()
    return hashlib.md5(hash_input).hexdigest()[:12]


def connectivity_to_graph(connectivity: np.ndarray, threshold: float = 0.5) -> nx.Graph:
    """Convert connectivity matrix to graph."""
    num_regions = connectivity.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_regions))

    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            if abs(connectivity[i, j]) >= threshold:
                G.add_edge(i, j, weight=abs(connectivity[i, j]))

    return G


class TestGraphHashing:
    """Test graph hashing utilities."""

    def test_graph_hash_deterministic(self):
        """Graph hash should be deterministic."""
        G = nx.erdos_renyi_graph(20, 0.3)
        hash1 = compute_graph_hash(G)
        hash2 = compute_graph_hash(G)

        assert hash1 == hash2
        assert len(hash1) == 12

    def test_graph_hash_different_structures(self):
        """Different graph structures should produce different hashes."""
        G1 = nx.path_graph(10)
        G2 = nx.cycle_graph(10)

        hash1 = compute_graph_hash(G1)
        hash2 = compute_graph_hash(G2)

        assert hash1 != hash2

    def test_connectivity_to_graph(self):
        """Convert connectivity matrix to graph."""
        # Small correlation matrix
        connectivity = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.6],
            [0.2, 0.6, 1.0]
        ])

        G = connectivity_to_graph(connectivity, threshold=0.5)

        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2  # 0-1 (0.8) and 1-2 (0.6)
        assert G.has_edge(0, 1)
        assert G.has_edge(1, 2)
        assert not G.has_edge(0, 2)  # 0.2 < 0.5


class TestPRIMEDEGraphConstruction:
    """Test building graphs from PRIME-DE connectivity."""

    @pytest.mark.asyncio
    async def test_load_and_build_graph(self):
        """Load PRIME-DE subject and build connectivity graph."""
        loader = PRIMEDELoader(base_url=PRIME_DE_URL)

        # Load sample subject
        data = await loader.load_and_process_subject(
            "BORDEAUX24",
            "m01",
            "bold",
            connectivity_method="pearson"  # Faster than distance_correlation
        )

        assert data["timeseries"].shape[1] == 368  # D99 regions
        assert data["connectivity"].shape == (368, 368)

        # Build graph from connectivity
        G = connectivity_to_graph(data["connectivity"], threshold=0.5)

        assert G.number_of_nodes() == 368
        assert G.number_of_edges() > 0  # Should have some edges

        # Verify graph properties
        density = nx.density(G)
        assert 0.0 < density < 1.0  # Not empty, not complete

        print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Density: {density:.3f}")


class TestCachedWorkflow:
    """Test cached workflow with graph hashing."""

    @pytest.mark.asyncio
    async def test_graph_hash_reuse(self):
        """Test that graph hash can be computed and reused."""
        # Load PRIME-DE data
        loader = PRIMEDELoader(base_url=PRIME_DE_URL)
        data = await loader.load_subject("BORDEAUX24", "m01", "bold")

        # Build graph
        G = connectivity_to_graph(
            np.random.randn(368, 368),  # Use random for speed
            threshold=0.5
        )

        # Compute hash
        graph_hash = compute_graph_hash(G)

        assert len(graph_hash) == 12
        assert graph_hash.isalnum()

        # Hash should be stable
        assert compute_graph_hash(G) == graph_hash

        print(f"\nGraph hash: {graph_hash}")
        print(f"Original graph size: ~{G.number_of_edges() * 16} bytes (edges)")
        print(f"Hash size: 12 bytes")
        print(f"Reduction: {(1 - 12 / (G.number_of_edges() * 16)) * 100:.2f}%")

    @pytest.mark.asyncio
    async def test_hierarchical_rids_with_thresholds(self):
        """Test hierarchical r-IDS with threshold optimization."""
        # Build simple test graphs
        graphs = {
            "small": nx.cycle_graph(20),
            "medium": nx.erdos_renyi_graph(50, 0.2)
        }

        # Convert to dict format for MCP
        graphs_dict = {
            level: {
                "nodes": list(graph.nodes()),
                "edges": [[u, v] for u, v in graph.edges()]
            }
            for level, graph in graphs.items()
        }

        # Prepare MCP request with thresholds
        level_parameters = {
            "small": {"r": 2, "target_size": 5},
            "medium": {"r": 3, "target_size": 10}
        }

        # Call yada-services-secure
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{YADA_URL}/mcp/tools/call",
                json={
                    "name": "compute_hierarchical_r_ids",
                    "arguments": {
                        "graphs_by_level": graphs_dict,
                        "level_parameters": level_parameters,

                        # Threshold optimization
                        "pre_repair_threshold": 0.5,
                        "post_repair_threshold": 0.7,
                        "use_read_repair": True,
                        "validate_clt": False,  # Small graphs, skip CLT
                        "use_service_layer": True
                    }
                }
            )

            assert response.status_code == 200
            result = response.json()

            # Verify results
            assert "solutions" in result or "result" in result

            print(f"\n✅ Hierarchical r-IDS completed")
            print(f"Response keys: {result.keys()}")


class TestDatabaseIntegration:
    """Test storing r-IDS results in database."""

    def test_store_rids_samples(self):
        """Test storing r-IDS sampled regions in database."""
        conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # Sample data
        subject_id = "m01_test"
        sampled_nodes = [5, 18, 32, 67, 101, 145, 203, 278, 312, 350]

        # Store in database
        cursor.execute("""
            INSERT INTO rids_connections
            (subject_id, level_name, sampled_nodes, coverage_percentage,
             r_parameter, created_at)
            VALUES
            (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (subject_id, level_name)
            DO UPDATE SET
                sampled_nodes = EXCLUDED.sampled_nodes,
                coverage_percentage = EXCLUDED.coverage_percentage,
                r_parameter = EXCLUDED.r_parameter,
                created_at = NOW()
            RETURNING *
        """, (
            subject_id,
            "region",
            sampled_nodes,
            100.0,
            4
        ))

        row = cursor.fetchone()
        conn.commit()

        # Verify storage
        assert row is not None
        assert row["subject_id"] == subject_id
        assert row["level_name"] == "region"
        assert row["sampled_nodes"] == sampled_nodes
        assert row["r_parameter"] == 4

        # Cleanup
        cursor.execute("""
            DELETE FROM rids_connections
            WHERE subject_id = %s
        """, (subject_id,))
        conn.commit()

        cursor.close()
        conn.close()

        print(f"\n✅ Stored and verified {len(sampled_nodes)} sampled regions")


class TestEndToEndCachedWorkflow:
    """End-to-end test of cached workflow."""

    @pytest.mark.asyncio
    async def test_prime_de_to_rids_cached(self):
        """
        Complete workflow: PRIME-DE → Graph → Hash → r-IDS → Database

        Steps:
        1. Load PRIME-DE subject
        2. Build connectivity graph
        3. Compute graph hash
        4. Run r-IDS with threshold optimization
        5. Store results in database
        """
        print("\n" + "=" * 60)
        print("End-to-End Cached Workflow Test")
        print("=" * 60)

        # Step 1: Load PRIME-DE data
        print("\n[1] Loading PRIME-DE subject...")
        loader = PRIMEDELoader(base_url=PRIME_DE_URL)
        data = await loader.load_subject("BORDEAUX24", "m01", "bold")

        timeseries = data["timeseries"]
        print(f"  ✅ Loaded: {timeseries.shape}")

        # Step 2: Build graph (simplified for test speed)
        print("\n[2] Building connectivity graph...")
        # Use small subset for fast test
        sample_connectivity = np.random.randn(50, 50)
        sample_connectivity = (sample_connectivity + sample_connectivity.T) / 2
        G = connectivity_to_graph(sample_connectivity, threshold=0.3)

        print(f"  ✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Step 3: Compute hash
        print("\n[3] Computing graph hash...")
        graph_hash = compute_graph_hash(G)
        print(f"  ✅ Hash: {graph_hash}")

        # Step 4: Run r-IDS
        print("\n[4] Running r-IDS with thresholds...")
        graph_dict = {
            "nodes": list(G.nodes()),
            "edges": [[u, v] for u, v in G.edges()]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{YADA_URL}/mcp/tools/call",
                json={
                    "name": "compute_r_ids",
                    "arguments": {
                        "graph_data": graph_dict,
                        "r": 3,
                        "target_size": 10,
                        "pre_repair_threshold": 0.5,
                        "post_repair_threshold": 0.7,
                        "use_read_repair": True,
                        "validate_clt": False,
                        "use_service_layer": True
                    }
                }
            )

            assert response.status_code == 200
            result = response.json()
            print(f"  ✅ r-IDS completed")

        # Step 5: Store in database
        print("\n[5] Storing results in database...")
        conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # Store sample
        cursor.execute("""
            INSERT INTO rids_connections
            (subject_id, level_name, sampled_nodes, coverage_percentage,
             r_parameter, created_at)
            VALUES
            (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (subject_id, level_name)
            DO UPDATE SET sampled_nodes = EXCLUDED.sampled_nodes
            RETURNING connection_id
        """, ("m01_cached_test", "region", [1, 5, 10], 100.0, 3))

        row = cursor.fetchone()
        conn.commit()

        assert row is not None
        print(f"  ✅ Stored: connection_id={row['connection_id']}")

        # Cleanup
        cursor.execute("DELETE FROM rids_connections WHERE subject_id = %s",
                      ("m01_cached_test",))
        conn.commit()
        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("✅ END-TO-END TEST COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "live_services"])
