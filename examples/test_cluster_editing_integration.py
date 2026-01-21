"""Test cluster-editing-vs integration with merge2docs.

Simple test with small graph to verify the integration works before
scaling up to larger brain models.
"""

import sys
from pathlib import Path
import asyncio
import numpy as np
import networkx as nx

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_hierarchical_brain_graph,
    detect_communities_cluster_editing,
    detect_communities_simple,
    evaluate_clustering_quality
)


async def test_cluster_editing_simple():
    """Test cluster-editing-vs with small simple graph."""

    print("=" * 70)
    print("Testing cluster-editing-vs Integration")
    print("=" * 70)
    print()

    # Generate small clustered graph
    print("Step 1: Generate small test graph...")
    n_samples = 50
    n_clusters = 3
    k = 5

    G, positions, ground_truth_labels, stats = generate_hierarchical_brain_graph(
        n_samples=n_samples,
        n_clusters=n_clusters,
        connection_type="knn",
        k=k,
        random_state=42
    )

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Ground truth: {n_clusters} clusters")
    print()

    # Test 1: Simple baseline
    print("Step 2: Baseline (connected components)...")
    communities_simple = detect_communities_simple(G, method="connected_components")
    quality_simple = evaluate_clustering_quality(G, communities_simple)

    print(f"  Detected: {quality_simple['n_clusters']} communities")
    print(f"  Modularity: {quality_simple['modularity']:.3f}")
    print(f"  Internal edges: {quality_simple['internal_edges']:.1%}")
    print()

    # Test 2: Try cluster-editing-vs
    print("Step 3: Testing cluster-editing-vs integration...")
    print("  Calling merge2docs algorithm service...")

    try:
        communities_ce = await detect_communities_cluster_editing(
            G,
            method="cluster_editing_vs",
            k=None,  # Auto-estimate
            use_gpu=False  # Start with CPU version
        )

        if communities_ce is None:
            print("  ❌ cluster-editing-vs returned None")
            print("  This likely means:")
            print("     - merge2docs service is not running")
            print("     - GraphDependencies initialization failed")
            print("     - Algorithm service call failed")
            print()
            print("  Checking merge2docs availability...")
            check_merge2docs_status()
        else:
            print("  ✅ cluster-editing-vs succeeded!")
            quality_ce = evaluate_clustering_quality(G, communities_ce)

            print(f"  Detected: {quality_ce['n_clusters']} communities")
            print(f"  Modularity: {quality_ce['modularity']:.3f}")
            print(f"  Internal edges: {quality_ce['internal_edges']:.1%}")
            print()

            # Compare methods
            print("Comparison:")
            print(f"  Ground truth:        {n_clusters} clusters")
            print(f"  Simple method:       {quality_simple['n_clusters']} clusters (modularity={quality_simple['modularity']:.3f})")
            print(f"  cluster-editing-vs:  {quality_ce['n_clusters']} clusters (modularity={quality_ce['modularity']:.3f})")
            print()

            if quality_ce['modularity'] > quality_simple['modularity']:
                print("  ✅ cluster-editing-vs found better clustering!")
            elif quality_ce['n_clusters'] == n_clusters:
                print("  ✅ cluster-editing-vs matched ground truth cluster count!")
            else:
                print("  ℹ️  Results differ - expected for different algorithms")

    except Exception as e:
        print(f"  ❌ Error during cluster-editing-vs: {e}")
        print(f"  Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print()
        check_merge2docs_status()

    print()
    print("=" * 70)


def check_merge2docs_status():
    """Check if merge2docs is available and properly configured."""
    print()
    print("Checking merge2docs status:")
    print("-" * 70)

    # Check if merge2docs path exists
    try:
        from backend.integration.merge2docs_bridge import (
            MERGE2DOCS_AVAILABLE, MERGE2DOCS_PATH, get_deps
        )

        print(f"  merge2docs path: {MERGE2DOCS_PATH}")
        print(f"  merge2docs available: {MERGE2DOCS_AVAILABLE}")

        if not MERGE2DOCS_AVAILABLE:
            print("  ❌ merge2docs not found at expected location")
            print()
            print("  Solutions:")
            print("  1. Ensure merge2docs is cloned at: ../merge2docs")
            print("  2. Or update MERGE2DOCS_PATH in merge2docs_bridge.py")
            return

        # Check GraphDependencies
        print("  Attempting to initialize GraphDependencies...")
        deps = get_deps()

        if deps is None:
            print("  ❌ GraphDependencies initialization failed")
            print()
            print("  Possible causes:")
            print("  1. merge2docs services not running")
            print("  2. A2A context creation failed")
            print("  3. Import errors in merge2docs")
        else:
            print("  ✅ GraphDependencies initialized successfully")
            print()
            print("  Note: Algorithm service calls may still fail if:")
            print("  - AlgorithmService is not registered")
            print("  - cluster_editing_vs algorithm is not available")
            print("  - Service method call fails")

    except ImportError as e:
        print(f"  ❌ Cannot import merge2docs_bridge: {e}")
        print()
        print("  Check that the bridge module exists at:")
        print("  src/backend/integration/merge2docs_bridge.py")

    print("-" * 70)


async def test_with_manual_clusters():
    """Test with a manually constructed graph with clear cluster structure."""

    print()
    print("=" * 70)
    print("Bonus Test: Manual Graph with Clear Clusters")
    print("=" * 70)
    print()

    # Build graph with 3 clear clusters (cliques)
    G = nx.Graph()

    # Cluster 1: nodes 0-9 (fully connected)
    for i in range(10):
        for j in range(i+1, 10):
            G.add_edge(i, j)

    # Cluster 2: nodes 10-19 (fully connected)
    for i in range(10, 20):
        for j in range(i+1, 20):
            G.add_edge(i, j)

    # Cluster 3: nodes 20-29 (fully connected)
    for i in range(20, 30):
        for j in range(i+1, 30):
            G.add_edge(i, j)

    # Add a few inter-cluster edges
    G.add_edge(5, 15)   # Connect cluster 1 and 2
    G.add_edge(15, 25)  # Connect cluster 2 and 3

    print(f"Created test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Structure: 3 cliques (10 nodes each) + 2 bridge edges")
    print()

    # Test simple method
    print("Simple method:")
    communities = detect_communities_simple(G, method="connected_components")
    quality = evaluate_clustering_quality(G, communities)
    print(f"  Clusters: {quality['n_clusters']}, Modularity: {quality['modularity']:.3f}")
    print()

    # Test cluster-editing-vs
    print("cluster-editing-vs:")
    try:
        communities_ce = await detect_communities_cluster_editing(
            G,
            method="cluster_editing_vs",
            k=10,  # Allow up to 10 edge edits
            use_gpu=False
        )

        if communities_ce is not None:
            quality_ce = evaluate_clustering_quality(G, communities_ce)
            print(f"  ✅ Clusters: {quality_ce['n_clusters']}, Modularity: {quality_ce['modularity']:.3f}")

            if quality_ce['n_clusters'] == 3:
                print("  ✅ Correctly identified 3 clusters!")
        else:
            print("  ❌ Algorithm returned None")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()


async def main():
    """Run all tests."""
    await test_cluster_editing_simple()
    await test_with_manual_clusters()

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    print("If cluster-editing-vs failed, this is expected if:")
    print("  - merge2docs is not installed at ../merge2docs")
    print("  - merge2docs services are not running")
    print("  - A2A service architecture is not initialized")
    print()
    print("The hierarchical brain model works with fallback (connected components)")
    print("and will automatically use cluster-editing-vs when available.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
