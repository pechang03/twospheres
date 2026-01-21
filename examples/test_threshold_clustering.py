"""Test threshold-based graph for better cluster separation."""

import sys
from pathlib import Path
import asyncio

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_hierarchical_brain_graph,
    detect_communities_cluster_editing,
    evaluate_clustering_quality
)


async def test_threshold_clustering():
    """Test different graph construction methods for better clustering."""

    print("=" * 70)
    print("Testing Threshold-Based Graph Construction")
    print("=" * 70)
    print()

    n_samples = 100
    n_clusters = 10

    # Test 1: k-NN with k=3 (fewer connections)
    print("Test 1: k-NN with k=3 (sparse connections)")
    G1, pos1, labels1, stats1 = generate_hierarchical_brain_graph(
        n_samples=n_samples,
        n_clusters=n_clusters,
        connection_type="knn",
        k=3,  # Very sparse
        random_state=42
    )

    inter_cluster = sum(1 for u, v in G1.edges() if labels1[u] != labels1[v])
    print(f"  Graph: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"  Inter-cluster edges: {inter_cluster}")

    communities1 = await detect_communities_cluster_editing(
        G1, method="cluster_editing_vs", k=100, use_gpu=False
    )
    if communities1:
        quality1 = evaluate_clustering_quality(G1, communities1)
        print(f"  Result: {quality1['n_clusters']} clusters, modularity={quality1['modularity']:.3f}")
    print()

    # Test 2: Threshold with distance=1.0 (tight threshold)
    print("Test 2: Threshold-based with threshold=1.0")
    G2, pos2, labels2, stats2 = generate_hierarchical_brain_graph(
        n_samples=n_samples,
        n_clusters=n_clusters,
        connection_type="threshold",
        threshold=1.0,  # Tight threshold
        random_state=42
    )

    inter_cluster = sum(1 for u, v in G2.edges() if labels2[u] != labels2[v])
    print(f"  Graph: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
    print(f"  Inter-cluster edges: {inter_cluster}")

    communities2 = await detect_communities_cluster_editing(
        G2, method="cluster_editing_vs", k=100, use_gpu=False
    )
    if communities2:
        quality2 = evaluate_clustering_quality(G2, communities2)
        print(f"  Result: {quality2['n_clusters']} clusters, modularity={quality2['modularity']:.3f}")
    print()

    # Test 3: Threshold with distance=0.8 (very tight)
    print("Test 3: Threshold-based with threshold=0.8 (very tight)")
    G3, pos3, labels3, stats3 = generate_hierarchical_brain_graph(
        n_samples=n_samples,
        n_clusters=n_clusters,
        connection_type="threshold",
        threshold=0.8,
        random_state=42
    )

    inter_cluster = sum(1 for u, v in G3.edges() if labels3[u] != labels3[v])
    print(f"  Graph: {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")
    print(f"  Inter-cluster edges: {inter_cluster}")

    communities3 = await detect_communities_cluster_editing(
        G3, method="cluster_editing_vs", k=100, use_gpu=False
    )
    if communities3:
        quality3 = evaluate_clustering_quality(G3, communities3)
        print(f"  Result: {quality3['n_clusters']} clusters, modularity={quality3['modularity']:.3f}")
    print()

    # Test 4: Increase cluster separation
    print("Test 4: Increased cluster separation (cluster_std=0.2)")
    from backend.visualization.hierarchical_brain_model import (
        generate_clustered_brain_nodes,
        build_knn_graph,
        compute_cluster_statistics
    )

    positions_tight, labels_tight = generate_clustered_brain_nodes(
        n_samples=n_samples,
        n_clusters=n_clusters,
        cluster_std=0.2,  # Tighter clusters (default 0.3)
        random_state=42
    )
    G4 = build_knn_graph(positions_tight, k=5)
    stats4 = compute_cluster_statistics(labels_tight, positions_tight)

    inter_cluster = sum(1 for u, v in G4.edges() if labels_tight[u] != labels_tight[v])
    print(f"  Graph: {G4.number_of_nodes()} nodes, {G4.number_of_edges()} edges")
    print(f"  Inter-cluster edges: {inter_cluster}")

    communities4 = await detect_communities_cluster_editing(
        G4, method="cluster_editing_vs", k=100, use_gpu=False
    )
    if communities4:
        quality4 = evaluate_clustering_quality(G4, communities4)
        print(f"  Result: {quality4['n_clusters']} clusters, modularity={quality4['modularity']:.3f}")

        from sklearn.metrics import adjusted_rand_score
        gt_labels = [labels_tight[i] for i in sorted(G4.nodes())]
        pred_labels = [communities4[i] for i in sorted(G4.nodes())]
        ari = adjusted_rand_score(gt_labels, pred_labels)
        print(f"  ARI vs ground truth: {ari:.3f}")
    print()

    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    print()
    print("Best approach for 10 clusters:")
    print("  - Use tighter cluster_std (0.2 instead of 0.3)")
    print("  - Use threshold-based graph with tight threshold")
    print("  - Or use k-NN with k=3 for sparse connectivity")
    print()


if __name__ == "__main__":
    asyncio.run(test_threshold_clustering())
