"""Test automatic clustering with adaptive parameter selection."""

import sys
from pathlib import Path
import asyncio

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_clustered_brain_nodes,
    build_knn_graph,
    compute_cluster_statistics,
    detect_communities_auto,
    evaluate_clustering_quality
)


async def test_auto_clustering():
    """Test automatic clustering parameter selection."""

    print("=" * 70)
    print("Testing Automatic Clustering Parameter Selection")
    print("=" * 70)
    print()

    # Generate tighter clusters for better separation
    n_samples = 100
    n_clusters = 10
    cluster_std = 0.2  # Tight clusters

    print(f"Generating {n_samples} nodes in {n_clusters} clusters (cluster_std={cluster_std})...")
    positions, ground_truth_labels = generate_clustered_brain_nodes(
        n_samples=n_samples,
        n_clusters=n_clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    # Build graph with k=5
    G = build_knn_graph(positions, k=5)
    stats = compute_cluster_statistics(ground_truth_labels, positions)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    # Count inter-cluster edges
    inter_cluster = sum(1 for u, v in G.edges() if ground_truth_labels[u] != ground_truth_labels[v])
    print(f"Inter-cluster edges: {inter_cluster} ({inter_cluster/G.number_of_edges()*100:.1f}%)")
    print()

    # Test auto-tuning WITHOUT target
    print("Test 1: Auto-tuning without target (pure modularity optimization)")
    print("-" * 70)
    communities1 = await detect_communities_auto(G, target_clusters=None, use_gpu=False)

    if communities1:
        quality1 = evaluate_clustering_quality(G, communities1)
        from sklearn.metrics import adjusted_rand_score
        gt_labels = [ground_truth_labels[i] for i in sorted(G.nodes())]
        pred_labels1 = [communities1[i] for i in sorted(G.nodes())]
        ari1 = adjusted_rand_score(gt_labels, pred_labels1)

        print(f"✅ Found {quality1['n_clusters']} clusters")
        print(f"   Modularity: {quality1['modularity']:.3f}")
        print(f"   ARI vs ground truth: {ari1:.3f}")
    print()

    # Test auto-tuning WITH target
    print(f"Test 2: Auto-tuning with target={n_clusters} clusters")
    print("-" * 70)
    communities2 = await detect_communities_auto(G, target_clusters=n_clusters, use_gpu=False)

    if communities2:
        quality2 = evaluate_clustering_quality(G, communities2)
        pred_labels2 = [communities2[i] for i in sorted(G.nodes())]
        ari2 = adjusted_rand_score(gt_labels, pred_labels2)

        print(f"✅ Found {quality2['n_clusters']} clusters")
        print(f"   Modularity: {quality2['modularity']:.3f}")
        print(f"   ARI vs ground truth: {ari2:.3f}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Ground truth: {n_clusters} clusters")
    print(f"Without target: {quality1['n_clusters']} clusters (ARI={ari1:.3f})")
    print(f"With target: {quality2['n_clusters']} clusters (ARI={ari2:.3f})")
    print()

    if quality2['n_clusters'] >= 8:
        print("✅ Auto-tuning successfully found close to ground truth!")
    elif quality2['n_clusters'] >= 6:
        print("⚠️  Auto-tuning found reasonable clustering (6-7 clusters)")
    else:
        print("❌ Auto-tuning needs further improvement")
    print()


if __name__ == "__main__":
    asyncio.run(test_auto_clustering())
