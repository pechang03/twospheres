"""Debug clustering to understand why we're not finding 10 clusters."""

import sys
from pathlib import Path
import numpy as np
import asyncio

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_hierarchical_brain_graph,
    detect_communities_cluster_editing,
    detect_communities_simple,
    evaluate_clustering_quality
)


async def debug_clustering():
    """Analyze why cluster-editing finds fewer clusters than ground truth."""

    print("=" * 70)
    print("Debugging Cluster Detection")
    print("=" * 70)
    print()

    # Generate graph with 10 ground truth clusters
    n_samples = 100
    n_clusters = 10
    k = 5  # Fewer neighbors

    print(f"Generating {n_samples} nodes in {n_clusters} ground truth clusters...")
    G, positions, ground_truth_labels, stats = generate_hierarchical_brain_graph(
        n_samples=n_samples,
        n_clusters=n_clusters,
        connection_type="knn",
        k=k,
        random_state=42
    )

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()

    # Analyze ground truth cluster structure
    print("Ground Truth Cluster Statistics:")
    for cluster_id, cluster_stats in stats.items():
        print(f"  Cluster {cluster_id}: {cluster_stats['size']} nodes, radius={cluster_stats['radius']:.2f}")
    print()

    # Count inter-cluster edges
    inter_cluster_edges = 0
    intra_cluster_edges = 0

    for u, v in G.edges():
        if ground_truth_labels[u] == ground_truth_labels[v]:
            intra_cluster_edges += 1
        else:
            inter_cluster_edges += 1

    print("Edge Analysis:")
    print(f"  Intra-cluster edges: {intra_cluster_edges} ({intra_cluster_edges/G.number_of_edges()*100:.1f}%)")
    print(f"  Inter-cluster edges: {inter_cluster_edges} ({inter_cluster_edges/G.number_of_edges()*100:.1f}%)")
    print(f"  Budget needed: ~{inter_cluster_edges} edge removals")
    print()

    # Test different k values for cluster-editing
    print("Testing cluster-editing-vs with different budgets:")
    print()

    for k_budget in [10, 50, 100, 200]:
        print(f"k={k_budget}:")
        communities = await detect_communities_cluster_editing(
            G,
            method="cluster_editing_vs",
            k=k_budget,
            use_gpu=False
        )

        if communities:
            quality = evaluate_clustering_quality(G, communities)
            print(f"  Found {quality['n_clusters']} clusters, modularity={quality['modularity']:.3f}")

            # Check agreement with ground truth
            from sklearn.metrics import adjusted_rand_score
            gt_labels = [ground_truth_labels[i] for i in sorted(G.nodes())]
            pred_labels = [communities[i] for i in sorted(G.nodes())]
            ari = adjusted_rand_score(gt_labels, pred_labels)
            print(f"  Adjusted Rand Index vs ground truth: {ari:.3f}")
        else:
            print(f"  ❌ Failed")
        print()

    # Compare with baseline
    print("Baseline (connected components):")
    communities_simple = detect_communities_simple(G, method="connected_components")
    quality_simple = evaluate_clustering_quality(G, communities_simple)
    print(f"  Found {quality_simple['n_clusters']} clusters, modularity={quality_simple['modularity']:.3f}")
    print()

    print("=" * 70)
    print("Analysis:")
    print("=" * 70)
    print()
    print(f"To separate {n_clusters} clusters, we need to remove ~{inter_cluster_edges} inter-cluster edges")
    print(f"This requires k >= {inter_cluster_edges}")
    print()
    print("Solutions:")
    print("  1. Increase k budget for cluster-editing")
    print("  2. Reduce k in k-NN to create fewer inter-cluster edges")
    print("  3. Increase cluster separation (adjust cluster_std in make_blobs)")
    print("  4. Use threshold-based graph instead of k-NN")
    print()


if __name__ == "__main__":
    asyncio.run(debug_clustering())
