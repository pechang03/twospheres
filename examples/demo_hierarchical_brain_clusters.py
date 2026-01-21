"""Demonstration of hierarchical brain model with clustered thinking hubs.

Complete pipeline:
1. Generate clustered synthetic data (make_blobs)
2. Build similarity graph (k-NN)
3. Detect communities (simple method - cluster editing integration pending)
4. Contract clusters to super-nodes
5. Map to sphere with spherical spring layout
6. Visualize with transparent cluster regions

This models different "thinking hubs" in brain architecture.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_hierarchical_brain_graph,
    generate_clustered_brain_nodes,
    build_knn_graph,
    compute_cluster_statistics,
    detect_communities_auto,
    detect_communities_simple,
    evaluate_clustering_quality,
    contract_clusters_to_supernodes,
    compute_cluster_regions
)
from backend.visualization.spherical_spring_layout import (
    spherical_spring_layout_with_scale
)
from backend.visualization.graph_on_sphere import (
    create_sphere_mesh
)


async def main():
    """Run complete hierarchical brain visualization pipeline."""

    print("=" * 70)
    print("Hierarchical Brain Model with Clustered Thinking Hubs")
    print("=" * 70)
    print()

    # Phase 1 & 2: Generate clustered brain graph with tighter clusters
    print("Phase 1-2: Generating clustered brain graph...")
    n_samples = 200
    n_ground_truth_clusters = 10
    k_neighbors = 5
    cluster_std = 0.2  # Tighter clusters for better separation

    print(f"  Using cluster_std={cluster_std} for better separation")

    # Generate tighter clusters
    positions, ground_truth_labels = generate_clustered_brain_nodes(
        n_samples=n_samples,
        n_clusters=n_ground_truth_clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    # Build k-NN graph
    G = build_knn_graph(positions, k=k_neighbors)
    stats = compute_cluster_statistics(ground_truth_labels, positions)

    print(f"  Generated {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Ground truth: {n_ground_truth_clusters} clusters")

    # Analyze graph structure
    inter_cluster_edges = sum(1 for u, v in G.edges() if ground_truth_labels[u] != ground_truth_labels[v])
    print(f"  Inter-cluster edges: {inter_cluster_edges} ({inter_cluster_edges/G.number_of_edges()*100:.1f}%)")
    print()

    # Phase 3: Detect communities using auto-tuning
    print("Phase 3: Detecting communities with auto-tuning...")
    print("  Using cluster-editing-vs with automatic parameter selection")

    communities = await detect_communities_auto(
        G,
        target_clusters=n_ground_truth_clusters,
        method="cluster_editing_vs",
        use_gpu=False
    )

    if communities is None:
        print("  ⚠️  Auto-tuning failed, falling back to connected components")
        communities = detect_communities_simple(G, method="connected_components")

    quality = evaluate_clustering_quality(G, communities)

    # Compute ARI
    from sklearn.metrics import adjusted_rand_score
    gt_labels = [ground_truth_labels[i] for i in sorted(G.nodes())]
    pred_labels = [communities[i] for i in sorted(G.nodes())]
    ari = adjusted_rand_score(gt_labels, pred_labels)

    print(f"  Detected {quality['n_clusters']} communities")
    print(f"  Modularity: {quality['modularity']:.3f}")
    print(f"  Internal edges: {quality['internal_edges']:.1%}")
    print(f"  Avg cluster size: {quality['avg_cluster_size']:.1f} nodes")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    print()

    # Phase 4: Contract clusters
    print("Phase 4: Contracting clusters to super-nodes...")
    G_contracted, cluster_positions_dict, cluster_members = contract_clusters_to_supernodes(
        G, communities, positions
    )

    print(f"  Contracted: {G.number_of_nodes()} -> {G_contracted.number_of_nodes()} super-nodes")
    print(f"  Edges: {G.number_of_edges()} -> {G_contracted.number_of_edges()}")
    print()

    # Compute cluster regions for visualization
    cluster_regions = compute_cluster_regions(positions, communities, expansion_factor=1.5)

    # Phase 5: Map to sphere
    print("Phase 5: Mapping to sphere surface...")
    sphere_radius = 1.5
    sphere_center = np.array([0.0, 0.0, 0.0])

    # Map contracted graph to sphere using spherical spring layout
    contracted_pos_3d_dict = spherical_spring_layout_with_scale(
        G_contracted,
        radius=sphere_radius,
        center=sphere_center,
        iterations=200,
        k=0.15,
        K=0.02,
        d0=0.5,
        eta0=0.15,
        seed=42
    )

    print(f"  Mapped {G_contracted.number_of_nodes()} super-nodes to sphere")
    print()

    # Phase 6: Visualize
    print("Phase 6: Creating visualization...")
    fig = plt.figure(figsize=(16, 12))

    # Left plot: Original clustered data in 3D space
    ax1 = fig.add_subplot(121, projection='3d')
    plot_original_clusters(ax1, positions, communities, cluster_regions, G)

    # Right plot: Hierarchical view on sphere
    ax2 = fig.add_subplot(122, projection='3d')
    plot_sphere_hierarchy(
        ax2, G_contracted, contracted_pos_3d_dict,
        cluster_regions, sphere_center, sphere_radius
    )

    plt.tight_layout()

    # Save
    output_path = 'hierarchical_brain_clusters.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Synthetic brain nodes: {n_samples}")
    print(f"Ground truth clusters: {n_ground_truth_clusters}")
    print(f"Detected communities: {quality['n_clusters']}")
    print(f"Modularity: {quality['modularity']:.3f}")
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Graph connectivity: k={k_neighbors} nearest neighbors")
    print(f"Cluster separation: cluster_std={cluster_std}")
    print()
    print("✅ All phases complete:")
    print("  Phase 1-2: Clustered data generation")
    print("  Phase 3: Auto-tuned cluster-editing-vs")
    print("  Phase 4: Cluster contraction")
    print("  Phase 5-6: Sphere mapping and visualization")
    print()

    if quality['n_clusters'] >= 8:
        print("🎯 Excellent clustering: Found 8+ thinking hubs")
    elif quality['n_clusters'] >= 6:
        print("✅ Good clustering: Found 6-7 major thinking hubs")
    else:
        print("⚠️  Clustering merged many regions")
    print()


def plot_original_clusters(ax, positions, communities, regions, G):
    """Plot original clustered data in 3D space."""
    ax.set_title('Original Clustered Brain Data\n(3D Euclidean Space)', fontsize=12, fontweight='bold')

    # Get unique cluster colors
    unique_clusters = sorted(set(communities.values()))
    cluster_to_color = {cid: regions[cid]['color'] for cid in unique_clusters}

    # Plot nodes colored by cluster
    for node, cluster_id in communities.items():
        pos = positions[node]
        color = cluster_to_color[cluster_id]
        ax.scatter(*pos, c=[color], s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

    # Plot edges (sample to avoid clutter)
    edge_sample = list(G.edges())[:300]  # Sample edges
    for u, v in edge_sample:
        if communities[u] == communities[v]:
            # Internal edge
            pos_u, pos_v = positions[u], positions[v]
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                    c='gray', alpha=0.2, linewidth=0.5, zorder=1)

    # Plot cluster regions as wireframe spheres
    for cluster_id, region in regions.items():
        center = region['center']
        radius = region['radius']
        color = region['color']

        # Create wireframe sphere
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color=color, alpha=0.15, edgecolor=color,
                        linewidth=0.5, zorder=0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal aspect ratio
    max_range = 3.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


def plot_sphere_hierarchy(ax, G_contracted, contracted_pos_3d, regions,
                          center, radius):
    """Plot hierarchical view on sphere surface."""
    ax.set_title('Hierarchical Brain Structure\n(Sphere Surface with Cluster Regions)',
                 fontsize=12, fontweight='bold')

    # Draw brain surface sphere
    X, Y, Z = create_sphere_mesh(center, radius)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.15, edgecolor='gray',
                    linewidth=0.3)

    # Plot super-nodes (cluster centroids)
    for cluster_id in G_contracted.nodes():
        pos = contracted_pos_3d[cluster_id]
        cluster_size = G_contracted.nodes[cluster_id]['size']
        color = regions[cluster_id]['color']

        # Size proportional to cluster size
        node_size = 100 + cluster_size * 2

        ax.scatter(*pos, c=[color], s=node_size, alpha=0.9,
                   edgecolors='black', linewidths=2, zorder=100, marker='o')

    # Plot edges between super-nodes
    for u, v in G_contracted.edges():
        pos_u = np.array(contracted_pos_3d[u])
        pos_v = np.array(contracted_pos_3d[v])
        edge_weight = G_contracted[u][v]['weight']

        # Edge thickness proportional to weight
        linewidth = 1.0 + edge_weight * 0.1

        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                c='darkblue', alpha=0.7, linewidth=linewidth, zorder=50)

    # Draw cluster region indicators
    for cluster_id, region in regions.items():
        if cluster_id in contracted_pos_3d:
            pos_sphere = np.array(contracted_pos_3d[cluster_id])
            color = region['color']

            # Draw small transparent sphere around super-node
            region_radius = 0.2  # Fixed visual size
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 8)
            x = pos_sphere[0] + region_radius * np.outer(np.cos(u), np.sin(v))
            y = pos_sphere[1] + region_radius * np.outer(np.sin(u), np.sin(v))
            z = pos_sphere[2] + region_radius * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color=color, alpha=0.25, edgecolor=color,
                            linewidth=0.5, zorder=80)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = radius * 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
