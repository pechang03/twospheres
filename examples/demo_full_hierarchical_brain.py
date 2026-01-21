"""Complete Hierarchical Brain Model with Backbone and Atlas Integration.

Three-level hierarchy:
1. Level 1: Backbone hubs (Connected Dominating Set)
2. Level 2: Communities (Cluster-editing-vs)
3. Level 3: Individual regions

Optionally uses real brain atlas data (D99 macaque, Allen mouse, etc.)
"""

import sys
from pathlib import Path
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    generate_clustered_brain_nodes,
    build_knn_graph,
    compute_cluster_statistics,
    detect_communities_auto,
    evaluate_clustering_quality,
    contract_clusters_to_supernodes,
    compute_cluster_regions
)
from backend.visualization.spherical_spring_layout import (
    spherical_spring_layout_with_scale
)
from backend.visualization.graph_on_sphere import create_sphere_mesh


async def compute_backbone_hubs(G, r=2):
    """Compute backbone hubs using Connected Dominating Set.

    Args:
        G: NetworkX graph
        r: Domination radius

    Returns:
        Set of backbone hub nodes
    """
    try:
        from backend.integration.merge2docs_bridge import call_algorithm_service

        print(f"  Computing CDS backbone (r={r})...")

        # Call struction_rids algorithm for r-IDS + CDS backbone
        graph_data = {
            'nodes': list(G.nodes()),
            'edges': list(G.edges())
        }

        result = await call_algorithm_service(
            algorithm_name="struction_rids",
            graph_data=graph_data,
            r=r,
            use_gpu=False,
            use_struction=True,
            compute_backbone=True
        )

        if result and hasattr(result, 'result_data'):
            data = result.result_data
            if 'backbone' in data:
                backbone = set(data['backbone'])
                print(f"    Found {len(backbone)} backbone hubs")
                return backbone

        print("    Backbone computation unavailable, using degree centrality")
        return _fallback_backbone_hubs(G)

    except Exception as e:
        print(f"    Error computing backbone: {e}")
        return _fallback_backbone_hubs(G)


def _fallback_backbone_hubs(G, fraction=0.15):
    """Fallback: select high-degree nodes as backbone.

    Args:
        G: NetworkX graph
        fraction: Fraction of nodes to select as hubs

    Returns:
        Set of backbone hub nodes
    """
    import networkx as nx

    # Use betweenness centrality (nodes that connect communities)
    centrality = nx.betweenness_centrality(G)
    n_hubs = max(3, int(len(G.nodes()) * fraction))

    # Select top-k by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    backbone = {node for node, _ in sorted_nodes[:n_hubs]}

    print(f"    Selected {len(backbone)} hubs (top {fraction:.0%} by betweenness centrality)")
    return backbone


async def main():
    """Run complete 3-level hierarchical brain model."""

    print("=" * 70)
    print("Complete Hierarchical Brain Model")
    print("Level 1: Backbone Hubs (CDS) → Level 2: Communities → Level 3: Nodes")
    print("=" * 70)
    print()

    # Configuration
    n_samples = 200
    n_ground_truth_clusters = 10
    k_neighbors = 5
    cluster_std = 0.2
    r_backbone = 2  # Backbone domination radius

    # Phase 1: Generate brain graph
    print("Phase 1: Generating brain graph...")
    print(f"  {n_samples} nodes, {n_ground_truth_clusters} ground truth clusters")
    print(f"  cluster_std={cluster_std}, k={k_neighbors}")

    positions, ground_truth_labels = generate_clustered_brain_nodes(
        n_samples=n_samples,
        n_clusters=n_ground_truth_clusters,
        cluster_std=cluster_std,
        random_state=42
    )

    G = build_knn_graph(positions, k=k_neighbors)
    stats = compute_cluster_statistics(ground_truth_labels, positions)

    inter_cluster_edges = sum(1 for u, v in G.edges() if ground_truth_labels[u] != ground_truth_labels[v])
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Inter-cluster edges: {inter_cluster_edges} ({inter_cluster_edges/G.number_of_edges()*100:.1f}%)")
    print()

    # Phase 2: Detect communities (Level 2)
    print("Phase 2: Detecting communities (Level 2)...")
    communities = await detect_communities_auto(
        G,
        target_clusters=n_ground_truth_clusters,
        method="cluster_editing_vs",
        use_gpu=False
    )

    quality = evaluate_clustering_quality(G, communities)
    from sklearn.metrics import adjusted_rand_score
    gt_labels = [ground_truth_labels[i] for i in sorted(G.nodes())]
    pred_labels = [communities[i] for i in sorted(G.nodes())]
    ari = adjusted_rand_score(gt_labels, pred_labels)

    print(f"  Detected {quality['n_clusters']} communities")
    print(f"  Modularity: {quality['modularity']:.3f}, ARI: {ari:.3f}")
    print()

    # Phase 3: Compute backbone hubs (Level 1)
    print("Phase 3: Computing backbone hubs (Level 1)...")
    backbone_nodes = await compute_backbone_hubs(G, r=r_backbone)

    # Analyze backbone
    backbone_communities = set(communities[node] for node in backbone_nodes)
    print(f"  Backbone: {len(backbone_nodes)} nodes spanning {len(backbone_communities)} communities")
    print()

    # Phase 4: Build hierarchy
    print("Phase 4: Building 3-level hierarchy...")

    # Level 3 → Level 2: Contract communities
    G_communities, cluster_pos, cluster_members = contract_clusters_to_supernodes(
        G, communities, positions
    )

    # Level 2 → Level 1: Extract backbone super-nodes
    backbone_community_ids = {communities[node] for node in backbone_nodes}
    G_backbone = G_communities.subgraph(backbone_community_ids).copy()

    print(f"  Level 3: {G.number_of_nodes()} individual nodes")
    print(f"  Level 2: {G_communities.number_of_nodes()} community super-nodes")
    print(f"  Level 1: {G_backbone.number_of_nodes()} backbone hubs")
    print()

    # Phase 5: Visualize on sphere
    print("Phase 5: Mapping to sphere and visualizing...")

    # Map Level 2 (communities) to sphere
    sphere_radius = 1.5
    sphere_center = np.array([0.0, 0.0, 0.0])

    pos_sphere = spherical_spring_layout_with_scale(
        G_communities,
        radius=sphere_radius,
        center=sphere_center,
        iterations=200,
        k=0.15,
        K=0.02,
        d0=0.5,
        eta0=0.15,
        seed=42
    )

    # Compute cluster regions for visualization
    cluster_regions = compute_cluster_regions(positions, communities, expansion_factor=1.5)

    # Create visualization
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Full hierarchy in 3D space
    ax1 = fig.add_subplot(131, projection='3d')
    plot_3d_hierarchy(ax1, G, positions, communities, backbone_nodes, cluster_regions)

    # Plot 2: Sphere with communities (Level 2)
    ax2 = fig.add_subplot(132, projection='3d')
    plot_sphere_communities(ax2, G_communities, pos_sphere, backbone_community_ids,
                           cluster_regions, sphere_center, sphere_radius)

    # Plot 3: Backbone network (Level 1)
    ax3 = fig.add_subplot(133, projection='3d')
    plot_backbone_network(ax3, G_backbone, pos_sphere, cluster_regions,
                         sphere_center, sphere_radius)

    plt.tight_layout()

    output_path = 'full_hierarchical_brain.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("Hierarchical Brain Model Summary")
    print("=" * 70)
    print(f"Level 1 (Backbone Hubs): {len(backbone_nodes)} nodes")
    print(f"  - Connected dominating set (r={r_backbone})")
    print(f"  - Span {len(backbone_communities)} communities")
    print(f"Level 2 (Communities): {quality['n_clusters']} clusters")
    print(f"  - Modularity: {quality['modularity']:.3f}")
    print(f"  - Adjusted Rand Index: {ari:.3f}")
    print(f"Level 3 (Individual Nodes): {n_samples} nodes")
    print(f"  - Organized into communities via cluster-editing-vs")
    print()
    print("🎯 Multi-scale brain representation complete!")
    print()


def plot_3d_hierarchy(ax, G, positions, communities, backbone_nodes, regions):
    """Plot full 3-level hierarchy in 3D space."""
    ax.set_title('3-Level Hierarchy\n(3D Space)', fontsize=12, fontweight='bold')

    # Plot community regions (Level 2)
    for cluster_id, region in regions.items():
        center = region['center']
        radius = region['radius']
        color = region['color']

        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color=color, alpha=0.1, edgecolor=color,
                       linewidth=0.5, zorder=1)

    # Plot nodes (Level 3)
    for node, pos in enumerate(positions):
        cluster_id = communities[node]
        color = regions[cluster_id]['color']

        if node in backbone_nodes:
            # Backbone hub
            ax.scatter(*pos, c='gold', s=150, marker='*', edgecolors='black',
                      linewidths=2, zorder=100, alpha=0.9)
        else:
            # Regular node
            ax.scatter(*pos, c=[color], s=20, alpha=0.6, edgecolors='black',
                      linewidths=0.3, zorder=50)

    # Plot backbone connections
    for u in backbone_nodes:
        for v in backbone_nodes:
            if u < v and G.has_edge(u, v):
                pos_u, pos_v = positions[u], positions[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                       c='gold', alpha=0.8, linewidth=2, zorder=90)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = 3.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


def plot_sphere_communities(ax, G_communities, pos_sphere, backbone_ids, regions,
                            center, radius):
    """Plot Level 2 (communities) on sphere."""
    ax.set_title('Level 2: Communities\n(Sphere Surface)', fontsize=12, fontweight='bold')

    # Draw sphere
    X, Y, Z = create_sphere_mesh(center, radius)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.15, edgecolor='gray',
                   linewidth=0.3)

    # Plot community super-nodes
    for cluster_id in G_communities.nodes():
        pos = pos_sphere[cluster_id]
        cluster_size = G_communities.nodes[cluster_id]['size']
        color = regions[cluster_id]['color']

        is_backbone = cluster_id in backbone_ids
        node_size = 200 if is_backbone else 100 + cluster_size

        marker = '*' if is_backbone else 'o'
        edge_color = 'gold' if is_backbone else 'black'
        edge_width = 3 if is_backbone else 1

        ax.scatter(*pos, c=[color], s=node_size, marker=marker,
                  edgecolors=edge_color, linewidths=edge_width,
                  zorder=100, alpha=0.9)

    # Plot edges between communities
    for u, v in G_communities.edges():
        pos_u = np.array(pos_sphere[u])
        pos_v = np.array(pos_sphere[v])
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
               c='darkblue', alpha=0.5, linewidth=1, zorder=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = radius * 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


def plot_backbone_network(ax, G_backbone, pos_sphere, regions, center, radius):
    """Plot Level 1 (backbone) network."""
    ax.set_title('Level 1: Backbone Hubs\n(Major Connectors)', fontsize=12, fontweight='bold')

    # Draw sphere
    X, Y, Z = create_sphere_mesh(center, radius)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.1, edgecolor='gray',
                   linewidth=0.2)

    # Plot backbone nodes
    for node in G_backbone.nodes():
        pos = pos_sphere[node]
        color = regions[node]['color']

        ax.scatter(*pos, c=[color], s=300, marker='*',
                  edgecolors='gold', linewidths=3,
                  zorder=100, alpha=1.0)

    # Plot backbone edges
    for u, v in G_backbone.edges():
        pos_u = np.array(pos_sphere[u])
        pos_v = np.array(pos_sphere[v])
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
               c='gold', alpha=0.9, linewidth=3, zorder=90)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = radius * 2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


if __name__ == "__main__":
    asyncio.run(main())
