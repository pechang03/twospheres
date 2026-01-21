"""Phase 7: Brain Atlas Integration - Real anatomical regions.

Replaces synthetic make_blobs with real brain regions from D99 macaque atlas.

Pipeline:
1. Query D99 atlas for cortical regions
2. Build anatomical connectivity graph from neighbors
3. Apply hierarchical clustering (cluster-editing-vs)
4. Compute backbone hubs (CDS)
5. Map to sphere and visualize

This creates the syntactic (anatomical) hierarchy for the YADA dual structure.
"""

import sys
from pathlib import Path
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import networkx as nx

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.hierarchical_brain_model import (
    detect_communities_auto,
    evaluate_clustering_quality,
    contract_clusters_to_supernodes,
    compute_cluster_regions
)
from backend.visualization.spherical_spring_layout import (
    spherical_spring_layout_with_scale
)
from backend.visualization.graph_on_sphere import create_sphere_mesh


# Brain atlas API configuration
ATLAS_API = "http://localhost:8007"


class BrainAtlasClient:
    """Client for brain atlas HTTP API."""

    def __init__(self, base_url: str = ATLAS_API):
        self.base_url = base_url

    def list_regions(self, species: str = "macaque", atlas: str = "D99",
                    filter_type: str = None, limit: int = None):
        """List brain regions."""
        payload = {"species": species, "atlas": atlas}
        if filter_type:
            payload["filter_type"] = filter_type

        response = requests.post(f"{self.base_url}/api/list_regions", json=payload)
        response.raise_for_status()
        data = response.json()

        regions = data["regions"]
        if limit:
            regions = regions[:limit]

        return regions

    def get_neighbors(self, region_id: int, species: str = "macaque", atlas: str = "D99"):
        """Get neighboring regions."""
        payload = {
            "species": species,
            "atlas": atlas,
            "region": str(region_id)
        }

        response = requests.post(f"{self.base_url}/api/get_neighbors", json=payload)
        if response.status_code != 200:
            return []

        data = response.json()
        return data.get("neighbors", [])


def build_anatomical_graph(regions, atlas_client, connectivity_threshold=0.3):
    """Build graph from anatomical regions and their connectivity.

    Args:
        regions: List of region dicts from atlas
        atlas_client: BrainAtlasClient instance
        connectivity_threshold: Fraction of regions to connect randomly (fallback)

    Returns:
        G: NetworkX graph with anatomical connectivity
        positions: 3D positions (synthesized for visualization)
    """
    G = nx.Graph()

    # Add nodes
    for region in regions:
        G.add_node(
            region["id"],
            name=region["name"],
            abbreviation=region["abbreviation"],
            region_type=region.get("region_type", "unknown"),
            hemisphere=region.get("hemisphere", "unknown")
        )

    # Build connectivity from neighbors
    print(f"  Querying neighbor relationships for {len(regions)} regions...")
    edge_count = 0

    for i, region in enumerate(regions):
        if i % 20 == 0:
            print(f"    Progress: {i}/{len(regions)} regions...")

        # Get neighbors from atlas
        neighbors = atlas_client.get_neighbors(region["id"])

        for neighbor in neighbors:
            neighbor_id = neighbor["id"]
            if neighbor_id in G.nodes():
                G.add_edge(region["id"], neighbor_id)
                edge_count += 1

    print(f"  Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # If graph is too sparse, add k-NN connections based on synthetic positions
    if G.number_of_edges() < G.number_of_nodes():
        print(f"  Graph too sparse, adding k-NN edges for visualization...")
        positions = synthesize_positions(G, regions)
        add_knn_edges(G, positions, k=3)
        print(f"  Enhanced graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        positions = synthesize_positions(G, regions)

    return G, positions


def synthesize_positions(G, regions, spread=2.0):
    """Generate 3D positions for regions based on hemisphere and ordering.

    This creates a spatial layout that reflects anatomical organization.
    """
    positions = {}

    # Separate by hemisphere
    left_regions = [r for r in regions if r.get("hemisphere") in ["left", "L"]]
    right_regions = [r for r in regions if r.get("hemisphere") in ["right", "R"]]
    bilateral_regions = [r for r in regions if r.get("hemisphere") in ["bilateral", "both"]]

    # Position left hemisphere (negative X)
    for i, region in enumerate(left_regions):
        angle = 2 * np.pi * i / max(len(left_regions), 1)
        positions[region["id"]] = np.array([
            -spread + spread * 0.3 * np.cos(angle),
            spread * np.sin(angle),
            spread * 0.5 * np.random.randn()
        ])

    # Position right hemisphere (positive X)
    for i, region in enumerate(right_regions):
        angle = 2 * np.pi * i / max(len(right_regions), 1)
        positions[region["id"]] = np.array([
            spread - spread * 0.3 * np.cos(angle),
            spread * np.sin(angle),
            spread * 0.5 * np.random.randn()
        ])

    # Position bilateral regions (near midline)
    for i, region in enumerate(bilateral_regions):
        angle = 2 * np.pi * i / max(len(bilateral_regions), 1)
        positions[region["id"]] = np.array([
            0.2 * spread * np.random.randn(),
            spread * np.sin(angle),
            spread * np.cos(angle)
        ])

    return positions


def add_knn_edges(G, positions, k=3):
    """Add k-nearest neighbor edges to ensure connectivity."""
    from scipy.spatial.distance import cdist

    nodes = list(G.nodes())
    pos_array = np.array([positions[n] for n in nodes])

    # Compute distances
    dist_matrix = cdist(pos_array, pos_array, metric='euclidean')

    # Add k-NN edges
    for i, node in enumerate(nodes):
        # Get k nearest neighbors (excluding self)
        neighbors_idx = np.argsort(dist_matrix[i])[1:k+1]
        for j in neighbors_idx:
            neighbor = nodes[j]
            G.add_edge(node, neighbor)


async def compute_backbone_hubs(G, r=4):
    """Compute backbone hubs using r-IDS (radius-r Independent Dominating Set).

    Args:
        G: NetworkX graph
        r: Coverage radius (default 4, optimal for brain LID≈4-7)

    Returns:
        backbone: Set of backbone hub nodes
    """
    print(f"  Computing r-IDS backbone (r={r})...")

    try:
        from backend.integration.merge2docs_bridge import call_algorithm_service

        # Call merge2docs 'ids' algorithm with r=4
        result = await call_algorithm_service(
            algorithm_name="ids",
            graph_data=G,
            r=r,
            use_gpu=False
        )

        if result and hasattr(result, 'result_data'):
            data = result.result_data

            # Try multiple possible keys for independent set
            for key in ['independent_set', 'dominating_set', 'ids', 'result']:
                if data and key in data:
                    backbone = set(data[key])
                    print(f"    Found {len(backbone)} r-IDS hubs (r={r})")
                    return backbone

        # If result exists but wrong format, try direct access
        if result and hasattr(result, 'independent_set'):
            backbone = set(result.independent_set)
            print(f"    Found {len(backbone)} r-IDS hubs (r={r})")
            return backbone

        print("    r-IDS computation unavailable, falling back to betweenness centrality")
        print(f"    (Result type: {type(result)}, has result_data: {hasattr(result, 'result_data') if result else False})")
        return await compute_backbone_hubs_fallback(G, fraction=0.10)

    except Exception as e:
        print(f"    Error computing r-IDS: {e}")
        return await compute_backbone_hubs_fallback(G, fraction=0.10)


async def compute_backbone_hubs_fallback(G, fraction=0.10):
    """Fallback: Betweenness centrality (use if r-IDS service unavailable).

    Args:
        G: NetworkX graph
        fraction: Fraction of nodes to select as hubs

    Returns:
        backbone: Set of backbone hub nodes
    """
    print(f"    Using fallback: top {fraction:.0%} by betweenness centrality")

    centrality = nx.betweenness_centrality(G)
    n_hubs = max(3, int(len(G.nodes()) * fraction))

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    backbone = {node for node, _ in sorted_nodes[:n_hubs]}

    print(f"    Selected {len(backbone)} hubs")
    return backbone


async def main():
    """Run Phase 7: Brain Atlas Integration."""

    print("=" * 70)
    print("Phase 7: Brain Atlas Integration")
    print("Real anatomical regions from D99 macaque atlas")
    print("=" * 70)
    print()

    # Configuration
    species = "macaque"
    atlas_name = "D99"
    region_limit = 100  # Use subset for demo (full atlas has 368 regions)
    target_communities = 10  # Expected functional systems

    # Phase 7.1: Query brain atlas
    print("Phase 7.1: Querying D99 macaque atlas...")
    atlas_client = BrainAtlasClient()

    regions = atlas_client.list_regions(
        species=species,
        atlas=atlas_name,
        filter_type="cortical",
        limit=region_limit
    )

    print(f"  Retrieved {len(regions)} cortical regions")
    print(f"  Sample regions: {', '.join([r['abbreviation'] for r in regions[:5]])}...")
    print()

    # Phase 7.2: Build anatomical connectivity graph
    print("Phase 7.2: Building anatomical connectivity graph...")
    G, positions = build_anatomical_graph(regions, atlas_client)
    print()

    # Phase 7.3: Detect functional communities (cluster-editing-vs)
    print("Phase 7.3: Detecting functional communities...")
    print(f"  Target: {target_communities} communities")

    communities = await detect_communities_auto(
        G,
        target_clusters=target_communities,
        method="cluster_editing_vs",
        use_gpu=False
    )

    if communities is None:
        print("  ⚠️  Clustering failed, using connected components")
        communities = {node: i for i, comp in enumerate(nx.connected_components(G))
                      for node in comp}

    quality = evaluate_clustering_quality(G, communities)
    print(f"  Detected {quality['n_clusters']} functional communities")
    print(f"  Modularity: {quality['modularity']:.3f}")
    print()

    # Phase 7.4: Compute backbone hubs
    print("Phase 7.4: Computing backbone hubs...")
    backbone_nodes = await compute_backbone_hubs(G, r=4)  # Use r=4 (optimal for brain LID≈4-7)
    backbone_communities = {communities[node] for node in backbone_nodes}
    print(f"  Backbone: {len(backbone_nodes)} hubs spanning {len(backbone_communities)} communities")
    print()

    # Phase 7.5: Build hierarchy
    print("Phase 7.5: Building 3-level hierarchy...")

    # Convert positions dict to array indexed by node ID (not sequential)
    # contract_clusters_to_supernodes expects positions[node_id] to work
    max_node_id = max(G.nodes())
    pos_array = np.zeros((max_node_id + 1, 3))
    for node, pos in positions.items():
        pos_array[node] = pos

    # Contract communities
    G_communities, cluster_pos, cluster_members = contract_clusters_to_supernodes(
        G, communities, pos_array
    )

    # Extract backbone super-nodes
    backbone_community_ids = {communities[node] for node in backbone_nodes}
    G_backbone = G_communities.subgraph(backbone_community_ids).copy()

    print(f"  Level 3: {G.number_of_nodes()} anatomical regions")
    print(f"  Level 2: {G_communities.number_of_nodes()} functional communities")
    print(f"  Level 1: {G_backbone.number_of_nodes()} backbone hubs")
    print()

    # Phase 7.6: Map to sphere and visualize
    print("Phase 7.6: Mapping to sphere and visualizing...")

    sphere_radius = 1.5
    sphere_center = np.array([0.0, 0.0, 0.0])

    # Map communities to sphere
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

    # Compute cluster regions
    cluster_regions = compute_cluster_regions(pos_array, communities, expansion_factor=1.5)

    # Create visualization
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Anatomical graph in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    plot_anatomical_graph(ax1, G, positions, communities, regions, cluster_regions)

    # Plot 2: Functional communities on sphere
    ax2 = fig.add_subplot(132, projection='3d')
    plot_functional_communities(ax2, G_communities, pos_sphere, backbone_community_ids,
                                cluster_regions, sphere_center, sphere_radius)

    # Plot 3: Backbone network
    ax3 = fig.add_subplot(133, projection='3d')
    plot_backbone_network(ax3, G_backbone, pos_sphere, cluster_regions,
                         sphere_center, sphere_radius)

    plt.tight_layout()

    output_path = 'atlas_hierarchical_brain.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("Brain Atlas Integration Summary")
    print("=" * 70)
    print(f"Atlas: {atlas_name} {species}")
    print(f"Anatomical regions: {len(regions)} cortical")
    print()
    print(f"Level 1 (Backbone): {len(backbone_nodes)} hubs")
    print(f"  - Span {len(backbone_communities)} communities")
    print(f"Level 2 (Communities): {quality['n_clusters']} functional systems")
    print(f"  - Modularity: {quality['modularity']:.3f}")
    print(f"Level 3 (Regions): {len(regions)} anatomical")
    print()
    print("🎯 Syntactic (anatomical) hierarchy complete!")
    print("   Next: Phase 8 - Semantic (functional) hierarchy from activity patterns")
    print()


def plot_anatomical_graph(ax, G, positions, communities, regions, cluster_regions):
    """Plot anatomical graph in 3D space."""
    ax.set_title('D99 Anatomical Regions\n(Syntactic Hierarchy)',
                 fontsize=12, fontweight='bold')

    # Create region lookup
    region_map = {r["id"]: r for r in regions}

    # Plot nodes
    for node in G.nodes():
        pos = positions[node]
        cluster_id = communities[node]
        color = cluster_regions[cluster_id]['color']

        region = region_map.get(node, {})
        size = 30 if region.get("region_type") == "cortical" else 20

        ax.scatter(*pos, c=[color], s=size, alpha=0.7,
                  edgecolors='black', linewidths=0.5, zorder=50)

    # Plot sample edges
    edge_sample = list(G.edges())[:200]
    for u, v in edge_sample:
        if u in positions and v in positions:
            pos_u, pos_v = positions[u], positions[v]
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]],
                   c='gray', alpha=0.2, linewidth=0.5, zorder=1)

    ax.set_xlabel('X (L-R)')
    ax.set_ylabel('Y (A-P)')
    ax.set_zlabel('Z (I-S)')

    max_range = 3.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


def plot_functional_communities(ax, G_communities, pos_sphere, backbone_ids,
                                cluster_regions, center, radius):
    """Plot functional communities on sphere."""
    ax.set_title('Functional Communities\n(Level 2)', fontsize=12, fontweight='bold')

    # Draw sphere
    X, Y, Z = create_sphere_mesh(center, radius)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.15, edgecolor='gray',
                   linewidth=0.3)

    # Plot community super-nodes
    for cluster_id in G_communities.nodes():
        pos = pos_sphere[cluster_id]
        cluster_size = G_communities.nodes[cluster_id]['size']
        color = cluster_regions[cluster_id]['color']

        is_backbone = cluster_id in backbone_ids
        node_size = 200 if is_backbone else 100 + cluster_size
        marker = '*' if is_backbone else 'o'
        edge_color = 'gold' if is_backbone else 'black'
        edge_width = 3 if is_backbone else 1

        ax.scatter(*pos, c=[color], s=node_size, marker=marker,
                  edgecolors=edge_color, linewidths=edge_width,
                  zorder=100, alpha=0.9)

    # Plot edges
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


def plot_backbone_network(ax, G_backbone, pos_sphere, cluster_regions, center, radius):
    """Plot backbone network."""
    ax.set_title('Backbone Hubs\n(Level 1)', fontsize=12, fontweight='bold')

    # Draw sphere
    X, Y, Z = create_sphere_mesh(center, radius)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.1, edgecolor='gray',
                   linewidth=0.2)

    # Plot backbone nodes
    for node in G_backbone.nodes():
        pos = pos_sphere[node]
        color = cluster_regions[node]['color']

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
