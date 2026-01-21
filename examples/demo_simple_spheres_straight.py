"""Simple two-sphere visualization with STRAIGHT edges.

This is the most basic version - just draws graphs on spheres
with straight line edges through space (not geodesic).
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.graph_on_sphere import (
    SphereGraphConfig,
    generate_graph,
    ensure_graph_connectivity,
    visualize_two_spheres_with_graphs
)

# Temporarily disable geodesic routing for this demo
import backend.visualization.graph_on_sphere as sphere_module


def visualize_two_spheres_straight_edges(
    config,
    G1=None,
    G2=None,
    inter_sphere_edges=None,
    save_path=None
):
    """Visualize with straight edges (original behavior)."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate graphs if not provided
    if G1 is None:
        G1 = generate_graph(config.graph_type, config.n_nodes, seed=42)
    if G2 is None:
        G2 = generate_graph(config.graph_type, config.n_nodes, seed=43)

    if config.ensure_connected:
        G1 = ensure_graph_connectivity(G1)
        G2 = ensure_graph_connectivity(G2)

    # Map graphs to spheres
    from backend.visualization.graph_on_sphere import (
        map_graph_to_sphere,
        create_sphere_mesh,
        find_hub_nodes,
        rotate_graph_to_place_node_at_pole
    )

    # Find and position hubs
    hubs1 = find_hub_nodes(G1, min_degree=5)
    hubs2 = find_hub_nodes(G2, min_degree=5)

    if hubs1 and hubs2:
        rotate_graph_to_place_node_at_pole(G1, hubs1[0], pole="south")
        rotate_graph_to_place_node_at_pole(G2, hubs2[0], pole="north")
        inter_sphere_edges = [(hubs1[0], hubs2[0])]
        hub1_idx, hub2_idx = hubs1[0], hubs2[0]
    else:
        hub1_idx, hub2_idx = None, None

    pos1_3d, edges1 = map_graph_to_sphere(G1, config, config.center1, config.radius1)
    pos2_3d, edges2 = map_graph_to_sphere(G2, config, config.center2, config.radius2)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw spheres
    X1, Y1, Z1 = create_sphere_mesh(config.center1, config.radius1)
    X2, Y2, Z2 = create_sphere_mesh(config.center2, config.radius2)

    ax.plot_surface(X1, Y1, Z1, color=config.sphere1_color, alpha=config.sphere_alpha)
    ax.plot_surface(X2, Y2, Z2, color=config.sphere2_color, alpha=config.sphere_alpha)

    # Draw nodes and STRAIGHT edges on sphere 1
    for i, (x, y, z) in enumerate(pos1_3d):
        if i == hub1_idx:
            ax.scatter(x, y, z, c='gold', s=150, marker='*', edgecolors='black', linewidths=2, zorder=100)
        else:
            ax.scatter(x, y, z, c=config.node_color, s=30, marker='o', edgecolors='black', linewidths=0.5)

    for u, v in edges1:
        p1 = pos1_3d[u]
        p2 = pos1_3d[v]
        # STRAIGHT LINE (original behavior)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c=config.edge_color, alpha=0.6, linewidth=1)

    # Draw nodes and STRAIGHT edges on sphere 2
    for i, (x, y, z) in enumerate(pos2_3d):
        if i == hub2_idx:
            ax.scatter(x, y, z, c='gold', s=150, marker='*', edgecolors='black', linewidths=2, zorder=100)
        else:
            ax.scatter(x, y, z, c=config.node_color, s=30, marker='o', edgecolors='black', linewidths=0.5)

    for u, v in edges2:
        p1 = pos2_3d[u]
        p2 = pos2_3d[v]
        # STRAIGHT LINE
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c=config.edge_color, alpha=0.6, linewidth=1)

    # Corpus callosum
    if config.show_inter_sphere_edges and inter_sphere_edges:
        for u, v in inter_sphere_edges:
            if u < len(pos1_3d) and v < len(pos2_3d):
                p1 = pos1_3d[u]
                p2 = pos2_3d[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        c='gold', alpha=0.9, linewidth=4, zorder=99)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = config.radius * 3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_title(
        f'Two-Sphere Brain Architecture - STRAIGHT EDGES\n({config.graph_type})',
        fontsize=14, fontweight='bold'
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def main():
    """Generate simple two-sphere visualization with straight edges."""

    print("=" * 70)
    print("Simple Two-Sphere Visualization - STRAIGHT EDGES")
    print("=" * 70)
    print()

    config = SphereGraphConfig(
        radius1=1.2,
        radius2=1.2,
        graph_type="erdos_renyi",
        n_nodes=60,
        rotation_x_deg=30.0,
        rotation_y_deg=45.0,
        edge_color="blue",
        node_color="red",
        sphere1_color="cyan",
        sphere2_color="magenta",
        sphere_alpha=0.25,
        show_inter_sphere_edges=True,
        ensure_connected=True
    )

    print(f"Configuration:")
    print(f"  Graph type: {config.graph_type}")
    print(f"  Nodes: {config.n_nodes} per hemisphere")
    print(f"  Radii: {config.radius1}, {config.radius2}")
    print(f"  Edge rendering: STRAIGHT LINES (through space)")
    print()

    fig = visualize_two_spheres_straight_edges(
        config,
        save_path="simple_spheres_straight.png"
    )

    print("✅ Saved: simple_spheres_straight.png")
    print()
    print("NOTE: Edges are straight lines (not geodesic).")
    print("      Some edges will appear to pass through sphere volume.")
    print()


if __name__ == "__main__":
    main()
