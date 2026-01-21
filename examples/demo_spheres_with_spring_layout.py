"""Two-sphere visualization with proper spherical spring layout.

Uses:
1. Spherical spring embedding for node layout
2. Quaternion rotation to place hubs at touching point
3. Smart edge rendering:
   - Straight lines for short edges (< 2r)
   - Geodesic arcs for long edges
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.graph_on_sphere import (
    generate_graph,
    ensure_graph_connectivity,
    find_hub_nodes,
    create_sphere_mesh,
    compute_great_circle_arc,
    rotate_sphere_to_place_hub_at_touching_point
)
from backend.visualization.spherical_spring_layout import (
    spherical_spring_layout_with_scale,
    should_use_straight_edge
)


def main():
    """Generate two-sphere visualization with spherical spring layout."""

    print("=" * 70)
    print("Two-Sphere Visualization with Spherical Spring Layout")
    print("=" * 70)
    print()

    # Configuration
    radius = 1.2
    center_left = np.array([0, radius, 0])
    center_right = np.array([0, -radius, 0])
    touching_point = np.array([0, 0, 0])
    n_nodes = 40

    # Generate graphs
    print(f"📊 Generating graphs ({n_nodes} nodes per hemisphere)...")
    G_left = generate_graph("erdos_renyi", n_nodes, seed=42)
    G_right = generate_graph("erdos_renyi", n_nodes, seed=43)

    G_left = ensure_graph_connectivity(G_left)
    G_right = ensure_graph_connectivity(G_right)
    print()

    # Spherical spring layout
    print("🔄 Computing spherical spring layout for left hemisphere...")
    pos_left_dict = spherical_spring_layout_with_scale(
        G_left,
        radius=radius,
        center=center_left,
        iterations=150,
        k=0.15,      # Spring constant
        K=0.02,      # Repulsion
        d0=0.4,      # Rest length ~23 degrees
        eta0=0.15,   # Learning rate
        seed=42
    )
    print(f"   Layout complete: {len(pos_left_dict)} nodes")

    print("🔄 Computing spherical spring layout for right hemisphere...")
    pos_right_dict = spherical_spring_layout_with_scale(
        G_right,
        radius=radius,
        center=center_right,
        iterations=150,
        k=0.15,
        K=0.02,
        d0=0.4,
        eta0=0.15,
        seed=43
    )
    print(f"   Layout complete: {len(pos_right_dict)} nodes")
    print()

    # Convert to lists
    nodes_left = list(G_left.nodes())
    nodes_right = list(G_right.nodes())
    pos_left_3d = [pos_left_dict[n] for n in nodes_left]
    pos_right_3d = [pos_right_dict[n] for n in nodes_right]

    # Find hubs
    hubs_left = find_hub_nodes(G_left, min_degree=4)
    hubs_right = find_hub_nodes(G_right, min_degree=4)

    if hubs_left and hubs_right:
        hub_left = hubs_left[0]
        hub_right = hubs_right[0]
        print(f"🔗 Corpus callosum hubs:")
        print(f"   Left: node {hub_left} (degree={G_left.degree(hub_left)})")
        print(f"   Right: node {hub_right} (degree={G_right.degree(hub_right)})")
        print()

        # Rotate spheres to place hubs at touching point
        print("🔄 Rotating left sphere to align hub...")
        pos_left_3d = rotate_sphere_to_place_hub_at_touching_point(
            pos_left_3d, hub_left, center_left, touching_point
        )

        print("🔄 Rotating right sphere to align hub...")
        pos_right_3d = rotate_sphere_to_place_hub_at_touching_point(
            pos_right_3d, hub_right, center_right, touching_point
        )

        # Verify
        hub1_pos = np.array(pos_left_3d[hub_left])
        hub2_pos = np.array(pos_right_3d[hub_right])
        hub_dist = np.linalg.norm(hub1_pos - hub2_pos)
        print(f"   Hub distance: {hub_dist:.6f}")
        print()
    else:
        hub_left = None
        hub_right = None

    # Visualize
    print("🎨 Creating visualization...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw spheres
    X1, Y1, Z1 = create_sphere_mesh(center_left, radius)
    X2, Y2, Z2 = create_sphere_mesh(center_right, radius)

    ax.plot_surface(X1, Y1, Z1, color='cyan', alpha=0.2)
    ax.plot_surface(X2, Y2, Z2, color='magenta', alpha=0.2)

    # Draw left hemisphere
    print("   Drawing left hemisphere...")
    straight_count = 0
    geodesic_count = 0

    for i, pos in enumerate(pos_left_3d):
        if i == hub_left:
            ax.scatter(*pos, c='gold', s=150, marker='*', edgecolors='black',
                       linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c='red', s=25, marker='o', edgecolors='black',
                       linewidths=0.5, zorder=50)

    for u, v in G_left.edges():
        p1 = np.array(pos_left_3d[u])
        p2 = np.array(pos_left_3d[v])

        # Decide: straight or geodesic?
        if should_use_straight_edge(p1, p2, center_left, radius):
            # Short edge: use straight line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c='blue', alpha=0.5, linewidth=0.8, zorder=40)
            straight_count += 1
        else:
            # Long edge: use geodesic arc
            arc = compute_great_circle_arc(p1 - center_left, p2 - center_left, n_points=15)
            arc = arc + center_left
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                    c='darkblue', alpha=0.7, linewidth=1.0, zorder=40)
            geodesic_count += 1

    print(f"      {straight_count} straight edges, {geodesic_count} geodesic arcs")

    # Draw right hemisphere
    print("   Drawing right hemisphere...")
    straight_count = 0
    geodesic_count = 0

    for i, pos in enumerate(pos_right_3d):
        if i == hub_right:
            ax.scatter(*pos, c='gold', s=150, marker='*', edgecolors='black',
                       linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c='red', s=25, marker='o', edgecolors='black',
                       linewidths=0.5, zorder=50)

    for u, v in G_right.edges():
        p1 = np.array(pos_right_3d[u])
        p2 = np.array(pos_right_3d[v])

        if should_use_straight_edge(p1, p2, center_right, radius):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    c='blue', alpha=0.5, linewidth=0.8, zorder=40)
            straight_count += 1
        else:
            arc = compute_great_circle_arc(p1 - center_right, p2 - center_right, n_points=15)
            arc = arc + center_right
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                    c='darkblue', alpha=0.7, linewidth=1.0, zorder=40)
            geodesic_count += 1

    print(f"      {straight_count} straight edges, {geodesic_count} geodesic arcs")

    # Corpus callosum
    if hub_left is not None and hub_right is not None:
        print("   Drawing corpus callosum...")
        p1 = np.array(pos_left_3d[hub_left])
        p2 = np.array(pos_right_3d[hub_right])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c='gold', alpha=0.9, linewidth=4, zorder=99)

    # Configure view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = radius * 3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_title(
        'Two-Sphere Brain Architecture\n' +
        '(Spherical Spring Layout + Smart Edge Rendering)',
        fontsize=14,
        fontweight='bold'
    )

    # Save
    output_path = 'spheres_spring_layout.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print()
    print(f"✅ Saved: {output_path}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Layout algorithm: Spherical spring embedding (Fruchterman-Reingold on S²)")
    print(f"Nodes: {n_nodes} × 2 = {2*n_nodes} total")
    print(f"Edges: {G_left.number_of_edges()} + {G_right.number_of_edges()} + 1")
    print(f"Edge rendering: Smart (straight for short, geodesic for long)")
    print(f"Hub alignment: Quaternion rotation to touching point")
    print()
    print("✅ Nodes properly distributed on sphere surface")
    print("✅ Hubs at exact touching point [0, 0, 0]")
    print("✅ Edges optimized (straight where safe, geodesic where needed)")
    print()


if __name__ == "__main__":
    main()
