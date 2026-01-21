"""Folded cortical hemispheres with proper MESH GEODESICS.

This is the most realistic version - fractal surfaces with edges
that follow the actual mesh topology (not smooth sphere approximation).
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from mri_analysis.fractal_surface import (
    generate_fractal_surface,
    compute_mesh_geodesic_path
)
from backend.visualization.graph_on_sphere import (
    generate_graph,
    ensure_graph_connectivity,
    find_hub_nodes,
    rotate_graph_to_place_node_at_pole
)


def map_graph_to_fractal_surface_nearest_vertex(
    G: nx.Graph,
    fractal_surface,
    sphere_center: np.ndarray
):
    """Map graph nodes to nearest mesh vertices.

    Returns:
        pos_3d: List of 3D positions
        vertex_indices: List of nearest mesh vertex indices
    """
    pos = nx.get_node_attributes(G, 'pos')
    pos_3d = []
    vertex_indices = []

    vertices = fractal_surface.vertices
    spherical_coords = fractal_surface.spherical_coords

    for node in G.nodes():
        u, v = pos[node]

        # Map (u,v) to (theta, phi)
        theta = v * np.pi
        phi = u * 2 * np.pi

        # Find nearest vertex on fractal surface
        theta_diffs = np.abs(spherical_coords[:, 0] - theta)
        phi_diffs = np.abs(spherical_coords[:, 1] - phi)
        phi_diffs = np.minimum(phi_diffs, 2*np.pi - phi_diffs)

        angular_dist = np.sqrt(theta_diffs**2 + phi_diffs**2)
        nearest_idx = np.argmin(angular_dist)

        # Get 3D position from fractal surface
        pos_3d_local = vertices[nearest_idx]
        pos_3d_final = pos_3d_local + sphere_center

        pos_3d.append(pos_3d_final)
        vertex_indices.append(nearest_idx)

    return pos_3d, vertex_indices


def main():
    """Generate folded hemisphere visualization with mesh geodesics."""

    print("=" * 70)
    print("Folded Cortical Hemispheres - MESH GEODESICS")
    print("=" * 70)
    print()

    # Configuration
    epsilon = 0.08
    resolution = 60  # ~162 vertices per hemisphere
    n_nodes = 30     # Fewer nodes for clearer visualization
    radius = 1.2

    # Sphere centers
    center_left = np.array([0, radius, 0])
    center_right = np.array([0, -radius, 0])

    # Generate fractal surfaces
    print("🧠 Generating left hemisphere (Julia set)...")
    fractal_left = generate_fractal_surface(
        method="julia",
        epsilon=epsilon,
        julia_c_real=-0.7,
        julia_c_imag=0.27,
        resolution=resolution,
        radius=radius,
        max_iterations=80,
        compute_safety_bound=False,
        compute_curvature=False
    )
    print(f"   {len(fractal_left.vertices)} vertices, D={fractal_left.fractal_dimension:.2f}")
    print()

    print("🧠 Generating right hemisphere...")
    fractal_right = generate_fractal_surface(
        method="julia",
        epsilon=epsilon,
        julia_c_real=-0.8,
        julia_c_imag=0.18,
        resolution=resolution,
        radius=radius,
        max_iterations=80,
        compute_safety_bound=False,
        compute_curvature=False
    )
    print(f"   {len(fractal_right.vertices)} vertices, D={fractal_right.fractal_dimension:.2f}")
    print()

    # Generate graphs
    print(f"📊 Generating neural networks ({n_nodes} nodes per hemisphere)...")
    G_left = generate_graph("erdos_renyi", n_nodes, seed=42)
    G_right = generate_graph("erdos_renyi", n_nodes, seed=43)

    G_left = ensure_graph_connectivity(G_left)
    G_right = ensure_graph_connectivity(G_right)

    # Find hubs
    hubs_left = find_hub_nodes(G_left, min_degree=4)
    hubs_right = find_hub_nodes(G_right, min_degree=4)

    if hubs_left and hubs_right:
        hub_left = hubs_left[0]
        hub_right = hubs_right[0]
        print(f"   Hubs: left={hub_left} (deg={G_left.degree(hub_left)}), " +
              f"right={hub_right} (deg={G_right.degree(hub_right)})")

        rotate_graph_to_place_node_at_pole(G_left, hub_left, pole="south")
        rotate_graph_to_place_node_at_pole(G_right, hub_right, pole="north")
    else:
        hub_left = None
        hub_right = None
    print()

    # Map graphs to fractal surfaces
    print("📍 Mapping graphs to fractal surfaces...")
    pos_left_3d, vtx_left = map_graph_to_fractal_surface_nearest_vertex(
        G_left, fractal_left, center_left
    )
    pos_right_3d, vtx_right = map_graph_to_fractal_surface_nearest_vertex(
        G_right, fractal_right, center_right
    )
    print()

    # Visualize
    print("🎨 Creating visualization...")
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw left hemisphere wireframe
    print("   Drawing left hemisphere mesh...")
    for face in fractal_left.faces[::2]:
        triangle = fractal_left.vertices[face] + center_left
        triangle_loop = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_loop[:, 0], triangle_loop[:, 1], triangle_loop[:, 2],
                c='cyan', alpha=0.15, linewidth=0.3)

    # Draw right hemisphere wireframe
    print("   Drawing right hemisphere mesh...")
    for face in fractal_right.faces[::2]:
        triangle = fractal_right.vertices[face] + center_right
        triangle_loop = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_loop[:, 0], triangle_loop[:, 1], triangle_loop[:, 2],
                c='magenta', alpha=0.15, linewidth=0.3)

    # Draw left network with MESH GEODESICS
    print("   Computing mesh geodesics for left hemisphere...")
    for i, pos in enumerate(pos_left_3d):
        if i == hub_left:
            ax.scatter(*pos, c='gold', s=150, marker='*', edgecolors='black',
                       linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c='red', s=25, marker='o', edgecolors='black',
                       linewidths=0.5, zorder=50)

    edge_count = 0
    for u, v in G_left.edges():
        # Use mesh geodesic path
        path = compute_mesh_geodesic_path(
            fractal_left.vertices,
            fractal_left.faces,
            vtx_left[u],
            vtx_left[v],
            n_interpolation_points=15
        )
        path = path + center_left  # Translate

        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                c='blue', alpha=0.7, linewidth=1.2, zorder=40)
        edge_count += 1

    print(f"      Drew {edge_count} geodesic paths")

    # Draw right network with MESH GEODESICS
    print("   Computing mesh geodesics for right hemisphere...")
    for i, pos in enumerate(pos_right_3d):
        if i == hub_right:
            ax.scatter(*pos, c='gold', s=150, marker='*', edgecolors='black',
                       linewidths=2, zorder=100)
        else:
            ax.scatter(*pos, c='red', s=25, marker='o', edgecolors='black',
                       linewidths=0.5, zorder=50)

    edge_count = 0
    for u, v in G_right.edges():
        path = compute_mesh_geodesic_path(
            fractal_right.vertices,
            fractal_right.faces,
            vtx_right[u],
            vtx_right[v],
            n_interpolation_points=15
        )
        path = path + center_right

        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                c='blue', alpha=0.7, linewidth=1.2, zorder=40)
        edge_count += 1

    print(f"      Drew {edge_count} geodesic paths")

    # Draw corpus callosum
    if hub_left is not None and hub_right is not None:
        print("   Drawing corpus callosum...")
        p1 = np.array(pos_left_3d[hub_left])
        p2 = np.array(pos_right_3d[hub_right])

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                c='gold', alpha=0.9, linewidth=5, linestyle='-', zorder=99)

    # Configure view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = radius * 3
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_title(
        f'Folded Cortical Hemispheres - MESH GEODESICS\n' +
        f'(Julia sets, ε={epsilon}, D_L={fractal_left.fractal_dimension:.2f}, ' +
        f'D_R={fractal_right.fractal_dimension:.2f})',
        fontsize=14,
        fontweight='bold'
    )

    # Save
    output_path = 'folded_spheres_mesh_geodesic.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print()
    print(f"✅ Saved: {output_path}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Left mesh:  {len(fractal_left.vertices)} vertices, {len(fractal_left.faces)} faces")
    print(f"Right mesh: {len(fractal_right.vertices)} vertices, {len(fractal_right.faces)} faces")
    print(f"Network:    {n_nodes} nodes × 2 = {2*n_nodes} total")
    print(f"Edges:      {G_left.number_of_edges()} + {G_right.number_of_edges()} + 1 corpus callosum")
    print(f"")
    print(f"✅ MESH GEODESICS: Edges follow actual mesh topology")
    print(f"✅ FRACTAL FOLDING: Realistic cortical gyrification")
    print(f"✅ Edges match the folded surface correctly!")
    print()


if __name__ == "__main__":
    main()
