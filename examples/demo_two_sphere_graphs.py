#!/usr/bin/env python3
"""Demo: Map planar graphs onto two spheres using quaternion rotation.

Demonstrates visualization of paired brain regions (left/right hemispheres)
with network connectivity patterns mapped onto sphere surfaces.

Based on ~/MRISpheres/twospheres work.

Usage:
    python examples/demo_two_sphere_graphs.py
    python examples/demo_two_sphere_graphs.py --graph-type small_world
    python examples/demo_two_sphere_graphs.py --show-inter-edges
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.visualization.graph_on_sphere import (
    create_two_sphere_graph_visualization,
    SphereGraphConfig,
    visualize_two_spheres_with_graphs,
    generate_graph
)


def demo_basic():
    """Basic demo with random geometric graph."""
    print("=== Demo 1: Random Geometric Graph on Two Spheres ===")

    result = create_two_sphere_graph_visualization(
        graph_type="random_geometric",
        n_nodes=80,
        radius=1.0,
        rotation_x=30.0,
        rotation_y=45.0,
        show_inter_edges=False
    )

    print(f"Graph type: {result['graph_type']}")
    print(f"Nodes on sphere 1: {result['n_nodes_sphere1']}")
    print(f"Edges on sphere 1: {result['n_edges_sphere1']}")
    print(f"Average degree: {result['avg_degree_sphere1']:.2f}")
    print(f"Clustering coefficient: {result['clustering_sphere1']:.3f}")

    plt.show()
    print()


def demo_small_world():
    """Demo with small-world graph."""
    print("=== Demo 2: Small-World Graph (Watts-Strogatz) ===")

    result = create_two_sphere_graph_visualization(
        graph_type="small_world",
        n_nodes=60,
        radius=1.0,
        rotation_x=20.0,
        rotation_y=60.0,
        show_inter_edges=False
    )

    print(f"Graph type: {result['graph_type']}")
    print(f"Clustering coefficient: {result['clustering_sphere1']:.3f}")
    print("(High clustering typical of small-world networks)")

    plt.show()
    print()


def demo_scale_free():
    """Demo with scale-free graph."""
    print("=== Demo 3: Scale-Free Graph (Barabási-Albert) ===")

    result = create_two_sphere_graph_visualization(
        graph_type="scale_free",
        n_nodes=70,
        radius=1.0,
        rotation_x=15.0,
        rotation_y=30.0,
        show_inter_edges=False
    )

    print(f"Graph type: {result['graph_type']}")
    print(f"Average degree: {result['avg_degree_sphere1']:.2f}")
    print("(Power-law degree distribution typical of scale-free networks)")

    plt.show()
    print()


def demo_erdos_renyi():
    """Demo with Erdős-Rényi random graph."""
    print("=== Demo 4: Erdős-Rényi Random Graph ===")

    result = create_two_sphere_graph_visualization(
        graph_type="erdos_renyi",
        n_nodes=100,
        radius=1.0,
        rotation_x=25.0,
        rotation_y=40.0,
        show_inter_edges=False
    )

    print(f"Graph type: {result['graph_type']}")
    print(f"Nodes: {result['n_nodes_sphere1']}")
    print(f"Edges: {result['n_edges_sphere1']}")
    print(f"Average degree: {result['avg_degree_sphere1']:.2f}")
    print(f"Clustering coefficient: {result['clustering_sphere1']:.3f}")
    print("(Classic G(n,p) random graph model)")

    plt.show()
    print()


def demo_with_inter_edges():
    """Demo showing inter-sphere connectivity."""
    print("=== Demo 5: With Inter-Sphere Edges (Corpus Callosum-like) ===")

    result = create_two_sphere_graph_visualization(
        graph_type="random_geometric",
        n_nodes=50,
        radius=1.0,
        rotation_x=30.0,
        rotation_y=45.0,
        show_inter_edges=True
    )

    print(f"Inter-sphere edges: {result['inter_sphere_edges']}")
    print("(Green dashed lines represent connections between hemispheres)")

    plt.show()
    print()


def demo_quaternion_rotations():
    """Demo showing different quaternion rotations."""
    print("=== Demo 6: Quaternion Rotation Variations ===")

    configs = [
        (0, 0, "No rotation"),
        (45, 0, "45° around X-axis"),
        (0, 45, "45° around Y-axis"),
        (30, 45, "30° X, 45° Y (default)")
    ]

    for rot_x, rot_y, label in configs:
        print(f"\n{label}:")
        result = create_two_sphere_graph_visualization(
            graph_type="grid",
            n_nodes=49,  # 7x7 grid
            radius=1.0,
            rotation_x=rot_x,
            rotation_y=rot_y,
            show_inter_edges=False
        )
        print(f"  Rotation: X={rot_x}°, Y={rot_y}°")
        plt.show()

    print()


def demo_custom_config():
    """Demo with custom configuration for publication-quality figure."""
    print("=== Demo 7: Custom Configuration (Publication Quality) ===")

    config = SphereGraphConfig(
        radius=1.0,
        center1=[0, 1.2, 0],  # Slightly more separated
        center2=[0, -1.2, 0],
        graph_type="random_geometric",
        n_nodes=100,
        rotation_x_deg=25.0,
        rotation_y_deg=35.0,
        rotation_z_deg=10.0,  # Add z-rotation
        edge_color="darkblue",
        node_color="red",
        sphere1_color="lightcyan",
        sphere2_color="lightpink",
        sphere_alpha=0.2,  # More transparent
        show_inter_sphere_edges=True
    )

    # Generate graphs
    G1 = generate_graph("random_geometric", 100, seed=42)
    G2 = generate_graph("random_geometric", 100, seed=43)

    # Create inter-sphere edges
    inter_edges = [(i, i) for i in range(min(len(G1), len(G2)))]

    # Visualize
    fig = visualize_two_spheres_with_graphs(
        config, G1, G2, inter_edges,
        save_path="two_sphere_graph_publication.png",
        figsize=(14, 12)
    )

    print("Custom configuration created with:")
    print(f"  Sphere separation: 2.4 (vs 2.0 default)")
    print(f"  3-axis rotation: X={config.rotation_x_deg}°, Y={config.rotation_y_deg}°, Z={config.rotation_z_deg}°")
    print(f"  Transparency: {config.sphere_alpha}")
    print(f"  Saved to: two_sphere_graph_publication.png")

    plt.show()
    print()


def main():
    """Run demonstrations."""
    parser = argparse.ArgumentParser(description="Two-sphere graph mapping demo")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["basic", "small_world", "scale_free", "erdos_renyi", "inter_edges", "rotations", "custom", "all"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["random_geometric", "erdos_renyi", "small_world", "scale_free", "grid"],
        help="Override graph type for basic demo"
    )
    parser.add_argument(
        "--show-inter-edges",
        action="store_true",
        help="Show inter-sphere edges"
    )
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=80,
        help="Number of nodes (default: 80)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Two-Sphere Graph Mapping Demo")
    print("Using Quaternion Rotation")
    print("=" * 70)
    print()

    if args.demo == "all":
        demo_basic()
        demo_small_world()
        demo_scale_free()
        demo_erdos_renyi()
        demo_with_inter_edges()
        demo_quaternion_rotations()
        demo_custom_config()
    elif args.demo == "basic":
        if args.graph_type:
            result = create_two_sphere_graph_visualization(
                graph_type=args.graph_type,
                n_nodes=args.n_nodes,
                show_inter_edges=args.show_inter_edges
            )
            print(f"Created {args.graph_type} graph with {args.n_nodes} nodes")
            print(f"Statistics: {result}")
            plt.show()
        else:
            demo_basic()
    elif args.demo == "small_world":
        demo_small_world()
    elif args.demo == "scale_free":
        demo_scale_free()
    elif args.demo == "erdos_renyi":
        demo_erdos_renyi()
    elif args.demo == "inter_edges":
        demo_with_inter_edges()
    elif args.demo == "rotations":
        demo_quaternion_rotations()
    elif args.demo == "custom":
        demo_custom_config()

    print("=" * 70)
    print("Demo complete!")
    print()
    print("Key features demonstrated:")
    print("  ✓ Quaternion-based rotation (avoids gimbal lock)")
    print("  ✓ Multiple graph types (random geometric, small-world, scale-free, grid)")
    print("  ✓ Paired sphere visualization (for bilateral brain regions)")
    print("  ✓ Inter-sphere connectivity (corpus callosum-like)")
    print("  ✓ Customizable appearance and rotation")
    print()
    print("For MCP tool usage, call 'two_sphere_graph_mapping' with parameters.")
    print("=" * 70)


if __name__ == "__main__":
    main()
