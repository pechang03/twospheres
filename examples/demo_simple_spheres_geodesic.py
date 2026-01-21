"""Simple two-sphere visualization with GEODESIC arcs.

Uses compute_great_circle_arc() for smooth geodesic routing
on simple spheres (not folded).
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.graph_on_sphere import (
    SphereGraphConfig,
    visualize_two_spheres_with_graphs
)


def main():
    """Generate simple two-sphere visualization with geodesic edges."""

    print("=" * 70)
    print("Simple Two-Sphere Visualization - GEODESIC ARCS")
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
    print(f"  Edge rendering: GEODESIC ARCS (great circles)")
    print()

    print("Generating visualization...")
    fig = visualize_two_spheres_with_graphs(
        config,
        save_path="simple_spheres_geodesic.png",
        figsize=(12, 10)
    )

    print()
    print("✅ Saved: simple_spheres_geodesic.png")
    print()
    print("Features:")
    print("  ✓ Geodesic arcs follow sphere surfaces")
    print("  ✓ SLERP interpolation (20 points per arc)")
    print("  ✓ Hub-based corpus callosum")
    print("  ✓ No volume penetration")
    print()


if __name__ == "__main__":
    main()
