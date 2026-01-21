"""Demonstration of geodesic edge routing on two-sphere brain architecture.

This demo shows the corrected implementation with edges following sphere surfaces
(geodesic arcs) rather than passing through sphere volume.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.graph_on_sphere import (
    SphereGraphConfig,
    visualize_two_spheres_with_graphs
)


def main():
    """Generate visualization with geodesic routing."""

    # Configuration for brain-like hemisphere architecture
    config = SphereGraphConfig(
        radius1=1.2,  # Left hemisphere
        radius2=1.2,  # Right hemisphere
        graph_type="erdos_renyi",
        n_nodes=60,
        rotation_x_deg=30.0,
        rotation_y_deg=45.0,
        rotation_z_deg=0.0,
        edge_color="blue",
        node_color="red",
        sphere1_color="cyan",
        sphere2_color="magenta",
        sphere_alpha=0.25,
        show_inter_sphere_edges=True,
        ensure_connected=True
    )

    print("🧠 Generating two-sphere brain architecture with GEODESIC edge routing")
    print("=" * 70)
    print(f"Graph type: {config.graph_type}")
    print(f"Nodes per hemisphere: {config.n_nodes}")
    print(f"Sphere radii: {config.radius1}, {config.radius2}")
    print()
    print("✅ GEODESIC ROUTING ENABLED:")
    print("   - Edges follow sphere surface (great circle arcs)")
    print("   - No volume penetration")
    print("   - Mathematically correct distance metric")
    print()

    # Create visualization
    fig = visualize_two_spheres_with_graphs(
        config,
        save_path="geodesic_brain_architecture.png",
        figsize=(14, 12)
    )

    print()
    print("📊 Visualization saved to: geodesic_brain_architecture.png")
    print()
    print("🔬 Key Features:")
    print("   ⭐ Gold stars = Hub nodes (corpus callosum connection points)")
    print("   🟡 Gold line = Corpus callosum (inter-hemisphere connection)")
    print("   🔵 Blue curves = Geodesic edges on sphere surface")
    print("   🔴 Red dots = Regular nodes")
    print()
    print("✅ All edges now follow sphere surfaces!")
    print("   (Compare with previous implementation where edges passed through volume)")
    print()

    # Show statistics
    print("📈 Implementation Details:")
    print("   - Geodesic distance: arc length on sphere surface")
    print("   - Great circle arcs: SLERP interpolation (20 points)")
    print("   - Hub detection: degree ≥ 5")
    print("   - Corpus callosum: single gold connection through hubs")
    print()

    plt.show()


if __name__ == "__main__":
    main()
