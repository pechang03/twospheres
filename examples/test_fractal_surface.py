"""Test fractal surface generation."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from mri_analysis.fractal_surface import generate_fractal_surface


def main():
    """Test fractal surface generation with Julia sets."""

    print("=" * 70)
    print("Fractal Cortical Surface Generation Test")
    print("=" * 70)
    print()

    # Generate fractal surface
    result = generate_fractal_surface(
        method="julia",
        epsilon=0.10,  # 10% perturbation
        julia_c_real=-0.7,
        julia_c_imag=0.27,
        resolution=80,  # ~2500 vertices
        radius=1.2,
        max_iterations=100,
        compute_safety_bound=True,
        compute_curvature=False
    )

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Vertices:          {len(result.vertices)}")
    print(f"Faces:             {len(result.faces)}")
    print(f"Surface area:      {result.surface_area:.3f}")
    print(f"Sphere area:       {4*np.pi*1.2**2:.3f}")
    print(f"Area ratio:        {result.surface_area / (4*np.pi*1.2**2):.3f}")
    print(f"Fractal dimension: {result.fractal_dimension:.3f}")
    print(f"Safe epsilon max:  {result.epsilon_max:.3f}")
    print(f"Current epsilon:   0.100")
    if result.epsilon_max > 0.10:
        print("✅ epsilon is within safe bounds!")
    else:
        print("⚠️  epsilon exceeds safe bounds!")
    print()

    # Visualize
    print("Creating visualization...")
    fig = plt.figure(figsize=(14, 6))

    # Color vertices by displacement
    colors = result.f_values
    vmin, vmax = colors.min(), colors.max()

    # Left plot: fractal surface as wireframe
    ax1 = fig.add_subplot(121, projection='3d')

    # Draw edges
    for face in result.faces[::2]:  # Sample every 2nd face for performance
        triangle = result.vertices[face]
        # Close the triangle loop
        triangle_loop = np.vstack([triangle, triangle[0]])
        ax1.plot(triangle_loop[:, 0], triangle_loop[:, 1], triangle_loop[:, 2],
                 c='black', alpha=0.2, linewidth=0.3)

    ax1.set_title(f'Fractal Cortical Surface\n(Julia set, ε={0.10})', fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1,1,1])

    # Right plot: displacement field
    ax2 = fig.add_subplot(122, projection='3d')

    # Scatter plot colored by displacement
    scatter = ax2.scatter(
        result.vertices[:, 0],
        result.vertices[:, 1],
        result.vertices[:, 2],
        c=colors,
        cmap='coolwarm',
        s=10,
        alpha=0.6
    )

    ax2.set_title(f'Displacement Field\n(D={result.fractal_dimension:.2f})', fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect([1,1,1])

    plt.colorbar(scatter, ax=ax2, label='Displacement f(θ,φ)', shrink=0.6)

    plt.tight_layout()
    plt.savefig('fractal_surface_test.png', dpi=200, bbox_inches='tight')
    print("Saved: fractal_surface_test.png")
    print()


if __name__ == "__main__":
    main()
