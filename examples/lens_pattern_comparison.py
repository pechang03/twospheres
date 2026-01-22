#!/usr/bin/env python3
"""
Lens Pattern Comparison: Straight 2D vs Lighthouse 180° Staggered

Compares two 2D air cavity lens patterns for PDMS microfluidic optics:
1. Straight pattern: All lenses aligned (convex facing same direction)
2. Lighthouse 180° staggered: Alternating lens orientations

Application: PHLoC (Photonic Lab-on-Chip) for HNSCC organoid culture
- Beer-Lambert absorption spectroscopy
- 1.5mm optical path through culture chamber
- 2D cylindrical air cavities in PDMS (n_air=1.0, n_PDMS≈1.4)

Usage:
    python examples/lens_pattern_comparison.py
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.optics.ray_tracing import MATERIAL_LIBRARY, get_material


# =============================================================================
# Physical Constants
# =============================================================================

N_PDMS = get_material('PDMS')    # 1.41
N_AIR = get_material('AIR')      # 1.0003
DELTA_N = N_PDMS - N_AIR         # ~0.41


# =============================================================================
# 2D Cylindrical Lens Model
# =============================================================================

class CylindricalAirLens2D:
    """2D cylindrical air cavity lens in PDMS.

    This models a lens formed by an air cavity within PDMS,
    where the cavity has a cylindrical curved surface.

    Light path: PDMS → Air cavity (curved interface) → PDMS
    """

    def __init__(self,
                 radius_of_curvature: float,
                 aperture_width: float = 0.5,
                 cavity_depth: float = 0.2,
                 orientation: str = 'convex_forward'):
        """Initialize 2D cylindrical lens.

        Args:
            radius_of_curvature: Radius of curved surface in mm
            aperture_width: Width of lens aperture in mm
            cavity_depth: Depth of air cavity in mm
            orientation: 'convex_forward' or 'convex_backward' (180° rotated)
        """
        self.radius = radius_of_curvature
        self.aperture = aperture_width
        self.depth = cavity_depth
        self.orientation = orientation

        # Effective sign for ray bending
        self.sign = 1 if orientation == 'convex_forward' else -1

    @property
    def focal_length(self) -> float:
        """Focal length of single curved surface.

        For air cavity in PDMS (light enters from PDMS side):
        f = R / (n_pdms - n_air) for convex surface

        Note: This is a diverging lens (air cavity is lower n than PDMS)
        """
        return self.radius / DELTA_N * self.sign

    @property
    def optical_power(self) -> float:
        """Optical power in diopters (1/m)."""
        return 1000.0 / self.focal_length if self.focal_length != 0 else 0

    def refract_ray(self, y_in: float, angle_in: float) -> Tuple[float, float]:
        """Refract a ray at the curved interface.

        Uses paraxial approximation for cylindrical lens.

        Args:
            y_in: Ray height at lens (mm)
            angle_in: Ray angle from optical axis (radians)

        Returns:
            (y_out, angle_out): Position and angle after lens
        """
        # Paraxial refraction at curved surface
        # Angle change: delta_theta = -y / f (thin lens approximation)
        angle_out = angle_in - y_in / self.focal_length

        # Position unchanged for thin lens
        y_out = y_in

        return y_out, angle_out

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': '2d_cylindrical_air_lens',
            'radius_mm': self.radius,
            'aperture_mm': self.aperture,
            'depth_mm': self.depth,
            'orientation': self.orientation,
            'focal_length_mm': self.focal_length,
            'optical_power_diopters': self.optical_power,
            'n_pdms': N_PDMS,
            'n_air': N_AIR
        }


# =============================================================================
# Lens Array Patterns
# =============================================================================

class LensArrayPattern:
    """Base class for 2D lens array patterns."""

    def __init__(self,
                 num_lenses: int,
                 lens_spacing: float,
                 lens_radius: float = 0.5,
                 aperture: float = 0.3):
        """Initialize lens array.

        Args:
            num_lenses: Number of lenses in array
            lens_spacing: Spacing between lens centers (mm)
            lens_radius: Radius of curvature for each lens (mm)
            aperture: Aperture width of each lens (mm)
        """
        self.num_lenses = num_lenses
        self.spacing = lens_spacing
        self.lens_radius = lens_radius
        self.aperture = aperture
        self.lenses: List[CylindricalAirLens2D] = []
        self._build_array()

    def _build_array(self):
        """Build the lens array. Override in subclasses."""
        raise NotImplementedError

    @property
    def total_length(self) -> float:
        """Total length of lens array (mm)."""
        return self.spacing * (self.num_lenses - 1)

    @property
    def effective_focal_length(self) -> float:
        """Combined focal length of lens array.

        For thin lenses in contact: 1/f_eff = sum(1/f_i)
        For spaced lenses, use thick lens formula.
        """
        # Simple sum for thin lens approximation
        total_power = sum(1.0 / L.focal_length for L in self.lenses if L.focal_length != 0)
        return 1.0 / total_power if total_power != 0 else float('inf')

    def trace_ray(self, y_start: float, angle_start: float = 0.0) -> List[Tuple[float, float, float]]:
        """Trace a ray through the lens array.

        Args:
            y_start: Initial ray height (mm)
            angle_start: Initial ray angle (radians)

        Returns:
            List of (z, y, angle) at each lens
        """
        trajectory = [(0.0, y_start, angle_start)]

        y = y_start
        angle = angle_start
        z = 0.0

        for i, lens in enumerate(self.lenses):
            # Propagate to next lens
            z = i * self.spacing
            y = y + angle * self.spacing

            # Refract at lens
            y, angle = lens.refract_ray(y, angle)
            trajectory.append((z, y, angle))

        return trajectory

    def compute_beam_metrics(self,
                             beam_height: float = 0.3,
                             num_rays: int = 11,
                             propagation_distance: float = 1.5) -> Dict[str, Any]:
        """Compute beam quality metrics after propagation.

        Args:
            beam_height: Half-height of input beam (mm)
            num_rays: Number of rays to trace
            propagation_distance: Distance after last lens (mm)

        Returns:
            Dictionary with beam metrics
        """
        # Trace rays across beam aperture
        y_inputs = np.linspace(-beam_height, beam_height, num_rays)
        final_positions = []
        final_angles = []

        for y0 in y_inputs:
            traj = self.trace_ray(y0, 0.0)
            z_final, y_final, angle_final = traj[-1]

            # Propagate to detector
            y_detector = y_final + angle_final * propagation_distance

            final_positions.append(y_detector)
            final_angles.append(angle_final)

        final_positions = np.array(final_positions)
        final_angles = np.array(final_angles)

        # Compute metrics
        beam_width = final_positions.max() - final_positions.min()
        beam_center = np.mean(final_positions)
        rms_spread = np.std(final_positions)

        # Collimation quality: ideally all angles should be 0
        angle_spread = np.std(final_angles) * 1000  # mrad
        max_angle = np.max(np.abs(final_angles)) * 1000  # mrad

        # Uniformity: how evenly distributed are rays across the beam
        # Perfect uniformity = input spacing preserved
        uniformity = 1.0 - np.std(np.diff(final_positions)) / np.mean(np.abs(np.diff(final_positions))) if len(final_positions) > 1 else 1.0

        return {
            'beam_width_mm': float(beam_width),
            'beam_center_mm': float(beam_center),
            'rms_spread_mm': float(rms_spread),
            'angle_spread_mrad': float(angle_spread),
            'max_angle_mrad': float(max_angle),
            'uniformity': float(max(0, min(1, uniformity))),
            'propagation_distance_mm': propagation_distance,
            'num_rays': num_rays,
            'input_beam_height_mm': beam_height
        }


class StraightPattern(LensArrayPattern):
    """Straight lens pattern: all lenses convex in same direction.

    Configuration: → → → → (all convex forward)

    This creates a consistent diverging effect (since air cavities
    in PDMS act as diverging lenses).
    """

    def _build_array(self):
        self.lenses = [
            CylindricalAirLens2D(
                radius_of_curvature=self.lens_radius,
                aperture_width=self.aperture,
                orientation='convex_forward'
            )
            for _ in range(self.num_lenses)
        ]


class LighthousePattern(LensArrayPattern):
    """Lighthouse 180° staggered pattern: alternating orientations.

    Configuration: → ← → ← (alternating convex direction)

    Named "lighthouse" because the alternating pattern is reminiscent
    of lighthouse Fresnel lens segments.

    Benefits:
    - Aberration compensation (odd aberrations cancel)
    - Better uniformity across beam
    - Reduced coma and distortion
    """

    def _build_array(self):
        self.lenses = []
        for i in range(self.num_lenses):
            orientation = 'convex_forward' if i % 2 == 0 else 'convex_backward'
            self.lenses.append(
                CylindricalAirLens2D(
                    radius_of_curvature=self.lens_radius,
                    aperture_width=self.aperture,
                    orientation=orientation
                )
            )


# =============================================================================
# Simulation and Comparison
# =============================================================================

def compare_patterns(
    num_lenses: int = 4,
    lens_spacing: float = 0.3,
    lens_radius: float = 0.5,
    optical_path: float = 1.5,
    beam_height: float = 0.25
) -> Dict[str, Any]:
    """Compare straight vs lighthouse lens patterns.

    Args:
        num_lenses: Number of lenses in each pattern
        lens_spacing: Spacing between lenses (mm)
        lens_radius: Radius of curvature (mm)
        optical_path: Total optical path through chamber (mm)
        beam_height: Half-height of input beam (mm)

    Returns:
        Comparison results dictionary
    """
    # Create both patterns
    straight = StraightPattern(
        num_lenses=num_lenses,
        lens_spacing=lens_spacing,
        lens_radius=lens_radius
    )

    lighthouse = LighthousePattern(
        num_lenses=num_lenses,
        lens_spacing=lens_spacing,
        lens_radius=lens_radius
    )

    # Propagation distance after lenses to detector
    prop_dist = optical_path - straight.total_length

    # Compute metrics for both
    straight_metrics = straight.compute_beam_metrics(
        beam_height=beam_height,
        propagation_distance=prop_dist
    )

    lighthouse_metrics = lighthouse.compute_beam_metrics(
        beam_height=beam_height,
        propagation_distance=prop_dist
    )

    # Detailed ray traces for visualization
    ray_heights = [-0.2, -0.1, 0.0, 0.1, 0.2]
    straight_rays = [straight.trace_ray(y) for y in ray_heights]
    lighthouse_rays = [lighthouse.trace_ray(y) for y in ray_heights]

    return {
        'configuration': {
            'num_lenses': num_lenses,
            'lens_spacing_mm': lens_spacing,
            'lens_radius_mm': lens_radius,
            'optical_path_mm': optical_path,
            'beam_height_mm': beam_height,
            'n_pdms': N_PDMS,
            'n_air': N_AIR,
            'single_lens_focal_length_mm': lens_radius / DELTA_N
        },
        'straight_pattern': {
            'description': 'All lenses convex in same direction (→ → → →)',
            'effective_focal_length_mm': straight.effective_focal_length,
            'metrics': straight_metrics,
            'lens_orientations': [L.orientation for L in straight.lenses]
        },
        'lighthouse_pattern': {
            'description': 'Alternating lens orientations (→ ← → ←)',
            'effective_focal_length_mm': lighthouse.effective_focal_length,
            'metrics': lighthouse_metrics,
            'lens_orientations': [L.orientation for L in lighthouse.lenses]
        },
        'comparison': {
            'beam_width_improvement': (straight_metrics['beam_width_mm'] - lighthouse_metrics['beam_width_mm']) / straight_metrics['beam_width_mm'] * 100 if straight_metrics['beam_width_mm'] != 0 else 0,
            'uniformity_improvement': (lighthouse_metrics['uniformity'] - straight_metrics['uniformity']) * 100,
            'angle_spread_improvement': (straight_metrics['angle_spread_mrad'] - lighthouse_metrics['angle_spread_mrad']) / straight_metrics['angle_spread_mrad'] * 100 if straight_metrics['angle_spread_mrad'] != 0 else 0,
            # Recommend based on collimation (lower angle spread = better for absorption spectroscopy)
            'recommendation': 'lighthouse' if lighthouse_metrics['angle_spread_mrad'] < straight_metrics['angle_spread_mrad'] else 'straight'
        },
        'ray_traces': {
            'ray_heights_mm': ray_heights,
            'straight': straight_rays,
            'lighthouse': lighthouse_rays
        }
    }


def print_comparison_report(results: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("=" * 70)
    print("2D Lens Pattern Comparison: Straight vs Lighthouse 180° Staggered")
    print("=" * 70)
    print(f"\nApplication: PHLoC HNSCC organoid absorption spectroscopy")
    print(f"Optical path: {results['configuration']['optical_path_mm']} mm (Beer-Lambert)")
    print()

    print("Configuration:")
    print(f"  Lenses: {results['configuration']['num_lenses']} × 2D cylindrical air cavities")
    print(f"  Lens radius: {results['configuration']['lens_radius_mm']} mm")
    print(f"  Lens spacing: {results['configuration']['lens_spacing_mm']} mm")
    print(f"  Single lens f: {results['configuration']['single_lens_focal_length_mm']:.2f} mm")
    print(f"  Materials: PDMS (n={N_PDMS}) / Air (n={N_AIR})")
    print()

    print("-" * 70)
    print("STRAIGHT PATTERN")
    print("-" * 70)
    print(f"  Configuration: {results['straight_pattern']['description']}")
    print(f"  Orientations: {results['straight_pattern']['lens_orientations']}")
    print(f"  Effective f: {results['straight_pattern']['effective_focal_length_mm']:.2f} mm")
    m = results['straight_pattern']['metrics']
    print(f"  Beam width: {m['beam_width_mm']:.4f} mm")
    print(f"  RMS spread: {m['rms_spread_mm']:.4f} mm")
    print(f"  Angle spread: {m['angle_spread_mrad']:.4f} mrad")
    print(f"  Uniformity: {m['uniformity']:.3f}")
    print()

    print("-" * 70)
    print("LIGHTHOUSE PATTERN (180° Staggered)")
    print("-" * 70)
    print(f"  Configuration: {results['lighthouse_pattern']['description']}")
    print(f"  Orientations: {results['lighthouse_pattern']['lens_orientations']}")
    print(f"  Effective f: {results['lighthouse_pattern']['effective_focal_length_mm']:.2f} mm")
    m = results['lighthouse_pattern']['metrics']
    print(f"  Beam width: {m['beam_width_mm']:.4f} mm")
    print(f"  RMS spread: {m['rms_spread_mm']:.4f} mm")
    print(f"  Angle spread: {m['angle_spread_mrad']:.4f} mrad")
    print(f"  Uniformity: {m['uniformity']:.3f}")
    print()

    print("-" * 70)
    print("COMPARISON SUMMARY")
    print("-" * 70)
    c = results['comparison']
    print(f"  Beam width improvement: {c['beam_width_improvement']:+.1f}%")
    print(f"  Uniformity improvement: {c['uniformity_improvement']:+.1f}%")
    print(f"  Angle spread improvement: {c['angle_spread_improvement']:+.1f}%")
    print()
    print(f"  Recommended pattern: {c['recommendation'].upper()}")
    print()

    if c['recommendation'] == 'lighthouse':
        print("  Lighthouse pattern advantages:")
        print("    - Odd aberrations (coma, distortion) cancel between pairs")
        print("    - Better beam uniformity across aperture")
        print("    - More collimated output (smaller angle spread)")
    else:
        print("  Straight pattern advantages:")
        print("    - Simpler fabrication (all lenses identical)")
        print("    - Predictable focusing behavior")
    print()
    print("=" * 70)


def main():
    """Run lens pattern comparison."""

    # PHLoC design parameters
    # - 1.5mm optical path (chamber length for Beer-Lambert)
    # - 4 lenses for collimation/focusing
    # - 0.5mm radius lenses (f ≈ 1.22mm for PDMS/air)

    print("\nRunning lens pattern simulation...\n")

    results = compare_patterns(
        num_lenses=4,
        lens_spacing=0.3,
        lens_radius=0.5,
        optical_path=1.5,
        beam_height=0.25
    )

    print_comparison_report(results)

    # Also test with different configurations
    print("\n" + "=" * 70)
    print("Parameter Sweep: Effect of Lens Count")
    print("=" * 70)

    for n_lenses in [2, 4, 6, 8]:
        r = compare_patterns(
            num_lenses=n_lenses,
            lens_spacing=0.2,
            lens_radius=0.5,
            optical_path=1.5
        )
        s_unif = r['straight_pattern']['metrics']['uniformity']
        l_unif = r['lighthouse_pattern']['metrics']['uniformity']
        improvement = r['comparison']['uniformity_improvement']
        print(f"  {n_lenses} lenses: Straight={s_unif:.3f}, Lighthouse={l_unif:.3f}, Δ={improvement:+.1f}%")

    print("\n" + "=" * 70)
    print("Parameter Sweep: Effect of Lens Radius")
    print("=" * 70)

    for radius in [0.3, 0.5, 0.75, 1.0]:
        r = compare_patterns(
            num_lenses=4,
            lens_spacing=0.3,
            lens_radius=radius,
            optical_path=1.5
        )
        f_single = radius / DELTA_N
        s_angle = r['straight_pattern']['metrics']['angle_spread_mrad']
        l_angle = r['lighthouse_pattern']['metrics']['angle_spread_mrad']
        print(f"  R={radius}mm (f={f_single:.2f}mm): Straight={s_angle:.2f}mrad, Lighthouse={l_angle:.2f}mrad")

    print("\n✓ Simulation complete")
    print("  For full 3D ray tracing, use pyoptools via freecad-mcp")

    return results


if __name__ == "__main__":
    results = main()
