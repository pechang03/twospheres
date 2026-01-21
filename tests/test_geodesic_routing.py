"""Tests for geodesic routing on sphere surfaces.

Tests the geodesic distance computation and great circle arc generation
to ensure edges follow sphere surfaces rather than passing through volume.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.visualization.graph_on_sphere import (
    geodesic_distance_on_sphere,
    compute_great_circle_arc
)


class TestGeodesicDistance:
    """Test geodesic distance computation on sphere surface."""

    def test_geodesic_distance_north_south_poles(self):
        """Test geodesic distance between north and south poles.

        Expected: π·r (half circumference of sphere)
        """
        radius = 1.0

        # North pole (z = +r)
        p1 = np.array([0.0, 0.0, radius])
        # South pole (z = -r)
        p2 = np.array([0.0, 0.0, -radius])

        dist = geodesic_distance_on_sphere(p1, p2, radius)
        expected = np.pi * radius  # Half circumference

        assert np.isclose(dist, expected, rtol=1e-5), \
            f"Expected {expected}, got {dist}"

    def test_geodesic_distance_quarter_sphere(self):
        """Test geodesic distance for 90-degree separation.

        Expected: (π/2)·r (quarter circumference)
        """
        radius = 1.0

        # Point on x-axis
        p1 = np.array([radius, 0.0, 0.0])
        # Point on z-axis
        p2 = np.array([0.0, 0.0, radius])

        dist = geodesic_distance_on_sphere(p1, p2, radius)
        expected = (np.pi / 2) * radius  # Quarter circumference

        assert np.isclose(dist, expected, rtol=1e-5), \
            f"Expected {expected}, got {dist}"

    def test_geodesic_distance_identical_points(self):
        """Test geodesic distance for identical points.

        Expected: 0
        """
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([radius, 0.0, 0.0])

        dist = geodesic_distance_on_sphere(p1, p2, radius)

        assert np.isclose(dist, 0.0, atol=1e-10), \
            f"Expected 0, got {dist}"

    def test_geodesic_distance_various_radii(self):
        """Test that geodesic distance scales linearly with radius."""
        # North and south poles
        p1_unit = np.array([0.0, 0.0, 1.0])
        p2_unit = np.array([0.0, 0.0, -1.0])

        for radius in [0.5, 1.0, 2.0, 10.0]:
            p1 = p1_unit * radius
            p2 = p2_unit * radius

            dist = geodesic_distance_on_sphere(p1, p2, radius)
            expected = np.pi * radius

            assert np.isclose(dist, expected, rtol=1e-5), \
                f"For radius {radius}: expected {expected}, got {dist}"

    def test_geodesic_vs_euclidean_distance(self):
        """Test that geodesic distance is always >= Euclidean distance.

        For points on sphere surface, geodesic (arc) is longer than
        straight line through volume (Euclidean).
        """
        radius = 1.0

        # North pole
        p1 = np.array([0.0, 0.0, radius])
        # South pole
        p2 = np.array([0.0, 0.0, -radius])

        geodesic_dist = geodesic_distance_on_sphere(p1, p2, radius)
        euclidean_dist = np.linalg.norm(p1 - p2)

        # Geodesic distance = π·r ≈ 3.14·r
        # Euclidean distance = 2·r
        # Ratio should be π/2 ≈ 1.57
        assert geodesic_dist > euclidean_dist, \
            f"Geodesic ({geodesic_dist}) should be > Euclidean ({euclidean_dist})"

        ratio = geodesic_dist / euclidean_dist
        expected_ratio = np.pi / 2
        assert np.isclose(ratio, expected_ratio, rtol=1e-5), \
            f"Ratio {ratio} should equal π/2 ≈ {expected_ratio}"


class TestGreatCircleArc:
    """Test great circle arc generation using SLERP."""

    def test_arc_endpoints_match(self):
        """Test that arc starts and ends at correct points."""
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([0.0, radius, 0.0])

        arc = compute_great_circle_arc(p1, p2, n_points=50)

        # First point should match p1
        assert np.allclose(arc[0], p1, atol=1e-10), \
            f"Arc start {arc[0]} doesn't match p1 {p1}"

        # Last point should match p2
        assert np.allclose(arc[-1], p2, atol=1e-10), \
            f"Arc end {arc[-1]} doesn't match p2 {p2}"

    def test_arc_points_on_sphere(self):
        """Test that all arc points lie exactly on sphere surface."""
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, radius])

        arc = compute_great_circle_arc(p1, p2, n_points=100)

        # All points should have distance = radius from origin
        radii = np.linalg.norm(arc, axis=1)

        assert np.allclose(radii, radius, rtol=1e-5), \
            f"Arc points not on sphere: radii range from {radii.min()} to {radii.max()}, expected {radius}"

    def test_arc_smoothness(self):
        """Test that arc has approximately constant angular velocity.

        SLERP should produce constant angular spacing between points.
        """
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, radius])

        n_points = 50
        arc = compute_great_circle_arc(p1, p2, n_points=n_points)

        # Compute angles between consecutive points
        angles = []
        for i in range(len(arc) - 1):
            v1 = arc[i] / np.linalg.norm(arc[i])
            v2 = arc[i+1] / np.linalg.norm(arc[i+1])
            cos_angle = np.dot(v1, v2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)

        angles = np.array(angles)

        # All angles should be approximately equal (constant angular velocity)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)

        # Standard deviation should be very small relative to mean
        assert std_angle / mean_angle < 0.01, \
            f"Arc not smooth: angle std={std_angle}, mean={mean_angle}"

    def test_arc_identical_points(self):
        """Test arc generation for identical start and end points."""
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([radius, 0.0, 0.0])

        arc = compute_great_circle_arc(p1, p2, n_points=20)

        # All points should be identical (or very close)
        for point in arc:
            assert np.allclose(point, p1, atol=1e-5), \
                f"Point {point} deviates from {p1} for identical endpoints"

    def test_arc_antipodal_points(self):
        """Test arc generation for antipodal (opposite) points."""
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([-radius, 0.0, 0.0])

        arc = compute_great_circle_arc(p1, p2, n_points=50)

        # Arc should pass through equator (x=0 plane)
        # Find point closest to x=0
        x_coords = np.abs(arc[:, 0])
        min_x_idx = np.argmin(x_coords)

        # This point should be near the equator (y or z should be ±radius)
        equator_point = arc[min_x_idx]
        distance_from_origin = np.linalg.norm(equator_point)

        assert np.isclose(distance_from_origin, radius, rtol=1e-5), \
            f"Antipodal arc doesn't stay on sphere at midpoint"

    def test_arc_various_radii(self):
        """Test arc generation for different sphere radii."""
        p1_unit = np.array([1.0, 0.0, 0.0])
        p2_unit = np.array([0.0, 1.0, 0.0])

        for radius in [0.5, 1.0, 2.0, 10.0]:
            p1 = p1_unit * radius
            p2 = p2_unit * radius

            arc = compute_great_circle_arc(p1, p2, n_points=50)

            # All points should be at correct radius
            radii = np.linalg.norm(arc, axis=1)
            assert np.allclose(radii, radius, rtol=1e-5), \
                f"Arc not on sphere of radius {radius}"


class TestGeodesicIntegration:
    """Integration tests comparing geodesic vs. Euclidean distances."""

    def test_geodesic_euclidean_ratios(self):
        """Test geodesic/Euclidean distance ratios for various separations.

        Expected ratios from SESSION_SUMMARY.md:
        - Adjacent (θ=10°): 1.00
        - Quarter sphere (θ=90°): 1.11
        - Opposite poles (θ=180°): 1.57
        """
        radius = 1.0

        test_cases = [
            # (angle_deg, expected_ratio)
            (10, 1.00),
            (90, 1.11),
            (180, 1.57),
        ]

        for angle_deg, expected_ratio in test_cases:
            # Create two points separated by angle
            angle_rad = np.radians(angle_deg)
            p1 = np.array([radius, 0.0, 0.0])
            p2 = np.array([
                radius * np.cos(angle_rad),
                radius * np.sin(angle_rad),
                0.0
            ])

            geodesic_dist = geodesic_distance_on_sphere(p1, p2, radius)
            euclidean_dist = np.linalg.norm(p1 - p2)

            ratio = geodesic_dist / euclidean_dist

            # Allow 2% tolerance
            assert np.isclose(ratio, expected_ratio, rtol=0.02), \
                f"Angle {angle_deg}°: ratio {ratio:.2f} != expected {expected_ratio}"

    def test_no_volume_penetration_visual(self):
        """Visual verification helper: ensure arc stays on surface.

        This test computes the minimum distance from arc points to sphere surface.
        All points should be ON the surface (distance ≈ 0).
        """
        radius = 1.0
        p1 = np.array([radius, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, radius])

        arc = compute_great_circle_arc(p1, p2, n_points=100)

        # Compute distance from each arc point to sphere surface
        # Distance to surface = |r_point - r_sphere|
        for point in arc:
            r_point = np.linalg.norm(point)
            distance_to_surface = abs(r_point - radius)

            assert distance_to_surface < 1e-5, \
                f"Arc point at distance {distance_to_surface} from surface (should be ~0)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
