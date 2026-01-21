"""Tests for two-sphere geometry and geodesic calculations."""

import numpy as np
import pytest

from src.backend.mri.sphere_mapping import (
    SphericalPoint,
    spherical_to_cartesian,
    cartesian_to_spherical,
    compute_geodesic_distance,
    quaternion_rotate,
    create_two_sphere_model,
    compute_interhemispheric_distance,
)


class TestSphericalPoint:
    """Tests for SphericalPoint class."""

    def test_initialization(self):
        """Test point initialization."""
        point = SphericalPoint(theta=0, phi=np.pi/2, radius=1.0)
        assert point.theta == 0
        assert point.phi == np.pi/2
        assert point.radius == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        point = SphericalPoint(theta=np.pi/4, phi=np.pi/3, radius=2.0)
        d = point.to_dict()
        assert d["theta"] == np.pi/4
        assert d["phi"] == np.pi/3
        assert d["radius"] == 2.0


class TestCoordinateTransformations:
    """Tests for coordinate transformations."""

    @pytest.mark.asyncio
    async def test_spherical_to_cartesian_equator(self):
        """Point on equator (φ=π/2) at θ=0 should be at [1, 0, 0]."""
        cart = await spherical_to_cartesian(theta=0, phi=np.pi/2, radius=1.0)
        expected = np.array([1.0, 0.0, 0.0])
        assert np.allclose(cart, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_spherical_to_cartesian_north_pole(self):
        """North pole (φ=0) should be at [0, 0, 1]."""
        cart = await spherical_to_cartesian(theta=0, phi=0, radius=1.0)
        expected = np.array([0.0, 0.0, 1.0])
        assert np.allclose(cart, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_spherical_to_cartesian_south_pole(self):
        """South pole (φ=π) should be at [0, 0, -1]."""
        cart = await spherical_to_cartesian(theta=0, phi=np.pi, radius=1.0)
        expected = np.array([0.0, 0.0, -1.0])
        assert np.allclose(cart, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_spherical_to_cartesian_with_center(self):
        """Test with non-origin center."""
        center = np.array([1.0, 2.0, 3.0])
        cart = await spherical_to_cartesian(
            theta=0, phi=np.pi/2, radius=1.0, center=center
        )
        expected = np.array([2.0, 2.0, 3.0])  # center + [1, 0, 0]
        assert np.allclose(cart, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_cartesian_to_spherical_roundtrip(self):
        """Round-trip conversion should recover original coordinates."""
        theta_orig = np.pi / 4
        phi_orig = np.pi / 3
        radius_orig = 2.0

        # Convert to Cartesian
        cart = await spherical_to_cartesian(theta_orig, phi_orig, radius_orig)

        # Convert back to spherical
        theta, phi, radius = await cartesian_to_spherical(cart)

        assert np.isclose(theta, theta_orig, atol=1e-10)
        assert np.isclose(phi, phi_orig, atol=1e-10)
        assert np.isclose(radius, radius_orig, atol=1e-10)

    @pytest.mark.asyncio
    async def test_cartesian_to_spherical_at_origin(self):
        """Point at origin should return zero radius."""
        origin = np.array([0.0, 0.0, 0.0])
        theta, phi, radius = await cartesian_to_spherical(origin)
        assert radius == 0.0


class TestGeodesicDistance:
    """Tests for geodesic distance computation."""

    @pytest.mark.asyncio
    async def test_geodesic_distance_equator_quarter_circle(self):
        """Two points on equator, 90° apart = π/2 arc length."""
        p1 = {"theta": 0, "phi": np.pi/2}
        p2 = {"theta": np.pi/2, "phi": np.pi/2}

        dist = await compute_geodesic_distance(p1, p2, radius=1.0)
        expected = np.pi / 2  # Quarter circle

        assert np.isclose(dist, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_geodesic_distance_same_point(self):
        """Distance from point to itself should be zero."""
        p = {"theta": np.pi/4, "phi": np.pi/3}

        dist = await compute_geodesic_distance(p, p, radius=1.0)

        assert np.isclose(dist, 0.0, atol=1e-10)

    @pytest.mark.asyncio
    async def test_geodesic_distance_antipodal(self):
        """Antipodal points (opposite sides of sphere) should be π apart."""
        # North pole
        p1 = {"theta": 0, "phi": 0}
        # South pole
        p2 = {"theta": 0, "phi": np.pi}

        dist = await compute_geodesic_distance(p1, p2, radius=1.0)
        expected = np.pi  # Half circumference

        assert np.isclose(dist, expected, atol=1e-10)

    @pytest.mark.asyncio
    async def test_geodesic_distance_scaled_radius(self):
        """Geodesic distance should scale linearly with radius."""
        p1 = {"theta": 0, "phi": np.pi/2}
        p2 = {"theta": np.pi/2, "phi": np.pi/2}

        dist_r1 = await compute_geodesic_distance(p1, p2, radius=1.0)
        dist_r2 = await compute_geodesic_distance(p1, p2, radius=2.0)

        assert np.isclose(dist_r2, 2 * dist_r1, atol=1e-10)

    @pytest.mark.asyncio
    async def test_geodesic_vs_euclidean(self):
        """Geodesic distance should be >= Euclidean distance."""
        p1 = {"theta": 0, "phi": np.pi/4}
        p2 = {"theta": np.pi/3, "phi": np.pi/3}
        radius = 1.0

        geodesic = await compute_geodesic_distance(p1, p2, radius)

        # Compute Euclidean distance
        cart1 = await spherical_to_cartesian(p1["theta"], p1["phi"], radius)
        cart2 = await spherical_to_cartesian(p2["theta"], p2["phi"], radius)
        euclidean = np.linalg.norm(cart1 - cart2)

        assert geodesic >= euclidean


class TestQuaternionRotation:
    """Tests for quaternion-based rotation."""

    @pytest.mark.asyncio
    async def test_quaternion_rotate_z_axis_90deg(self):
        """Rotate point 90° around z-axis."""
        # Point at (1, 0, 0) on equator
        p = {"theta": 0, "phi": np.pi/2}
        axis = np.array([0, 0, 1])
        angle = np.pi / 2

        p_rot = await quaternion_rotate(p, angle, axis, radius=1.0)

        # Should rotate to (0, 1, 0), which is θ=π/2, φ=π/2
        assert np.isclose(p_rot["theta"], np.pi/2, atol=1e-6)
        assert np.isclose(p_rot["phi"], np.pi/2, atol=1e-6)

    @pytest.mark.asyncio
    async def test_quaternion_rotate_preserves_radius(self):
        """Rotation should preserve distance from center."""
        p = {"theta": np.pi/6, "phi": np.pi/4}
        axis = np.array([1, 1, 0])
        angle = np.pi / 3
        radius = 2.0

        p_rot = await quaternion_rotate(p, angle, axis, radius=radius)

        assert np.isclose(p_rot["radius"], radius, atol=1e-6)

    @pytest.mark.asyncio
    async def test_quaternion_rotate_identity(self):
        """Rotation by 0 radians should leave point unchanged."""
        p = {"theta": np.pi/4, "phi": np.pi/3}
        axis = np.array([1, 0, 0])
        angle = 0.0

        p_rot = await quaternion_rotate(p, angle, axis, radius=1.0)

        assert np.isclose(p_rot["theta"], p["theta"], atol=1e-6)
        assert np.isclose(p_rot["phi"], p["phi"], atol=1e-6)

    @pytest.mark.asyncio
    async def test_quaternion_rotate_preserves_geodesic_distance(self):
        """Rotation should preserve geodesic distances between points."""
        p1 = {"theta": 0, "phi": np.pi/4}
        p2 = {"theta": np.pi/6, "phi": np.pi/3}
        radius = 1.0

        # Distance before rotation
        dist_before = await compute_geodesic_distance(p1, p2, radius)

        # Rotate both points by same rotation
        axis = np.array([0, 0, 1])
        angle = np.pi / 4
        p1_rot = await quaternion_rotate(p1, angle, axis, radius)
        p2_rot = await quaternion_rotate(p2, angle, axis, radius)

        # Distance after rotation
        dist_after = await compute_geodesic_distance(p1_rot, p2_rot, radius)

        assert np.isclose(dist_before, dist_after, atol=1e-6)


class TestTwoSphereModel:
    """Tests for two-sphere brain model."""

    @pytest.mark.asyncio
    async def test_create_two_sphere_model_default(self):
        """Test default two-sphere model creation."""
        model = await create_two_sphere_model()

        assert "sphere1" in model
        assert "sphere2" in model
        assert model["sphere1"]["label"] == "right_hemisphere"
        assert model["sphere2"]["label"] == "left_hemisphere"
        assert model["separation"] == 0.0

    @pytest.mark.asyncio
    async def test_two_sphere_model_centers(self):
        """Test sphere centers are correctly positioned."""
        radius = 1.0
        model = await create_two_sphere_model(radius=radius)

        center1 = model["sphere1"]["center"]
        center2 = model["sphere2"]["center"]

        # Right hemisphere at y = +radius
        assert np.isclose(center1[1], radius, atol=1e-10)

        # Left hemisphere at y = -radius
        assert np.isclose(center2[1], -radius, atol=1e-10)

    @pytest.mark.asyncio
    async def test_two_sphere_model_with_separation(self):
        """Test model with separation between spheres."""
        radius = 1.0
        separation = 0.5
        model = await create_two_sphere_model(radius=radius, separation=separation)

        center1 = model["sphere1"]["center"]
        center2 = model["sphere2"]["center"]

        distance_between = np.linalg.norm(center1 - center2)
        expected_distance = 2 * radius + separation

        assert np.isclose(distance_between, expected_distance, atol=1e-10)

    @pytest.mark.asyncio
    async def test_interhemispheric_distance(self):
        """Test interhemispheric distance calculation."""
        model = await create_two_sphere_model(radius=1.0)

        # Points on equator of each sphere (closest approach)
        p1 = {"theta": 0, "phi": np.pi/2}  # On sphere 1
        p2 = {"theta": 0, "phi": np.pi/2}  # On sphere 2

        dist = await compute_interhemispheric_distance(p1, p2, model)

        # For touching spheres at equator, distance should be ~0
        # (actually 2*radius since centers are at [0, ±1, 0])
        assert dist >= 0
        assert dist <= 3.0  # Reasonable upper bound


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    @pytest.mark.asyncio
    async def test_geodesic_distance_numerical_stability(self):
        """Test numerical stability for very close points."""
        p1 = {"theta": 0, "phi": np.pi/2}
        p2 = {"theta": 1e-10, "phi": np.pi/2}  # Very close

        dist = await compute_geodesic_distance(p1, p2, radius=1.0)

        # Should be very small but non-negative
        assert dist >= 0
        assert dist < 1e-8

    @pytest.mark.asyncio
    async def test_spherical_coordinates_theta_wrap(self):
        """Test that theta wraps correctly to [0, 2π]."""
        # Start at θ = -π/4 (negative)
        cart = await spherical_to_cartesian(theta=-np.pi/4, phi=np.pi/2, radius=1.0)
        theta, phi, radius = await cartesian_to_spherical(cart)

        # Should wrap to positive value
        assert theta >= 0
        assert theta < 2 * np.pi
