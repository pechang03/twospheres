"""Two-sphere geometry and geodesic distance calculations.

Implements spherical coordinate transformations and geodesic distance
computations for brain hemisphere modeling (MRISpheres/twospheres pattern).

Key concepts:
- Brain hemispheres as two touching spheres
- Geodesic distance = great circle arc length
- Haversine formula for surface distance
- Quaternion rotation without gimbal lock
"""

import asyncio
from typing import Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray


class SphericalPoint:
    """Point on sphere surface in spherical coordinates."""

    def __init__(self, theta: float, phi: float, radius: float = 1.0):
        """
        Initialize spherical point.

        Args:
            theta: Azimuthal angle (longitude) in radians [0, 2π]
            phi: Polar angle (latitude) in radians [0, π]
            radius: Sphere radius (default 1.0)
        """
        self.theta = theta
        self.phi = phi
        self.radius = radius

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"theta": self.theta, "phi": self.phi, "radius": self.radius}


async def spherical_to_cartesian(
    theta: float,
    phi: float,
    radius: float = 1.0,
    center: Optional[NDArray] = None
) -> NDArray:
    """
    Convert spherical coordinates to Cartesian.

    Spherical coordinate convention:
    - theta: Azimuthal angle (longitude), [0, 2π]
    - phi: Polar angle from +z axis (latitude), [0, π]

    Cartesian formula:
        x = center[0] + r * sin(φ) * cos(θ)
        y = center[1] + r * sin(φ) * sin(θ)
        z = center[2] + r * cos(φ)

    Args:
        theta: Azimuthal angle in radians
        phi: Polar angle in radians
        radius: Sphere radius
        center: Sphere center [x, y, z] (default: origin)

    Returns:
        Cartesian coordinates [x, y, z]

    Example:
        >>> # Point at equator (φ=π/2), longitude 0
        >>> cart = await spherical_to_cartesian(0, np.pi/2, 1.0)
        >>> # Returns [1, 0, 0]
    """
    def _convert():
        if center is None:
            center_arr = np.array([0.0, 0.0, 0.0])
        else:
            center_arr = np.asarray(center)

        x = center_arr[0] + radius * np.sin(phi) * np.cos(theta)
        y = center_arr[1] + radius * np.sin(phi) * np.sin(theta)
        z = center_arr[2] + radius * np.cos(phi)

        return np.array([x, y, z])

    return await asyncio.to_thread(_convert)


async def cartesian_to_spherical(
    point: NDArray,
    center: Optional[NDArray] = None
) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical.

    Args:
        point: Cartesian coordinates [x, y, z]
        center: Sphere center [x, y, z] (default: origin)

    Returns:
        Tuple of (theta, phi, radius)

    Example:
        >>> cart = np.array([1, 0, 0])
        >>> theta, phi, r = await cartesian_to_spherical(cart)
        >>> # Returns (0, π/2, 1.0)
    """
    def _convert():
        if center is None:
            center_arr = np.array([0.0, 0.0, 0.0])
        else:
            center_arr = np.asarray(center)

        # Translate to sphere center
        p = np.asarray(point) - center_arr

        # Compute radius
        radius = np.linalg.norm(p)

        if radius < 1e-10:
            # At center, coordinates are undefined
            return 0.0, 0.0, 0.0

        # Compute angles
        phi = np.arccos(p[2] / radius)  # Polar angle from +z
        theta = np.arctan2(p[1], p[0])  # Azimuthal angle

        # Ensure theta in [0, 2π]
        if theta < 0:
            theta += 2 * np.pi

        return float(theta), float(phi), float(radius)

    return await asyncio.to_thread(_convert)


async def compute_geodesic_distance(
    point1: Dict[str, float],
    point2: Dict[str, float],
    radius: float = 1.0
) -> float:
    """
    Compute geodesic distance (great circle arc) between two points on sphere.

    Uses Haversine formula:
        d = 2r * arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δθ/2)))

    Or equivalently:
        d = r * arccos(sin(φ₁)sin(φ₂) + cos(φ₁)cos(φ₂)cos(Δθ))

    Args:
        point1: {"theta": θ₁, "phi": φ₁} in radians
        point2: {"theta": θ₂, "phi": φ₂} in radians
        radius: Sphere radius

    Returns:
        Geodesic distance (arc length along surface)

    Example:
        >>> # Two points on equator, 90° apart
        >>> p1 = {"theta": 0, "phi": np.pi/2}
        >>> p2 = {"theta": np.pi/2, "phi": np.pi/2}
        >>> dist = await compute_geodesic_distance(p1, p2, radius=1.0)
        >>> # Returns π/2 ≈ 1.571 (quarter circle)
    """
    def _compute():
        theta1 = point1["theta"]
        phi1 = point1["phi"]
        theta2 = point2["theta"]
        phi2 = point2["phi"]

        # Spherical law of cosines with polar angle (colatitude)
        # For polar coordinates (θ=azimuth, φ=polar from +z):
        # cos(d) = cos(φ₁)cos(φ₂) + sin(φ₁)sin(φ₂)cos(Δθ)

        delta_theta = theta2 - theta1

        # Central angle
        cos_angle = (np.cos(phi1) * np.cos(phi2) +
                     np.sin(phi1) * np.sin(phi2) * np.cos(delta_theta))

        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)

        # Arc length = radius * angle
        distance = radius * angle

        return float(distance)

    return await asyncio.to_thread(_compute)


async def quaternion_rotate(
    point: Dict[str, float],
    angle: float,
    axis: NDArray,
    radius: float = 1.0
) -> Dict[str, float]:
    """
    Rotate point on sphere using quaternion rotation.

    Quaternion rotation avoids gimbal lock and is ideal for
    rotating brain hemisphere networks on sphere surface.

    Args:
        point: {"theta": θ, "phi": φ} spherical coordinates
        angle: Rotation angle in radians
        axis: Rotation axis [x, y, z] (will be normalized)
        radius: Sphere radius

    Returns:
        Rotated point in spherical coordinates {"theta": θ', "phi": φ'}

    Example:
        >>> # Rotate point 90° around z-axis
        >>> p = {"theta": 0, "phi": np.pi/2}
        >>> axis = np.array([0, 0, 1])
        >>> p_rot = await quaternion_rotate(p, np.pi/2, axis)
        >>> # Result: point rotated along equator
    """
    def _rotate():
        # Convert to Cartesian
        theta = point["theta"]
        phi = point["phi"]

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        cart = np.array([x, y, z])

        # Normalize rotation axis
        axis_norm = axis / np.linalg.norm(axis)

        # Quaternion rotation using Rodrigues' formula
        # v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Cross product k × v
        cross = np.cross(axis_norm, cart)

        # Dot product k · v
        dot = np.dot(axis_norm, cart)

        # Rodrigues' rotation formula
        rotated = (cart * cos_angle +
                   cross * sin_angle +
                   axis_norm * dot * (1 - cos_angle))

        # Convert back to spherical
        r_rotated = np.linalg.norm(rotated)
        phi_new = np.arccos(rotated[2] / r_rotated)
        theta_new = np.arctan2(rotated[1], rotated[0])

        if theta_new < 0:
            theta_new += 2 * np.pi

        return {"theta": float(theta_new), "phi": float(phi_new), "radius": float(r_rotated)}

    return await asyncio.to_thread(_rotate)


async def create_two_sphere_model(
    radius: float = 1.0,
    separation: float = 0.0
) -> Dict[str, any]:
    """
    Create two-sphere brain model (left/right hemispheres).

    Args:
        radius: Sphere radius
        separation: Distance between sphere centers (0 = touching)

    Returns:
        Dict with sphere1, sphere2 parameters

    Example:
        >>> model = await create_two_sphere_model(radius=1.0)
        >>> # Returns two touching spheres at [0, ±1, 0]
    """
    def _create():
        # Right hemisphere (positive y)
        sphere1_center = np.array([0.0, radius + separation/2, 0.0])

        # Left hemisphere (negative y)
        sphere2_center = np.array([0.0, -radius - separation/2, 0.0])

        return {
            "sphere1": {
                "center": sphere1_center,
                "radius": radius,
                "label": "right_hemisphere"
            },
            "sphere2": {
                "center": sphere2_center,
                "radius": radius,
                "label": "left_hemisphere"
            },
            "separation": separation,
            "equator": {
                "plane": "y=0",
                "description": "Mid-sagittal plane (corpus callosum)"
            }
        }

    return await asyncio.to_thread(_create)


async def compute_interhemispheric_distance(
    point1_sphere1: Dict[str, float],
    point2_sphere2: Dict[str, float],
    model: Dict[str, any]
) -> float:
    """
    Compute distance between points on different hemispheres.

    For interhemispheric connections (e.g., via corpus callosum),
    compute shortest path crossing the equator.

    Args:
        point1_sphere1: Point on sphere 1 (right hemisphere)
        point2_sphere2: Point on sphere 2 (left hemisphere)
        model: Two-sphere model from create_two_sphere_model()

    Returns:
        Interhemispheric distance (approximate)

    Note:
        This is an approximation. True interhemispheric distance
        involves path through corpus callosum (at equator).
    """
    def _compute():
        # Convert to Cartesian
        theta1, phi1 = point1_sphere1["theta"], point1_sphere1["phi"]
        radius1 = model["sphere1"]["radius"]
        center1 = model["sphere1"]["center"]

        x1 = center1[0] + radius1 * np.sin(phi1) * np.cos(theta1)
        y1 = center1[1] + radius1 * np.sin(phi1) * np.sin(theta1)
        z1 = center1[2] + radius1 * np.cos(phi1)
        cart1 = np.array([x1, y1, z1])

        theta2, phi2 = point2_sphere2["theta"], point2_sphere2["phi"]
        radius2 = model["sphere2"]["radius"]
        center2 = model["sphere2"]["center"]

        x2 = center2[0] + radius2 * np.sin(phi2) * np.cos(theta2)
        y2 = center2[1] + radius2 * np.sin(phi2) * np.sin(theta2)
        z2 = center2[2] + radius2 * np.cos(phi2)
        cart2 = np.array([x2, y2, z2])

        # Euclidean distance (approximate interhemispheric path)
        distance = np.linalg.norm(cart1 - cart2)

        return float(distance)

    return await asyncio.to_thread(_compute)
