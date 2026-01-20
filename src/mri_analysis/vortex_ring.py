"""Vortex ring and trefoil knot modeling for neural connectivity.

Based on VortexRingInterpolation.py from MRISpheres/twospheres.
"""

from typing import Tuple, Optional
import numpy as np

__all__ = ['VortexRing', 'trefoil_knot', 'compute_frenet_frame']


class VortexRing:
    """Model vortex ring structures for neural pathway analysis."""
    
    def __init__(self, scale_x: float = 5.0, scale_y: float = 0.5, 
                 scale_z: float = 1.0, n_turns: int = 3):
        """Initialize vortex ring.
        
        Args:
            scale_x: Scale factor for x-coordinate
            scale_y: Scale factor for y-coordinate  
            scale_z: Scale factor for z-coordinate
            n_turns: Number of turns (3 for trefoil knot)
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.n_turns = n_turns
        self._curve = None
    
    def compute_curve(self, num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the trefoil knot curve.
        
        Args:
            num_points: Number of points on the curve
            
        Returns:
            Tuple of (x, y, z) coordinate arrays
        """
        t = np.linspace(0, 4 * np.pi, num_points)
        
        x = self.scale_x * np.sin(self.n_turns * t)
        y = self.scale_y * (np.cos(t) + 2 * np.cos(2 * t))
        z = self.scale_z * (np.sin(t) - 2 * np.sin(2 * t))
        
        self._curve = (x, y, z)
        return x, y, z
    
    def interpolate_smooth(self, num_points: int = 1000):
        """Get smoothly interpolated curve using splines.
        
        Args:
            num_points: Number of output points
            
        Returns:
            Tuple of (xi, yi, zi) interpolated coordinates
        """
        from scipy.interpolate import splprep, splev
        
        if self._curve is None:
            self.compute_curve(num_points)
        
        x, y, z = self._curve
        tck, u = splprep([x, y, z], s=0)
        xi, yi, zi = splev(u, tck)
        
        return xi, yi, zi
    
    def compute_tube_surface(self, tube_radius: float = 0.2, 
                             circle_points: int = 100) -> np.ndarray:
        """Compute tube surface around the knot curve.
        
        Uses Frenet-Serret frame to orient circles along the curve.
        
        Args:
            tube_radius: Radius of the tube
            circle_points: Number of points per circle
            
        Returns:
            Array of shape (N, 3) with surface points
        """
        xi, yi, zi = self.interpolate_smooth()
        
        theta = np.linspace(0, 2 * np.pi, circle_points)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        
        surface_points = []
        
        for i in range(1, len(xi) - 1):
            # Compute tangent using central difference
            tangent = np.array([
                xi[i+1] - xi[i-1],
                yi[i+1] - yi[i-1],
                zi[i+1] - zi[i-1]
            ])
            tangent = tangent / np.linalg.norm(tangent)
            
            # Find orthogonal vectors
            if abs(tangent[0]) < 0.9:
                helper = np.array([1, 0, 0])
            else:
                helper = np.array([0, 1, 0])
            
            normal = np.cross(tangent, helper)
            normal = normal / np.linalg.norm(normal)
            binormal = np.cross(tangent, normal)
            
            # Create rotation matrix
            R = np.column_stack([normal, binormal, tangent])
            
            # Transform circle points
            center = np.array([xi[i], yi[i], zi[i]])
            for cp in circle:
                point = center + tube_radius * (cp[0] * normal + cp[1] * binormal)
                surface_points.append(point)
        
        return np.array(surface_points)
    
    def compute_length(self) -> float:
        """Compute the curve length."""
        xi, yi, zi = self.interpolate_smooth()
        ds = np.sqrt(np.diff(xi)**2 + np.diff(yi)**2 + np.diff(zi)**2)
        return float(np.sum(ds))
    
    def bounding_box(self) -> dict:
        """Get bounding box of the curve."""
        xi, yi, zi = self.interpolate_smooth()
        return {
            'x': [float(np.min(xi)), float(np.max(xi))],
            'y': [float(np.min(yi)), float(np.max(yi))],
            'z': [float(np.min(zi)), float(np.max(zi))]
        }


def trefoil_knot(t: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate trefoil knot coordinates.
    
    Args:
        t: Parameter array (typically 0 to 4*pi)
        scale: Overall scale factor
        
    Returns:
        Tuple of (x, y, z) arrays
    """
    x = scale * np.sin(3 * t)
    y = scale * (np.cos(t) + 2 * np.cos(2 * t)) / 3
    z = scale * (np.sin(t) - 2 * np.sin(2 * t)) / 3
    return x, y, z


def compute_frenet_frame(curve_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Frenet-Serret frame (T, N, B) along a curve.
    
    Args:
        curve_points: Array of shape (N, 3) with curve points
        
    Returns:
        Tuple of (tangent, normal, binormal) arrays, each of shape (N-2, 3)
    """
    tangents = []
    normals = []
    binormals = []
    
    for i in range(1, len(curve_points) - 1):
        # Tangent from central difference
        T = curve_points[i+1] - curve_points[i-1]
        T = T / np.linalg.norm(T)
        
        # Second derivative approximation
        d2 = curve_points[i+1] - 2*curve_points[i] + curve_points[i-1]
        
        # Normal perpendicular to tangent
        N = d2 - np.dot(d2, T) * T
        if np.linalg.norm(N) > 1e-10:
            N = N / np.linalg.norm(N)
        else:
            # Fallback for straight sections
            if abs(T[0]) < 0.9:
                N = np.cross(T, [1, 0, 0])
            else:
                N = np.cross(T, [0, 1, 0])
            N = N / np.linalg.norm(N)
        
        # Binormal
        B = np.cross(T, N)
        
        tangents.append(T)
        normals.append(N)
        binormals.append(B)
    
    return np.array(tangents), np.array(normals), np.array(binormals)
