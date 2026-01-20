"""Two-sphere model for paired brain region analysis.

Based on the MRISpheres/twospheres research for Alzheimer's disease analysis.
"""

from typing import List, Tuple, Optional
import numpy as np

__all__ = ['TwoSphereModel', 'compute_sphere_mesh', 'compute_distance']


class TwoSphereModel:
    """Model paired brain regions as two spheres for correlation analysis."""
    
    def __init__(self, radius: float = 1.0):
        """Initialize two-sphere model.
        
        Args:
            radius: Default radius for spheres
        """
        self.default_radius = radius
        self.spheres = []
    
    def add_sphere(self, center: List[float], label: str = None, 
                   radius: Optional[float] = None):
        """Add a sphere to the model.
        
        Args:
            center: [x, y, z] center coordinates
            label: Optional label for the brain region
            radius: Optional custom radius (uses default if None)
        """
        self.spheres.append({
            'center': np.array(center),
            'radius': radius or self.default_radius,
            'label': label or f'Region_{len(self.spheres) + 1}'
        })
    
    def compute_distance(self, idx1: int = 0, idx2: int = 1) -> float:
        """Compute distance between two sphere centers.
        
        Args:
            idx1: Index of first sphere
            idx2: Index of second sphere
            
        Returns:
            Euclidean distance between centers
        """
        if len(self.spheres) < 2:
            raise ValueError("Need at least 2 spheres")
        
        c1 = self.spheres[idx1]['center']
        c2 = self.spheres[idx2]['center']
        return float(np.linalg.norm(c1 - c2))
    
    def check_overlap(self, idx1: int = 0, idx2: int = 1) -> bool:
        """Check if two spheres overlap.
        
        Args:
            idx1: Index of first sphere
            idx2: Index of second sphere
            
        Returns:
            True if spheres overlap
        """
        dist = self.compute_distance(idx1, idx2)
        r1 = self.spheres[idx1]['radius']
        r2 = self.spheres[idx2]['radius']
        return dist < (r1 + r2)
    
    def generate_mesh(self, sphere_idx: int = 0, 
                      resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate mesh coordinates for a sphere.
        
        Args:
            sphere_idx: Index of sphere to mesh
            resolution: Number of points in each dimension
            
        Returns:
            Tuple of (X, Y, Z) coordinate arrays
        """
        sphere = self.spheres[sphere_idx]
        center = sphere['center']
        radius = sphere['radius']
        
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        U, V = np.meshgrid(u, v)
        
        X = center[0] + radius * np.sin(V) * np.cos(U)
        Y = center[1] + radius * np.sin(V) * np.sin(U)
        Z = center[2] + radius * np.cos(V)
        
        return X, Y, Z
    
    def to_dict(self) -> dict:
        """Convert model to dictionary representation."""
        return {
            'default_radius': self.default_radius,
            'spheres': [
                {
                    'center': s['center'].tolist(),
                    'radius': s['radius'],
                    'label': s['label']
                }
                for s in self.spheres
            ],
            'num_spheres': len(self.spheres)
        }


def compute_sphere_mesh(center: List[float], radius: float, 
                        resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mesh coordinates for a sphere.
    
    Args:
        center: [x, y, z] center coordinates
        radius: Sphere radius
        resolution: Number of points
        
    Returns:
        Tuple of (X, Y, Z) coordinate arrays
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    U, V = np.meshgrid(u, v)
    
    X = center[0] + radius * np.sin(V) * np.cos(U)
    Y = center[1] + radius * np.sin(V) * np.sin(U)
    Z = center[2] + radius * np.cos(V)
    
    return X, Y, Z


def compute_distance(center1: List[float], center2: List[float]) -> float:
    """Compute Euclidean distance between two points.
    
    Args:
        center1: First point [x, y, z]
        center2: Second point [x, y, z]
        
    Returns:
        Distance
    """
    c1 = np.array(center1)
    c2 = np.array(center2)
    return float(np.linalg.norm(c1 - c2))
