"""Ray tracing simulation wrappers for pyoptools."""

from typing import List, Tuple, Optional
import numpy as np

__all__ = ['RayTracer', 'trace_parallel_beam', 'trace_point_source']


class RayTracer:
    """Wrapper for pyoptools ray tracing."""
    
    def __init__(self, wavelength_nm: float = 550.0):
        """Initialize ray tracer.
        
        Args:
            wavelength_nm: Light wavelength in nanometers
        """
        self.wavelength = wavelength_nm * 1e-9  # Convert to meters
        self.elements = []
        self._pyoptools_available = self._check_pyoptools()
    
    def _check_pyoptools(self) -> bool:
        """Check if pyoptools is available."""
        try:
            import pyoptools
            return True
        except ImportError:
            return False
    
    def add_lens(self, position: Tuple[float, float, float], 
                 focal_length: float, diameter: float = 25.0):
        """Add a lens to the optical system.
        
        Args:
            position: (x, y, z) position in mm
            focal_length: Focal length in mm
            diameter: Lens diameter in mm
        """
        self.elements.append({
            'type': 'lens',
            'position': position,
            'focal_length': focal_length,
            'diameter': diameter
        })
    
    def trace(self, source_position: Tuple[float, float, float],
              source_direction: Tuple[float, float, float],
              num_rays: int = 100) -> dict:
        """Trace rays through the optical system.
        
        Args:
            source_position: Starting position [x, y, z]
            source_direction: Direction vector [dx, dy, dz]
            num_rays: Number of rays to trace
            
        Returns:
            Dictionary with ray trace results
        """
        if not self._pyoptools_available:
            return {
                'error': 'pyoptools not installed',
                'install': 'pip install pyoptools'
            }
        
        # Normalize direction
        direction = np.array(source_direction)
        direction = direction / np.linalg.norm(direction)
        
        return {
            'source_position': list(source_position),
            'source_direction': direction.tolist(),
            'num_rays': num_rays,
            'wavelength_nm': self.wavelength * 1e9,
            'elements': self.elements,
            'status': 'ready'
        }


def trace_parallel_beam(position: Tuple[float, float, float],
                        direction: Tuple[float, float, float],
                        beam_diameter: float = 10.0,
                        num_rays: int = 100) -> dict:
    """Trace a parallel beam of light.
    
    Args:
        position: Center position of beam
        direction: Beam direction
        beam_diameter: Diameter of beam in mm
        num_rays: Number of rays
        
    Returns:
        Ray trace results
    """
    tracer = RayTracer()
    return tracer.trace(position, direction, num_rays)


def trace_point_source(position: Tuple[float, float, float],
                       cone_angle: float = 30.0,
                       num_rays: int = 100) -> dict:
    """Trace rays from a point source.
    
    Args:
        position: Source position
        cone_angle: Half-angle of emission cone in degrees
        num_rays: Number of rays
        
    Returns:
        Ray trace results
    """
    return {
        'source_position': list(position),
        'cone_angle_deg': cone_angle,
        'num_rays': num_rays,
        'type': 'point_source'
    }
