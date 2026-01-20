"""Ray tracing simulation wrappers for pyoptools.

Supports standard glass optics and microfluidic PDMS/air lenses.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

__all__ = ['RayTracer', 'PDMSLens', 'trace_parallel_beam', 'trace_point_source',
           'MATERIAL_LIBRARY', 'create_pdms_air_lens']


# =============================================================================
# Material Library
# =============================================================================

# Refractive indices at 550nm (visible)
MATERIAL_LIBRARY = {
    # Standard optical glasses
    'BK7': 1.5168,       # Borosilicate crown glass at 550nm
    'N-BK7': 1.5168,
    'SF11': 1.7847,      # Dense flint glass
    'FUSED_SILICA': 1.4585,
    
    # Microfluidic / soft lithography materials
    'PDMS': 1.41,        # Polydimethylsiloxane (Sylgard 184)
    'PDMS_CURED': 1.43,  # Fully cured PDMS (slightly higher)
    'SU8': 1.596,        # SU-8 photoresist
    'PMMA': 1.492,       # Polymethyl methacrylate (acrylic)
    'NOA61': 1.56,       # Norland Optical Adhesive 61
    'NOA81': 1.56,       # Norland Optical Adhesive 81
    
    # Common fluids
    'AIR': 1.0003,
    'WATER': 1.333,
    'GLYCEROL': 1.473,
    'OIL_IMMERSION': 1.515,
    
    # Biological
    'CYTOPLASM': 1.36,   # Typical cell cytoplasm
    'BRAIN_TISSUE': 1.37,  # Gray matter approximate
}

# NIR refractive indices (800-1000nm) - dispersion correction
MATERIAL_LIBRARY_NIR = {
    'BK7': 1.5108,       # ~0.4% lower at NIR
    'N-BK7': 1.5108,
    'SF11': 1.7720,
    'FUSED_SILICA': 1.4533,
    'PDMS': 1.403,       # PDMS at 900nm
    'PDMS_800': 1.405,   # PDMS at 800nm
    'PDMS_1000': 1.400,  # PDMS at 1000nm
    'PDMS_CURED': 1.423,
    'SU8': 1.583,
    'PMMA': 1.484,
    'WATER': 1.328,
    'AIR': 1.0003,
}


def get_material_nir(name: str, wavelength_nm: float = 900) -> float:
    """Get refractive index for a material at NIR wavelengths.
    
    Args:
        name: Material name (case-insensitive)
        wavelength_nm: Wavelength in nm (800-1000nm range)
        
    Returns:
        Refractive index at specified wavelength
    """
    key = name.upper().replace('-', '_').replace(' ', '_')
    
    # Check NIR library first
    if key in MATERIAL_LIBRARY_NIR:
        base_n = MATERIAL_LIBRARY_NIR[key]
    elif key in MATERIAL_LIBRARY:
        # Apply approximate dispersion correction for NIR
        base_n = MATERIAL_LIBRARY[key] * 0.995  # ~0.5% lower in NIR
    else:
        raise ValueError(f"Unknown material: {name}")
    
    # Fine wavelength adjustment for PDMS (main use case)
    if 'PDMS' in key and key not in ['PDMS_800', 'PDMS_1000']:
        # Linear interpolation: 1.405 at 800nm, 1.400 at 1000nm
        if 800 <= wavelength_nm <= 1000:
            base_n = 1.405 - (wavelength_nm - 800) * 0.005 / 200
    
    return base_n


def get_material(name: str) -> float:
    """Get refractive index for a material.
    
    Args:
        name: Material name (case-insensitive)
        
    Returns:
        Refractive index at ~550nm
        
    Raises:
        ValueError: If material not found
    """
    key = name.upper().replace('-', '_').replace(' ', '_')
    if key in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[key]
    raise ValueError(f"Unknown material: {name}. Available: {list(MATERIAL_LIBRARY.keys())}")


# =============================================================================
# PDMS/Air Lens for Microfluidics
# =============================================================================

class PDMSLens:
    """PDMS-based microfluidic lens with air or fluid cavity.
    
    Common in lab-on-chip devices and soft lithography optics.
    The lens is formed by a PDMS/air or PDMS/fluid interface.
    """
    
    def __init__(self, 
                 radius_of_curvature: float,
                 diameter: float = 5.0,
                 pdms_thickness: float = 2.0,
                 cavity_material: str = 'AIR',
                 pdms_type: str = 'PDMS'):
        """Initialize PDMS lens.
        
        Args:
            radius_of_curvature: Radius of curved interface in mm (positive = convex into cavity)
            diameter: Lens diameter in mm
            pdms_thickness: PDMS layer thickness in mm
            cavity_material: Material filling the cavity ('AIR', 'WATER', etc.)
            pdms_type: PDMS variant ('PDMS' or 'PDMS_CURED')
        """
        self.radius = radius_of_curvature
        self.diameter = diameter
        self.pdms_thickness = pdms_thickness
        self.n_pdms = get_material(pdms_type)
        self.n_cavity = get_material(cavity_material)
        self.cavity_material = cavity_material
        
    @property
    def curvature(self) -> float:
        """Curvature (1/R) in 1/mm."""
        return 1.0 / self.radius if self.radius != 0 else 0
    
    @property
    def focal_length(self) -> float:
        """Approximate focal length using lensmaker equation for single surface.
        
        f = R / (n2 - n1) for a single refracting surface.
        """
        delta_n = self.n_pdms - self.n_cavity
        if abs(delta_n) < 1e-6:
            return float('inf')  # No refraction
        return self.radius / delta_n
    
    @property
    def numerical_aperture(self) -> float:
        """Approximate NA based on geometry."""
        half_angle = np.arctan((self.diameter / 2) / abs(self.focal_length))
        return self.n_cavity * np.sin(half_angle)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': 'pdms_lens',
            'radius_of_curvature_mm': self.radius,
            'diameter_mm': self.diameter,
            'pdms_thickness_mm': self.pdms_thickness,
            'n_pdms': self.n_pdms,
            'n_cavity': self.n_cavity,
            'cavity_material': self.cavity_material,
            'focal_length_mm': self.focal_length,
            'numerical_aperture': self.numerical_aperture
        }
    
    def create_pyoptools_component(self):
        """Create pyoptools component for this lens.
        
        For PDMS/air microlens: light enters PDMS through flat surface,
        exits through curved PDMS/air interface into the air cavity.
        
        Returns:
            pyoptools SphericalLens or None if pyoptools unavailable
        """
        try:
            from pyoptools.raytrace.comp_lib import SphericalLens
            
            # Plano-convex lens geometry:
            # - S1 (flat): light enters from air into PDMS
            # - S2 (curved): light exits PDMS into air
            # For a converging plano-convex lens, S2 curvature should be negative
            # (center of curvature is on the incident side)
            return SphericalLens(
                radius=self.diameter / 2,
                curvature_s1=0,  # Flat entry surface
                curvature_s2=-self.curvature,  # Convex exit surface (negative for converging)
                thickness=self.pdms_thickness,
                material=self.n_pdms
            )
        except ImportError:
            return None
    
    def create_pyoptools_system(self, detector_distance: float = None):
        """Create a complete pyoptools system with this lens and a detector.
        
        Args:
            detector_distance: Distance to detector in mm (default: focal length)
            
        Returns:
            Tuple of (System, CCD) or (None, None) if pyoptools unavailable
        """
        try:
            from pyoptools.raytrace.system import System
            from pyoptools.raytrace.comp_lib import CCD
            
            lens = self.create_pyoptools_component()
            if lens is None:
                return None, None
            
            if detector_distance is None:
                detector_distance = abs(self.focal_length)
            
            ccd = CCD(size=(self.diameter, self.diameter))
            
            system = System(
                complist=[
                    (lens, (0, 0, 0), (0, 0, 0)),
                    (ccd, (0, 0, detector_distance), (0, 0, 0))
                ],
                n=1.0  # Air surrounding
            )
            
            return system, ccd
        except ImportError:
            return None, None


def create_pdms_air_lens(radius: float, diameter: float = 5.0) -> PDMSLens:
    """Convenience function to create a PDMS/air lens.
    
    Args:
        radius: Radius of curvature in mm
        diameter: Lens diameter in mm
        
    Returns:
        PDMSLens instance
    """
    return PDMSLens(
        radius_of_curvature=radius,
        diameter=diameter,
        cavity_material='AIR'
    )


# =============================================================================
# General Ray Tracer
# =============================================================================

class RayTracer:
    """Wrapper for pyoptools ray tracing.
    
    Supports both standard glass optics and PDMS microfluidic lenses.
    """
    
    def __init__(self, wavelength_nm: float = 550.0):
        """Initialize ray tracer.
        
        Args:
            wavelength_nm: Light wavelength in nanometers
        """
        self.wavelength = wavelength_nm * 1e-9  # Convert to meters
        self.wavelength_nm = wavelength_nm
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
                 focal_length: float, diameter: float = 25.0,
                 material: str = 'BK7'):
        """Add a glass lens to the optical system.
        
        Args:
            position: (x, y, z) position in mm
            focal_length: Focal length in mm
            diameter: Lens diameter in mm
            material: Material name from MATERIAL_LIBRARY
        """
        n = get_material(material)
        self.elements.append({
            'type': 'lens',
            'position': position,
            'focal_length': focal_length,
            'diameter': diameter,
            'material': material,
            'refractive_index': n
        })
    
    def add_pdms_lens(self, position: Tuple[float, float, float],
                      radius_of_curvature: float,
                      diameter: float = 5.0,
                      cavity_material: str = 'AIR'):
        """Add a PDMS microfluidic lens.
        
        Args:
            position: (x, y, z) position in mm
            radius_of_curvature: Radius in mm (positive = convex into cavity)
            diameter: Lens diameter in mm
            cavity_material: 'AIR', 'WATER', etc.
        """
        lens = PDMSLens(
            radius_of_curvature=radius_of_curvature,
            diameter=diameter,
            cavity_material=cavity_material
        )
        self.elements.append({
            'type': 'pdms_lens',
            'position': position,
            'pdms_lens': lens,
            **lens.to_dict()
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
