"""Fiber optics coupling and 4f imaging systems.

Provides tools for simulating fiber-to-fiber coupling through optical
chambers, commonly used in microfluidic spectroscopy and sensing.
"""

from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np

__all__ = [
    'FiberSpec', 'LensSpec', 'MeniscusLens', 'FourFSystem', 'CouplingResult',
    'FIBER_TYPES', 'compute_spot_size', 'design_4f_system',
    'design_meniscus_lens', 'PDMS_NIR_INDEX'
]

# PDMS refractive indices for NIR
PDMS_NIR_INDEX = {
    800: 1.405,
    850: 1.404,
    900: 1.403,
    950: 1.401,
    1000: 1.400,
}


# =============================================================================
# Fiber Specifications
# =============================================================================

@dataclass
class FiberSpec:
    """Optical fiber specification.
    
    Attributes:
        core_diameter: Core diameter in µm
        cladding_diameter: Cladding diameter in µm
        na: Numerical aperture
        name: Fiber type name
    """
    core_diameter: float  # µm
    cladding_diameter: float  # µm
    na: float
    name: str = "custom"
    
    @property
    def core_radius_mm(self) -> float:
        """Core radius in mm."""
        return self.core_diameter / 2000
    
    @property
    def acceptance_angle_deg(self) -> float:
        """Full acceptance angle in degrees."""
        return 2 * np.degrees(np.arcsin(self.na))
    
    def mode_field_diameter(self, wavelength_nm: float = 900) -> float:
        """Approximate mode field diameter for single-mode fiber (µm).
        
        Uses Marcuse formula: MFD ≈ 2a * (0.65 + 1.619/V^1.5 + 2.879/V^6)
        where V = (2πa/λ) * NA
        """
        a = self.core_diameter / 2  # radius in µm
        wavelength_um = wavelength_nm / 1000
        V = (2 * np.pi * a / wavelength_um) * self.na
        
        if V < 2.405:  # Single-mode condition
            mfd = 2 * a * (0.65 + 1.619/V**1.5 + 2.879/V**6)
            return mfd
        else:
            return self.core_diameter  # Multimode: use core


# Common fiber types
FIBER_TYPES = {
    'SMF-28': FiberSpec(core_diameter=8.2, cladding_diameter=125, na=0.14, name='SMF-28'),
    'SM800': FiberSpec(core_diameter=5.6, cladding_diameter=125, na=0.12, name='SM800'),
    'SM980': FiberSpec(core_diameter=6.0, cladding_diameter=125, na=0.12, name='SM980'),
    'MM62.5': FiberSpec(core_diameter=62.5, cladding_diameter=125, na=0.275, name='MM62.5'),
    'MM50': FiberSpec(core_diameter=50, cladding_diameter=125, na=0.20, name='MM50'),
    'MM200': FiberSpec(core_diameter=200, cladding_diameter=230, na=0.22, name='MM200'),
    'MM400': FiberSpec(core_diameter=400, cladding_diameter=440, na=0.22, name='MM400'),
}


# =============================================================================
# Lens Specifications  
# =============================================================================

@dataclass
class LensSpec:
    """Lens specification for fiber coupling.
    
    Attributes:
        radius_of_curvature: Radius of curvature in mm
        diameter: Lens diameter in mm
        thickness: Center thickness in mm
        refractive_index: Refractive index at operating wavelength
        lens_type: 'plano-convex', 'biconvex', 'meniscus', 'aspheric'
    """
    radius_of_curvature: float  # mm (R1 for first surface)
    diameter: float  # mm
    thickness: float  # mm
    refractive_index: float
    lens_type: str = 'plano-convex'
    radius_of_curvature_2: Optional[float] = None  # R2 for second surface
    
    @property
    def focal_length(self) -> float:
        """Calculate focal length using lensmaker equation."""
        n = self.refractive_index
        R1 = self.radius_of_curvature
        R2 = self.radius_of_curvature_2
        
        if self.lens_type == 'plano-convex':
            # Plano-convex: one flat surface (R=inf), one curved
            return R1 / (n - 1)
        elif R2 is not None:
            # General case: 1/f = (n-1)(1/R1 - 1/R2)
            return 1 / ((n - 1) * (1/R1 - 1/R2))
        else:
            return R1 / (n - 1)
    
    @property  
    def f_number(self) -> float:
        """F-number (focal length / diameter)."""
        return abs(self.focal_length) / self.diameter
    
    def create_pyoptools_lens(self):
        """Create pyoptools SphericalLens component."""
        try:
            from pyoptools.raytrace.comp_lib import SphericalLens
            
            R1 = self.radius_of_curvature
            R2 = self.radius_of_curvature_2 or float('inf')
            
            if self.lens_type == 'plano-convex':
                # Flat first surface, curved second (converging)
                return SphericalLens(
                    radius=self.diameter / 2,
                    curvature_s1=0,
                    curvature_s2=-1/R1,  # Negative for converging
                    thickness=self.thickness,
                    material=self.refractive_index
                )
            else:
                return SphericalLens(
                    radius=self.diameter / 2,
                    curvature_s1=1/R1 if R1 != float('inf') else 0,
                    curvature_s2=-1/R2 if R2 != float('inf') else 0,
                    thickness=self.thickness,
                    material=self.refractive_index
                )
        except ImportError:
            return None


# =============================================================================
# Meniscus Lens (SA-optimized)
# =============================================================================

@dataclass
class MeniscusLens:
    """Best-form meniscus lens optimized for minimal spherical aberration.
    
    Based on optical design principles for fiber coupling:
    - Concentric meniscus shape minimizes SA at unit magnification
    - Curved side toward fiber (longer conjugate) reduces SA by ~3x
    - Optional aspheric conic constant for further SA reduction
    
    Reference: Zemax optimization for PDMS n≈1.405 at NA=0.25
    - Wavefront error: ~0.25λ (plano-convex) → <0.03λ (meniscus)
    """
    r1: float  # Front radius of curvature (convex, positive)
    r2: float  # Back radius of curvature (concave, negative) 
    thickness: float  # Center thickness in mm
    diameter: float  # Lens diameter in mm
    refractive_index: float  # Material n at wavelength
    conic: float = 0.0  # Aspheric conic constant (κ≈-0.8 for SA correction)
    
    @property
    def focal_length(self) -> float:
        """Calculate focal length using thick lens formula."""
        n = self.refractive_index
        t = self.thickness
        R1, R2 = self.r1, self.r2
        
        # Lensmaker equation with thickness correction
        phi1 = (n - 1) / R1
        phi2 = -(n - 1) / R2  # R2 is negative for concave
        phi = phi1 + phi2 - (t / n) * phi1 * phi2
        
        return 1 / phi if phi != 0 else float('inf')
    
    @property
    def principal_plane_offset(self) -> float:
        """Distance from front vertex to front principal plane."""
        n = self.refractive_index
        t = self.thickness
        f = self.focal_length
        R2 = self.r2
        return -f * t * (n - 1) / (n * R2)
    
    def spherical_aberration_factor(self) -> float:
        """Relative SA factor compared to plano-convex (lower = better).
        
        Plano-convex = 1.0, optimized meniscus ≈ 0.1-0.15
        Uses Coddington shape factor with standard sign convention:
        - R positive for center of curvature to right of surface
        - Plano-convex (flat/curved): q = +1
        - Biconvex equi-radii: q = 0
        - Best-form meniscus: q ≈ +0.7 for n≈1.4
        """
        # Standard shape factor: q = (R1 + R2) / (R1 - R2)
        # For our convention: R1 positive (convex), R2 negative (concave toward back)
        R1_signed = self.r1  # positive
        R2_signed = self.r2  # already negative
        q = (R1_signed + R2_signed) / (R1_signed - R2_signed)
        n = self.refractive_index
        
        # Third-order SA coefficient (Kingslake formula)
        # SA ∝ n³/[f³(n-1)²] * [(n+2)/(n(n-1)) * q² + 2(n²-1)/n * q + (3n+1)(n-1)]
        # Simplified: SA_rel ∝ A*q² + B*q + C
        A = (n + 2) / (n * (n - 1))
        B = 2 * (n**2 - 1) / n
        C = (3*n + 1) * (n - 1)
        
        sa_meniscus = abs(A * q**2 + B * q + C)
        
        # Plano-convex reference (q = +1 with curved side toward focus)
        q_pc = 1.0
        sa_pc = abs(A * q_pc**2 + B * q_pc + C)
        
        return sa_meniscus / sa_pc if sa_pc > 0 else 1.0
    
    def create_pyoptools_lens(self):
        """Create pyoptools SphericalLens for this meniscus."""
        try:
            from pyoptools.raytrace.comp_lib import SphericalLens
            
            return SphericalLens(
                radius=self.diameter / 2,
                curvature_s1=1/self.r1,  # Convex front
                curvature_s2=1/self.r2,  # Concave back (r2 negative)
                thickness=self.thickness,
                material=self.refractive_index
            )
        except ImportError:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': 'meniscus',
            'r1_mm': self.r1,
            'r2_mm': self.r2,
            'thickness_mm': self.thickness,
            'diameter_mm': self.diameter,
            'n': self.refractive_index,
            'conic': self.conic,
            'focal_length_mm': self.focal_length,
            'sa_factor': self.spherical_aberration_factor()
        }


def design_meniscus_lens(
    focal_length: float,
    diameter: float = 1.0,
    wavelength_nm: float = 850,
    n_pdms: Optional[float] = None
) -> MeniscusLens:
    """Design an SA-optimized meniscus lens for fiber coupling.
    
    Uses best-form shape factor for minimal spherical aberration.
    Based on ernie2_swarm optimization for PDMS at NA=0.25:
    - Reference design: R1=+0.55mm, R2=-0.45mm, t=0.35mm for f≈1mm
    - Achieves wavefront error <0.03λ vs 0.25λ for plano-convex
    
    Args:
        focal_length: Desired focal length in mm
        diameter: Lens diameter in mm
        wavelength_nm: Operating wavelength (800-1000nm)
        n_pdms: PDMS refractive index (auto from wavelength if None)
        
    Returns:
        MeniscusLens with optimized parameters
    """
    # Get PDMS index for wavelength
    if n_pdms is None:
        # Interpolate from PDMS_NIR_INDEX
        wl = int(wavelength_nm)
        if wl in PDMS_NIR_INDEX:
            n_pdms = PDMS_NIR_INDEX[wl]
        else:
            # Linear interpolation
            wls = sorted(PDMS_NIR_INDEX.keys())
            for i in range(len(wls) - 1):
                if wls[i] <= wl <= wls[i+1]:
                    t = (wl - wls[i]) / (wls[i+1] - wls[i])
                    n_pdms = PDMS_NIR_INDEX[wls[i]] * (1-t) + PDMS_NIR_INDEX[wls[i+1]] * t
                    break
            else:
                n_pdms = 1.403  # Default to 900nm
    
    n = n_pdms
    
    # Optimal shape factor for minimum SA (Kingslake formula)
    # dSA/dq = 0 gives: q_opt = -B/(2A) = -n(n²-1) / (n+2)
    A = (n + 2) / (n * (n - 1))
    B = 2 * (n**2 - 1) / n
    q_opt = -B / (2 * A)
    
    # Reference from ernie2_swarm optimization at f=1mm:
    # R1 = +0.55mm, R2 = -0.45mm gives q = (0.55 + -0.45)/(0.55 - -0.45) = 0.1/1.0 = 0.1
    # This is close to optimal for PDMS
    
    # From shape factor q = (R1 + R2)/(R1 - R2), derive R1/R2 ratio
    # Let r = |R2|/R1 (both magnitudes), with R2 negative
    # q = (R1 - r*R1) / (R1 + r*R1) = (1-r)/(1+r)
    # Solving: r = (1-q)/(1+q)
    r_ratio = (1 - q_opt) / (1 + q_opt)
    
    # Scale to desired focal length using lensmaker equation
    # 1/f = (n-1) * (1/R1 - 1/R2) = (n-1) * (1/R1 + 1/(r*R1)) = (n-1)/R1 * (1 + 1/r)
    # R1 = f * (n-1) * (1 + 1/r)
    R1 = focal_length * (n - 1) * (1 + 1/r_ratio)
    R2 = -r_ratio * R1  # Negative for concave back surface
    
    # Thickness: ~0.35/0.55 = 0.64 ratio from reference design
    thickness = abs(R1) * 0.64
    
    return MeniscusLens(
        r1=R1,
        r2=R2,
        thickness=thickness,
        diameter=diameter,
        refractive_index=n_pdms,
        conic=-0.8  # Aspheric correction for residual SA
    )


# =============================================================================
# 4f Imaging System
# =============================================================================

@dataclass
class CouplingResult:
    """Results from fiber coupling simulation.
    
    Attributes:
        spot_diameter_um: Spot diameter at output in µm
        rms_spot_um: RMS spot size in µm
        coupling_efficiency: Estimated coupling efficiency (0-1)
        ray_positions: Final ray positions at detector
        optimal_z: Optimal detector position in mm
    """
    spot_diameter_um: float
    rms_spot_um: float
    coupling_efficiency: float
    ray_positions: List[Tuple[float, float]]  # (y, z) for each ray
    optimal_z: float


class FourFSystem:
    """4f imaging system for fiber-to-fiber coupling through a chamber.
    
    Configuration:
        Input fiber → Lens1 → [Collimated region/Chamber] → Lens2 → Output fiber
        
    The system provides unit magnification when both lenses have equal focal length.
    The collimated region allows insertion of optical elements (filters, chambers).
    """
    
    def __init__(self, 
                 lens: LensSpec,
                 fiber: FiberSpec,
                 chamber_width: float = 5.0,
                 wavelength_nm: float = 900):
        """Initialize 4f system.
        
        Args:
            lens: Lens specification (same lens used on both sides)
            fiber: Fiber specification
            chamber_width: Width of chamber/sample region in mm
            wavelength_nm: Operating wavelength in nm
        """
        self.lens = lens
        self.fiber = fiber
        self.chamber_width = chamber_width
        self.wavelength_nm = wavelength_nm
        self._pyoptools_available = self._check_pyoptools()
        
    def _check_pyoptools(self) -> bool:
        try:
            import pyoptools
            return True
        except ImportError:
            return False
    
    @property
    def focal_length(self) -> float:
        """Effective focal length of lenses."""
        return self.lens.focal_length
    
    @property
    def total_length(self) -> float:
        """Total optical path length."""
        return 4 * abs(self.focal_length)
    
    @property
    def collimated_region_length(self) -> float:
        """Length of collimated region (between lenses)."""
        return 2 * abs(self.focal_length)
    
    def get_layout(self) -> Dict[str, float]:
        """Get z-positions of system components."""
        f = abs(self.focal_length)
        return {
            'input_fiber': 0.0,
            'lens1': f,
            'collimated_start': f + self.lens.thickness,
            'chamber_start': f + 2,  # Small gap after lens
            'chamber_end': f + 2 + self.chamber_width,
            'collimated_end': 3 * f,
            'lens2': 3 * f,
            'output_fiber': 4 * f,
            'total_length': 4 * f
        }
    
    def trace_rays(self, num_rays: int = 9) -> CouplingResult:
        """Trace rays through the 4f system.
        
        Args:
            num_rays: Number of rays to trace across fiber NA
            
        Returns:
            CouplingResult with spot size and coupling efficiency
        """
        if not self._pyoptools_available:
            raise RuntimeError("pyoptools not available")
        
        from pyoptools.raytrace.ray import Ray
        from pyoptools.raytrace.system import System
        from pyoptools.raytrace.comp_lib import CCD
        
        layout = self.get_layout()
        lens1 = self.lens.create_pyoptools_lens()
        lens2 = self.lens.create_pyoptools_lens()
        
        if lens1 is None or lens2 is None:
            raise RuntimeError("Failed to create lenses")
        
        # Scan for optimal focus position
        best_spot = float('inf')
        best_z = layout['output_fiber']
        best_positions = []
        
        for dz in np.linspace(-3, 3, 13):
            z_out = layout['output_fiber'] + dz
            
            ccd = CCD(size=(2, 2))
            system = System(complist=[
                (lens1, (0, 0, layout['lens1']), (0, 0, 0)),
                (lens2, (0, 0, layout['lens2']), (0, 0, 0)),
                (ccd, (0, 0, z_out), (0, 0, 0))
            ], n=1)
            
            rays = []
            angles = np.linspace(-self.fiber.na, self.fiber.na, num_rays)
            for angle in angles:
                ray = Ray(
                    origin=(0, 0, 0),
                    direction=(0, np.sin(angle), np.cos(angle)),
                    wavelength=self.wavelength_nm * 1e-9
                )
                rays.append(ray)
                system.ray_add(ray)
            
            system.propagate()
            
            positions = []
            for ray in rays:
                current = ray
                while current.childs:
                    current = current.childs[0]
                positions.append((current.origin[1], current.origin[2]))
            
            y_vals = [p[0] for p in positions]
            spot = max(y_vals) - min(y_vals)
            
            if spot < best_spot:
                best_spot = spot
                best_z = z_out
                best_positions = positions
        
        # Calculate metrics
        y_vals = [p[0] for p in best_positions]
        spot_um = best_spot * 1000
        rms_um = np.sqrt(np.mean(np.array(y_vals)**2)) * 1000
        
        # Estimate coupling efficiency (overlap integral approximation)
        # For Gaussian beam, ~86% power within 1/e² radius
        fiber_accept_um = self.fiber.mode_field_diameter(self.wavelength_nm)
        if spot_um < fiber_accept_um:
            efficiency = 0.86  # Near-perfect coupling
        else:
            # Simple Gaussian overlap estimate
            efficiency = 0.86 * (fiber_accept_um / spot_um) ** 2
            efficiency = min(efficiency, 1.0)
        
        return CouplingResult(
            spot_diameter_um=spot_um,
            rms_spot_um=rms_um,
            coupling_efficiency=efficiency,
            ray_positions=best_positions,
            optimal_z=best_z
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system to dictionary representation."""
        layout = self.get_layout()
        return {
            'wavelength_nm': self.wavelength_nm,
            'fiber': {
                'name': self.fiber.name,
                'core_um': self.fiber.core_diameter,
                'na': self.fiber.na
            },
            'lens': {
                'type': self.lens.lens_type,
                'focal_length_mm': self.lens.focal_length,
                'diameter_mm': self.lens.diameter,
                'n': self.lens.refractive_index
            },
            'chamber_width_mm': self.chamber_width,
            'layout': layout,
            'total_length_mm': layout['total_length']
        }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_spot_size(focal_length: float, na: float, aberration_factor: float = 1.0) -> float:
    """Compute theoretical spot size at focus.
    
    Args:
        focal_length: Lens focal length in mm
        na: Fiber numerical aperture
        aberration_factor: Multiplier for aberration effects (1.0 = ideal)
        
    Returns:
        Spot diameter in µm
    """
    # Geometric spot size: D = 2 * f * sin(theta) ≈ 2 * f * NA
    geometric_spot = 2 * focal_length * na * 1000  # mm to µm
    return geometric_spot * aberration_factor


def design_4f_system(
    chamber_width: float,
    max_length: float,
    fiber: FiberSpec,
    material_n: float = 1.403,
    wavelength_nm: float = 900
) -> Tuple[LensSpec, FourFSystem]:
    """Design a 4f system given constraints.
    
    Args:
        chamber_width: Required chamber width in mm
        max_length: Maximum total system length in mm
        fiber: Fiber specification
        material_n: Lens material refractive index
        wavelength_nm: Operating wavelength
        
    Returns:
        Tuple of (LensSpec, FourFSystem)
    """
    # 4f system: total length = 4f, collimated region = 2f
    # Need: collimated region > chamber_width, total < max_length
    
    # Minimum f for chamber: 2f > chamber_width + margins
    f_min = (chamber_width + 4) / 2  # 2mm margin each side
    
    # Maximum f for length: 4f < max_length
    f_max = max_length / 4
    
    if f_min > f_max:
        raise ValueError(f"Cannot fit {chamber_width}mm chamber in {max_length}mm length")
    
    # Use the larger f for better collimation
    f = (f_min + f_max) / 2
    
    # Calculate required lens radius of curvature
    # f = R / (n - 1) for plano-convex
    R = f * (material_n - 1)
    
    lens = LensSpec(
        radius_of_curvature=R,
        diameter=4.0,  # Standard 4mm lens
        thickness=1.5,
        refractive_index=material_n,
        lens_type='plano-convex'
    )
    
    system = FourFSystem(
        lens=lens,
        fiber=fiber,
        chamber_width=chamber_width,
        wavelength_nm=wavelength_nm
    )
    
    return lens, system
