"""Wavefront analysis using Zernike polynomials."""

from typing import List, Optional, Tuple
import numpy as np

__all__ = ['WavefrontAnalyzer', 'zernike_polynomial', 'compute_rms', 'compute_pv']


def zernike_polynomial(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute Zernike polynomial Z_n^m.
    
    Args:
        n: Radial order (n >= 0)
        m: Azimuthal frequency (-n <= m <= n, n-|m| even)
        rho: Normalized radial coordinate (0 <= rho <= 1)
        theta: Angular coordinate in radians
        
    Returns:
        Zernike polynomial values
    """
    # Simplified implementation for first few terms
    if n == 0 and m == 0:
        return np.ones_like(rho)  # Piston
    elif n == 1 and m == 1:
        return rho * np.cos(theta)  # Tilt X
    elif n == 1 and m == -1:
        return rho * np.sin(theta)  # Tilt Y
    elif n == 2 and m == 0:
        return 2 * rho**2 - 1  # Defocus
    elif n == 2 and m == 2:
        return rho**2 * np.cos(2 * theta)  # Astigmatism
    elif n == 2 and m == -2:
        return rho**2 * np.sin(2 * theta)  # Astigmatism 45
    elif n == 3 and m == 1:
        return (3 * rho**3 - 2 * rho) * np.cos(theta)  # Coma X
    elif n == 3 and m == -1:
        return (3 * rho**3 - 2 * rho) * np.sin(theta)  # Coma Y
    elif n == 4 and m == 0:
        return 6 * rho**4 - 6 * rho**2 + 1  # Spherical
    else:
        return np.zeros_like(rho)


class WavefrontAnalyzer:
    """Analyze optical wavefronts using Zernike decomposition."""
    
    def __init__(self, aperture_radius: float = 10.0, resolution: int = 256):
        """Initialize wavefront analyzer.
        
        Args:
            aperture_radius: Aperture radius in mm
            resolution: Grid resolution
        """
        self.aperture_radius = aperture_radius
        self.resolution = resolution
        
        # Create coordinate grids
        x = np.linspace(-aperture_radius, aperture_radius, resolution)
        y = np.linspace(-aperture_radius, aperture_radius, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2) / aperture_radius
        self.Theta = np.arctan2(self.Y, self.X)
        self.mask = self.R <= 1
    
    def set_coefficients(self, coefficients: List[float]) -> np.ndarray:
        """Set Zernike coefficients and compute wavefront.
        
        Args:
            coefficients: List of Zernike coefficients (Noll ordering)
            
        Returns:
            Wavefront array
        """
        W = np.zeros_like(self.R)
        
        # Map to (n, m) pairs (simplified Noll ordering)
        noll_map = [
            (0, 0),   # 1: Piston
            (1, 1),   # 2: Tilt X
            (1, -1),  # 3: Tilt Y
            (2, 0),   # 4: Defocus
            (2, -2),  # 5: Astigmatism 45
            (2, 2),   # 6: Astigmatism 0
            (3, -1),  # 7: Coma Y
            (3, 1),   # 8: Coma X
            (3, -3),  # 9: Trefoil Y
            (3, 3),   # 10: Trefoil X
            (4, 0),   # 11: Spherical
        ]
        
        for i, coeff in enumerate(coefficients):
            if i < len(noll_map) and coeff != 0:
                n, m = noll_map[i]
                W += coeff * zernike_polynomial(n, m, self.R, self.Theta)
        
        W = np.where(self.mask, W, np.nan)
        self.wavefront = W
        return W
    
    def compute_metrics(self) -> dict:
        """Compute wavefront metrics.
        
        Returns:
            Dictionary with RMS, PV, and other metrics
        """
        if not hasattr(self, 'wavefront'):
            return {'error': 'No wavefront set. Call set_coefficients first.'}
        
        W = self.wavefront
        return {
            'rms': float(np.nanstd(W)),
            'pv': float(np.nanmax(W) - np.nanmin(W)),
            'mean': float(np.nanmean(W)),
            'aperture_radius_mm': self.aperture_radius,
            'resolution': self.resolution
        }


def compute_rms(wavefront: np.ndarray) -> float:
    """Compute RMS wavefront error."""
    return float(np.nanstd(wavefront))


def compute_pv(wavefront: np.ndarray) -> float:
    """Compute peak-to-valley wavefront error."""
    return float(np.nanmax(wavefront) - np.nanmin(wavefront))
