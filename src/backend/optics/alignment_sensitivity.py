"""Alignment sensitivity analysis for LOC fabrication tolerance budgeting.

Monte Carlo simulation of coupling efficiency vs. lateral and angular misalignment.
Adapts tolerance analysis methodology for fiber-to-chip coupling in integrated photonics.

Based on:
- Coupling efficiency models from fiber_optics.py
- Statistical tolerance analysis for manufacturing
"""

import asyncio
from typing import Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class AlignmentToleranceSpec:
    """Fabrication tolerance specifications for alignment analysis.

    Attributes:
        lateral_tolerance_um: Lateral (x,y) positioning tolerance (±3σ) in µm
        angular_tolerance_deg: Angular (tip/tilt) tolerance (±3σ) in degrees
        z_tolerance_um: Axial (z) positioning tolerance (±3σ) in µm
        wavelength_tolerance_nm: Wavelength variation (±3σ) in nm
    """
    lateral_tolerance_um: float = 1.0  # Typical: ±1 µm for active alignment
    angular_tolerance_deg: float = 0.5  # Typical: ±0.5° for fiber coupling
    z_tolerance_um: float = 2.0  # Typical: ±2 µm for focal position
    wavelength_tolerance_nm: float = 1.0  # Typical: ±1 nm for laser diode


class AlignmentSensitivityAnalyzer:
    """Monte Carlo analyzer for LOC alignment tolerance and coupling efficiency.

    Simulates fabrication variations and computes statistical coupling efficiency
    distributions for fiber-to-chip and chip-to-chip optical interfaces.
    """

    def __init__(
        self,
        wavelength_nm: float = 1550,
        fiber_mfd_um: float = 10.4,  # SMF-28 mode field diameter
        spot_size_um: float = 3.0,   # Waveguide spot size
        na_fiber: float = 0.14,       # Numerical aperture
        working_distance_um: float = 10.0
    ):
        """Initialize alignment sensitivity analyzer.

        Args:
            wavelength_nm: Operating wavelength in nm
            fiber_mfd_um: Fiber mode field diameter in µm (1/e² intensity)
            spot_size_um: Waveguide/chip spot size in µm (1/e² intensity)
            na_fiber: Numerical aperture of fiber
            working_distance_um: Working distance from fiber facet to chip
        """
        self.wavelength_nm = wavelength_nm
        self.fiber_mfd_um = fiber_mfd_um
        self.spot_size_um = spot_size_um
        self.na_fiber = na_fiber
        self.working_distance_um = working_distance_um

        # Gaussian beam parameters
        self.w0_fiber = fiber_mfd_um / 2  # Beam waist radius (µm)
        self.w0_chip = spot_size_um / 2    # Chip mode waist radius (µm)

    async def compute_coupling_efficiency(
        self,
        lateral_offset_um: float = 0.0,
        angular_offset_deg: float = 0.0,
        z_offset_um: float = 0.0
    ) -> float:
        """Compute coupling efficiency for given misalignment.

        Uses Gaussian beam overlap integral for butt coupling efficiency:
            η = (4 * (w1*w2)^2) / ((w1^2 + w2^2)^2) * exp(-2*Δx^2/(w1^2+w2^2))

        Args:
            lateral_offset_um: Lateral (x or y) misalignment in µm
            angular_offset_deg: Angular (tilt) misalignment in degrees
            z_offset_um: Axial (z) misalignment in µm

        Returns:
            Coupling efficiency (0 to 1, typically converted to dB as 10*log10(η))
        """
        def _compute():
            # Convert angles to radians
            theta_rad = np.deg2rad(angular_offset_deg)

            # Beam propagation: spot size grows with z-offset
            # w(z) = w0 * sqrt(1 + (z*λ/(π*w0^2))^2)
            wavelength_um = self.wavelength_nm / 1000
            z_rayleigh = np.pi * self.w0_fiber**2 / wavelength_um  # Rayleigh range

            w_fiber_at_chip = self.w0_fiber * np.sqrt(1 + ((self.working_distance_um + z_offset_um) / z_rayleigh)**2)

            # Mode mismatch factor (Gaussian beam overlap)
            w1 = w_fiber_at_chip
            w2 = self.w0_chip
            mode_mismatch = (4 * (w1 * w2)**2) / ((w1**2 + w2**2)**2)

            # Lateral offset loss
            lateral_loss = np.exp(-2 * lateral_offset_um**2 / (w1**2 + w2**2))

            # Angular offset loss (phase front mismatch)
            # Approximation: η_angular ≈ exp(-(π*w*θ/λ)^2)
            w_avg = (w1 + w2) / 2
            angular_loss = np.exp(-((np.pi * w_avg * np.tan(theta_rad)) / wavelength_um)**2)

            # Total coupling efficiency
            efficiency = mode_mismatch * lateral_loss * angular_loss

            return float(np.clip(efficiency, 0.0, 1.0))

        return await asyncio.to_thread(_compute)

    async def run_monte_carlo(
        self,
        tolerance_spec: AlignmentToleranceSpec,
        n_samples: int = 10000,
        correlation_xy: float = 0.0
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation of alignment errors.

        Samples from Gaussian distributions representing fabrication process
        variations and computes statistical coupling efficiency distribution.

        Args:
            tolerance_spec: Fabrication tolerance specifications (±3σ values)
            n_samples: Number of Monte Carlo samples
            correlation_xy: Correlation coefficient between x and y errors (0-1)

        Returns:
            Dict containing:
                - mean_efficiency: Mean coupling efficiency
                - std_efficiency: Standard deviation
                - efficiency_dB_mean: Mean efficiency in dB
                - efficiency_dB_std: Std dev in dB
                - percentiles: 1%, 5%, 50%, 95%, 99% efficiency values
                - yield_90_percent: Fraction of samples with η > 0.9
                - yield_50_percent: Fraction of samples with η > 0.5
                - worst_case_efficiency: Minimum efficiency in simulation
                - samples: Optional array of all efficiency values
        """
        def _simulate():
            # Convert ±3σ tolerances to 1σ standard deviations
            sigma_lateral = tolerance_spec.lateral_tolerance_um / 3.0
            sigma_angular = tolerance_spec.angular_tolerance_deg / 3.0
            sigma_z = tolerance_spec.z_tolerance_um / 3.0

            # Generate correlated x-y lateral offsets
            if correlation_xy > 0:
                # Cholesky decomposition for correlated sampling
                cov_matrix = np.array([
                    [sigma_lateral**2, correlation_xy * sigma_lateral**2],
                    [correlation_xy * sigma_lateral**2, sigma_lateral**2]
                ])
                lateral_offsets = np.random.multivariate_normal(
                    [0, 0], cov_matrix, n_samples
                )
                x_offsets = lateral_offsets[:, 0]
                y_offsets = lateral_offsets[:, 1]
            else:
                # Independent x and y
                x_offsets = np.random.normal(0, sigma_lateral, n_samples)
                y_offsets = np.random.normal(0, sigma_lateral, n_samples)

            # Radial lateral offset: r = sqrt(x^2 + y^2)
            radial_offsets = np.sqrt(x_offsets**2 + y_offsets**2)

            # Angular and z-offsets (independent)
            angular_offsets = np.random.normal(0, sigma_angular, n_samples)
            z_offsets = np.random.normal(0, sigma_z, n_samples)

            return radial_offsets, angular_offsets, z_offsets

        # Generate Monte Carlo samples
        radial_offsets, angular_offsets, z_offsets = await asyncio.to_thread(_simulate)

        # Compute coupling efficiency for each sample
        efficiencies = []
        for i in range(n_samples):
            eff = await self.compute_coupling_efficiency(
                lateral_offset_um=radial_offsets[i],
                angular_offset_deg=angular_offsets[i],
                z_offset_um=z_offsets[i]
            )
            efficiencies.append(eff)

        efficiencies = np.array(efficiencies)

        # Convert to dB (avoid log(0) with clipping)
        efficiencies_dB = 10 * np.log10(np.clip(efficiencies, 1e-10, 1.0))

        # Statistical analysis
        percentiles = np.percentile(efficiencies, [1, 5, 50, 95, 99])
        yield_90 = np.mean(efficiencies > 0.9)
        yield_50 = np.mean(efficiencies > 0.5)

        return {
            "mean_efficiency": float(np.mean(efficiencies)),
            "std_efficiency": float(np.std(efficiencies)),
            "efficiency_dB_mean": float(np.mean(efficiencies_dB)),
            "efficiency_dB_std": float(np.std(efficiencies_dB)),
            "percentiles": {
                "1%": float(percentiles[0]),
                "5%": float(percentiles[1]),
                "50%": float(percentiles[2]),
                "95%": float(percentiles[3]),
                "99%": float(percentiles[4])
            },
            "percentiles_dB": {
                "1%": float(10 * np.log10(percentiles[0])),
                "5%": float(10 * np.log10(percentiles[1])),
                "50%": float(10 * np.log10(percentiles[2])),
                "95%": float(10 * np.log10(percentiles[3])),
                "99%": float(10 * np.log10(percentiles[4]))
            },
            "yield_90_percent": float(yield_90),
            "yield_50_percent": float(yield_50),
            "worst_case_efficiency": float(np.min(efficiencies)),
            "worst_case_efficiency_dB": float(np.min(efficiencies_dB)),
            "n_samples": n_samples,
            "tolerance_spec": {
                "lateral_tolerance_um": tolerance_spec.lateral_tolerance_um,
                "angular_tolerance_deg": tolerance_spec.angular_tolerance_deg,
                "z_tolerance_um": tolerance_spec.z_tolerance_um
            }
        }

    async def compute_sensitivity_map(
        self,
        lateral_range_um: Tuple[float, float] = (-5.0, 5.0),
        angular_range_deg: Tuple[float, float] = (-2.0, 2.0),
        n_points: int = 50
    ) -> Dict[str, Any]:
        """Compute 2D sensitivity map of coupling efficiency vs. misalignment.

        Creates a heatmap of coupling efficiency as a function of lateral
        and angular offsets for visualization and tolerance analysis.

        Args:
            lateral_range_um: (min, max) lateral offset range in µm
            angular_range_deg: (min, max) angular offset range in degrees
            n_points: Number of grid points per axis

        Returns:
            Dict containing:
                - lateral_offsets: Array of lateral offset values
                - angular_offsets: Array of angular offset values
                - efficiency_map: 2D array (n_points x n_points) of efficiencies
                - efficiency_dB_map: 2D array of efficiencies in dB
        """
        def _compute():
            lateral_offsets = np.linspace(lateral_range_um[0], lateral_range_um[1], n_points)
            angular_offsets = np.linspace(angular_range_deg[0], angular_range_deg[1], n_points)

            efficiency_map = np.zeros((n_points, n_points))

            return lateral_offsets, angular_offsets, efficiency_map

        lateral_offsets, angular_offsets, efficiency_map = await asyncio.to_thread(_compute)

        # Compute efficiency for each (lateral, angular) pair
        for i, lateral in enumerate(lateral_offsets):
            for j, angular in enumerate(angular_offsets):
                efficiency_map[i, j] = await self.compute_coupling_efficiency(
                    lateral_offset_um=lateral,
                    angular_offset_deg=angular,
                    z_offset_um=0.0
                )

        # Convert to dB
        efficiency_dB_map = 10 * np.log10(np.clip(efficiency_map, 1e-10, 1.0))

        return {
            "lateral_offsets_um": lateral_offsets.tolist(),
            "angular_offsets_deg": angular_offsets.tolist(),
            "efficiency_map": efficiency_map.tolist(),
            "efficiency_dB_map": efficiency_dB_map.tolist(),
            "grid_shape": (n_points, n_points)
        }

    async def compute_tolerance_budget(
        self,
        target_efficiency: float = 0.5
    ) -> Dict[str, float]:
        """Compute allowable tolerance budget for target coupling efficiency.

        Determines maximum allowable lateral and angular misalignment
        to maintain specified coupling efficiency threshold.

        Args:
            target_efficiency: Target minimum efficiency (0 to 1)
            z_offset_um: Nominal z-offset in µm

        Returns:
            Dict with:
                - lateral_tolerance_um: Max lateral offset for target efficiency
                - angular_tolerance_deg: Max angular offset for target efficiency
                - target_efficiency: Input target efficiency
                - target_efficiency_dB: Target efficiency in dB
        """
        def _compute():
            # Binary search for lateral tolerance
            lateral_low, lateral_high = 0.0, 20.0  # µm
            for _ in range(20):  # Binary search iterations
                lateral_mid = (lateral_low + lateral_high) / 2
                # Need to call async function - will compute synchronously for simplicity
                # In production, use proper async iteration
                lateral_high = lateral_mid  # Placeholder

            # Binary search for angular tolerance
            angular_low, angular_high = 0.0, 5.0  # degrees
            for _ in range(20):
                angular_mid = (angular_low + angular_high) / 2
                angular_high = angular_mid  # Placeholder

            # Simplified: use analytical approximation
            # For lateral: η = exp(-2*Δx^2/w^2) ≈ target_efficiency
            # => Δx = w * sqrt(-ln(target_efficiency) / 2)
            w_avg = (self.w0_fiber + self.w0_chip) / 2
            lateral_tol = w_avg * np.sqrt(-np.log(target_efficiency) / 2)

            # For angular: η ≈ exp(-(π*w*θ/λ)^2)
            # => θ ≈ (λ/(π*w)) * sqrt(-ln(target_efficiency))
            wavelength_um = self.wavelength_nm / 1000
            angular_tol_rad = (wavelength_um / (np.pi * w_avg)) * np.sqrt(-np.log(target_efficiency))
            angular_tol_deg = np.rad2deg(angular_tol_rad)

            return lateral_tol, angular_tol_deg

        lateral_tol, angular_tol_deg = await asyncio.to_thread(_compute)

        return {
            "lateral_tolerance_um": float(lateral_tol),
            "angular_tolerance_deg": float(angular_tol_deg),
            "target_efficiency": float(target_efficiency),
            "target_efficiency_dB": float(10 * np.log10(target_efficiency))
        }
