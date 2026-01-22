"""
Glymphatic Flow Simulator - Perivascular CSF Flow and Microfluidic Channel Model

Provides simulation of:
- Perivascular space flow (brain glymphatic system)
- Microfluidic channel flow (Lab-on-Chip devices)
- Both use Stokes flow (Re << 1) physics

Based on PH-7 design: Glymphatic-Microfluidics Integration
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


# =============================================================================
# Physical Constants
# =============================================================================

# CSF properties at 37°C
CSF_VISCOSITY = 0.001      # Pa·s (similar to water)
CSF_DENSITY = 1007         # kg/m³

# Water properties at 25°C (for LOC devices)
WATER_VISCOSITY = 0.001    # Pa·s
WATER_DENSITY = 997        # kg/m³

# Physiological parameters
CARDIAC_FREQUENCY = 1.0    # Hz (60 bpm)
ARTERIAL_PULSE_AMPLITUDE = 0.1  # relative wall displacement

# Clearance parameters (Xie et al. 2013)
AWAKE_CLEARANCE_COEFF = 0.10   # α for awake state
SLEEP_CLEARANCE_COEFF = 0.16   # α for sleep state (~60% increase)


# =============================================================================
# Geometry Classes
# =============================================================================

@dataclass
class PerivascularSpace:
    """Model of a perivascular space around a blood vessel."""

    vessel_radius_um: float      # Inner radius (blood vessel wall)
    gap_thickness_um: float      # Width of perivascular space (3-50 µm typical)
    length_mm: float             # Length of vessel segment
    vessel_type: str = 'artery'  # 'artery' or 'vein'

    @property
    def outer_radius_um(self) -> float:
        """Outer radius (tissue boundary)."""
        return self.vessel_radius_um + self.gap_thickness_um

    @property
    def cross_section_area_um2(self) -> float:
        """Cross-sectional area of perivascular space."""
        r_out = self.outer_radius_um
        r_in = self.vessel_radius_um
        return np.pi * (r_out**2 - r_in**2)

    @property
    def hydraulic_diameter_um(self) -> float:
        """Hydraulic diameter for annular flow."""
        return 2 * self.gap_thickness_um

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vessel_radius_um': self.vessel_radius_um,
            'gap_thickness_um': self.gap_thickness_um,
            'outer_radius_um': self.outer_radius_um,
            'length_mm': self.length_mm,
            'vessel_type': self.vessel_type,
            'cross_section_area_um2': float(self.cross_section_area_um2),
            'hydraulic_diameter_um': float(self.hydraulic_diameter_um)
        }


@dataclass
class MicrofluidicChannel:
    """Model of a rectangular microfluidic channel for LOC devices."""

    width_um: float       # Channel width (typical: 100-500 µm)
    height_um: float      # Channel height (typical: 50-200 µm)
    length_mm: float      # Channel length
    material: str = 'PDMS'  # Channel material

    @property
    def cross_section_area_um2(self) -> float:
        """Cross-sectional area."""
        return self.width_um * self.height_um

    @property
    def hydraulic_diameter_um(self) -> float:
        """Hydraulic diameter for rectangular channel: D_h = 4A/P."""
        A = self.cross_section_area_um2
        P = 2 * (self.width_um + self.height_um)
        return 4 * A / P

    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio."""
        return self.width_um / self.height_um

    def to_dict(self) -> Dict[str, Any]:
        return {
            'width_um': self.width_um,
            'height_um': self.height_um,
            'length_mm': self.length_mm,
            'material': self.material,
            'cross_section_area_um2': float(self.cross_section_area_um2),
            'hydraulic_diameter_um': float(self.hydraulic_diameter_um),
            'aspect_ratio': float(self.aspect_ratio)
        }


# =============================================================================
# Glymphatic Flow Simulator
# =============================================================================

class GlymphaticFlowSimulator:
    """Simulator for glymphatic CSF flow in perivascular spaces.

    Uses validated PHLoC microfluidic physics scaled to brain dimensions.
    Same Stokes flow (Re << 1) applies to both systems.
    """

    def __init__(self, state: str = 'awake'):
        """Initialize simulator.

        Args:
            state: 'awake' or 'sleep' (affects clearance coefficient)
        """
        self.state = state
        self.clearance_coeff = SLEEP_CLEARANCE_COEFF if state == 'sleep' else AWAKE_CLEARANCE_COEFF

    def compute_steady_flow(self,
                           pvs: PerivascularSpace,
                           pressure_gradient_Pa_m: float) -> Dict[str, Any]:
        """Compute steady-state flow in perivascular space.

        Uses annular Poiseuille flow solution (exact for Stokes flow).

        Args:
            pvs: Perivascular space geometry
            pressure_gradient_Pa_m: Pressure gradient (Pa/m)

        Returns:
            Flow metrics dictionary
        """
        # Convert to SI units
        r_in = pvs.vessel_radius_um * 1e-6  # m
        r_out = pvs.outer_radius_um * 1e-6  # m
        L = pvs.length_mm * 1e-3  # m
        delta = pvs.gap_thickness_um * 1e-6  # m

        mu = CSF_VISCOSITY
        dP_dz = pressure_gradient_Pa_m

        # Annular flow solution (exact)
        # Q = (π/8μ) * (dP/dz) * [r_out⁴ - r_in⁴ - (r_out² - r_in²)²/ln(r_out/r_in)]
        if r_in > 0:
            term1 = r_out**4 - r_in**4
            term2 = (r_out**2 - r_in**2)**2 / np.log(r_out / r_in)
            Q = (np.pi / (8 * mu)) * dP_dz * (term1 - term2)
        else:
            # Circular pipe (no inner wall)
            Q = (np.pi * r_out**4 / (8 * mu)) * dP_dz

        # Average velocity
        A = np.pi * (r_out**2 - r_in**2)
        v_avg = Q / A if A > 0 else 0

        # Reynolds number
        D_h = 2 * delta
        Re = CSF_DENSITY * abs(v_avg) * D_h / mu

        # Wall shear stress (at inner wall)
        tau_wall = abs(dP_dz) * r_in / 2 if r_in > 0 else abs(dP_dz) * r_out / 4

        # Flow rate in convenient units
        Q_uL_min = abs(Q) * 1e9 * 60  # m³/s → µL/min

        # Residence time
        residence_time_s = L / abs(v_avg) if v_avg != 0 else float('inf')

        return {
            'geometry': pvs.to_dict(),
            'pressure_gradient_Pa_m': pressure_gradient_Pa_m,
            'flow_rate_m3_s': float(Q),
            'flow_rate_uL_min': float(Q_uL_min),
            'average_velocity_um_s': float(v_avg * 1e6),
            'reynolds_number': float(Re),
            'wall_shear_stress_Pa': float(tau_wall),
            'wall_shear_stress_dyn_cm2': float(tau_wall * 10),  # Pa to dyn/cm²
            'residence_time_s': float(residence_time_s),
            'flow_regime': 'Stokes (Re << 1)' if Re < 0.1 else 'Laminar',
            'state': self.state,
            'clearance_coefficient': self.clearance_coeff
        }

    def compute_pulsatile_flow(self,
                               pvs: PerivascularSpace,
                               mean_pressure_gradient: float,
                               pulse_amplitude: float = 0.3,
                               frequency_Hz: float = CARDIAC_FREQUENCY,
                               num_cycles: int = 5) -> Dict[str, Any]:
        """Compute pulsatile flow driven by cardiac cycle.

        Args:
            pvs: Perivascular space geometry
            mean_pressure_gradient: Mean pressure gradient (Pa/m)
            pulse_amplitude: Relative amplitude of pulsation (0-1)
            frequency_Hz: Cardiac frequency (default: 1 Hz = 60 bpm)
            num_cycles: Number of cycles to simulate

        Returns:
            Time-resolved flow metrics with statistics
        """
        points_per_cycle = 50
        period = 1.0 / frequency_Hz
        t = np.linspace(0, num_cycles * period, num_cycles * points_per_cycle)

        # Pulsatile pressure gradient
        dP_dz = mean_pressure_gradient * (1 + pulse_amplitude * np.sin(2 * np.pi * frequency_Hz * t))

        # Compute flow at each time point (quasi-steady approximation)
        flow_rates = []
        velocities = []
        for dp in dP_dz:
            result = self.compute_steady_flow(pvs, dp)
            flow_rates.append(result['flow_rate_uL_min'])
            velocities.append(result['average_velocity_um_s'])

        flow_rates = np.array(flow_rates)
        velocities = np.array(velocities)

        # Net displacement over one cycle
        dt = t[1] - t[0]
        net_displacement_um = np.sum(velocities) * dt

        return {
            'geometry': pvs.to_dict(),
            'mean_pressure_gradient_Pa_m': mean_pressure_gradient,
            'pulse_amplitude': pulse_amplitude,
            'frequency_Hz': frequency_Hz,
            'num_cycles': num_cycles,
            'flow_rate_mean_uL_min': float(np.mean(flow_rates)),
            'flow_rate_max_uL_min': float(np.max(flow_rates)),
            'flow_rate_min_uL_min': float(np.min(flow_rates)),
            'velocity_mean_um_s': float(np.mean(velocities)),
            'velocity_amplitude_um_s': float((np.max(velocities) - np.min(velocities)) / 2),
            'net_displacement_per_cycle_um': float(net_displacement_um / num_cycles),
            'state': self.state,
            'clearance_coefficient': self.clearance_coeff
        }

    def estimate_clearance(self,
                          pvs: PerivascularSpace,
                          flow_rate_uL_min: float,
                          solute_name: str = 'amyloid-beta',
                          initial_concentration_uM: float = 1.0) -> Dict[str, Any]:
        """Estimate solute clearance rate based on flow.

        Uses simplified convection-diffusion model.

        Args:
            pvs: Perivascular space geometry
            flow_rate_uL_min: Flow rate in µL/min
            solute_name: Name of solute (for reference)
            initial_concentration_uM: Initial concentration in µM

        Returns:
            Clearance metrics
        """
        # Volume of PVS segment (µL)
        V = pvs.cross_section_area_um2 * 1e-12 * pvs.length_mm  # µL

        # Turnover time (how long to replace entire volume)
        turnover_time_min = V / flow_rate_uL_min if flow_rate_uL_min > 0 else float('inf')

        # Clearance half-time (assuming exponential decay with α coefficient)
        half_time_min = np.log(2) / self.clearance_coeff if self.clearance_coeff > 0 else float('inf')

        # Effective clearance rate
        clearance_rate = flow_rate_uL_min * self.clearance_coeff  # µL/min effective

        return {
            'solute': solute_name,
            'initial_concentration_uM': initial_concentration_uM,
            'pvs_volume_uL': float(V),
            'flow_rate_uL_min': flow_rate_uL_min,
            'turnover_time_min': float(turnover_time_min),
            'clearance_coefficient': self.clearance_coeff,
            'clearance_half_time_min': float(half_time_min),
            'effective_clearance_rate_uL_min': float(clearance_rate),
            'state': self.state,
            'note': f"Sleep state provides ~60% higher clearance than awake"
        }

    def design_brain_chip_channel(self,
                                  target_shear_Pa: float = 0.1,
                                  target_residence_time_s: float = 60.0,
                                  length_mm: float = 10.0) -> Dict[str, Any]:
        """Design a microfluidic channel that mimics brain PVS conditions.

        Args:
            target_shear_Pa: Target wall shear stress (0.03-0.15 Pa physiological)
            target_residence_time_s: Target residence time in seconds
            length_mm: Channel length in mm

        Returns:
            Recommended channel dimensions and flow parameters
        """
        # For rectangular channel with aspect ratio ~4:1 (width:height)
        # Approximate as parallel plate: τ = 6μQ/(wh²)
        # And v = Q/(wh), so residence time t = L/v = Lwh/Q

        # Start with typical dimensions
        width_um = 300   # µm
        height_um = 75   # µm

        # From shear stress: Q = τ·w·h²/(6μ)
        mu = WATER_VISCOSITY
        w = width_um * 1e-6
        h = height_um * 1e-6
        L = length_mm * 1e-3

        Q = target_shear_Pa * w * h**2 / (6 * mu)  # m³/s
        Q_uL_min = Q * 1e9 * 60

        # Check residence time
        v = Q / (w * h)
        actual_residence = L / v

        # Adjust height to match residence time if needed
        if abs(actual_residence - target_residence_time_s) / target_residence_time_s > 0.1:
            # Iterate to find better dimensions
            h_new = (target_shear_Pa * w * L / (6 * mu * target_residence_time_s))**(1/3)
            height_um = h_new * 1e6
            h = h_new
            Q = target_shear_Pa * w * h**2 / (6 * mu)
            Q_uL_min = Q * 1e9 * 60
            v = Q / (w * h)
            actual_residence = L / v

        # Reynolds number
        D_h = 4 * w * h / (2 * (w + h))
        Re = WATER_DENSITY * v * D_h / mu

        return {
            'recommended_dimensions': {
                'width_um': float(width_um),
                'height_um': float(height_um),
                'length_mm': length_mm,
                'hydraulic_diameter_um': float(D_h * 1e6)
            },
            'operating_parameters': {
                'flow_rate_uL_min': float(Q_uL_min),
                'average_velocity_um_s': float(v * 1e6),
                'wall_shear_stress_Pa': target_shear_Pa,
                'wall_shear_stress_dyn_cm2': float(target_shear_Pa * 10),
                'residence_time_s': float(actual_residence),
                'reynolds_number': float(Re)
            },
            'targets': {
                'target_shear_Pa': target_shear_Pa,
                'target_residence_time_s': target_residence_time_s
            },
            'validation': {
                'flow_regime': 'Stokes (Re << 1)' if Re < 0.1 else 'Laminar',
                'physiological_shear_range': '0.03-0.15 Pa',
                'matches_pvs_conditions': 0.03 <= target_shear_Pa <= 0.15
            }
        }


# =============================================================================
# Microfluidic Channel Flow (for LOC devices)
# =============================================================================

def compute_channel_flow(channel: MicrofluidicChannel,
                        flow_rate_uL_min: float,
                        fluid_viscosity: float = WATER_VISCOSITY,
                        fluid_density: float = WATER_DENSITY) -> Dict[str, Any]:
    """Compute flow parameters in a rectangular microfluidic channel.

    Args:
        channel: Channel geometry
        flow_rate_uL_min: Volumetric flow rate in µL/min
        fluid_viscosity: Dynamic viscosity in Pa·s
        fluid_density: Fluid density in kg/m³

    Returns:
        Flow parameters dictionary
    """
    # Convert units
    Q = flow_rate_uL_min * 1e-9 / 60  # m³/s
    w = channel.width_um * 1e-6  # m
    h = channel.height_um * 1e-6  # m
    L = channel.length_mm * 1e-3  # m

    # Cross-section and hydraulic diameter
    A = w * h
    D_h = 4 * A / (2 * (w + h))

    # Average velocity
    v_avg = Q / A

    # Reynolds number
    Re = fluid_density * v_avg * D_h / fluid_viscosity

    # Pressure drop (parallel plate approximation for high aspect ratio)
    # ΔP = 12μLQ/(wh³) for w >> h
    if channel.aspect_ratio > 2:
        dP = 12 * fluid_viscosity * L * Q / (w * h**3)
    else:
        # More accurate for low aspect ratio
        dP = 12 * fluid_viscosity * L * Q / (w * h**3) * (1 - 0.63 * h / w)

    # Wall shear stress (at top/bottom walls)
    tau_wall = 6 * fluid_viscosity * Q / (w * h**2)

    # Residence time
    residence_time = L / v_avg if v_avg > 0 else float('inf')

    # Peclet number (for mass transfer, assume D ~ 1e-9 m²/s)
    D_solute = 1e-9  # Typical small molecule diffusivity
    Pe = v_avg * D_h / D_solute

    return {
        'geometry': channel.to_dict(),
        'flow_rate_uL_min': flow_rate_uL_min,
        'flow_rate_m3_s': float(Q),
        'average_velocity_um_s': float(v_avg * 1e6),
        'average_velocity_mm_s': float(v_avg * 1e3),
        'reynolds_number': float(Re),
        'pressure_drop_Pa': float(dP),
        'pressure_drop_mbar': float(dP / 100),
        'wall_shear_stress_Pa': float(tau_wall),
        'wall_shear_stress_dyn_cm2': float(tau_wall * 10),
        'residence_time_s': float(residence_time),
        'peclet_number': float(Pe),
        'flow_regime': 'Stokes (Re << 1)' if Re < 0.1 else ('Laminar' if Re < 2000 else 'Transitional'),
        'transport_regime': 'Diffusion-dominated' if Pe < 1 else 'Convection-dominated'
    }
