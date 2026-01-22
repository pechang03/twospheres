#!/usr/bin/env python3
"""
Glymphatic CFD Simulator - Perivascular CSF Flow Model

Extends the PHLoC microfluidic simulation to model brain glymphatic flow:
- Perivascular space geometry (annular gap around blood vessels)
- Cardiac pulse-driven pulsatile flow
- Sleep/wake state transitions
- Amyloid-β clearance rate estimation

Based on Stokes flow (Re << 1) - same physics as PHLoC microfluidics.

Usage:
    python prototypes/glymphatic_cfd_simulator.py

Task: PH-7 (Glymphatic-Microfluidics Integration)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

try:
    from bin.twosphere_mcp import handle_cfd_microfluidics
    HAS_CFD = True
except ImportError:
    HAS_CFD = False


# =============================================================================
# Physical Constants
# =============================================================================

# CSF properties at 37°C
CSF_VISCOSITY = 0.001  # Pa·s (similar to water)
CSF_DENSITY = 1007     # kg/m³

# Physiological parameters
CARDIAC_FREQUENCY = 1.0  # Hz (60 bpm)
ARTERIAL_PULSE_AMPLITUDE = 0.1  # relative wall displacement

# Clearance parameters
AWAKE_CLEARANCE_COEFF = 0.10   # α for awake state
SLEEP_CLEARANCE_COEFF = 0.16   # α for sleep state (~60% increase)

# Amyloid-β parameters
ABETA_DIFFUSION_COEFF = 1e-10  # m²/s (approximate)
ABETA_PRODUCTION_RATE = 1e-12  # mol/(m³·s) (varies by region)


# =============================================================================
# Perivascular Space Geometry
# =============================================================================

@dataclass
class PerivascularSpace:
    """Model of a perivascular space around a blood vessel."""

    vessel_radius_um: float      # Inner radius (blood vessel wall)
    gap_thickness_um: float      # Width of perivascular space
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
        # D_h = 2 * gap for thin annulus
        return 2 * self.gap_thickness_um

    def to_dict(self) -> Dict[str, Any]:
        return {
            'vessel_radius_um': self.vessel_radius_um,
            'gap_thickness_um': self.gap_thickness_um,
            'outer_radius_um': self.outer_radius_um,
            'length_mm': self.length_mm,
            'vessel_type': self.vessel_type,
            'cross_section_area_um2': self.cross_section_area_um2,
            'hydraulic_diameter_um': self.hydraulic_diameter_um
        }


# =============================================================================
# Glymphatic Flow Calculator
# =============================================================================

class GlymphaticFlowSimulator:
    """Simulator for glymphatic CSF flow in perivascular spaces.

    Uses validated PHLoC microfluidic physics scaled to brain dimensions.
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

        Uses annular Poiseuille flow solution.

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
            Q = (np.pi / (8 * mu)) * dP_dz * L * (term1 - term2) / L
        else:
            # Circular pipe (no inner wall)
            Q = (np.pi * r_out**4 / (8 * mu)) * dP_dz

        # Thin gap approximation (for comparison)
        # Q ≈ (π·R·δ³/6μL) * ΔP
        R_avg = (r_in + r_out) / 2
        Q_thin = (np.pi * R_avg * delta**3 / (6 * mu)) * dP_dz

        # Average velocity
        A = np.pi * (r_out**2 - r_in**2)
        v_avg = Q / A if A > 0 else 0

        # Reynolds number
        D_h = 2 * delta
        Re = CSF_DENSITY * v_avg * D_h / mu

        # Wall shear stress (at inner wall)
        # τ = μ * du/dr|_{r=r_in}
        if r_in > 0:
            # For annular flow: τ_in = (dP/dz) * r_in/2 * [1 - (r_in/r_out)² + ...]
            tau_wall = abs(dP_dz) * r_in / 2
        else:
            tau_wall = abs(dP_dz) * r_out / 4

        # Flow rate in convenient units
        Q_uL_min = Q * 1e9 * 60  # m³/s → µL/min

        return {
            'geometry': pvs.to_dict(),
            'pressure_gradient_Pa_m': pressure_gradient_Pa_m,
            'flow_rate_m3_s': float(Q),
            'flow_rate_uL_min': float(Q_uL_min),
            'flow_rate_thin_approx_m3_s': float(Q_thin),
            'average_velocity_um_s': float(v_avg * 1e6),
            'reynolds_number': float(Re),
            'wall_shear_stress_Pa': float(tau_wall),
            'flow_regime': 'Stokes (Re << 1)' if Re < 0.1 else 'Laminar',
            'state': self.state,
            'clearance_coefficient': self.clearance_coeff
        }

    def compute_pulsatile_flow(self,
                               pvs: PerivascularSpace,
                               mean_pressure_gradient: float,
                               pulse_amplitude: float = 0.3,
                               frequency_Hz: float = CARDIAC_FREQUENCY,
                               num_cycles: int = 5,
                               points_per_cycle: int = 100) -> Dict[str, Any]:
        """Compute pulsatile flow driven by cardiac cycle.

        Args:
            pvs: Perivascular space geometry
            mean_pressure_gradient: Mean pressure gradient (Pa/m)
            pulse_amplitude: Relative amplitude of pulsation (0-1)
            frequency_Hz: Cardiac frequency
            num_cycles: Number of cycles to simulate
            points_per_cycle: Time resolution per cycle

        Returns:
            Time-resolved flow metrics
        """
        # Time array
        period = 1.0 / frequency_Hz
        t = np.linspace(0, num_cycles * period, num_cycles * points_per_cycle)

        # Pulsatile pressure gradient
        # Simple sinusoidal model: dP/dz = mean * (1 + amplitude * sin(2πft))
        dP_dz = mean_pressure_gradient * (1 + pulse_amplitude * np.sin(2 * np.pi * frequency_Hz * t))

        # Compute flow at each time point (quasi-steady approximation)
        # Valid when Womersley number << 1 (thin gaps, low frequency)
        flows = []
        for dP in dP_dz:
            result = self.compute_steady_flow(pvs, dP)
            flows.append(result['flow_rate_m3_s'])

        flows = np.array(flows)

        # Statistics
        Q_mean = np.mean(flows)
        Q_max = np.max(flows)
        Q_min = np.min(flows)
        Q_amplitude = (Q_max - Q_min) / 2

        # Womersley number (flow unsteadiness parameter)
        delta = pvs.gap_thickness_um * 1e-6
        omega = 2 * np.pi * frequency_Hz
        alpha = delta * np.sqrt(omega * CSF_DENSITY / CSF_VISCOSITY)

        return {
            'geometry': pvs.to_dict(),
            'cardiac_frequency_Hz': frequency_Hz,
            'pulse_amplitude': pulse_amplitude,
            'mean_pressure_gradient_Pa_m': mean_pressure_gradient,
            'womersley_number': float(alpha),
            'quasi_steady_valid': alpha < 1.0,
            'flow_statistics': {
                'mean_flow_m3_s': float(Q_mean),
                'max_flow_m3_s': float(Q_max),
                'min_flow_m3_s': float(Q_min),
                'amplitude_m3_s': float(Q_amplitude),
                'mean_flow_uL_min': float(Q_mean * 1e9 * 60),
            },
            'time_series': {
                'time_s': t.tolist(),
                'flow_m3_s': flows.tolist(),
                'pressure_gradient_Pa_m': dP_dz.tolist()
            },
            'state': self.state
        }

    def estimate_clearance_rate(self,
                                pvs: PerivascularSpace,
                                flow_result: Dict[str, Any],
                                metabolic_rate: float = ABETA_PRODUCTION_RATE) -> Dict[str, Any]:
        """Estimate waste clearance rate based on flow.

        Uses the metabolic coupling constraint: ∑q_e = α·m_v

        Args:
            pvs: Perivascular space geometry
            flow_result: Result from compute_steady_flow or compute_pulsatile_flow
            metabolic_rate: Local metabolic waste production rate (mol/m³/s)

        Returns:
            Clearance metrics
        """
        # Get flow rate
        if 'flow_statistics' in flow_result:
            Q = flow_result['flow_statistics']['mean_flow_m3_s']
        else:
            Q = flow_result['flow_rate_m3_s']

        # Volume of tissue served (approximate as cylinder around PVS)
        # Assume clearance radius ~100 µm from vessel
        clearance_radius = 100e-6  # m
        L = pvs.length_mm * 1e-3
        V_tissue = np.pi * clearance_radius**2 * L

        # Metabolic production in this volume
        waste_production = metabolic_rate * V_tissue  # mol/s

        # Clearance capacity (advective)
        # Assuming waste concentration C, clearance = Q * C
        # At steady state: Q * C_out = α * m_v * V
        # So effective clearance rate = Q / V_tissue (volume turnover)

        volume_turnover_rate = Q / V_tissue if V_tissue > 0 else 0  # 1/s
        clearance_time_s = 1 / volume_turnover_rate if volume_turnover_rate > 0 else float('inf')

        # Peclet number (advection vs diffusion)
        v_avg = flow_result.get('average_velocity_um_s', 0) * 1e-6  # m/s
        L_char = pvs.gap_thickness_um * 1e-6
        Pe = v_avg * L_char / ABETA_DIFFUSION_COEFF if ABETA_DIFFUSION_COEFF > 0 else float('inf')

        return {
            'geometry': pvs.to_dict(),
            'state': self.state,
            'clearance_coefficient': self.clearance_coeff,
            'tissue_volume_mm3': float(V_tissue * 1e9),
            'waste_production_rate_mol_s': float(waste_production),
            'volume_turnover_rate_per_s': float(volume_turnover_rate),
            'clearance_time_s': float(clearance_time_s),
            'clearance_time_min': float(clearance_time_s / 60),
            'peclet_number': float(Pe),
            'transport_mode': 'advection-dominated' if Pe > 10 else 'diffusion-dominated' if Pe < 0.1 else 'mixed',
            'sleep_improvement_factor': SLEEP_CLEARANCE_COEFF / AWAKE_CLEARANCE_COEFF
        }


# =============================================================================
# Network-Level Glymphatic Model
# =============================================================================

class GlymphaticNetwork:
    """Network model of glymphatic system with multiple perivascular paths.

    Connects to disc dimension analysis for topology → efficiency mapping.
    """

    def __init__(self, state: str = 'awake'):
        """Initialize network model.

        Args:
            state: 'awake' or 'sleep'
        """
        self.simulator = GlymphaticFlowSimulator(state)
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int, PerivascularSpace]] = []

    def add_node(self, name: str, metabolic_rate: float = ABETA_PRODUCTION_RATE):
        """Add a brain region node."""
        self.nodes.append({
            'id': len(self.nodes),
            'name': name,
            'metabolic_rate': metabolic_rate
        })
        return len(self.nodes) - 1

    def add_perivascular_edge(self,
                              node1: int,
                              node2: int,
                              vessel_radius_um: float = 25,
                              gap_thickness_um: float = 10,
                              length_mm: float = 5,
                              vessel_type: str = 'artery'):
        """Add a perivascular space connecting two nodes."""
        pvs = PerivascularSpace(
            vessel_radius_um=vessel_radius_um,
            gap_thickness_um=gap_thickness_um,
            length_mm=length_mm,
            vessel_type=vessel_type
        )
        self.edges.append((node1, node2, pvs))

    def compute_network_flow(self,
                            inlet_pressure_Pa: float = 100,
                            outlet_pressure_Pa: float = 0) -> Dict[str, Any]:
        """Compute flow through entire network.

        Simplified model: parallel paths from inlet to outlet.

        Args:
            inlet_pressure_Pa: Pressure at CSF inlet (subarachnoid)
            outlet_pressure_Pa: Pressure at drainage (cervical lymph)

        Returns:
            Network flow results
        """
        total_flow = 0
        edge_flows = []

        for i, (n1, n2, pvs) in enumerate(self.edges):
            # Pressure gradient (simplified: linear drop along each path)
            dP_dz = (inlet_pressure_Pa - outlet_pressure_Pa) / (pvs.length_mm * 1e-3)

            result = self.simulator.compute_steady_flow(pvs, dP_dz)
            edge_flows.append({
                'edge_id': i,
                'from_node': self.nodes[n1]['name'] if n1 < len(self.nodes) else n1,
                'to_node': self.nodes[n2]['name'] if n2 < len(self.nodes) else n2,
                'flow_result': result
            })
            total_flow += result['flow_rate_m3_s']

        # Network statistics
        flow_rates = [e['flow_result']['flow_rate_m3_s'] for e in edge_flows]

        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'inlet_pressure_Pa': inlet_pressure_Pa,
            'outlet_pressure_Pa': outlet_pressure_Pa,
            'total_flow_m3_s': float(total_flow),
            'total_flow_uL_min': float(total_flow * 1e9 * 60),
            'mean_edge_flow_m3_s': float(np.mean(flow_rates)) if flow_rates else 0,
            'std_edge_flow_m3_s': float(np.std(flow_rates)) if flow_rates else 0,
            'edge_flows': edge_flows,
            'state': self.simulator.state
        }

    def estimate_network_clearance(self,
                                   inlet_pressure_Pa: float = 100) -> Dict[str, Any]:
        """Estimate total network clearance capacity.

        Returns:
            Network clearance metrics
        """
        network_flow = self.compute_network_flow(inlet_pressure_Pa)

        # Total metabolic load
        total_metabolic = sum(n['metabolic_rate'] for n in self.nodes)

        # Total tissue volume (sum over edges)
        total_tissue_volume = 0
        for _, _, pvs in self.edges:
            clearance_radius = 100e-6
            L = pvs.length_mm * 1e-3
            total_tissue_volume += np.pi * clearance_radius**2 * L

        # Network clearance time
        Q_total = network_flow['total_flow_m3_s']
        if Q_total > 0 and total_tissue_volume > 0:
            clearance_time = total_tissue_volume / Q_total
        else:
            clearance_time = float('inf')

        return {
            'network_flow': network_flow,
            'total_metabolic_rate_mol_s': float(total_metabolic),
            'total_tissue_volume_mm3': float(total_tissue_volume * 1e9),
            'network_clearance_time_s': float(clearance_time),
            'network_clearance_time_min': float(clearance_time / 60),
            'state': self.simulator.state,
            'sleep_vs_awake_improvement': SLEEP_CLEARANCE_COEFF / AWAKE_CLEARANCE_COEFF
        }


# =============================================================================
# Example Simulations
# =============================================================================

def example_single_vessel():
    """Simulate flow in a single perivascular space."""
    print("=" * 70)
    print("Example 1: Single Perivascular Space")
    print("=" * 70)

    # Typical penetrating arteriole
    pvs = PerivascularSpace(
        vessel_radius_um=25,      # 50 µm diameter arteriole
        gap_thickness_um=10,      # 10 µm perivascular space
        length_mm=5.0,            # 5 mm segment
        vessel_type='artery'
    )

    print(f"\nGeometry:")
    print(f"  Vessel radius: {pvs.vessel_radius_um} µm")
    print(f"  Gap thickness: {pvs.gap_thickness_um} µm")
    print(f"  Length: {pvs.length_mm} mm")
    print(f"  Hydraulic diameter: {pvs.hydraulic_diameter_um:.1f} µm")

    # Steady flow - awake
    sim_awake = GlymphaticFlowSimulator(state='awake')
    result_awake = sim_awake.compute_steady_flow(pvs, pressure_gradient_Pa_m=10)

    print(f"\nAwake State (α = {AWAKE_CLEARANCE_COEFF}):")
    print(f"  Pressure gradient: {result_awake['pressure_gradient_Pa_m']} Pa/m")
    print(f"  Flow rate: {result_awake['flow_rate_uL_min']:.6f} µL/min")
    print(f"  Average velocity: {result_awake['average_velocity_um_s']:.2f} µm/s")
    print(f"  Reynolds number: {result_awake['reynolds_number']:.6f}")
    print(f"  Flow regime: {result_awake['flow_regime']}")

    # Steady flow - sleep
    sim_sleep = GlymphaticFlowSimulator(state='sleep')
    result_sleep = sim_sleep.compute_steady_flow(pvs, pressure_gradient_Pa_m=16)  # 60% higher

    print(f"\nSleep State (α = {SLEEP_CLEARANCE_COEFF}):")
    print(f"  Pressure gradient: {result_sleep['pressure_gradient_Pa_m']} Pa/m (60% increase)")
    print(f"  Flow rate: {result_sleep['flow_rate_uL_min']:.6f} µL/min")
    print(f"  Average velocity: {result_sleep['average_velocity_um_s']:.2f} µm/s")

    # Clearance estimate
    clearance = sim_sleep.estimate_clearance_rate(pvs, result_sleep)
    print(f"\nClearance Metrics:")
    print(f"  Tissue volume served: {clearance['tissue_volume_mm3']:.4f} mm³")
    print(f"  Clearance time: {clearance['clearance_time_min']:.1f} min")
    print(f"  Transport mode: {clearance['transport_mode']}")
    print(f"  Peclet number: {clearance['peclet_number']:.1f}")

    return result_awake, result_sleep, clearance


def example_pulsatile_flow():
    """Simulate cardiac-driven pulsatile flow."""
    print("\n" + "=" * 70)
    print("Example 2: Pulsatile Flow (Cardiac Cycle)")
    print("=" * 70)

    pvs = PerivascularSpace(
        vessel_radius_um=25,
        gap_thickness_um=10,
        length_mm=5.0,
        vessel_type='artery'
    )

    sim = GlymphaticFlowSimulator(state='awake')
    result = sim.compute_pulsatile_flow(
        pvs,
        mean_pressure_gradient=10,  # Pa/m
        pulse_amplitude=0.3,        # 30% variation
        frequency_Hz=1.0,           # 60 bpm
        num_cycles=3
    )

    print(f"\nPulsatile Flow Parameters:")
    print(f"  Cardiac frequency: {result['cardiac_frequency_Hz']} Hz")
    print(f"  Pulse amplitude: {result['pulse_amplitude']*100}%")
    print(f"  Womersley number: {result['womersley_number']:.3f}")
    print(f"  Quasi-steady valid: {result['quasi_steady_valid']}")

    print(f"\nFlow Statistics:")
    stats = result['flow_statistics']
    print(f"  Mean flow: {stats['mean_flow_uL_min']:.6f} µL/min")
    print(f"  Max flow: {stats['max_flow_m3_s']*1e9*60:.6f} µL/min")
    print(f"  Min flow: {stats['min_flow_m3_s']*1e9*60:.6f} µL/min")

    return result


def example_simple_network():
    """Simulate a simple glymphatic network."""
    print("\n" + "=" * 70)
    print("Example 3: Simple Glymphatic Network")
    print("=" * 70)

    # Create network
    network = GlymphaticNetwork(state='sleep')

    # Add brain regions (nodes)
    csf = network.add_node("CSF_inlet", metabolic_rate=0)
    cortex1 = network.add_node("Cortex_A", metabolic_rate=ABETA_PRODUCTION_RATE)
    cortex2 = network.add_node("Cortex_B", metabolic_rate=ABETA_PRODUCTION_RATE)
    deep = network.add_node("Deep_Gray", metabolic_rate=ABETA_PRODUCTION_RATE * 1.5)
    drain = network.add_node("Lymph_drain", metabolic_rate=0)

    # Add perivascular connections
    network.add_perivascular_edge(csf, cortex1, vessel_radius_um=30, gap_thickness_um=15, length_mm=10)
    network.add_perivascular_edge(csf, cortex2, vessel_radius_um=30, gap_thickness_um=15, length_mm=10)
    network.add_perivascular_edge(cortex1, deep, vessel_radius_um=20, gap_thickness_um=8, length_mm=15)
    network.add_perivascular_edge(cortex2, deep, vessel_radius_um=20, gap_thickness_um=8, length_mm=15)
    network.add_perivascular_edge(deep, drain, vessel_radius_um=25, gap_thickness_um=10, length_mm=20)

    print(f"\nNetwork Structure:")
    print(f"  Nodes: {[n['name'] for n in network.nodes]}")
    print(f"  Edges: {len(network.edges)} perivascular paths")

    # Compute network clearance
    clearance = network.estimate_network_clearance(inlet_pressure_Pa=100)

    print(f"\nNetwork Flow (Sleep State):")
    print(f"  Total flow: {clearance['network_flow']['total_flow_uL_min']:.4f} µL/min")
    print(f"  Total tissue volume: {clearance['total_tissue_volume_mm3']:.2f} mm³")
    print(f"  Network clearance time: {clearance['network_clearance_time_min']:.1f} min")

    return clearance


def example_compare_with_phloc():
    """Compare glymphatic parameters with PHLoC design."""
    print("\n" + "=" * 70)
    print("Example 4: Cross-Domain Comparison (Glymphatic vs PHLoC)")
    print("=" * 70)

    if not HAS_CFD:
        print("  [Skipped: handle_cfd_microfluidics not available]")
        return None

    import asyncio

    async def run_comparison():
        # Glymphatic parameters
        glymph = await handle_cfd_microfluidics({
            'geometry_type': 'glymphatic',
            'channel_diameter_um': 20,  # 10 µm gap × 2
            'velocity_um_s': 15,
            'fluid': 'csf',
            'length_mm': 10
        })

        # PHLoC at similar scale
        phloc = await handle_cfd_microfluidics({
            'geometry_type': 'phloc',
            'channel_diameter_um': 20,
            'velocity_um_s': 15,
            'fluid': 'water',
            'length_mm': 10
        })

        print(f"\n{'Parameter':<25} {'Glymphatic':<20} {'PHLoC':<20}")
        print("-" * 65)
        print(f"{'Channel diameter':<25} {glymph['geometry']['channel_diameter_um']} µm{'':<14} {phloc['geometry']['channel_diameter_um']} µm")
        print(f"{'Velocity':<25} {glymph['flow_conditions']['velocity_um_s']} µm/s{'':<11} {phloc['flow_conditions']['velocity_um_s']} µm/s")
        print(f"{'Reynolds number':<25} {glymph['dimensionless_numbers']['reynolds_number']:.6f}{'':<12} {phloc['dimensionless_numbers']['reynolds_number']:.6f}")
        print(f"{'Wall shear (Pa)':<25} {glymph['results']['wall_shear_stress_Pa']:.6f}{'':<12} {phloc['results']['wall_shear_stress_Pa']:.6f}")
        print(f"{'Flow regime':<25} {glymph['dimensionless_numbers']['flow_regime'][:18]:<20} {phloc['dimensionless_numbers']['flow_regime'][:18]}")

        print(f"\n✓ Same Stokes flow physics applies to both systems!")
        print(f"  PHLoC designs can validate glymphatic flow models.")

        return glymph, phloc

    return asyncio.run(run_comparison())


def main():
    """Run all examples."""
    print("Glymphatic CFD Simulator")
    print("Task: PH-7 (Glymphatic-Microfluidics Integration)")
    print("=" * 70)

    example_single_vessel()
    example_pulsatile_flow()
    example_simple_network()
    example_compare_with_phloc()

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
