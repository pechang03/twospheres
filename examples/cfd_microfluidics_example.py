#!/usr/bin/env python3
"""
CFD Microfluidics Example - PHLoC and Glymphatic Simulation

Demonstrates the cross-domain overlap between:
1. PHLoC (Photonic Lab-on-Chip) organoid culture
2. Brain glymphatic system CSF flow

Both operate in Stokes flow regime (Re << 1).

Usage:
    python examples/cfd_microfluidics_example.py

Requirements:
    - twosphere-mcp installed
    - For full CFD: FreeCAD + CfdOF + OpenFOAM (see docs/CFDOF_INSTALL.md)
"""

import asyncio
import sys
import json
from pathlib import Path

# Add bin to path for MCP tools
sys.path.insert(0, str(Path(__file__).parent.parent / 'bin'))

from twosphere_mcp import handle_cfd_microfluidics


async def example_phloc_organoid():
    """
    PHLoC organoid culture microfluidics.

    Design parameters:
    - 10 µm channels (gentle fluid exchange)
    - ~10 mm/s flow velocity
    - Water/culture media

    Key constraint: Wall shear stress < 0.5 Pa for organoid health
    """
    print("=" * 60)
    print("Example 1: PHLoC Organoid Culture")
    print("=" * 60)

    # Initial design - may have too high shear
    result = await handle_cfd_microfluidics({
        'geometry_type': 'phloc',
        'channel_diameter_um': 10,
        'velocity_um_s': 10000,  # 10 mm/s
        'fluid': 'culture_media',
        'length_mm': 5.0
    })

    print(f"\nApplication: {result['application']}")
    print(f"Channel: {result['geometry']['channel_diameter_um']} µm diameter")
    print(f"Velocity: {result['flow_conditions']['velocity_mm_s']} mm/s")
    print(f"Flow rate: {result['flow_conditions']['flow_rate_uL_min']:.4f} µL/min")
    print(f"\nReynolds number: {result['dimensionless_numbers']['reynolds_number']:.4f}")
    print(f"Flow regime: {result['dimensionless_numbers']['flow_regime']}")
    print(f"\nWall shear stress: {result['results']['wall_shear_stress_Pa']:.2f} Pa")
    print(f"Organoid tolerance: {result['biological_relevance']['organoid_shear_tolerance']}")

    # Check if shear is safe
    shear = result['results']['wall_shear_stress_Pa']
    if shear > 0.5:
        print(f"\n⚠️  WARNING: Shear stress ({shear:.2f} Pa) exceeds organoid tolerance!")
        print("   Reducing velocity to find safe operating point...")

        # Find safe velocity
        for velocity in [5000, 2000, 1000, 500, 200]:
            test = await handle_cfd_microfluidics({
                'geometry_type': 'phloc',
                'channel_diameter_um': 10,
                'velocity_um_s': velocity,
                'fluid': 'culture_media'
            })
            if test['results']['wall_shear_stress_Pa'] < 0.5:
                print(f"   ✓ Safe at {velocity} µm/s ({velocity/1000} mm/s): "
                      f"shear = {test['results']['wall_shear_stress_Pa']:.3f} Pa")
                break
    else:
        print(f"\n✓ Shear stress is within organoid tolerance")

    return result


async def example_glymphatic_csf():
    """
    Brain glymphatic system CSF flow simulation.

    The glymphatic system clears metabolic waste (Aβ, tau) via:
    - Perivascular spaces around arteries/veins
    - ~10-50 µm channel width
    - Very slow flow (~10-20 µm/s)
    - Driven by arterial pulsations
    - Increases ~60% during sleep
    """
    print("\n" + "=" * 60)
    print("Example 2: Glymphatic System CSF Flow")
    print("=" * 60)

    # Awake state
    print("\n--- Awake State ---")
    awake = await handle_cfd_microfluidics({
        'geometry_type': 'glymphatic',
        'channel_diameter_um': 20,  # Perivascular space
        'velocity_um_s': 10,  # ~10 µm/s awake
        'fluid': 'csf',
        'length_mm': 10.0  # 1 cm vessel segment
    })

    print(f"Application: {awake['application']}")
    print(f"Perivascular space: {awake['geometry']['channel_diameter_um']} µm")
    print(f"CSF velocity: {awake['flow_conditions']['velocity_um_s']} µm/s")
    print(f"Reynolds number: {awake['dimensionless_numbers']['reynolds_number']:.6f}")
    print(f"Flow regime: {awake['dimensionless_numbers']['flow_regime']}")
    print(f"Pressure drop: {awake['results']['pressure_drop_Pa']:.4f} Pa")

    # Sleep state (~60% increase)
    print("\n--- Sleep State (60% increased flow) ---")
    sleep = await handle_cfd_microfluidics({
        'geometry_type': 'glymphatic',
        'channel_diameter_um': 20,
        'velocity_um_s': 16,  # 60% increase
        'fluid': 'csf',
        'length_mm': 10.0
    })

    print(f"CSF velocity: {sleep['flow_conditions']['velocity_um_s']} µm/s")
    print(f"Flow rate increase: {(16-10)/10*100:.0f}%")
    print(f"Pressure drop: {sleep['results']['pressure_drop_Pa']:.4f} Pa")

    print(f"\nBiological relevance:")
    for key, value in awake['biological_relevance'].items():
        print(f"  - {key}: {value}")

    return awake, sleep


async def example_compare_systems():
    """
    Compare PHLoC and glymphatic systems side-by-side.

    Demonstrates the cross-domain physics overlap.
    """
    print("\n" + "=" * 60)
    print("Example 3: Cross-Domain Comparison")
    print("=" * 60)

    phloc = await handle_cfd_microfluidics({
        'geometry_type': 'phloc',
        'channel_diameter_um': 10,
        'velocity_um_s': 600,  # Reduced for organoid safety
        'fluid': 'culture_media'
    })

    glymph = await handle_cfd_microfluidics({
        'geometry_type': 'glymphatic',
        'channel_diameter_um': 20,
        'velocity_um_s': 10,
        'fluid': 'csf'
    })

    print(f"\n{'Parameter':<25} {'PHLoC':<20} {'Glymphatic':<20}")
    print("-" * 65)
    print(f"{'Channel diameter':<25} {phloc['geometry']['channel_diameter_um']} µm{'':<14} {glymph['geometry']['channel_diameter_um']} µm")
    print(f"{'Velocity':<25} {phloc['flow_conditions']['velocity_um_s']} µm/s{'':<11} {glymph['flow_conditions']['velocity_um_s']} µm/s")
    print(f"{'Fluid':<25} {phloc['fluid']['name']:<20} {glymph['fluid']['name']:<20}")
    print(f"{'Reynolds number':<25} {phloc['dimensionless_numbers']['reynolds_number']:.4f}{'':<14} {glymph['dimensionless_numbers']['reynolds_number']:.6f}")
    print(f"{'Flow regime':<25} {phloc['dimensionless_numbers']['flow_regime'][:18]:<20} {glymph['dimensionless_numbers']['flow_regime'][:18]:<20}")
    print(f"{'Wall shear (Pa)':<25} {phloc['results']['wall_shear_stress_Pa']:.4f}{'':<14} {glymph['results']['wall_shear_stress_Pa']:.6f}")

    print(f"\n✓ {phloc['cross_domain_note']}")
    print("  Same CFD tools and physics apply to both biological systems!")


async def example_parameter_sweep():
    """
    Parameter sweep to find optimal PHLoC operating conditions.

    Constraints:
    - Wall shear < 0.5 Pa (organoid health)
    - Adequate nutrient delivery (flow rate)
    - Reasonable pressure drop (pump capacity)
    """
    print("\n" + "=" * 60)
    print("Example 4: PHLoC Design Optimization")
    print("=" * 60)

    print("\nSweeping channel diameter and velocity...")
    print(f"\n{'D (µm)':<10} {'v (mm/s)':<12} {'Re':<12} {'Shear (Pa)':<12} {'Safe?':<8}")
    print("-" * 54)

    safe_designs = []

    for diameter in [10, 20, 50, 100]:
        for velocity_mm_s in [0.1, 0.5, 1.0, 5.0, 10.0]:
            result = await handle_cfd_microfluidics({
                'geometry_type': 'phloc',
                'channel_diameter_um': diameter,
                'velocity_um_s': velocity_mm_s * 1000,
                'fluid': 'culture_media'
            })

            shear = result['results']['wall_shear_stress_Pa']
            re = result['dimensionless_numbers']['reynolds_number']
            safe = shear < 0.5

            if safe:
                safe_designs.append({
                    'diameter': diameter,
                    'velocity': velocity_mm_s,
                    'shear': shear,
                    'flow_rate': result['flow_conditions']['flow_rate_uL_min']
                })

            print(f"{diameter:<10} {velocity_mm_s:<12} {re:<12.4f} {shear:<12.4f} {'✓' if safe else '✗':<8}")

    print(f"\n{len(safe_designs)} safe designs found (shear < 0.5 Pa)")
    if safe_designs:
        best = max(safe_designs, key=lambda x: x['flow_rate'])
        print(f"\nOptimal design (max flow rate while safe):")
        print(f"  Channel: {best['diameter']} µm")
        print(f"  Velocity: {best['velocity']} mm/s")
        print(f"  Flow rate: {best['flow_rate']:.4f} µL/min")
        print(f"  Wall shear: {best['shear']:.4f} Pa")


async def main():
    """Run all examples."""
    print("CFD Microfluidics Examples")
    print("Cross-domain: PHLoC Organoids ↔ Brain Glymphatics")
    print("Both in Stokes flow regime (Re << 1)")

    await example_phloc_organoid()
    await example_glymphatic_csf()
    await example_compare_systems()
    await example_parameter_sweep()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("For full CFD simulation in FreeCAD, see:")
    print("  examples/freecad_cfd_microfluidics.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
