#!/usr/bin/env python3
"""TwoSphere MCP Server - Optical physics and MRI spherical geometry analysis.

Integrates pyoptools for optical simulations with MRISpheres/twospheres research.

Usage:
    python bin/twosphere_mcp.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env_twosphere"
if env_path.exists():
    load_dotenv(env_path)

TWOSPHERES_PATH = os.getenv("TWOSPHERES_PATH", os.path.expanduser("~/MRISpheres/twospheres"))

# Initialize MCP server
server = Server("twosphere-mcp")


# =============================================================================
# Expert Query Integration (ernie2_swarm)
# =============================================================================

# Collection mappings for domain-expert queries
EXPERT_COLLECTIONS = {
    "interferometric_sensing": ["docs_library_neuroscience_MRI", "docs_library_physics_optics"],
    "lock_in_detection": ["docs_library_physics_optics", "docs_library_statistics"],
    "absorption_spectroscopy": ["docs_library_bioengineering_LOC", "docs_library_physics_optics"],
    "optimize_resonator": ["docs_library_physics_optics", "docs_library_mathematics"],
    "microfluidics": ["docs_library_bioengineering_LOC", "docs_library_neuroscience_MRI"],
}


async def query_domain_experts(
    question: str,
    tool_name: str,
    collections: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Query ernie2_swarm for domain-expert parameter suggestions.

    Args:
        question: Question to ask domain experts
        tool_name: Name of the tool (for collection selection)
        collections: Override default collections

    Returns:
        Dict with 'answer' and optionally 'parameters' extracted from response
    """
    try:
        from backend.services.ernie2_integration import YadaServicesMCPClient

        # Use tool-specific collections or provided ones
        cols = collections or EXPERT_COLLECTIONS.get(tool_name, [])

        client = YadaServicesMCPClient()
        result = await client.query_ernie2(
            question=question,
            collections=cols,
            max_steps=2,
            style="technical"
        )

        return result

    except Exception as e:
        return {"error": str(e), "answer": None}


# =============================================================================
# Tool Definitions
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available TwoSphere MCP tools."""
    return [
        # =============================================================================
        # Service Meta Tools (merge2docs pattern)
        # =============================================================================
        Tool(
            name="advice",
            description="Get guidance on using this service. Returns capabilities, use cases, "
                       "and recommendations for when to route queries here.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Optional domain focus (optics, microfluidics, brain, fluigi)",
                        "enum": ["optics", "microfluidics", "brain", "fluigi", "all"]
                    },
                    "verbose": {
                        "type": "boolean",
                        "description": "Include detailed tool descriptions",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="health_check",
            description="Verify service health and dependency status. Returns operational status "
                       "of all subsystems (Fluigi, FreeCAD, ernie2_swarm, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "check_fluigi": {
                        "type": "boolean",
                        "description": "Test Fluigi CLI availability",
                        "default": True
                    },
                    "check_freecad": {
                        "type": "boolean",
                        "description": "Test FreeCAD RPC connection",
                        "default": False
                    },
                    "check_ernie2": {
                        "type": "boolean",
                        "description": "Test ernie2_swarm availability",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="two_sphere_model",
            description="Create and visualize a two-sphere model for paired brain regions. "
                       "Returns 3D coordinates and can generate matplotlib visualization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "Radius of each sphere (default: 1.0)",
                        "default": 1.0
                    },
                    "center1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Center of first sphere [x, y, z] (default: [0, 1, 0])"
                    },
                    "center2": {
                        "type": "array", 
                        "items": {"type": "number"},
                        "description": "Center of second sphere [x, y, z] (default: [0, -1, 0])"
                    },
                    "resolution": {
                        "type": "integer",
                        "description": "Number of points for mesh (default: 100)",
                        "default": 100
                    },
                    "save_plot": {
                        "type": "string",
                        "description": "Optional path to save visualization PNG"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="vortex_ring",
            description="Generate a vortex ring with trefoil knot structure for modeling "
                       "neural connectivity patterns. Uses Frenet-Serret frame for tube surface.",
            inputSchema={
                "type": "object",
                "properties": {
                    "scale_x": {
                        "type": "number",
                        "description": "Scale factor for x-coordinate (default: 5.0)",
                        "default": 5.0
                    },
                    "scale_y": {
                        "type": "number",
                        "description": "Scale factor for y-coordinate (default: 0.5)",
                        "default": 0.5
                    },
                    "scale_z": {
                        "type": "number",
                        "description": "Scale factor for z-coordinate (default: 1.0)",
                        "default": 1.0
                    },
                    "n_turns": {
                        "type": "integer",
                        "description": "Number of turns in the knot (default: 3 for trefoil)",
                        "default": 3
                    },
                    "tube_radius": {
                        "type": "number",
                        "description": "Radius of the vortex tube (default: 0.2)",
                        "default": 0.2
                    },
                    "save_plot": {
                        "type": "string",
                        "description": "Optional path to save visualization PNG"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="fft_correlation",
            description="Compute frequency-domain correlation between two signals. "
                       "Useful for analyzing functional connectivity between brain regions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_a": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "First time-series signal"
                    },
                    "signal_b": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Second time-series signal"
                    },
                    "sampling_rate": {
                        "type": "number",
                        "description": "Sampling rate in Hz (default: 1000)",
                        "default": 1000
                    },
                    "return_spectrum": {
                        "type": "boolean",
                        "description": "Include frequency spectrum in output (default: false)",
                        "default": False
                    }
                },
                "required": ["signal_a", "signal_b"]
            }
        ),
        Tool(
            name="ray_trace",
            description="Perform optical ray tracing simulation using pyoptools. "
                       "Trace rays through optical elements like lenses and mirrors.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Light source position [x, y, z]"
                    },
                    "source_direction": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Light source direction vector [dx, dy, dz]"
                    },
                    "wavelength": {
                        "type": "number",
                        "description": "Light wavelength in nm (default: 550)",
                        "default": 550
                    },
                    "num_rays": {
                        "type": "integer",
                        "description": "Number of rays to trace (default: 100)",
                        "default": 100
                    },
                    "lens_focal_length": {
                        "type": "number",
                        "description": "Optional lens focal length in mm"
                    },
                    "save_plot": {
                        "type": "string",
                        "description": "Optional path to save ray trace visualization"
                    }
                },
                "required": ["source_position", "source_direction"]
            }
        ),
        Tool(
            name="wavefront_analysis",
            description="Analyze optical wavefront using Zernike polynomials. "
                       "Useful for characterizing optical aberrations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "aperture_radius": {
                        "type": "number",
                        "description": "Aperture radius in mm (default: 10)",
                        "default": 10
                    },
                    "zernike_coefficients": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Zernike polynomial coefficients (default: first 15)"
                    },
                    "resolution": {
                        "type": "integer",
                        "description": "Grid resolution (default: 256)",
                        "default": 256
                    },
                    "save_plot": {
                        "type": "string",
                        "description": "Optional path to save wavefront visualization"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_twosphere_files",
            description="List available files in the MRISpheres/twospheres research directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (default: *.py)",
                        "default": "*.py"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="interferometric_sensing",
            description="Fit interference pattern and compute visibility for biosensing. "
                       "Uses lmfit for automatic uncertainty propagation. Returns visibility V = A/(A + 2*C₀) "
                       "and refractive index shift from phase measurements. "
                       "Supports expert_query for domain-expert parameter suggestions (neuroscience_MRI, physics_optics).",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Spatial position or time coordinate (N points)"
                    },
                    "intensity": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Measured intensity values (N points)"
                    },
                    "wavelength_nm": {
                        "type": "number",
                        "description": "Operating wavelength in nm (default: 633)",
                        "default": 633
                    },
                    "path_length_mm": {
                        "type": "number",
                        "description": "Interferometer path length difference in mm (default: 10)",
                        "default": 10
                    },
                    "compute_delta_n": {
                        "type": "boolean",
                        "description": "Compute refractive index shift from visibility change (default: false)",
                        "default": False
                    },
                    "visibility_baseline": {
                        "type": "number",
                        "description": "Baseline visibility for Δn computation (required if compute_delta_n=true)"
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts for parameter suggestions before analysis. "
                                      "Example: 'What sensitivity for GABA detection in neural tissue?'"
                    }
                },
                "required": ["position", "intensity"]
            }
        ),
        Tool(
            name="lock_in_detection",
            description="Digital lock-in amplification for phase-sensitive detection. "
                       "Extracts signals buried in noise by correlating with reference frequency. "
                       "Returns I/Q channels, amplitude, and phase. "
                       "Supports expert_query for domain-expert guidance (physics_optics, statistics).",
            inputSchema={
                "type": "object",
                "properties": {
                    "signal": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Input signal to demodulate (N points)"
                    },
                    "time": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time vector in seconds (N points)"
                    },
                    "reference_frequency": {
                        "type": "number",
                        "description": "Reference frequency in Hz"
                    },
                    "time_constant": {
                        "type": "number",
                        "description": "Low-pass filter time constant in seconds (default: 1.0)",
                        "default": 1.0
                    },
                    "compute_error_signal": {
                        "type": "boolean",
                        "description": "Compute phase error signal for feedback control (default: false)",
                        "default": False
                    },
                    "setpoint_phase": {
                        "type": "number",
                        "description": "Target phase in radians for error signal (default: 0.0)",
                        "default": 0.0
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts for guidance. "
                                      "Example: 'What time constant for detecting 10 Hz neural oscillations?'"
                    }
                },
                "required": ["signal", "time", "reference_frequency"]
            }
        ),
        Tool(
            name="absorption_spectroscopy",
            description="Measure analyte concentration via Beer-Lambert law: A = ε·c·L. "
                       "Fits transmission spectrum to determine concentration with uncertainty. "
                       "Supports expert_query for domain-expert guidance (bioengineering_LOC, physics_optics).",
            inputSchema={
                "type": "object",
                "properties": {
                    "wavelength": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Wavelength array in nm (N points)"
                    },
                    "transmission": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Transmission spectrum (0 to 1, N points)"
                    },
                    "path_length_cm": {
                        "type": "number",
                        "description": "Optical path length in cm (default: 1.0)",
                        "default": 1.0
                    },
                    "extinction_coefficient": {
                        "type": "number",
                        "description": "Molar extinction coefficient ε in M⁻¹cm⁻¹ (if known)"
                    },
                    "reference_wavelength": {
                        "type": "number",
                        "description": "Wavelength for concentration measurement in nm (default: peak absorption)"
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts for guidance. "
                                      "Example: 'What extinction coefficient for hemoglobin at 540nm?'"
                    }
                },
                "required": ["wavelength", "transmission"]
            }
        ),
        Tool(
            name="cavity_ringdown_spectroscopy",
            description="Ultra-sensitive trace gas detection via cavity ringdown time measurement. "
                       "Fits exponential decay to determine absorption coefficient α = (1/τ - 1/τ₀)·(c/2L).",
            inputSchema={
                "type": "object",
                "properties": {
                    "time": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Time array in microseconds (N points)"
                    },
                    "intensity": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Cavity intensity decay (N points)"
                    },
                    "cavity_length_cm": {
                        "type": "number",
                        "description": "Cavity length in cm (default: 50)",
                        "default": 50
                    },
                    "baseline_ringdown_us": {
                        "type": "number",
                        "description": "Baseline (empty cavity) ringdown time in μs"
                    }
                },
                "required": ["time", "intensity"]
            }
        ),
        Tool(
            name="alignment_sensitivity_monte_carlo",
            description="Monte Carlo simulation of fiber-chip alignment tolerance. "
                       "Computes coupling efficiency statistics under fabrication variations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "wavelength_nm": {
                        "type": "number",
                        "description": "Operating wavelength in nm (default: 1550)",
                        "default": 1550
                    },
                    "fiber_mfd_um": {
                        "type": "number",
                        "description": "Fiber mode field diameter in µm (default: 10.4)",
                        "default": 10.4
                    },
                    "spot_size_um": {
                        "type": "number",
                        "description": "Waveguide spot size in µm (default: 3.0)",
                        "default": 3.0
                    },
                    "lateral_tolerance_um": {
                        "type": "number",
                        "description": "Lateral tolerance ±3σ in µm (default: 1.0)",
                        "default": 1.0
                    },
                    "angular_tolerance_deg": {
                        "type": "number",
                        "description": "Angular tolerance ±3σ in degrees (default: 0.5)",
                        "default": 0.5
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of Monte Carlo samples (default: 10000)",
                        "default": 10000
                    },
                    "query_experts": {
                        "type": "boolean",
                        "description": "Query ernie2_swarm for expert guidance (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="whole_brain_network_analysis",
            description="Complete whole-brain functional connectivity analysis from fMRI time-series. "
                       "Computes connectivity matrix, constructs network, and analyzes graph metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "region_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of brain region names (e.g., ['V1_L', 'V1_R', 'V4_L', 'V4_R'])"
                    },
                    "connectivity_method": {
                        "type": "string",
                        "description": "Connectivity estimation method",
                        "enum": ["distance_correlation", "fft_correlation", "phase_locking"],
                        "default": "distance_correlation"
                    },
                    "network_density": {
                        "type": "number",
                        "description": "Target network edge density 0-1 (default: 0.15)",
                        "default": 0.15
                    },
                    "n_timepoints": {
                        "type": "integer",
                        "description": "Number of time points in synthetic signals (default: 400)",
                        "default": 400
                    },
                    "query_experts": {
                        "type": "boolean",
                        "description": "Query ernie2_swarm for expert guidance (default: false)",
                        "default": False
                    }
                },
                "required": ["region_labels"]
            }
        ),
        Tool(
            name="multi_organ_ooc_simulation",
            description="Simulate multi-organ OOC pharmacokinetics and biomarker transport. "
                       "Models drug/biomarker distribution across interconnected organ chambers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "system_type": {
                        "type": "string",
                        "description": "Predefined OOC system configuration",
                        "enum": ["liver_kidney", "tumor_metastasis"],
                        "default": "liver_kidney"
                    },
                    "source_organ": {
                        "type": "string",
                        "description": "Organ producing biomarker (for biomarker simulation)"
                    },
                    "production_rate": {
                        "type": "number",
                        "description": "Biomarker production rate in µM/min (default: 1.0)",
                        "default": 1.0
                    },
                    "simulation_time_min": {
                        "type": "number",
                        "description": "Simulation time in minutes (default: 60)",
                        "default": 60.0
                    },
                    "query_experts": {
                        "type": "boolean",
                        "description": "Query ernie2_swarm for expert guidance (default: false)",
                        "default": False
                    }
                },
                "required": ["system_type"]
            }
        ),
        Tool(
            name="two_sphere_graph_mapping",
            description="Map planar graphs onto two sphere surfaces using quaternion rotation. "
                       "Visualizes paired brain regions (e.g., left/right hemispheres) with network connectivity. "
                       "Supports random geometric, Erdős-Rényi, small-world, scale-free, and grid graphs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "graph_type": {
                        "type": "string",
                        "description": "Type of graph to generate",
                        "enum": ["random_geometric", "erdos_renyi", "small_world", "scale_free", "grid"],
                        "default": "random_geometric"
                    },
                    "n_nodes": {
                        "type": "integer",
                        "description": "Number of nodes in each graph (default: 100)",
                        "default": 100
                    },
                    "radius": {
                        "type": "number",
                        "description": "Sphere radius (default: 1.0)",
                        "default": 1.0
                    },
                    "rotation_x": {
                        "type": "number",
                        "description": "Rotation around x-axis in degrees (default: 30)",
                        "default": 30.0
                    },
                    "rotation_y": {
                        "type": "number",
                        "description": "Rotation around y-axis in degrees (default: 45)",
                        "default": 45.0
                    },
                    "rotation_z": {
                        "type": "number",
                        "description": "Rotation around z-axis in degrees (default: 0)",
                        "default": 0.0
                    },
                    "show_inter_edges": {
                        "type": "boolean",
                        "description": "Show edges connecting corresponding nodes between spheres (default: false)",
                        "default": False
                    },
                    "save_plot": {
                        "type": "string",
                        "description": "Optional path to save visualization PNG"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="simulate_loc_chip",
            description="Simulate and optimize Lab-on-Chip (LOC) optical system design. "
                       "Phase 5 (F₅ Meta/Planning) tool using tensor routing, ernie2_swarm expert guidance, "
                       "and merge2docs optimization algorithms. Optimizes Strehl ratio, MTF, coupling efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "wavelength_nm": {
                        "type": "number",
                        "description": "Operating wavelength in nanometers (default: 633)",
                        "default": 633
                    },
                    "na_objective": {
                        "type": "number",
                        "description": "Numerical aperture of objective (0.1-1.45, default: 0.6)",
                        "default": 0.6
                    },
                    "pixel_size_um": {
                        "type": "number",
                        "description": "Camera pixel size in micrometers (default: 6.5)",
                        "default": 6.5
                    },
                    "target_strehl": {
                        "type": "number",
                        "description": "Target Strehl ratio (0-1, default: 0.8)",
                        "default": 0.8
                    },
                    "target_coupling_efficiency": {
                        "type": "number",
                        "description": "Target fiber-chip coupling efficiency (0-1, default: 0.7)",
                        "default": 0.7
                    },
                    "query_experts": {
                        "type": "boolean",
                        "description": "Query ernie2_swarm for expert guidance (default: true)",
                        "default": True
                    },
                    "use_tensor_routing": {
                        "type": "boolean",
                        "description": "Use tensor routing for tool selection (default: true)",
                        "default": True
                    },
                    "optimization_method": {
                        "type": "string",
                        "description": "Optimization algorithm to use",
                        "enum": ["nsga2", "bayesian", "differential_evolution", "monte_carlo"],
                        "default": "differential_evolution"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="cfd_microfluidics",
            description="CFD simulation for microfluidic systems using OpenFOAM/CfdOF. "
                       "Supports two applications with shared Stokes flow regime (Re << 1): "
                       "(1) PHLoC organoid culture - 10µm channels for nutrient exchange, "
                       "(2) Glymphatic system - brain perivascular CSF/ISF flow simulation. "
                       "Calculates Reynolds number, pressure drop, flow distribution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "geometry_type": {
                        "type": "string",
                        "description": "Geometry type: 'phloc' for lab-on-chip or 'glymphatic' for brain perivascular",
                        "enum": ["phloc", "glymphatic"],
                        "default": "phloc"
                    },
                    "channel_diameter_um": {
                        "type": "number",
                        "description": "Channel/space diameter in micrometers (default: 10 for phloc, 20 for glymphatic)",
                        "default": 10
                    },
                    "velocity_um_s": {
                        "type": "number",
                        "description": "Flow velocity in micrometers/second (default: 10000 for phloc, 10 for glymphatic)",
                        "default": 10000
                    },
                    "fluid": {
                        "type": "string",
                        "description": "Fluid type: 'water', 'csf' (cerebrospinal fluid), or 'culture_media'",
                        "enum": ["water", "csf", "culture_media"],
                        "default": "water"
                    },
                    "length_mm": {
                        "type": "number",
                        "description": "Channel/vessel segment length in mm (default: 5)",
                        "default": 5
                    },
                    "run_simulation": {
                        "type": "boolean",
                        "description": "If true and FreeCAD RPC available, create geometry in FreeCAD (default: false)",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        # =====================================================================
        # Glymphatic/Microfluidic Simulation Tools (PH-7)
        # =====================================================================
        Tool(
            name="simulate_perivascular_flow",
            description="Simulate CSF flow in brain perivascular spaces (glymphatic system). "
                       "Uses Stokes flow physics (Re << 1) - same as PHLoC microfluidics. "
                       "Supports expert_query for domain guidance (neuroscience_MRI, bioengineering_LOC).",
            inputSchema={
                "type": "object",
                "properties": {
                    "vessel_radius_um": {
                        "type": "number",
                        "description": "Blood vessel radius in µm (typical: 10-100 for arteries)",
                        "default": 50
                    },
                    "gap_thickness_um": {
                        "type": "number",
                        "description": "Perivascular space width in µm (typical: 3-50)",
                        "default": 20
                    },
                    "length_mm": {
                        "type": "number",
                        "description": "Vessel segment length in mm",
                        "default": 5
                    },
                    "pressure_gradient_Pa_m": {
                        "type": "number",
                        "description": "Pressure gradient in Pa/m (typical: 1-100)",
                        "default": 10
                    },
                    "state": {
                        "type": "string",
                        "description": "Brain state: 'awake' or 'sleep' (affects clearance ~60%)",
                        "enum": ["awake", "sleep"],
                        "default": "awake"
                    },
                    "pulsatile": {
                        "type": "boolean",
                        "description": "If true, simulate cardiac-driven pulsatile flow",
                        "default": False
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts. Example: 'What pressure gradient for cortical arteries?'"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="analyze_clearance_network",
            description="Analyze brain network topology for waste clearance efficiency. "
                       "Links disc dimension (graph embedding) to glymphatic clearance prediction. "
                       "Based on Paul et al. 2023 obstruction theory: pobs(tw) = {K₅, K₃,₃}. "
                       "Supports expert_query for domain guidance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "adjacency_matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "N×N adjacency matrix (symmetric, 0/1 or weighted)"
                    },
                    "network_name": {
                        "type": "string",
                        "description": "Optional name for the network",
                        "default": "brain_network"
                    },
                    "optimal_disc_dimension": {
                        "type": "number",
                        "description": "Optimal disc dimension for clearance (default: 2.5)",
                        "default": 2.5
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts. Example: 'What network topology optimizes glymphatic flow?'"
                    }
                },
                "required": ["adjacency_matrix"]
            }
        ),
        Tool(
            name="design_brain_chip_channel",
            description="Design microfluidic channel that mimics brain perivascular conditions. "
                       "Matches physiological shear stress (0.03-0.15 Pa) and residence time. "
                       "Supports expert_query for domain guidance (bioengineering_LOC, neuroscience_MRI).",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_shear_Pa": {
                        "type": "number",
                        "description": "Target wall shear stress in Pa (physiological: 0.03-0.15)",
                        "default": 0.1
                    },
                    "target_residence_time_s": {
                        "type": "number",
                        "description": "Target fluid residence time in seconds",
                        "default": 60
                    },
                    "length_mm": {
                        "type": "number",
                        "description": "Channel length in mm",
                        "default": 10
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts. Example: 'What shear stress for neural cell culture?'"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="analyze_brain_clearance",
            description="Analyze brain glymphatic clearance from fMRI connectivity data. "
                       "Predicts waste clearance efficiency using disc dimension theory "
                       "(Paul et al. 2023) and compares sleep vs awake clearance. "
                       "Includes amyloid-β accumulation risk modeling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connectivity_matrix": {
                        "type": "array",
                        "description": "NxN functional connectivity matrix (list of lists)",
                        "items": {"type": "array", "items": {"type": "number"}}
                    },
                    "region_labels": {
                        "type": "array",
                        "description": "List of brain region names",
                        "items": {"type": "string"}
                    },
                    "brain_state": {
                        "type": "string",
                        "enum": ["awake", "sleep", "anesthesia"],
                        "description": "Brain state for clearance estimation (default: awake)",
                        "default": "awake"
                    },
                    "compute_amyloid": {
                        "type": "boolean",
                        "description": "Whether to compute amyloid-β accumulation dynamics",
                        "default": False
                    },
                    "compare_sleep_wake": {
                        "type": "boolean",
                        "description": "Compare sleep vs wake clearance (Xie et al. 2013)",
                        "default": False
                    },
                    "expert_query": {
                        "type": "string",
                        "description": "Optional: Ask domain experts. Example: 'What connectivity thresholds for Alzheimer risk?'"
                    }
                },
                "required": ["connectivity_matrix"]
            }
        ),
        Tool(
            name="visualize_brain_clearance",
            description="Visualize glymphatic clearance on two-sphere brain model. "
                       "Colors nodes by clearance efficiency, edges by connectivity. "
                       "Can compare sleep vs wake states side-by-side.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connectivity_matrix": {
                        "type": "array",
                        "description": "NxN brain connectivity matrix",
                        "items": {"type": "array", "items": {"type": "number"}}
                    },
                    "region_labels": {
                        "type": "array",
                        "description": "List of brain region names",
                        "items": {"type": "string"}
                    },
                    "brain_state": {
                        "type": "string",
                        "enum": ["awake", "sleep"],
                        "description": "Brain state for visualization",
                        "default": "awake"
                    },
                    "compare_sleep_wake": {
                        "type": "boolean",
                        "description": "Show side-by-side sleep vs wake comparison",
                        "default": False
                    },
                    "show_risk": {
                        "type": "boolean",
                        "description": "Show amyloid risk overlay (red markers)",
                        "default": False
                    },
                    "save_path": {
                        "type": "string",
                        "description": "Path to save visualization PNG"
                    }
                },
                "required": ["connectivity_matrix"]
            }
        ),
        Tool(
            name="design_brain_chip",
            description="Design brain-mimetic microfluidic chip for glymphatic validation. "
                       "Creates channel networks matching brain topology (disc dimension). "
                       "Can export to FreeCAD for fabrication.",
            inputSchema={
                "type": "object",
                "properties": {
                    "connectivity_matrix": {
                        "type": "array",
                        "description": "Optional NxN brain connectivity matrix",
                        "items": {"type": "array", "items": {"type": "number"}}
                    },
                    "n_regions": {
                        "type": "integer",
                        "description": "Number of brain regions (if no matrix)",
                        "default": 5
                    },
                    "network_type": {
                        "type": "string",
                        "enum": ["planar", "non_planar", "tree"],
                        "description": "Network topology type for disc dimension comparison",
                        "default": "planar"
                    },
                    "channel_diameter_um": {
                        "type": "number",
                        "description": "Channel diameter in micrometers",
                        "default": 50
                    },
                    "export_to_freecad": {
                        "type": "boolean",
                        "description": "Export design to FreeCAD (requires FreeCAD RPC)",
                        "default": False
                    },
                    "comparison_set": {
                        "type": "boolean",
                        "description": "Generate planar/non-planar/tree comparison set",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="simulate_perivascular_flow_3d",
            description="3D CFD simulation of perivascular CSF flow. "
                       "Simulates flow in straight, curved, branching, or tortuous vessels. "
                       "Computes velocity fields, wall shear stress, and clearance rates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "vessel_radius_um": {
                        "type": "number",
                        "description": "Inner vessel radius in micrometers (default: 25)",
                        "default": 25
                    },
                    "gap_thickness_um": {
                        "type": "number",
                        "description": "Perivascular space width in micrometers (default: 15)",
                        "default": 15
                    },
                    "length_um": {
                        "type": "number",
                        "description": "Vessel segment length in micrometers (default: 1000)",
                        "default": 1000
                    },
                    "pressure_gradient_Pa_m": {
                        "type": "number",
                        "description": "Pressure gradient in Pa/m (default: 100)",
                        "default": 100
                    },
                    "geometry_type": {
                        "type": "string",
                        "enum": ["straight", "curved", "branching", "tortuous"],
                        "description": "Vessel geometry type (default: straight)",
                        "default": "straight"
                    },
                    "resolution": {
                        "type": "integer",
                        "description": "Grid resolution per dimension (default: 20, higher = more accurate but slower)",
                        "default": 20
                    },
                    "brain_state": {
                        "type": "string",
                        "enum": ["awake", "sleep"],
                        "description": "Brain state for clearance estimation (default: awake)",
                        "default": "awake"
                    },
                    "compare_geometries": {
                        "type": "boolean",
                        "description": "Compare flow across straight/curved/tortuous geometries",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        # =============================================================================
        # Fluigi/NetworkX Integration Tools (F₂ → F₃ pipeline)
        # =============================================================================
        Tool(
            name="design_to_networkx",
            description="Convert a ChipDesign to NetworkX graph for graph-based analysis "
                       "and Fluigi integration. Returns graph metrics and edge list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "design_json": {
                        "type": "object",
                        "description": "ChipDesign as JSON (from design_brain_chip or other design tools)"
                    },
                    "include_metrics": {
                        "type": "boolean",
                        "description": "Include graph metrics (density, clustering, etc.)",
                        "default": True
                    }
                },
                "required": ["design_json"]
            }
        ),
        Tool(
            name="design_to_mint",
            description="Convert a ChipDesign or NetworkX graph to MINT format for Fluigi. "
                       "MINT is the Microfluidic Netlist format used by Fluigi place-and-route.",
            inputSchema={
                "type": "object",
                "properties": {
                    "design_json": {
                        "type": "object",
                        "description": "ChipDesign as JSON (from design_brain_chip)"
                    },
                    "graph_data": {
                        "type": "object",
                        "description": "Alternative: NetworkX graph as node_link_data dict"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Name for the MINT device (default: 'chip')",
                        "default": "chip"
                    },
                    "default_channel_width": {
                        "type": "number",
                        "description": "Default channel width in micrometers",
                        "default": 100
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="fluigi_compile",
            description="Compile MINT code using Fluigi. Validates syntax and generates intermediate representation. "
                       "Part of F₂→F₃ design-to-code pipeline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mint_code": {
                        "type": "string",
                        "description": "MINT code to compile"
                    },
                    "mint_file": {
                        "type": "string",
                        "description": "Alternative: Path to .mint file"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for compiled files",
                        "default": "/tmp/fluigi_output"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="fluigi_parchmint",
            description="Convert MINT to Parchmint JSON format using Fluigi. "
                       "Parchmint is the intermediate format for microfluidic device description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mint_code": {
                        "type": "string",
                        "description": "MINT code to convert"
                    },
                    "mint_file": {
                        "type": "string",
                        "description": "Alternative: Path to .mint file"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for Parchmint JSON",
                        "default": "/tmp/fluigi_output"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="networkx_to_mint",
            description="Direct conversion from NetworkX graph to MINT format. "
                       "Supports various graph types (DiGraph, Graph) with automatic component type inference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "mint_type": {"type": "string", "enum": ["PORT", "NODE", "MIXER", "CHAMBER"]},
                                "port_radius": {"type": "number"},
                                "width": {"type": "number"},
                                "height": {"type": "number"}
                            }
                        },
                        "description": "List of nodes with attributes"
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "width": {"type": "number"}
                            }
                        },
                        "description": "List of edges with attributes"
                    },
                    "device_name": {
                        "type": "string",
                        "description": "Name for the MINT device",
                        "default": "device"
                    },
                    "default_channel_width": {
                        "type": "number",
                        "description": "Default channel width in micrometers",
                        "default": 100
                    }
                },
                "required": ["nodes", "edges"]
            }
        ),
    ]


# =============================================================================
# Tool Implementations
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        # Service meta tools
        if name == "advice":
            result = await handle_advice(arguments)
        elif name == "health_check":
            result = await handle_health_check(arguments)
        elif name == "two_sphere_model":
            result = await handle_two_sphere_model(arguments)
        elif name == "vortex_ring":
            result = await handle_vortex_ring(arguments)
        elif name == "fft_correlation":
            result = await handle_fft_correlation(arguments)
        elif name == "ray_trace":
            result = await handle_ray_trace(arguments)
        elif name == "wavefront_analysis":
            result = await handle_wavefront_analysis(arguments)
        elif name == "list_twosphere_files":
            result = await handle_list_files(arguments)
        elif name == "interferometric_sensing":
            result = await handle_interferometric_sensing(arguments)
        elif name == "lock_in_detection":
            result = await handle_lock_in_detection(arguments)
        elif name == "absorption_spectroscopy":
            result = await handle_absorption_spectroscopy(arguments)
        elif name == "cavity_ringdown_spectroscopy":
            result = await handle_cavity_ringdown_spectroscopy(arguments)
        elif name == "alignment_sensitivity_monte_carlo":
            result = await handle_alignment_sensitivity_monte_carlo(arguments)
        elif name == "whole_brain_network_analysis":
            result = await handle_whole_brain_network_analysis(arguments)
        elif name == "multi_organ_ooc_simulation":
            result = await handle_multi_organ_ooc_simulation(arguments)
        elif name == "two_sphere_graph_mapping":
            result = await handle_two_sphere_graph_mapping(arguments)
        elif name == "simulate_loc_chip":
            result = await handle_simulate_loc_chip(arguments)
        elif name == "cfd_microfluidics":
            result = await handle_cfd_microfluidics(arguments)
        elif name == "simulate_perivascular_flow":
            result = await handle_simulate_perivascular_flow(arguments)
        elif name == "analyze_clearance_network":
            result = await handle_analyze_clearance_network(arguments)
        elif name == "design_brain_chip_channel":
            result = await handle_design_brain_chip_channel(arguments)
        elif name == "analyze_brain_clearance":
            result = await handle_analyze_brain_clearance(arguments)
        elif name == "simulate_perivascular_flow_3d":
            result = await handle_simulate_perivascular_flow_3d(arguments)
        elif name == "design_brain_chip":
            result = await handle_design_brain_chip(arguments)
        elif name == "visualize_brain_clearance":
            result = await handle_visualize_brain_clearance(arguments)
        # Fluigi/NetworkX integration tools
        elif name == "design_to_networkx":
            result = await handle_design_to_networkx(arguments)
        elif name == "design_to_mint":
            result = await handle_design_to_mint(arguments)
        elif name == "fluigi_compile":
            result = await handle_fluigi_compile(arguments)
        elif name == "fluigi_parchmint":
            result = await handle_fluigi_parchmint(arguments)
        elif name == "networkx_to_mint":
            result = await handle_networkx_to_mint(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# =============================================================================
# Service Meta Handlers (merge2docs pattern)
# =============================================================================

# RLServiceAdvisor compatible service catalog
SERVICE_CATALOG = {
    # Optics cluster
    "ray_trace": {
        "description": "Ray tracing through optical surfaces for LOC imaging systems",
        "capabilities": ["ray_tracing", "optical_design", "surface_analysis"],
        "avg_time": 2.0,
        "reliability": 0.95
    },
    "wavefront_analysis": {
        "description": "Zernike polynomial wavefront analysis for aberration characterization",
        "capabilities": ["wavefront", "zernike", "aberration_analysis"],
        "avg_time": 1.5,
        "reliability": 0.95
    },
    "interferometric_sensing": {
        "description": "Phase-shift interferometry for precision measurements",
        "capabilities": ["interferometry", "phase_analysis", "precision_sensing"],
        "avg_time": 3.0,
        "reliability": 0.90
    },
    "absorption_spectroscopy": {
        "description": "Beer-Lambert absorption analysis for concentration measurement",
        "capabilities": ["spectroscopy", "concentration", "absorption"],
        "avg_time": 2.0,
        "reliability": 0.95
    },
    # Microfluidics cluster
    "cfd_microfluidics": {
        "description": "CFD simulation for microfluidic systems using Stokes flow",
        "capabilities": ["cfd", "flow_simulation", "microfluidics", "stokes_flow"],
        "avg_time": 15.0,
        "reliability": 0.85
    },
    "design_brain_chip": {
        "description": "Design brain-mimetic microfluidic chip from connectivity matrix",
        "capabilities": ["chip_design", "brain_mimetic", "microfluidics"],
        "avg_time": 5.0,
        "reliability": 0.90
    },
    "multi_organ_ooc_simulation": {
        "description": "Multi-organ organ-on-chip pharmacokinetics simulation",
        "capabilities": ["ooc", "pharmacokinetics", "multi_organ", "simulation"],
        "avg_time": 10.0,
        "reliability": 0.85
    },
    # Brain analysis cluster
    "two_sphere_model": {
        "description": "Two-sphere geometry for paired brain hemisphere analysis",
        "capabilities": ["geometry", "brain_modeling", "visualization"],
        "avg_time": 1.0,
        "reliability": 0.98
    },
    "analyze_brain_clearance": {
        "description": "Glymphatic clearance analysis from fMRI connectivity",
        "capabilities": ["glymphatic", "clearance", "fmri", "brain_analysis"],
        "avg_time": 8.0,
        "reliability": 0.85
    },
    "whole_brain_network_analysis": {
        "description": "Complete brain network analysis with graph metrics",
        "capabilities": ["network_analysis", "graph_metrics", "brain_connectivity"],
        "avg_time": 12.0,
        "reliability": 0.85
    },
    # Fluigi cluster (F2->F3 pipeline)
    "design_to_networkx": {
        "description": "Convert ChipDesign to NetworkX graph for analysis",
        "capabilities": ["graph_conversion", "networkx", "design_analysis"],
        "avg_time": 0.5,
        "reliability": 0.98
    },
    "design_to_mint": {
        "description": "Convert ChipDesign to MINT format for Fluigi",
        "capabilities": ["mint_generation", "design_conversion", "fluigi"],
        "avg_time": 0.5,
        "reliability": 0.95
    },
    "fluigi_parchmint": {
        "description": "Convert MINT to Parchmint JSON via Fluigi place-and-route",
        "capabilities": ["parchmint", "place_and_route", "fluigi", "fabrication"],
        "avg_time": 5.0,
        "reliability": 0.90
    },
}

# Q-table bootstrap templates for RLServiceAdvisor
CONTEXT_TEMPLATES = {
    "optical_design": {
        "ray_trace": 0.9, "wavefront_analysis": 0.8, "interferometric_sensing": 0.7,
        "absorption_spectroscopy": 0.6, "alignment_sensitivity_monte_carlo": 0.7
    },
    "microfluidic_simulation": {
        "cfd_microfluidics": 0.9, "simulate_perivascular_flow_3d": 0.8,
        "design_brain_chip": 0.7, "multi_organ_ooc_simulation": 0.75
    },
    "brain_analysis": {
        "analyze_brain_clearance": 0.9, "whole_brain_network_analysis": 0.85,
        "two_sphere_model": 0.7, "visualize_brain_clearance": 0.75
    },
    "design_to_fabrication": {
        "design_to_mint": 0.9, "fluigi_parchmint": 0.85,
        "design_to_networkx": 0.8, "fluigi_compile": 0.8
    },
    "graph_analysis": {
        "design_to_networkx": 0.9, "two_sphere_graph_mapping": 0.8,
        "whole_brain_network_analysis": 0.7
    }
}

# Cluster definitions for RLServiceAdvisor
SERVICE_CLUSTERS = {
    0: {
        "name": "optics",
        "services": ["ray_trace", "wavefront_analysis", "interferometric_sensing",
                    "lock_in_detection", "absorption_spectroscopy",
                    "cavity_ringdown_spectroscopy", "alignment_sensitivity_monte_carlo"],
        "complexity_indicators": ["wavefront", "aberration", "interferometer", "spectroscopy", "fiber"]
    },
    1: {
        "name": "microfluidics",
        "services": ["cfd_microfluidics", "simulate_perivascular_flow",
                    "simulate_perivascular_flow_3d", "design_brain_chip_channel",
                    "multi_organ_ooc_simulation"],
        "complexity_indicators": ["cfd", "flow", "channel", "organ-on-chip", "perivascular", "stokes"]
    },
    2: {
        "name": "brain",
        "services": ["two_sphere_model", "two_sphere_graph_mapping",
                    "whole_brain_network_analysis", "analyze_brain_clearance",
                    "visualize_brain_clearance", "design_brain_chip",
                    "analyze_clearance_network"],
        "complexity_indicators": ["glymphatic", "clearance", "connectivity", "fmri", "hemisphere", "brain"]
    },
    3: {
        "name": "fluigi",
        "services": ["design_to_networkx", "design_to_mint", "networkx_to_mint",
                    "fluigi_compile", "fluigi_parchmint"],
        "complexity_indicators": ["mint", "parchmint", "networkx", "place-and-route", "fabrication"]
    },
}

# Gateway required fields (service_gateway.py compatible)
GATEWAY_REQUIREMENTS = {
    "ray_trace": {"required": ["surface_data"], "types": {"surface_data": dict}},
    "design_to_networkx": {"required": ["design_json"], "types": {"design_json": dict}},
    "design_to_mint": {"required": [], "types": {}},  # design_json OR graph_data
    "fluigi_compile": {"required": [], "types": {}},  # mint_code OR mint_file
    "fluigi_parchmint": {"required": [], "types": {}},
    "networkx_to_mint": {"required": ["nodes", "edges"], "types": {"nodes": list, "edges": list}},
    "analyze_brain_clearance": {"required": ["connectivity_matrix"], "types": {"connectivity_matrix": list}},
}

SERVICE_ADVICE = {
    "service_name": "twosphere-mcp",
    "version": "1.0.0",
    "description": "Optical physics, MRI spherical geometry, brain microfluidics, and Fluigi integration service.",
    "functor_level": "F3",  # Code/Implementation level in F0-F4 hierarchy

    # RLServiceAdvisor compatible
    "service_catalog": SERVICE_CATALOG,
    "context_templates": CONTEXT_TEMPLATES,
    "clusters": SERVICE_CLUSTERS,
    "gateway_requirements": GATEWAY_REQUIREMENTS,

    "domains": {
        "optics": {
            "description": "Optical system design, ray tracing, interferometry, absorption spectroscopy",
            "tools": ["ray_trace", "wavefront_analysis", "interferometric_sensing", "lock_in_detection",
                     "absorption_spectroscopy", "cavity_ringdown_spectroscopy", "alignment_sensitivity_monte_carlo"],
            "use_when": "Designing optical components for LOC chips, analyzing wavefronts, fiber coupling"
        },
        "microfluidics": {
            "description": "Lab-on-chip design, CFD simulation, perivascular flow modeling",
            "tools": ["cfd_microfluidics", "simulate_perivascular_flow", "simulate_perivascular_flow_3d",
                     "design_brain_chip_channel", "multi_organ_ooc_simulation"],
            "use_when": "Designing microfluidic channels, simulating flow, organ-on-chip systems"
        },
        "brain": {
            "description": "Brain network analysis, glymphatic clearance, two-sphere geometry",
            "tools": ["two_sphere_model", "two_sphere_graph_mapping", "whole_brain_network_analysis",
                     "analyze_brain_clearance", "visualize_brain_clearance", "design_brain_chip",
                     "analyze_clearance_network"],
            "use_when": "Analyzing brain connectivity, modeling glymphatic flow, paired hemisphere analysis"
        },
        "fluigi": {
            "description": "Microfluidic design automation via Fluigi place-and-route",
            "tools": ["design_to_networkx", "design_to_mint", "networkx_to_mint",
                     "fluigi_compile", "fluigi_parchmint"],
            "use_when": "Converting designs to physical layouts, generating fabrication files, F2->F3 pipeline"
        }
    },
    "routing_hints": {
        "keywords": ["optical", "microfluidic", "brain", "glymphatic", "two-sphere", "LOC", "chip",
                    "CFD", "flow", "MINT", "Fluigi", "parchmint", "networkx"],
        "input_types": ["connectivity_matrix", "time_series", "graph", "design_json", "mint_code"],
        "output_types": ["visualization", "metrics", "mint_code", "parchmint_json", "networkx_graph"],
        "query_types": list(CONTEXT_TEMPLATES.keys())
    }
}


def advice() -> Dict[str, Any]:
    """
    Service advice method (generate_service_pseudocode.py compatible).

    This static method provides service metadata for:
    - RLServiceAdvisor Q-table bootstrapping
    - Pseudocode generation
    - Service discovery and routing
    """
    return {
        # Core metadata
        "description": "Optical physics, MRI spherical geometry, brain microfluidics, and Fluigi integration service (F3 level)",
        "version": "1.0.0",

        # Capabilities for clustering
        "capabilities": [
            "optical_design", "ray_tracing", "wavefront_analysis",
            "microfluidic_simulation", "cfd", "stokes_flow",
            "brain_network_analysis", "glymphatic_clearance", "two_sphere_geometry",
            "fluigi_integration", "mint_generation", "parchmint_conversion",
            "networkx_graphs", "design_automation"
        ],

        # Workflow for pseudocode generation
        "workflow": [
            "1. Receive design request (connectivity matrix, design JSON, or MINT code)",
            "2. Route to appropriate domain tool (optics/microfluidics/brain/fluigi)",
            "3. Execute simulation or conversion",
            "4. Return results (metrics, visualization, or fabrication files)"
        ],

        # Performance metrics for RLServiceAdvisor
        "performance": "Variable: 0.5s (conversions) to 15s (CFD simulations)",
        "memory_usage": "Low-Medium: 50MB-500MB depending on simulation complexity",

        # Best use cases
        "best_for": [
            "Lab-on-chip optical system design",
            "Brain-mimetic microfluidic chip design",
            "Glymphatic clearance modeling from fMRI data",
            "Converting designs to Fluigi MINT format",
            "Generating Parchmint JSON for fabrication"
        ],

        # RLServiceAdvisor integration
        "service_catalog": SERVICE_CATALOG,
        "context_templates": CONTEXT_TEMPLATES,
        "clusters": SERVICE_CLUSTERS,
        "gateway_requirements": GATEWAY_REQUIREMENTS,

        # Routing hints
        "complexity_indicators": ["wavefront", "cfd", "glymphatic", "connectivity", "mint", "parchmint"],
        "domain_tags": ["physics", "optics", "microfluidics", "neuroscience", "fabrication"],
        "functor_level": "F3"
    }


async def handle_advice(args: Dict[str, Any]) -> Dict[str, Any]:
    """Return service guidance for agent routing."""
    domain = args.get("domain", "all")
    verbose = args.get("verbose", False)

    # Get base advice from static method
    base_advice = advice()

    result = {
        "service_name": SERVICE_ADVICE["service_name"],
        "version": base_advice["version"],
        "description": base_advice["description"],
        "functor_level": base_advice["functor_level"],
        "routing_hints": SERVICE_ADVICE["routing_hints"],

        # Include advice() compatible fields for pseudocode generator
        "capabilities": base_advice["capabilities"],
        "workflow": base_advice["workflow"],
        "performance": base_advice["performance"],
        "memory_usage": base_advice["memory_usage"],
        "best_for": base_advice["best_for"],
    }

    if domain == "all":
        result["domains"] = SERVICE_ADVICE["domains"]
        result["service_catalog"] = SERVICE_CATALOG
        result["context_templates"] = CONTEXT_TEMPLATES
        if verbose:
            # Include all tools with their descriptions
            tools = await list_tools()
            result["tools_detail"] = {t.name: t.description for t in tools}
    else:
        if domain in SERVICE_ADVICE["domains"]:
            result["domain_info"] = SERVICE_ADVICE["domains"][domain]
            # Filter service catalog to domain
            domain_tools = SERVICE_ADVICE["domains"][domain]["tools"]
            result["service_catalog"] = {k: v for k, v in SERVICE_CATALOG.items() if k in domain_tools}
            if verbose:
                # Include detailed tool info for this domain
                tools = await list_tools()
                result["tools_detail"] = {
                    t.name: {"description": t.description, "schema": t.inputSchema}
                    for t in tools if t.name in domain_tools
                }
        else:
            result["error"] = f"Unknown domain: {domain}"
            result["available_domains"] = list(SERVICE_ADVICE["domains"].keys())

    result["pseudocode"] = """
SERVICE twosphere-mcp:
    # F3 Implementation Service for Physics/Microfluidics

    ## Meta Tools
    FUNCTION advice(domain?)
        # Returns service capabilities, workflow, and routing hints
        RETURN {description, capabilities, workflow, performance, best_for}
    END FUNCTION

    FUNCTION health_check(subsystems?)
        # Verifies Fluigi, FreeCAD, ernie2_swarm dependencies
        FOR each subsystem in [fluigi, freecad, ernie2, core_deps]
            status[subsystem] := check_availability(subsystem)
        ENDFOR
        RETURN {status, warnings}
    END FUNCTION

    ## Optical Tools
    FUNCTION ray_trace(surface_data, ray_config)
        rays := generate_rays(ray_config)
        traced := trace_through_surface(rays, surface_data)
        RETURN {paths, intersections, aberrations}
    END FUNCTION

    FUNCTION wavefront_analysis(wavefront_data)
        zernike := fit_zernike_polynomials(wavefront_data)
        RETURN {coefficients, rms_error, strehl_ratio}
    END FUNCTION

    ## Microfluidics Tools
    FUNCTION cfd_microfluidics(geometry, params)
        mesh := generate_mesh(geometry)
        solution := solve_stokes_flow(mesh, params)
        RETURN {velocity_field, pressure_drop, shear_stress}
    END FUNCTION

    FUNCTION design_brain_chip(connectivity_matrix)
        graph := matrix_to_graph(connectivity_matrix)
        layout := optimize_planar_layout(graph)
        channels := generate_channel_network(layout)
        RETURN {chambers, channels, ports, design_json}
    END FUNCTION

    ## Brain Analysis Tools
    FUNCTION analyze_brain_clearance(connectivity_matrix, brain_state)
        network := build_clearance_network(connectivity_matrix)
        clearance := compute_glymphatic_clearance(network, brain_state)
        RETURN {clearance_rates, risk_regions, disc_dimension}
    END FUNCTION

    ## Fluigi Pipeline (F2 Design -> F3 Code)
    FUNCTION design_to_networkx(design_json)
        G := DiGraph()
        FOR each chamber in design_json.chambers
            G.add_node(chamber, mint_type="NODE")
        ENDFOR
        FOR each channel in design_json.channels
            G.add_edge(channel.source, channel.target)
        ENDFOR
        RETURN {graph_data, metrics}
    END FUNCTION

    FUNCTION design_to_mint(design_json OR graph_data)
        IF graph_data THEN
            G := node_link_graph(graph_data)
        ELSE
            G := design_to_networkx(design_json)
        ENDIF
        mint_code := nx_to_mint(G, device_name)
        RETURN {mint_code, node_count, edge_count}
    END FUNCTION

    FUNCTION fluigi_parchmint(mint_code)
        CALL fluigi convert-to-parchmint mint_file -o output_dir
        parchmint := load_json(output_dir/device.json)
        RETURN {parchmint, components, connections}
    END FUNCTION
"""

    result["tool"] = "advice"
    return result


async def handle_health_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check service health and dependency status."""
    import subprocess
    import shutil

    check_fluigi = args.get("check_fluigi", True)
    check_freecad = args.get("check_freecad", False)
    check_ernie2 = args.get("check_ernie2", False)

    result = {
        "service": "twosphere-mcp",
        "status": "healthy",
        "checks": {},
        "warnings": []
    }

    # Core Python dependencies
    core_deps = {
        "numpy": False,
        "networkx": False,
        "scipy": False,
    }
    for dep in core_deps:
        try:
            __import__(dep)
            core_deps[dep] = True
        except ImportError:
            core_deps[dep] = False
            result["warnings"].append(f"Missing core dependency: {dep}")

    result["checks"]["core_dependencies"] = core_deps

    # Check Fluigi
    if check_fluigi:
        fluigi_status = {"available": False, "version": None, "path": None}
        try:
            fluigi_path = shutil.which("fluigi")
            if fluigi_path:
                fluigi_status["path"] = fluigi_path
                proc = subprocess.run(["fluigi", "--version"], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    fluigi_status["available"] = True
                    fluigi_status["version"] = proc.stdout.strip() or proc.stderr.strip()
                else:
                    fluigi_status["error"] = proc.stderr
            else:
                fluigi_status["error"] = "fluigi not in PATH"
                result["warnings"].append("Fluigi not available - install pyfluigi")
        except Exception as e:
            fluigi_status["error"] = str(e)

        result["checks"]["fluigi"] = fluigi_status

        # Check nx_bridge
        nx_bridge_status = {"available": False}
        try:
            sys.path.insert(0, os.path.expanduser("~/pyfluigi"))
            from fluigi.nx_bridge import nx_to_mint
            nx_bridge_status["available"] = True
        except ImportError as e:
            nx_bridge_status["error"] = str(e)
            result["warnings"].append("nx_bridge not available")

        result["checks"]["nx_bridge"] = nx_bridge_status

    # Check FreeCAD RPC
    if check_freecad:
        freecad_status = {"available": False}
        try:
            import xmlrpc.client
            proxy = xmlrpc.client.ServerProxy("http://localhost:9875")
            proxy.system.listMethods()
            freecad_status["available"] = True
            freecad_status["endpoint"] = "http://localhost:9875"
        except Exception as e:
            freecad_status["error"] = str(e)
            result["warnings"].append("FreeCAD RPC not available")

        result["checks"]["freecad"] = freecad_status

    # Check ernie2_swarm
    if check_ernie2:
        ernie2_status = {"available": False}
        try:
            from backend.services.ernie2_integration import YadaServicesMCPClient
            ernie2_status["available"] = True
        except ImportError as e:
            ernie2_status["error"] = str(e)

        result["checks"]["ernie2"] = ernie2_status

    # Overall status
    critical_failures = [
        not core_deps.get("numpy", False),
        not core_deps.get("networkx", False),
    ]
    if any(critical_failures):
        result["status"] = "degraded"

    if check_fluigi and not result["checks"].get("fluigi", {}).get("available"):
        result["status"] = "degraded" if result["status"] == "healthy" else result["status"]

    result["tool"] = "health_check"
    return result


async def handle_two_sphere_model(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create two-sphere model for paired brain regions."""
    import numpy as np
    
    radius = args.get("radius", 1.0)
    center1 = args.get("center1", [0, radius, 0])
    center2 = args.get("center2", [0, -radius, 0])
    resolution = args.get("resolution", 100)
    save_plot = args.get("save_plot")
    
    # Generate sphere meshes
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    U, V = np.meshgrid(u, v)
    
    # Sphere 1
    X1 = center1[0] + radius * np.sin(V) * np.cos(U)
    Y1 = center1[1] + radius * np.sin(V) * np.sin(U)
    Z1 = center1[2] + radius * np.cos(V)
    
    # Sphere 2
    X2 = center2[0] + radius * np.sin(V) * np.cos(U)
    Y2 = center2[1] + radius * np.sin(V) * np.sin(U)
    Z2 = center2[2] + radius * np.cos(V)
    
    # Calculate distance between centers
    distance = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(center1, center2)))
    
    result = {
        "sphere1": {
            "center": center1,
            "radius": radius,
            "num_points": resolution * resolution
        },
        "sphere2": {
            "center": center2,
            "radius": radius,
            "num_points": resolution * resolution
        },
        "distance_between_centers": distance,
        "overlap": distance < 2 * radius
    }
    
    if save_plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, Y1, Z1, color='b', alpha=0.5, label='Sphere 1')
        ax.plot_surface(X2, Y2, Z2, color='r', alpha=0.5, label='Sphere 2')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Two-Sphere Brain Region Model')
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        plt.close()
        result["plot_saved"] = save_plot
    
    return result


async def handle_vortex_ring(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate vortex ring with trefoil knot structure."""
    import numpy as np
    from scipy.interpolate import splprep, splev
    
    a = args.get("scale_x", 5.0)
    c = args.get("scale_y", 0.5)
    b = args.get("scale_z", 1.0)
    n_turns = args.get("n_turns", 3)
    tube_radius = args.get("tube_radius", 0.2)
    save_plot = args.get("save_plot")
    ntheta = 100
    
    # Create parameter grid
    t = np.linspace(0, 4 * np.pi, 1000)
    
    # Trefoil knot parametric equations
    x = a * np.sin(n_turns * t)
    y = c * (np.cos(t) + 2 * np.cos(2 * t))
    z = b * (np.sin(t) - 2 * np.sin(2 * t))
    
    # Interpolate for smooth curve
    tck, u_param = splprep([x, y, z], s=0)
    xi, yi, zi = splev(u_param, tck)
    
    result = {
        "knot_type": f"{n_turns}-turn trefoil",
        "tube_radius": tube_radius,
        "num_curve_points": len(xi),
        "bounding_box": {
            "x": [float(np.min(xi)), float(np.max(xi))],
            "y": [float(np.min(yi)), float(np.max(yi))],
            "z": [float(np.min(zi)), float(np.max(zi))]
        },
        "curve_length": float(np.sum(np.sqrt(np.diff(xi)**2 + np.diff(yi)**2 + np.diff(zi)**2)))
    }
    
    if save_plot:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xi, yi, zi, 'b-', linewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Vortex Ring - {n_turns}-Turn Trefoil Knot')
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        plt.close()
        result["plot_saved"] = save_plot
    
    return result


async def handle_fft_correlation(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compute FFT-based correlation between signals."""
    import numpy as np
    
    signal_a = np.array(args["signal_a"])
    signal_b = np.array(args["signal_b"])
    sampling_rate = args.get("sampling_rate", 1000)
    return_spectrum = args.get("return_spectrum", False)
    
    # Ensure same length
    min_len = min(len(signal_a), len(signal_b))
    signal_a = signal_a[:min_len]
    signal_b = signal_b[:min_len]
    
    # Compute FFT
    fft_a = np.fft.fft(signal_a)
    fft_b = np.fft.fft(signal_b)
    freqs = np.fft.fftfreq(min_len, 1/sampling_rate)
    
    # Cross-correlation in frequency domain
    cross_spectrum = fft_a * np.conj(fft_b)
    correlation = np.fft.ifft(cross_spectrum).real
    
    # Normalize
    norm_factor = np.sqrt(np.sum(signal_a**2) * np.sum(signal_b**2))
    if norm_factor > 0:
        correlation = correlation / norm_factor
    
    # Find peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = peak_idx if peak_idx < min_len // 2 else peak_idx - min_len
    
    result = {
        "peak_correlation": float(np.max(np.abs(correlation))),
        "peak_lag_samples": int(peak_lag),
        "peak_lag_seconds": float(peak_lag / sampling_rate),
        "mean_correlation": float(np.mean(correlation)),
        "signal_length": min_len,
        "sampling_rate": sampling_rate
    }
    
    if return_spectrum:
        # Return positive frequencies only
        pos_mask = freqs >= 0
        result["frequencies"] = freqs[pos_mask].tolist()
        result["power_a"] = (np.abs(fft_a[pos_mask])**2).tolist()
        result["power_b"] = (np.abs(fft_b[pos_mask])**2).tolist()
    
    return result


async def handle_ray_trace(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform optical ray tracing (pyoptools wrapper)."""
    try:
        from pyoptools.raytrace.ray import Ray
        from pyoptools.raytrace.system import System
        from pyoptools.raytrace.comp_lib import SphericalLens
        import numpy as np
        
        source_pos = args["source_position"]
        source_dir = args["source_direction"]
        wavelength = args.get("wavelength", 550) * 1e-9  # Convert nm to m
        num_rays = args.get("num_rays", 100)
        focal_length = args.get("lens_focal_length")
        save_plot = args.get("save_plot")
        
        # Normalize direction
        dir_array = np.array(source_dir)
        dir_array = dir_array / np.linalg.norm(dir_array)
        
        result = {
            "source_position": source_pos,
            "source_direction": dir_array.tolist(),
            "wavelength_nm": wavelength * 1e9,
            "num_rays": num_rays,
            "pyoptools_available": True
        }
        
        if focal_length:
            result["lens_focal_length_mm"] = focal_length
        
        return result
        
    except ImportError:
        return {
            "error": "pyoptools not installed",
            "install_hint": "pip install pyoptools",
            "source_position": args.get("source_position"),
            "source_direction": args.get("source_direction")
        }


async def handle_wavefront_analysis(args: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze optical wavefront using Zernike polynomials."""
    import numpy as np
    
    aperture_radius = args.get("aperture_radius", 10)
    resolution = args.get("resolution", 256)
    zernike_coeffs = args.get("zernike_coefficients", [0]*15)
    save_plot = args.get("save_plot")
    
    # Create polar grid
    x = np.linspace(-aperture_radius, aperture_radius, resolution)
    y = np.linspace(-aperture_radius, aperture_radius, resolution)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) / aperture_radius
    Theta = np.arctan2(Y, X)
    
    # Mask outside aperture
    mask = R <= 1
    
    # Simple Zernike-like wavefront (first few terms)
    W = np.zeros_like(R)
    if len(zernike_coeffs) > 0:
        W += zernike_coeffs[0] * np.ones_like(R)  # Piston
    if len(zernike_coeffs) > 1:
        W += zernike_coeffs[1] * R * np.cos(Theta)  # Tilt X
    if len(zernike_coeffs) > 2:
        W += zernike_coeffs[2] * R * np.sin(Theta)  # Tilt Y
    if len(zernike_coeffs) > 3:
        W += zernike_coeffs[3] * (2*R**2 - 1)  # Defocus
    if len(zernike_coeffs) > 4:
        W += zernike_coeffs[4] * R**2 * np.cos(2*Theta)  # Astigmatism
    
    W = np.where(mask, W, np.nan)
    
    result = {
        "aperture_radius_mm": aperture_radius,
        "resolution": resolution,
        "num_zernike_terms": len(zernike_coeffs),
        "rms_wavefront": float(np.nanstd(W)),
        "pv_wavefront": float(np.nanmax(W) - np.nanmin(W))
    }
    
    if save_plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(W, extent=[-aperture_radius, aperture_radius, 
                                   -aperture_radius, aperture_radius],
                       cmap='RdBu_r', origin='lower')
        plt.colorbar(im, label='Wavefront (waves)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Wavefront Analysis')
        ax.set_aspect('equal')
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        plt.close()
        result["plot_saved"] = save_plot
    
    return result


async def handle_list_files(args: Dict[str, Any]) -> Dict[str, Any]:
    """List files in MRISpheres/twospheres directory."""
    import glob
    
    pattern = args.get("pattern", "*.py")
    search_path = Path(TWOSPHERES_PATH) / pattern
    
    files = glob.glob(str(search_path))
    file_info = []
    
    for f in sorted(files):
        p = Path(f)
        file_info.append({
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "path": str(p)
        })
    
    return {
        "twospheres_path": TWOSPHERES_PATH,
        "pattern": pattern,
        "files": file_info,
        "count": len(file_info)
    }


async def handle_interferometric_sensing(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform interferometric visibility analysis for biosensing."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.services.sensing_service import InterferometricSensor
        import numpy as np

        # Check for expert query - get domain-expert parameter suggestions
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="interferometric_sensing"
            )

        # Extract arguments
        position = np.array(args["position"])
        intensity = np.array(args["intensity"])
        wavelength_nm = args.get("wavelength_nm", 633)
        path_length_mm = args.get("path_length_mm", 10)
        compute_delta_n = args.get("compute_delta_n", False)
        visibility_baseline = args.get("visibility_baseline")

        # Create sensor configuration
        config = {
            "wavelength_nm": wavelength_nm,
            "path_length_mm": path_length_mm,
            "refractive_index_sensitivity": 1e-6
        }

        # Initialize sensor and fit visibility
        sensor = InterferometricSensor(config)
        visibility, visibility_stderr, fit_params = await sensor.fit_visibility(
            position, intensity
        )

        result = {
            "visibility": float(visibility),
            "visibility_uncertainty": float(visibility_stderr),
            "amplitude": float(fit_params["amplitude"]),
            "background": float(fit_params["background"]),
            "phase_rad": float(fit_params["phase"]),
            "period": float(fit_params["period"]),
            "chi_square": float(fit_params["chi_square"]),
            "reduced_chi_square": float(fit_params["reduced_chi_square"]),
            "r_squared": float(fit_params["r_squared"]),
            "num_data_points": len(position),
            "wavelength_nm": wavelength_nm,
            "path_length_mm": path_length_mm
        }

        # Optionally compute refractive index shift
        if compute_delta_n and visibility_baseline is not None:
            delta_n, delta_n_stderr = await sensor.compute_refractive_index_shift(
                visibility_baseline, visibility, visibility_stderr
            )
            result["delta_n"] = float(delta_n)
            result["delta_n_uncertainty"] = float(delta_n_stderr)
            result["visibility_baseline"] = float(visibility_baseline)

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        return {"error": str(e), "tool": "interferometric_sensing"}


async def handle_lock_in_detection(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform digital lock-in amplification for phase-sensitive detection."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.optics.feedback_control import DigitalLockIn
        import numpy as np

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="lock_in_detection"
            )

        # Extract arguments
        signal_in = np.array(args["signal"])
        time = np.array(args["time"])
        reference_frequency = args["reference_frequency"]
        time_constant = args.get("time_constant", 1.0)
        compute_error_signal = args.get("compute_error_signal", False)
        setpoint_phase = args.get("setpoint_phase", 0.0)

        # Compute sampling rate from time vector
        if len(time) > 1:
            dt = np.mean(np.diff(time))
            sampling_rate = 1.0 / dt
        else:
            return {"error": "Time vector must have at least 2 points"}

        # Initialize lock-in amplifier
        lock_in = DigitalLockIn(reference_frequency, sampling_rate, time_constant)

        # Perform demodulation
        i_component, q_component, amplitude, phase = lock_in.demodulate(signal_in, time)

        result = {
            "i_mean": float(np.mean(i_component)),
            "q_mean": float(np.mean(q_component)),
            "amplitude_mean": float(np.mean(amplitude)),
            "phase_mean_rad": float(np.mean(phase)),
            "amplitude_std": float(np.std(amplitude)),
            "phase_std_rad": float(np.std(phase)),
            "reference_frequency_hz": reference_frequency,
            "sampling_rate_hz": sampling_rate,
            "time_constant_s": time_constant,
            "num_samples": len(signal_in),
            "noise_bandwidth_hz": 1.0 / (4 * time_constant)
        }

        # Optionally compute error signal for feedback
        if compute_error_signal:
            error_signal = lock_in.compute_error_signal(signal_in, time, setpoint_phase)
            result["error_signal_mean_rad"] = float(np.mean(error_signal))
            result["error_signal_std_rad"] = float(np.std(error_signal))
            result["setpoint_phase_rad"] = float(setpoint_phase)

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        return {"error": str(e), "tool": "lock_in_detection"}


async def handle_absorption_spectroscopy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Measure analyte concentration via Beer-Lambert law."""
    try:
        import numpy as np

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="absorption_spectroscopy"
            )

        # Extract arguments
        wavelength = np.array(args["wavelength"])
        transmission = np.array(args["transmission"])
        path_length_cm = args.get("path_length_cm", 1.0)
        epsilon = args.get("extinction_coefficient")
        ref_wavelength = args.get("reference_wavelength")

        # Compute absorbance: A = -log₁₀(T)
        # Add small epsilon to avoid log(0)
        transmission_safe = np.clip(transmission, 1e-10, 1.0)
        absorbance = -np.log10(transmission_safe)

        # Find peak absorption
        peak_idx = np.argmax(absorbance)
        peak_wavelength = wavelength[peak_idx]
        peak_absorbance = absorbance[peak_idx]

        # Use reference wavelength or peak
        if ref_wavelength is not None:
            # Find closest wavelength
            ref_idx = np.argmin(np.abs(wavelength - ref_wavelength))
            ref_wavelength_actual = wavelength[ref_idx]
            ref_absorbance = absorbance[ref_idx]
        else:
            ref_wavelength_actual = peak_wavelength
            ref_absorbance = peak_absorbance

        result = {
            "peak_wavelength_nm": float(peak_wavelength),
            "peak_absorbance": float(peak_absorbance),
            "reference_wavelength_nm": float(ref_wavelength_actual),
            "reference_absorbance": float(ref_absorbance),
            "path_length_cm": path_length_cm,
            "mean_absorbance": float(np.mean(absorbance)),
            "num_wavelengths": len(wavelength)
        }

        # Compute concentration if extinction coefficient is known
        # Beer-Lambert law: A = ε·c·L  =>  c = A/(ε·L)
        if epsilon is not None and epsilon > 0:
            concentration_M = ref_absorbance / (epsilon * path_length_cm)
            result["concentration_M"] = float(concentration_M)
            result["concentration_mM"] = float(concentration_M * 1000)
            result["extinction_coefficient_M_cm"] = epsilon
        else:
            result["note"] = "Provide extinction_coefficient to compute concentration"

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        return {"error": str(e), "tool": "absorption_spectroscopy"}


async def handle_cavity_ringdown_spectroscopy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-sensitive trace gas detection via cavity ringdown."""
    try:
        from lmfit import Model
        import numpy as np

        # Extract arguments
        time = np.array(args["time"])  # microseconds
        intensity = np.array(args["intensity"])
        cavity_length_cm = args.get("cavity_length_cm", 50)
        baseline_ringdown_us = args.get("baseline_ringdown_us")

        # Define exponential decay model: I(t) = I₀·exp(-t/τ)
        def exponential_decay(t, amplitude, tau, background):
            return amplitude * np.exp(-t / tau) + background

        # Create lmfit Model
        model = Model(exponential_decay, independent_vars=['t'])

        # Initial parameter guesses
        amplitude_guess = np.max(intensity) - np.min(intensity)
        background_guess = np.min(intensity)

        # Estimate tau from half-life
        half_max = (np.max(intensity) + np.min(intensity)) / 2
        half_idx = np.argmin(np.abs(intensity - half_max))
        tau_guess = time[half_idx] / np.log(2)  # τ = t_half / ln(2)

        # Set up parameters
        params = model.make_params(
            amplitude=amplitude_guess,
            tau=tau_guess,
            background=background_guess
        )
        params['amplitude'].min = 0
        params['tau'].min = 0
        params['background'].min = 0

        # Perform fit
        result_fit = model.fit(intensity, params=params, t=time)

        # Extract fitted ringdown time
        tau_us = result_fit.params['tau'].value
        tau_stderr = result_fit.params['tau'].stderr if result_fit.params['tau'].stderr is not None else 0.0

        result = {
            "ringdown_time_us": float(tau_us),
            "ringdown_time_uncertainty_us": float(tau_stderr),
            "amplitude": float(result_fit.params['amplitude'].value),
            "background": float(result_fit.params['background'].value),
            "chi_square": float(result_fit.chisqr),
            "reduced_chi_square": float(result_fit.redchi),
            "cavity_length_cm": cavity_length_cm,
            "num_data_points": len(time)
        }

        # Compute absorption coefficient if baseline is provided
        # α = (1/τ - 1/τ₀)·(c/2L)
        # where c = 3×10¹⁰ cm/s (speed of light)
        if baseline_ringdown_us is not None and baseline_ringdown_us > 0:
            c_cm_per_s = 3e10  # cm/s
            alpha_per_cm = (1/tau_us - 1/baseline_ringdown_us) * (c_cm_per_s / (2 * cavity_length_cm))
            # Convert to per-meter for standard units
            alpha_per_m = alpha_per_cm * 100

            result["absorption_coefficient_per_cm"] = float(alpha_per_cm)
            result["absorption_coefficient_per_m"] = float(alpha_per_m)
            result["baseline_ringdown_us"] = baseline_ringdown_us
        else:
            result["note"] = "Provide baseline_ringdown_us to compute absorption coefficient"

        return result

    except Exception as e:
        return {"error": str(e), "tool": "cavity_ringdown_spectroscopy"}


async def handle_alignment_sensitivity_monte_carlo(args: Dict[str, Any]) -> Dict[str, Any]:
    """Monte Carlo simulation of alignment tolerance for fiber-chip coupling."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.optics.alignment_sensitivity import (
            AlignmentSensitivityAnalyzer,
            AlignmentToleranceSpec
        )

        # Extract arguments
        wavelength_nm = args.get("wavelength_nm", 1550)
        fiber_mfd_um = args.get("fiber_mfd_um", 10.4)
        spot_size_um = args.get("spot_size_um", 3.0)
        lateral_tol = args.get("lateral_tolerance_um", 1.0)
        angular_tol = args.get("angular_tolerance_deg", 0.5)
        n_samples = args.get("n_samples", 10000)
        query_experts = args.get("query_experts", False)

        # Query experts if requested (Phase 5 integration)
        expert_insights = None
        if query_experts:
            try:
                from backend.services.ernie2_integration import query_expert_collections
                question = (
                    f"Analyze fiber-to-chip alignment sensitivity for {wavelength_nm}nm wavelength. "
                    f"Fiber MFD: {fiber_mfd_um}µm, waveguide spot size: {spot_size_um}µm. "
                    f"Tolerances: ±{lateral_tol}µm lateral, ±{angular_tol}° angular. "
                    f"What are the dominant coupling loss mechanisms and mitigation strategies?"
                )
                expert_result = await query_expert_collections(
                    question=question,
                    collections=['physics_optics', 'bioengineering_LOC'],
                    use_cloud=False
                )
                expert_insights = expert_result.get('answer', '')[:300]  # Truncate for brevity
            except Exception as e:
                expert_insights = f"Expert query failed: {str(e)}"

        # Initialize analyzer
        analyzer = AlignmentSensitivityAnalyzer(
            wavelength_nm=wavelength_nm,
            fiber_mfd_um=fiber_mfd_um,
            spot_size_um=spot_size_um
        )

        # Define tolerance spec
        tolerance_spec = AlignmentToleranceSpec(
            lateral_tolerance_um=lateral_tol,
            angular_tolerance_deg=angular_tol
        )

        # Run Monte Carlo simulation
        mc_results = await analyzer.run_monte_carlo(
            tolerance_spec,
            n_samples=n_samples
        )

        # Add expert insights to results if available
        if expert_insights:
            mc_results['expert_insights'] = expert_insights

        return mc_results

    except Exception as e:
        return {"error": str(e), "tool": "alignment_sensitivity_monte_carlo"}


async def handle_whole_brain_network_analysis(args: Dict[str, Any]) -> Dict[str, Any]:
    """Whole-brain functional connectivity network analysis."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.mri.whole_brain_network import WholeBrainNetworkAnalyzer
        import numpy as np

        # Extract arguments
        region_labels = args["region_labels"]
        connectivity_method = args.get("connectivity_method", "distance_correlation")
        network_density = args.get("network_density", 0.15)
        n_timepoints = args.get("n_timepoints", 400)
        query_experts = args.get("query_experts", False)

        # Query experts if requested (Phase 5 integration)
        expert_insights = None
        if query_experts:
            try:
                from backend.services.ernie2_integration import query_expert_collections
                question = (
                    f"Analyze functional connectivity network for brain regions: {', '.join(region_labels[:5])}... "
                    f"Using {connectivity_method} method with {network_density:.1%} network density. "
                    f"What graph metrics are most relevant for understanding information flow and network topology?"
                )
                expert_result = await query_expert_collections(
                    question=question,
                    collections=['mathematics', 'computer_science_papers', 'neuroscience_papers'],
                    use_cloud=False
                )
                expert_insights = expert_result.get('answer', '')[:300]  # Truncate for brevity
            except Exception as e:
                expert_insights = f"Expert query failed: {str(e)}"

        # Initialize analyzer
        analyzer = WholeBrainNetworkAnalyzer(
            region_labels=region_labels,
            sampling_rate_hz=0.5,
            connectivity_method=connectivity_method
        )

        # Generate synthetic fMRI signals (for demonstration)
        # In production, user would provide actual fMRI data
        np.random.seed(42)
        n_regions = len(region_labels)

        # Create correlated signals (simulate functional connectivity)
        shared_signal = np.sin(2 * np.pi * 0.01 * np.arange(n_timepoints))  # Slow oscillation
        time_series = []
        for i in range(n_regions):
            # Each region gets shared signal + noise + unique component
            region_signal = (
                0.5 * shared_signal +
                0.3 * np.sin(2 * np.pi * (0.01 + i * 0.005) * np.arange(n_timepoints)) +
                0.2 * np.random.randn(n_timepoints)
            )
            time_series.append(region_signal)

        # Run complete analysis
        results = await analyzer.run_complete_analysis(
            time_series,
            network_density=network_density
        )

        # Add expert insights to results if available
        if expert_insights:
            results['expert_insights'] = expert_insights

        return results

    except Exception as e:
        return {"error": str(e), "tool": "whole_brain_network_analysis"}


async def handle_multi_organ_ooc_simulation(args: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-organ OOC pharmacokinetics simulation."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.services.multi_organ_ooc import (
            MultiOrganOOC,
            create_liver_kidney_system,
            create_tumor_metastasis_system
        )

        # Extract arguments
        system_type = args["system_type"]
        source_organ = args.get("source_organ")
        production_rate = args.get("production_rate", 1.0)
        simulation_time_min = args.get("simulation_time_min", 60.0)
        query_experts = args.get("query_experts", False)

        # Query experts if requested (Phase 5 integration)
        expert_insights = None
        if query_experts:
            try:
                from backend.services.ernie2_integration import query_expert_collections
                question = (
                    f"Analyze {system_type.replace('_', '-')} organ-on-chip system pharmacokinetics. "
                    f"Biomarker source: {source_organ or 'unknown'}, production rate: {production_rate} µM/min, "
                    f"simulation time: {simulation_time_min} min. "
                    f"What are the key transport parameters and steady-state behavior?"
                )
                expert_result = await query_expert_collections(
                    question=question,
                    collections=['bioengineering_LOC', 'pharmacology_papers'],
                    use_cloud=False
                )
                expert_insights = expert_result.get('answer', '')[:300]  # Truncate for brevity
            except Exception as e:
                expert_insights = f"Expert query failed: {str(e)}"

        # Create OOC system
        if system_type == "liver_kidney":
            ooc_system = create_liver_kidney_system()
            if source_organ is None:
                source_organ = "liver"
        elif system_type == "tumor_metastasis":
            ooc_system = create_tumor_metastasis_system()
            if source_organ is None:
                source_organ = "tumor"
        else:
            return {"error": f"Unknown system_type: {system_type}"}

        # Run biomarker transfer simulation
        results = await ooc_system.compute_biomarker_transfer(
            source_organ=source_organ,
            biomarker_production_rate=production_rate,
            simulation_time_min=simulation_time_min
        )

        # Add expert insights to results if available
        if expert_insights:
            results['expert_insights'] = expert_insights

        return results

    except Exception as e:
        return {"error": str(e), "tool": "multi_organ_ooc_simulation"}


async def handle_two_sphere_graph_mapping(args: Dict[str, Any]) -> Dict[str, Any]:
    """Map planar graphs onto two sphere surfaces with quaternion rotation."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.visualization.graph_on_sphere import create_two_sphere_graph_visualization

        # Extract arguments
        graph_type = args.get("graph_type", "random_geometric")
        n_nodes = args.get("n_nodes", 100)
        radius = args.get("radius", 1.0)
        rotation_x = args.get("rotation_x", 30.0)
        rotation_y = args.get("rotation_y", 45.0)
        show_inter_edges = args.get("show_inter_edges", False)
        save_plot = args.get("save_plot")

        # Create visualization
        result = create_two_sphere_graph_visualization(
            graph_type=graph_type,
            n_nodes=n_nodes,
            radius=radius,
            rotation_x=rotation_x,
            rotation_y=rotation_y,
            show_inter_edges=show_inter_edges,
            save_path=save_plot
        )

        # Add visualization info
        result['message'] = f"Created {graph_type} graph with {n_nodes} nodes on two spheres"
        result['quaternion_rotation'] = {
            'x_degrees': rotation_x,
            'y_degrees': rotation_y,
            'z_degrees': 0.0
        }

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "two_sphere_graph_mapping"
        }


async def handle_simulate_loc_chip(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate and optimize Lab-on-Chip optical system (Phase 5 F₅ integration)."""
    try:
        # Import Phase 5 integration layer
        from backend.integration import (
            TensorRoutingClient,
            call_monte_carlo_service,
            MERGE2DOCS_AVAILABLE
        )
        from backend.services.ernie2_integration import query_expert_collections

        # Extract parameters
        wavelength_nm = args.get("wavelength_nm", 633)
        na_objective = args.get("na_objective", 0.6)
        pixel_size_um = args.get("pixel_size_um", 6.5)
        target_strehl = args.get("target_strehl", 0.8)
        target_coupling_efficiency = args.get("target_coupling_efficiency", 0.7)
        query_experts = args.get("query_experts", True)
        use_tensor_routing = args.get("use_tensor_routing", True)
        optimization_method = args.get("optimization_method", "differential_evolution")

        result = {
            "parameters": {
                "wavelength_nm": wavelength_nm,
                "na_objective": na_objective,
                "pixel_size_um": pixel_size_um,
                "target_strehl": target_strehl,
                "target_coupling_efficiency": target_coupling_efficiency
            },
            "integration_stack": []
        }

        # Layer 1: Tensor routing (if enabled)
        routing_info = None
        if use_tensor_routing:
            try:
                client = TensorRoutingClient()
                query = (
                    f"Optimize Lab-on-Chip optical system at {wavelength_nm}nm wavelength "
                    f"with NA={na_objective} objective, targeting Strehl ratio {target_strehl} "
                    f"and coupling efficiency {target_coupling_efficiency}"
                )
                routing_info = await client.route_query(query, domain_hint='physics')
                result["tensor_routing"] = {
                    "domain": routing_info.get('domain'),
                    "fi_level": routing_info.get('fi_level'),
                    "tools": routing_info.get('tools', []),
                    "cell_address": routing_info.get('routing_info', {}).get('cell_address')
                }
                result["integration_stack"].append("Tensor routing (F₅ planning)")
            except Exception as e:
                result["tensor_routing"] = {"error": str(e), "fallback": True}

        # Layer 2: Expert knowledge query (if enabled)
        expert_insights = ""
        if query_experts:
            try:
                question = (
                    f"Design a Lab-on-Chip optical imaging system at {wavelength_nm}nm wavelength "
                    f"with numerical aperture {na_objective}. The system uses a camera with "
                    f"{pixel_size_um}µm pixels. Target performance: Strehl ratio ≥ {target_strehl}, "
                    f"fiber-to-chip coupling efficiency ≥ {target_coupling_efficiency}. "
                    f"What are the critical design parameters and optimization strategies?"
                )
                expert_result = await query_expert_collections(
                    question=question,
                    collections=['physics_optics', 'bioengineering_LOC'],
                    use_cloud=False
                )
                expert_insights = expert_result.get('answer', 'No expert guidance available')
                result["expert_insights"] = {
                    "answer": expert_insights[:500] + "..." if len(expert_insights) > 500 else expert_insights,
                    "collections": expert_result.get('collections_searched', [])
                }
                result["integration_stack"].append("ernie2_swarm expert guidance")
            except Exception as e:
                result["expert_insights"] = {"error": str(e)}

        # Layer 3: Optimization using merge2docs algorithms
        optimization_result = {}

        if optimization_method == "monte_carlo" and MERGE2DOCS_AVAILABLE:
            try:
                # Use Monte Carlo for tolerance analysis
                mc_result = await call_monte_carlo_service(
                    simulation_type="RISK_ANALYSIS",
                    n_simulations=5000,
                    data={
                        "wavelength_nm": wavelength_nm,
                        "na_objective": na_objective,
                        "target_strehl": target_strehl
                    },
                    confidence_level=0.95
                )
                if mc_result:
                    optimization_result = {
                        "method": "monte_carlo",
                        "via": "merge2docs MonteCarloService (A2A)",
                        "statistics": getattr(mc_result, 'statistics', {}),
                        "confidence_intervals": getattr(mc_result, 'confidence_intervals', {}),
                        "n_samples": 5000
                    }
                    result["integration_stack"].append("merge2docs Monte Carlo optimization (A2A)")
            except Exception as e:
                optimization_result = {"method": "monte_carlo", "error": str(e)}
        else:
            # Simplified optimization (fallback)
            import numpy as np

            # Compute Airy disk radius
            airy_radius_um = 0.61 * (wavelength_nm / 1000) / na_objective

            # Check Nyquist sampling
            nyquist_pixel_size = airy_radius_um / 2
            is_nyquist_sampled = pixel_size_um <= nyquist_pixel_size

            # Estimate Strehl ratio (simplified)
            # Assumes diffraction-limited if well-sampled
            estimated_strehl = 0.8 if is_nyquist_sampled else 0.5

            # Estimate coupling efficiency (simplified model)
            mode_field_diameter = 10.4  # µm for SMF-28 at 1550nm (typical)
            spot_size = airy_radius_um * 2.44  # Full Airy disk
            size_ratio = spot_size / mode_field_diameter
            estimated_coupling = np.exp(-((size_ratio - 1.0) ** 2) / 0.5)

            optimization_result = {
                "method": optimization_method,
                "via": "simplified_model",
                "airy_radius_um": float(airy_radius_um),
                "nyquist_pixel_size_um": float(nyquist_pixel_size),
                "is_nyquist_sampled": bool(is_nyquist_sampled),
                "estimated_strehl_ratio": float(estimated_strehl),
                "estimated_coupling_efficiency": float(estimated_coupling),
                "meets_strehl_target": bool(estimated_strehl >= target_strehl),
                "meets_coupling_target": bool(estimated_coupling >= target_coupling_efficiency),
                "note": "Using simplified optical model. For full optimization, use merge2docs services."
            }
            result["integration_stack"].append("Simplified optical physics model")

        result["optimization"] = optimization_result

        # Layer 4: Design recommendations
        recommendations = []

        if optimization_result.get("method") == "simplified_model":
            if not optimization_result.get("is_nyquist_sampled"):
                recommendations.append(
                    f"Reduce pixel size to ≤ {optimization_result['nyquist_pixel_size_um']:.2f}µm "
                    "for Nyquist sampling"
                )

            if optimization_result.get("estimated_strehl_ratio", 0) < target_strehl:
                recommendations.append(
                    "Improve optical aberration correction to increase Strehl ratio"
                )

            if optimization_result.get("estimated_coupling_efficiency", 0) < target_coupling_efficiency:
                recommendations.append(
                    "Optimize spot size converter or use graded-index fiber for better mode matching"
                )

        if not recommendations:
            recommendations.append("System meets performance targets")

        result["recommendations"] = recommendations

        # Summary
        result["summary"] = {
            "phase": "P5 (F₅ Meta/Planning)",
            "integration_layers": len(result["integration_stack"]),
            "merge2docs_used": MERGE2DOCS_AVAILABLE and optimization_method == "monte_carlo",
            "a2a_pattern": True,
            "status": "optimized"
        }

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "simulate_loc_chip"
        }


async def handle_cfd_microfluidics(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    CFD simulation for microfluidic systems.

    Cross-domain overlap: same physics applies to:
    - PHLoC organoid culture (10µm channels)
    - Brain glymphatic system (10-50µm perivascular spaces)

    Both operate in Stokes flow regime (Re << 1).
    """
    import math

    geometry_type = args.get("geometry_type", "phloc")
    fluid = args.get("fluid", "water")
    length_mm = args.get("length_mm", 5.0)
    run_simulation = args.get("run_simulation", False)

    # Set defaults based on geometry type
    if geometry_type == "phloc":
        channel_diameter_um = args.get("channel_diameter_um", 10)
        velocity_um_s = args.get("velocity_um_s", 10000)  # 10 mm/s typical
    else:  # glymphatic
        channel_diameter_um = args.get("channel_diameter_um", 20)
        velocity_um_s = args.get("velocity_um_s", 10)  # ~10 µm/s in brain

    # Fluid properties
    fluid_properties = {
        "water": {"density": 998, "viscosity": 1.0e-3, "name": "Water"},
        "csf": {"density": 1007, "viscosity": 0.7e-3, "name": "Cerebrospinal Fluid"},
        "culture_media": {"density": 1010, "viscosity": 1.2e-3, "name": "Culture Media"}
    }

    props = fluid_properties.get(fluid, fluid_properties["water"])
    density = props["density"]  # kg/m³
    viscosity = props["viscosity"]  # Pa·s

    # Convert units
    diameter_m = channel_diameter_um * 1e-6
    velocity_ms = velocity_um_s * 1e-6
    length_m = length_mm * 1e-3

    # Calculate Reynolds number: Re = ρvD/μ
    reynolds = density * velocity_ms * diameter_m / viscosity

    # Flow regime classification
    if reynolds < 1:
        flow_regime = "Stokes flow (creeping flow)"
        regime_note = "Inertial effects negligible, viscous dominated"
    elif reynolds < 2300:
        flow_regime = "Laminar flow"
        regime_note = "Smooth, predictable streamlines"
    else:
        flow_regime = "Transitional/Turbulent"
        regime_note = "May have chaotic mixing"

    # Pressure drop (Hagen-Poiseuille for circular pipe): ΔP = 128μLQ/(πD⁴)
    # With Q = v * A = v * π(D/2)²
    area = math.pi * (diameter_m / 2) ** 2
    flow_rate = velocity_ms * area  # m³/s
    pressure_drop = 128 * viscosity * length_m * flow_rate / (math.pi * diameter_m**4)

    # Wall shear stress: τ = 8μv/D
    wall_shear = 8 * viscosity * velocity_ms / diameter_m

    result = {
        "geometry_type": geometry_type,
        "application": "PHLoC organoid culture" if geometry_type == "phloc" else "Glymphatic CSF flow",
        "cross_domain_note": "Same Stokes flow physics applies to both microfluidics and brain glymphatics",

        "geometry": {
            "channel_diameter_um": channel_diameter_um,
            "length_mm": length_mm,
            "cross_section_area_um2": area * 1e12
        },

        "fluid": {
            "type": fluid,
            "name": props["name"],
            "density_kg_m3": density,
            "viscosity_Pa_s": viscosity
        },

        "flow_conditions": {
            "velocity_um_s": velocity_um_s,
            "velocity_mm_s": velocity_um_s / 1000,
            "flow_rate_uL_min": flow_rate * 1e9 * 60
        },

        "dimensionless_numbers": {
            "reynolds_number": reynolds,
            "flow_regime": flow_regime,
            "regime_note": regime_note
        },

        "results": {
            "pressure_drop_Pa": pressure_drop,
            "pressure_drop_mbar": pressure_drop / 100,
            "wall_shear_stress_Pa": wall_shear,
            "wall_shear_stress_dyn_cm2": wall_shear * 10  # CGS units common in biology
        },

        "biological_relevance": {
            "phloc": {
                "organoid_shear_tolerance": "< 0.5 Pa recommended for organoids",
                "nutrient_exchange": "Flow ensures oxygen/nutrient delivery"
            },
            "glymphatic": {
                "waste_clearance": "CSF flow clears metabolic waste (Aβ, tau)",
                "sleep_dependence": "Flow increases ~60% during sleep",
                "perivascular_pumping": "Arterial pulsations drive flow"
            }
        }[geometry_type]
    }

    # Optional: Create geometry in FreeCAD
    if run_simulation:
        try:
            import xmlrpc.client
            server = xmlrpc.client.ServerProxy('http://localhost:9875', allow_none=True)
            if server.ping():
                result["freecad_status"] = "FreeCAD connected - use examples/freecad_cfd_microfluidics.py for full simulation"
            else:
                result["freecad_status"] = "FreeCAD RPC not responding"
        except Exception as e:
            result["freecad_status"] = f"FreeCAD not available: {e}"

    return result


# =============================================================================
# Glymphatic/Microfluidic Simulation Handlers (PH-7)
# =============================================================================

async def handle_simulate_perivascular_flow(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate CSF flow in brain perivascular spaces."""
    try:
        from backend.simulation.glymphatic_flow import (
            GlymphaticFlowSimulator,
            PerivascularSpace
        )

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="microfluidics",
                collections=["docs_library_bioengineering_LOC", "docs_library_neuroscience_MRI"]
            )

        # Create perivascular space geometry
        pvs = PerivascularSpace(
            vessel_radius_um=args.get("vessel_radius_um", 50),
            gap_thickness_um=args.get("gap_thickness_um", 20),
            length_mm=args.get("length_mm", 5),
            vessel_type="artery"
        )

        # Initialize simulator with brain state
        state = args.get("state", "awake")
        simulator = GlymphaticFlowSimulator(state=state)

        # Run simulation
        pressure_gradient = args.get("pressure_gradient_Pa_m", 10)
        pulsatile = args.get("pulsatile", False)

        if pulsatile:
            result = simulator.compute_pulsatile_flow(
                pvs=pvs,
                mean_pressure_gradient=pressure_gradient,
                pulse_amplitude=0.3,
                frequency_Hz=1.0,
                num_cycles=5
            )
        else:
            result = simulator.compute_steady_flow(
                pvs=pvs,
                pressure_gradient_Pa_m=pressure_gradient
            )

        # Add clearance estimate
        flow_rate = result.get('flow_rate_uL_min', result.get('flow_rate_mean_uL_min', 0))
        clearance = simulator.estimate_clearance(
            pvs=pvs,
            flow_rate_uL_min=flow_rate,
            solute_name='amyloid-beta'
        )
        result['clearance_estimate'] = clearance

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "simulate_perivascular_flow"
        }


async def handle_analyze_clearance_network(args: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze brain network topology for waste clearance prediction."""
    try:
        import numpy as np
        from backend.simulation.clearance_network import ClearanceNetworkAnalyzer

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="microfluidics",
                collections=["docs_library_neuroscience_MRI"]
            )

        # Get adjacency matrix
        adj_matrix = np.array(args["adjacency_matrix"])

        # Create analyzer
        optimal_disc = args.get("optimal_disc_dimension", 2.5)
        analyzer = ClearanceNetworkAnalyzer(optimal_disc_dim=optimal_disc)

        # Analyze network
        result = analyzer.analyze_network(adj_matrix)
        result["network_name"] = args.get("network_name", "brain_network")

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "analyze_clearance_network"
        }


async def handle_design_brain_chip_channel(args: Dict[str, Any]) -> Dict[str, Any]:
    """Design microfluidic channel mimicking brain perivascular conditions."""
    try:
        from backend.simulation.glymphatic_flow import GlymphaticFlowSimulator

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="microfluidics",
                collections=["docs_library_bioengineering_LOC", "docs_library_neuroscience_MRI"]
            )

        # Get design parameters
        target_shear = args.get("target_shear_Pa", 0.1)
        target_residence = args.get("target_residence_time_s", 60)
        length_mm = args.get("length_mm", 10)

        # Design channel
        simulator = GlymphaticFlowSimulator()
        result = simulator.design_brain_chip_channel(
            target_shear_Pa=target_shear,
            target_residence_time_s=target_residence,
            length_mm=length_mm
        )

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "design_brain_chip_channel"
        }


async def handle_visualize_brain_clearance(args: Dict[str, Any]) -> Dict[str, Any]:
    """Visualize brain clearance on two-sphere model."""
    try:
        import numpy as np
        from backend.visualization.clearance_visualization import visualize_brain_clearance

        # Get parameters
        connectivity_matrix = np.array(args.get("connectivity_matrix", []))
        if connectivity_matrix.size == 0:
            return {"error": "connectivity_matrix is required"}

        n_regions = connectivity_matrix.shape[0]
        region_labels = args.get("region_labels", [f"region_{i}" for i in range(n_regions)])
        brain_state = args.get("brain_state", "awake")
        compare_sleep_wake = args.get("compare_sleep_wake", False)
        show_risk = args.get("show_risk", False)
        save_path = args.get("save_path")

        # Generate visualization
        result = await visualize_brain_clearance(
            connectivity_matrix,
            region_labels,
            brain_state=brain_state,
            compare_sleep_wake=compare_sleep_wake,
            show_risk=show_risk,
            save_path=save_path,
        )

        result["tool"] = "visualize_brain_clearance"
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "visualize_brain_clearance"
        }


async def handle_design_brain_chip(args: Dict[str, Any]) -> Dict[str, Any]:
    """Design brain-mimetic microfluidic chip."""
    try:
        import numpy as np
        from backend.simulation.brain_chip_designer import (
            BrainChipDesigner,
            FreeCADExporter,
            design_brain_chip,
        )

        # Get parameters
        connectivity_matrix = args.get("connectivity_matrix")
        n_regions = args.get("n_regions", 5)
        network_type = args.get("network_type", "planar")
        channel_diameter_um = args.get("channel_diameter_um", 50)
        export_to_freecad = args.get("export_to_freecad", False)
        comparison_set = args.get("comparison_set", False)

        designer = BrainChipDesigner()

        if comparison_set:
            # Generate comparison set
            designs = designer.design_comparison_set(n_regions)
            result = {
                "comparison_set": True,
                "designs": {
                    name: design.to_dict()
                    for name, design in designs.items()
                },
            }

            if export_to_freecad:
                exporter = FreeCADExporter()
                if exporter.connect():
                    export_results = exporter.export_comparison_set(designs)
                    result["freecad_exports"] = export_results
                else:
                    result["freecad_error"] = "Could not connect to FreeCAD RPC"

        else:
            # Single design
            conn_matrix = np.array(connectivity_matrix) if connectivity_matrix else None
            result = design_brain_chip(
                connectivity_matrix=conn_matrix,
                n_regions=n_regions,
                network_type=network_type,
                channel_diameter_um=channel_diameter_um,
                export_to_freecad=export_to_freecad,
            )

        result["tool"] = "design_brain_chip"
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "design_brain_chip"
        }


async def handle_simulate_perivascular_flow_3d(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate 3D perivascular CSF flow."""
    try:
        from backend.simulation.cfd_3d import CFD3DSimulator

        # Get parameters
        vessel_radius = args.get("vessel_radius_um", 25)
        gap_thickness = args.get("gap_thickness_um", 15)
        length = args.get("length_um", 1000)
        pressure_gradient = args.get("pressure_gradient_Pa_m", 100)
        geometry_type = args.get("geometry_type", "straight")
        resolution = args.get("resolution", 20)
        brain_state = args.get("brain_state", "awake")
        compare_geometries = args.get("compare_geometries", False)

        # Create simulator
        simulator = CFD3DSimulator()

        if compare_geometries:
            # Compare different geometries
            result = simulator.compare_geometries(
                vessel_radius_um=vessel_radius,
                gap_thickness_um=gap_thickness,
                length_um=length,
                pressure_gradient_Pa_m=pressure_gradient,
            )
        else:
            # Single geometry simulation
            result = simulator.simulate_perivascular_flow_3d(
                vessel_radius_um=vessel_radius,
                gap_thickness_um=gap_thickness,
                length_um=length,
                pressure_gradient_Pa_m=pressure_gradient,
                geometry_type=geometry_type,
                resolution=resolution,
                brain_state=brain_state,
            )

        result["tool"] = "simulate_perivascular_flow_3d"
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "simulate_perivascular_flow_3d"
        }


async def handle_analyze_brain_clearance(args: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze brain glymphatic clearance from fMRI connectivity."""
    try:
        import numpy as np
        from backend.simulation.glymphatic_fmri_integration import (
            GlymphaticFMRIIntegrator,
            BrainState,
        )

        # Check for expert query
        expert_response = None
        expert_query = args.get("expert_query")
        if expert_query:
            expert_response = await query_domain_experts(
                question=expert_query,
                tool_name="microfluidics",
                collections=["docs_library_neuroscience_MRI", "docs_library_bioengineering_LOC"]
            )

        # Get connectivity matrix
        connectivity_matrix = np.array(args.get("connectivity_matrix", []))
        if connectivity_matrix.size == 0:
            return {"error": "connectivity_matrix is required"}

        # Get region labels or generate defaults
        n_regions = connectivity_matrix.shape[0]
        region_labels = args.get("region_labels", [f"region_{i}" for i in range(n_regions)])

        # Get brain state
        state_str = args.get("brain_state", "awake")
        brain_state = {
            "awake": BrainState.AWAKE,
            "sleep": BrainState.SLEEP,
            "anesthesia": BrainState.ANESTHESIA,
        }.get(state_str, BrainState.AWAKE)

        compute_amyloid = args.get("compute_amyloid", False)
        compare_sleep_wake = args.get("compare_sleep_wake", False)

        # Create integrator
        integrator = GlymphaticFMRIIntegrator()

        # Run analysis
        if compare_sleep_wake:
            # Comparison mode - analyze both states
            result = await integrator.compare_sleep_wake_clearance(
                connectivity_matrix,
                region_labels,
            )
        else:
            # Single state analysis
            analysis = await integrator.analyze_from_connectivity_matrix(
                connectivity_matrix,
                region_labels,
                brain_state=brain_state,
                compute_amyloid=compute_amyloid,
            )
            result = analysis.to_dict()

        # Include expert response if queried
        if expert_response and expert_response.get("answer"):
            result["expert_guidance"] = expert_response["answer"]
            result["expert_collections"] = expert_response.get("collections_queried", [])

        result["tool"] = "analyze_brain_clearance"
        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": "analyze_brain_clearance"
        }


# =============================================================================
# Fluigi/NetworkX Integration Handlers
# =============================================================================

async def handle_design_to_networkx(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ChipDesign to NetworkX graph."""
    try:
        import networkx as nx

        design_json = args.get("design_json", {})
        include_metrics = args.get("include_metrics", True)

        # Build NetworkX graph from design JSON
        G = nx.DiGraph()

        # Add chambers as nodes
        chambers = design_json.get("chambers", [])
        for chamber in chambers:
            name = chamber.get("name", f"chamber_{len(G.nodes)}")
            G.add_node(name,
                      mint_type="NODE",
                      width=chamber.get("width", 1000),
                      height=chamber.get("height", 1000),
                      pos=chamber.get("center", [0, 0])[:2] if chamber.get("center") else None)

        # Add ports
        inlet_ports = design_json.get("inlet_ports", [])
        for i, port in enumerate(inlet_ports):
            name = port.get("name", f"inlet_{i}")
            G.add_node(name, mint_type="PORT", port_radius=500)

        outlet_ports = design_json.get("outlet_ports", [])
        for i, port in enumerate(outlet_ports):
            name = port.get("name", f"outlet_{i}")
            G.add_node(name, mint_type="PORT", port_radius=500)

        # Add main inlet/outlet if present
        if design_json.get("main_inlet_position"):
            G.add_node("main_inlet", mint_type="PORT", port_radius=500)
        if design_json.get("main_outlet_position"):
            G.add_node("main_outlet", mint_type="PORT", port_radius=500)

        # Add channels as edges
        channels = design_json.get("channels", [])
        for channel in channels:
            source = channel.get("source", channel.get("from"))
            target = channel.get("target", channel.get("to"))
            if source and target:
                G.add_edge(source, target,
                          width=channel.get("width", 100),
                          name=channel.get("name"))

        # Compute metrics if requested
        result = {
            "nodes": list(G.nodes()),
            "edges": [(u, v) for u, v in G.edges()],
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        }

        if include_metrics and G.number_of_nodes() > 0:
            result["metrics"] = {
                "density": nx.density(G),
                "is_connected": nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G),
            }
            if G.number_of_nodes() > 1:
                try:
                    result["metrics"]["average_clustering"] = nx.average_clustering(G.to_undirected())
                except:
                    pass

        # Include node_link_data for serialization
        result["graph_data"] = nx.node_link_data(G)
        result["tool"] = "design_to_networkx"

        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "tool": "design_to_networkx"}


async def handle_design_to_mint(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ChipDesign or NetworkX graph to MINT format."""
    try:
        import networkx as nx
        sys.path.insert(0, os.path.expanduser("~/pyfluigi"))
        from fluigi.nx_bridge import nx_to_mint, MintComponentType

        design_json = args.get("design_json")
        graph_data = args.get("graph_data")
        device_name = args.get("device_name", "chip")
        default_channel_width = args.get("default_channel_width", 100)

        if graph_data:
            # Direct from NetworkX node_link_data
            G = nx.node_link_graph(graph_data)
        elif design_json:
            # Convert design to graph first
            G = nx.DiGraph()

            # Add chambers
            for chamber in design_json.get("chambers", []):
                name = chamber.get("name", f"chamber_{len(G.nodes)}")
                G.add_node(name, mint_type="NODE")

            # Add ports
            for i, port in enumerate(design_json.get("inlet_ports", [])):
                name = port.get("name", f"inlet_{i}")
                G.add_node(name, mint_type="PORT", port_radius=500)
            for i, port in enumerate(design_json.get("outlet_ports", [])):
                name = port.get("name", f"outlet_{i}")
                G.add_node(name, mint_type="PORT", port_radius=500)

            if design_json.get("main_inlet_position"):
                G.add_node("main_inlet", mint_type="PORT", port_radius=500)
            if design_json.get("main_outlet_position"):
                G.add_node("main_outlet", mint_type="PORT", port_radius=500)

            # Add channels
            for channel in design_json.get("channels", []):
                source = channel.get("source", channel.get("from"))
                target = channel.get("target", channel.get("to"))
                if source and target:
                    G.add_edge(source, target, width=channel.get("width", default_channel_width))
        else:
            return {"error": "Either design_json or graph_data required", "tool": "design_to_mint"}

        # Convert to MINT
        mint_code = nx_to_mint(G, device_name=device_name, default_channel_width=default_channel_width)

        return {
            "mint_code": mint_code,
            "device_name": device_name,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "tool": "design_to_mint"
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "tool": "design_to_mint"}


async def handle_fluigi_compile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compile MINT code using Fluigi."""
    try:
        import subprocess
        import tempfile

        mint_code = args.get("mint_code")
        mint_file = args.get("mint_file")
        output_dir = args.get("output_dir", "/tmp/fluigi_output")

        os.makedirs(output_dir, exist_ok=True)

        # Write MINT code to temp file if provided directly
        if mint_code:
            mint_file = os.path.join(output_dir, "input.mint")
            with open(mint_file, "w") as f:
                f.write(mint_code)

        if not mint_file:
            return {"error": "Either mint_code or mint_file required", "tool": "fluigi_compile"}

        # Run fluigi mint-compile
        result = subprocess.run(
            ["fluigi", "mint-compile", mint_file, "-o", output_dir],
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "output_dir": output_dir,
            "mint_file": mint_file,
            "tool": "fluigi_compile"
        }

    except subprocess.TimeoutExpired:
        return {"error": "Fluigi compile timed out", "tool": "fluigi_compile"}
    except FileNotFoundError:
        return {"error": "Fluigi not found. Ensure pyfluigi is installed and fluigi is in PATH.", "tool": "fluigi_compile"}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "tool": "fluigi_compile"}


async def handle_fluigi_parchmint(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MINT to Parchmint JSON using Fluigi."""
    try:
        import subprocess

        mint_code = args.get("mint_code")
        mint_file = args.get("mint_file")
        output_dir = args.get("output_dir", "/tmp/fluigi_output")

        os.makedirs(output_dir, exist_ok=True)

        # Write MINT code to temp file if provided directly
        if mint_code:
            mint_file = os.path.join(output_dir, "input.mint")
            with open(mint_file, "w") as f:
                f.write(mint_code)

        if not mint_file:
            return {"error": "Either mint_code or mint_file required", "tool": "fluigi_parchmint"}

        # Run fluigi convert-to-parchmint
        result = subprocess.run(
            ["fluigi", "convert-to-parchmint", mint_file, "-o", output_dir],
            capture_output=True,
            text=True,
            timeout=60
        )

        # Try to read the output JSON
        parchmint_json = None
        device_name = os.path.splitext(os.path.basename(mint_file))[0]
        json_path = os.path.join(output_dir, f"{device_name}.json")

        if os.path.exists(json_path):
            with open(json_path) as f:
                parchmint_json = json.load(f)

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "output_dir": output_dir,
            "json_path": json_path if os.path.exists(json_path) else None,
            "parchmint": parchmint_json,
            "tool": "fluigi_parchmint"
        }

    except subprocess.TimeoutExpired:
        return {"error": "Fluigi convert timed out", "tool": "fluigi_parchmint"}
    except FileNotFoundError:
        return {"error": "Fluigi not found. Ensure pyfluigi is installed and fluigi is in PATH.", "tool": "fluigi_parchmint"}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "tool": "fluigi_parchmint"}


async def handle_networkx_to_mint(args: Dict[str, Any]) -> Dict[str, Any]:
    """Direct conversion from node/edge lists to MINT format."""
    try:
        import networkx as nx
        sys.path.insert(0, os.path.expanduser("~/pyfluigi"))
        from fluigi.nx_bridge import nx_to_mint, MintComponentType

        nodes = args.get("nodes", [])
        edges = args.get("edges", [])
        device_name = args.get("device_name", "device")
        default_channel_width = args.get("default_channel_width", 100)

        # Build graph
        G = nx.DiGraph()

        # Add nodes
        for node in nodes:
            node_id = node.get("id")
            mint_type = node.get("mint_type", "NODE")
            attrs = {k: v for k, v in node.items() if k not in ("id",)}
            attrs["mint_type"] = mint_type
            G.add_node(node_id, **attrs)

        # Add edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            width = edge.get("width", default_channel_width)
            G.add_edge(source, target, width=width)

        # Convert to MINT
        mint_code = nx_to_mint(G, device_name=device_name, default_channel_width=default_channel_width)

        return {
            "mint_code": mint_code,
            "device_name": device_name,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "tool": "networkx_to_mint"
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "tool": "networkx_to_mint"}


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
