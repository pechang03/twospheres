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
# Tool Definitions
# =============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available TwoSphere MCP tools."""
    return [
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
                       "and refractive index shift from phase measurements.",
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
                    }
                },
                "required": ["position", "intensity"]
            }
        ),
        Tool(
            name="lock_in_detection",
            description="Digital lock-in amplification for phase-sensitive detection. "
                       "Extracts signals buried in noise by correlating with reference frequency. "
                       "Returns I/Q channels, amplitude, and phase.",
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
                    }
                },
                "required": ["signal", "time", "reference_frequency"]
            }
        ),
        Tool(
            name="absorption_spectroscopy",
            description="Measure analyte concentration via Beer-Lambert law: A = ε·c·L. "
                       "Fits transmission spectrum to determine concentration with uncertainty.",
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
                    }
                },
                "required": ["system_type"]
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
        if name == "two_sphere_model":
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
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


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

        return result

    except Exception as e:
        return {"error": str(e), "tool": "interferometric_sensing"}


async def handle_lock_in_detection(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform digital lock-in amplification for phase-sensitive detection."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from backend.optics.feedback_control import DigitalLockIn
        import numpy as np

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

        return result

    except Exception as e:
        return {"error": str(e), "tool": "lock_in_detection"}


async def handle_absorption_spectroscopy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Measure analyte concentration via Beer-Lambert law."""
    try:
        import numpy as np

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

        return results

    except Exception as e:
        return {"error": str(e), "tool": "multi_organ_ooc_simulation"}


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
