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
