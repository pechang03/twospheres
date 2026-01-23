#!/usr/bin/env python3
"""TwoSphere MCP HTTP Server - REST API for optical physics and MRI analysis.

Usage:
    python bin/twosphere_http_server.py --port 8006
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import tool handlers from MCP server
from twosphere_mcp import (
    # Meta tools
    handle_advice,
    handle_health_check,
    advice,
    SERVICE_CATALOG,
    # Core tools
    handle_two_sphere_model,
    handle_vortex_ring,
    handle_fft_correlation,
    handle_ray_trace,
    handle_wavefront_analysis,
    handle_list_files,
    # Fluigi/NetworkX tools
    handle_design_to_networkx,
    handle_design_to_mint,
    handle_fluigi_compile,
    handle_fluigi_parchmint,
    handle_networkx_to_mint,
)

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env_twosphere"
if env_path.exists():
    load_dotenv(env_path)


# =============================================================================
# API Routes
# =============================================================================

async def health_check(request):
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "twosphere-mcp",
        "version": "0.1.0"
    })


async def list_tools(request):
    """List available tools from SERVICE_CATALOG + meta tools."""
    # Build tools list from SERVICE_CATALOG
    tools = [
        {"name": "advice", "description": "Get service guidance, capabilities, and routing hints"},
        {"name": "health_check", "description": "Check service health and dependency status"},
    ]

    # Add all tools from SERVICE_CATALOG
    for name, info in SERVICE_CATALOG.items():
        tools.append({
            "name": name,
            "description": info.get("description", ""),
            "capabilities": info.get("capabilities", []),
            "avg_time": info.get("avg_time"),
            "reliability": info.get("reliability")
        })

    # Add legacy tools not in catalog
    legacy_tools = [
        {"name": "two_sphere_model", "description": "Create and visualize two-sphere model for brain regions"},
        {"name": "vortex_ring", "description": "Generate vortex ring with trefoil knot structure"},
        {"name": "fft_correlation", "description": "Compute FFT-based correlation between signals"},
        {"name": "ray_trace", "description": "Optical ray tracing simulation"},
        {"name": "wavefront_analysis", "description": "Analyze optical wavefront with Zernike polynomials"},
        {"name": "list_twosphere_files", "description": "List files in MRISpheres/twospheres directory"},
        {"name": "fluigi_compile", "description": "Compile MINT code using Fluigi"},
        {"name": "networkx_to_mint", "description": "Convert NetworkX node/edge lists to MINT format"},
    ]

    existing_names = {t["name"] for t in tools}
    for tool in legacy_tools:
        if tool["name"] not in existing_names:
            tools.append(tool)

    return JSONResponse({"tools": tools, "count": len(tools)})


async def call_tool(request):
    """Call a tool by name."""
    try:
        body = await request.json()
        name = body.get("name")
        arguments = body.get("arguments", {})

        handlers = {
            # Meta tools
            "advice": handle_advice,
            "health_check": handle_health_check,
            # Core tools
            "two_sphere_model": handle_two_sphere_model,
            "vortex_ring": handle_vortex_ring,
            "fft_correlation": handle_fft_correlation,
            "ray_trace": handle_ray_trace,
            "wavefront_analysis": handle_wavefront_analysis,
            "list_twosphere_files": handle_list_files,
            # Fluigi/NetworkX tools
            "design_to_networkx": handle_design_to_networkx,
            "design_to_mint": handle_design_to_mint,
            "fluigi_compile": handle_fluigi_compile,
            "fluigi_parchmint": handle_fluigi_parchmint,
            "networkx_to_mint": handle_networkx_to_mint,
        }

        if name not in handlers:
            return JSONResponse(
                {"error": f"Unknown tool: {name}", "available": list(handlers.keys())},
                status_code=400
            )

        result = await handlers[name](arguments)
        return JSONResponse({"result": result})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Application Setup
# =============================================================================

routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/tools", list_tools, methods=["GET"]),
    Route("/api/call_tool", call_tool, methods=["POST"]),
]

app = Starlette(routes=routes)


def main():
    parser = argparse.ArgumentParser(description="TwoSphere MCP HTTP Server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("TWOSPHERE_PORT", 8006)),
        help="Port to run server on (default: 8006)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    print(f"🔬 TwoSphere MCP HTTP Server starting on {args.host}:{args.port}")
    print(f"   Health: http://{args.host}:{args.port}/health")
    print(f"   Tools:  http://{args.host}:{args.port}/api/tools")
    print(f"   Call:   POST http://{args.host}:{args.port}/api/call_tool")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
