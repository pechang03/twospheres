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
    handle_two_sphere_model,
    handle_vortex_ring,
    handle_fft_correlation,
    handle_ray_trace,
    handle_wavefront_analysis,
    handle_list_files,
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
    """List available tools."""
    tools = [
        {
            "name": "two_sphere_model",
            "description": "Create and visualize two-sphere model for brain regions"
        },
        {
            "name": "vortex_ring",
            "description": "Generate vortex ring with trefoil knot structure"
        },
        {
            "name": "fft_correlation",
            "description": "Compute FFT-based correlation between signals"
        },
        {
            "name": "ray_trace",
            "description": "Optical ray tracing simulation"
        },
        {
            "name": "wavefront_analysis",
            "description": "Analyze optical wavefront with Zernike polynomials"
        },
        {
            "name": "list_twosphere_files",
            "description": "List files in MRISpheres/twospheres directory"
        }
    ]
    return JSONResponse({"tools": tools})


async def call_tool(request):
    """Call a tool by name."""
    try:
        body = await request.json()
        name = body.get("name")
        arguments = body.get("arguments", {})
        
        handlers = {
            "two_sphere_model": handle_two_sphere_model,
            "vortex_ring": handle_vortex_ring,
            "fft_correlation": handle_fft_correlation,
            "ray_trace": handle_ray_trace,
            "wavefront_analysis": handle_wavefront_analysis,
            "list_twosphere_files": handle_list_files,
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
