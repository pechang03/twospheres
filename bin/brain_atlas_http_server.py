#!/usr/bin/env python3
"""Brain Atlas MCP HTTP Server - Multi-species brain atlas queries.

A standalone service providing brain atlas lookups for:
- Macaque: D99 v2.0 (368 regions)
- Mouse: Allen CCF v3 (800+ regions) - requires allensdk
- Rat: Waxholm Space (222 regions) - requires download

Can be used by both twosphere-mcp and merge2docs.

Usage:
    python bin/brain_atlas_http_server.py --port 8007
    
Endpoints:
    GET  /health                  - Health check
    GET  /api/atlases             - List available atlases
    POST /api/lookup_region       - Look up region by name/coord
    POST /api/generate_mask       - Generate NIfTI ROI mask
    POST /api/list_regions        - List/filter regions
    POST /api/get_neighbors       - Get neighboring regions

Installation:
    # D99 Macaque (auto-installed)
    cd data/atlases && curl -O https://afni.nimh.nih.gov/.../D99_v2.0_dist.tgz && tar -xvf D99_v2.0_dist.tgz
    
    # Allen Mouse CCF (requires SDK)
    pip install allensdk
    python -c "from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache; ..."
    
    # Waxholm Rat
    # Download from https://www.nitrc.org/projects/whs-sd-atlas/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import atlas modules
try:
    from atlases import get_atlas, BrainAtlas, BrainRegion
    from atlases.d99_atlas import D99Atlas
    ATLASES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Atlas modules not fully available: {e}")
    ATLASES_AVAILABLE = False

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env_twosphere"
if env_path.exists():
    load_dotenv(env_path)


# =============================================================================
# Atlas Registry
# =============================================================================

AVAILABLE_ATLASES = {
    "macaque": {
        "D99": {
            "name": "D99 v2.0",
            "regions": 368,
            "description": "MRI-histology macaque atlas (Saleem & Logothetis)",
            "installed": False,
        }
    },
    "mouse": {
        "Allen_CCF": {
            "name": "Allen CCF v3",
            "regions": 800,
            "description": "Allen Institute Mouse Brain Common Coordinate Framework",
            "installed": False,
            "requires": "pip install allensdk",
        }
    },
    "rat": {
        "Waxholm": {
            "name": "Waxholm Space v4",
            "regions": 222,
            "description": "Sprague-Dawley rat brain atlas",
            "installed": False,
            "requires": "Download from NITRC",
        }
    }
}


def check_atlas_availability():
    """Check which atlases are actually installed."""
    atlas_dir = Path(__file__).parent.parent / "data" / "atlases"
    
    # D99
    d99_path = atlas_dir / "D99_v2.0_dist" / "D99_atlas_v2.0.nii.gz"
    AVAILABLE_ATLASES["macaque"]["D99"]["installed"] = d99_path.exists()
    
    # Allen CCF - check for downloaded files
    allen_manifest = atlas_dir / "Allen_CCF_mouse" / "mouse_connectivity_manifest.json"
    allen_structures = atlas_dir / "Allen_CCF_mouse" / "structure_graph.json"
    AVAILABLE_ATLASES["mouse"]["Allen_CCF"]["installed"] = (
        allen_manifest.exists() or allen_structures.exists()
    )
    
    # Waxholm
    waxholm_path = atlas_dir / "Waxholm_rat"
    waxholm_files = list(waxholm_path.glob("*.nii*")) if waxholm_path.exists() else []
    AVAILABLE_ATLASES["rat"]["Waxholm"]["installed"] = len(waxholm_files) > 0


# =============================================================================
# API Routes
# =============================================================================

async def health_check(request):
    """Health check endpoint."""
    check_atlas_availability()
    
    installed = []
    for species, atlases in AVAILABLE_ATLASES.items():
        for atlas_name, info in atlases.items():
            if info["installed"]:
                installed.append(f"{species}/{atlas_name}")
    
    return JSONResponse({
        "status": "healthy",
        "service": "brain-atlas-mcp",
        "version": "0.1.0",
        "installed_atlases": installed,
    })


async def list_atlases(request):
    """List all available atlases."""
    check_atlas_availability()
    return JSONResponse({"atlases": AVAILABLE_ATLASES})


async def lookup_region(request):
    """Look up a brain region.
    
    POST body:
    {
        "species": "macaque",
        "atlas": "D99",
        "query": "V1",
        "query_type": "name",  // name, abbreviation, id, coordinate
        "return_neighbors": false,
        "return_hierarchy": false
    }
    """
    try:
        body = await request.json()
        species = body.get("species", "macaque")
        atlas_name = body.get("atlas")
        query = body.get("query")
        query_type = body.get("query_type", "auto")
        return_neighbors = body.get("return_neighbors", False)
        return_hierarchy = body.get("return_hierarchy", False)
        
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)
        
        # Handle coordinate lookup
        if query_type == "coordinate" or query.startswith("coord:"):
            coord_str = query.replace("coord:", "").strip()
            try:
                x, y, z = [float(c.strip()) for c in coord_str.split(",")]
                atlas = get_atlas(species, atlas_name)
                region = atlas.get_region_at_coordinate(x, y, z)
            except Exception as e:
                return JSONResponse({"error": f"Invalid coordinate: {e}"}, status_code=400)
        else:
            atlas = get_atlas(species, atlas_name)
            region = atlas.get_region(query, query_type)
        
        if not region:
            return JSONResponse({
                "found": False,
                "query": query,
                "message": f"Region not found in {species}/{atlas_name or 'default'}"
            })
        
        result = {
            "found": True,
            "region": region.to_dict(),
            "atlas": {
                "name": atlas.name,
                "version": atlas.version,
                "species": atlas.species,
            }
        }
        
        if return_neighbors:
            neighbors = atlas.get_neighbors(region.id)
            result["neighbors"] = [n.to_dict() for n in neighbors[:10]]  # Limit to 10
        
        if return_hierarchy:
            result["hierarchy"] = atlas.get_hierarchy(region.id)
        
        return JSONResponse(result)
        
    except NotImplementedError as e:
        return JSONResponse({"error": str(e)}, status_code=501)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def list_regions(request):
    """List regions with optional filtering.
    
    POST body:
    {
        "species": "macaque",
        "atlas": "D99",
        "filter_type": "cortical",
        "filter_system": "visual",
        "search_pattern": "V[0-9]"
    }
    """
    try:
        body = await request.json()
        species = body.get("species", "macaque")
        atlas_name = body.get("atlas")
        filter_type = body.get("filter_type")
        search_pattern = body.get("search_pattern")
        
        atlas = get_atlas(species, atlas_name)
        regions = atlas.list_regions(
            region_type=filter_type,
            search_pattern=search_pattern
        )
        
        # Filter by functional system if requested
        filter_system = body.get("filter_system")
        if filter_system:
            regions = [r for r in regions if filter_system in r.functional_systems]
        
        return JSONResponse({
            "atlas": atlas.name,
            "species": atlas.species,
            "total_regions": len(regions),
            "regions": [r.to_dict() for r in regions]
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def generate_mask(request):
    """Generate NIfTI ROI mask.
    
    POST body:
    {
        "species": "macaque",
        "atlas": "D99",
        "regions": ["V1", "V2"],
        "output_path": "/path/to/mask.nii.gz",
        "dilate_mm": 0,
        "hemisphere": "both"
    }
    """
    try:
        body = await request.json()
        species = body.get("species", "macaque")
        atlas_name = body.get("atlas")
        region_names = body.get("regions", [])
        output_path = body.get("output_path")
        dilate_mm = body.get("dilate_mm", 0)
        hemisphere = body.get("hemisphere", "both")
        
        if not region_names:
            return JSONResponse({"error": "regions list required"}, status_code=400)
        if not output_path:
            return JSONResponse({"error": "output_path required"}, status_code=400)
        
        atlas = get_atlas(species, atlas_name)
        
        # Resolve region names to IDs
        region_ids = []
        resolved = []
        for name in region_names:
            region = atlas.get_region(name, "auto")
            if region:
                region_ids.append(region.id)
                resolved.append({"name": name, "id": region.id})
        
        if not region_ids:
            return JSONResponse({"error": "No valid regions found"}, status_code=400)
        
        # Generate mask
        mask_path = atlas.generate_mask(
            region_ids=region_ids,
            output_path=output_path,
            dilate_mm=dilate_mm,
            hemisphere=hemisphere
        )
        
        return JSONResponse({
            "mask_path": mask_path,
            "regions_included": resolved,
            "total_regions": len(resolved)
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_neighbors(request):
    """Get neighboring regions.
    
    POST body:
    {
        "species": "macaque",
        "atlas": "D99",
        "region": "V1"
    }
    """
    try:
        body = await request.json()
        species = body.get("species", "macaque")
        atlas_name = body.get("atlas")
        region_query = body.get("region")
        
        if not region_query:
            return JSONResponse({"error": "region required"}, status_code=400)
        
        atlas = get_atlas(species, atlas_name)
        region = atlas.get_region(region_query, "auto")
        
        if not region:
            return JSONResponse({"error": f"Region not found: {region_query}"}, status_code=404)
        
        neighbors = atlas.get_neighbors(region.id)
        
        return JSONResponse({
            "region": region.to_dict(),
            "neighbors": [n.to_dict() for n in neighbors],
            "neighbor_count": len(neighbors)
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_atlas_info(request):
    """Get detailed atlas information.
    
    POST body:
    {
        "species": "macaque",
        "atlas": "D99"
    }
    """
    try:
        body = await request.json()
        species = body.get("species", "macaque")
        atlas_name = body.get("atlas")
        
        atlas = get_atlas(species, atlas_name)
        info = atlas.info()
        
        return JSONResponse(info)
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Application Setup
# =============================================================================

routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/atlases", list_atlases, methods=["GET"]),
    Route("/api/atlas_info", get_atlas_info, methods=["POST"]),
    Route("/api/lookup_region", lookup_region, methods=["POST"]),
    Route("/api/list_regions", list_regions, methods=["POST"]),
    Route("/api/generate_mask", generate_mask, methods=["POST"]),
    Route("/api/get_neighbors", get_neighbors, methods=["POST"]),
]

app = Starlette(routes=routes)


def main():
    parser = argparse.ArgumentParser(description="Brain Atlas MCP HTTP Server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("BRAIN_ATLAS_PORT", 8007)),
        help="Port to run server on (default: 8007)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    # Check atlas availability on startup
    check_atlas_availability()
    
    installed = []
    for species, atlases in AVAILABLE_ATLASES.items():
        for atlas_name, info in atlases.items():
            status = "✅" if info["installed"] else "❌"
            installed.append(f"  {status} {species}/{atlas_name}: {info['name']}")
    
    print(f"🧠 Brain Atlas MCP HTTP Server starting on {args.host}:{args.port}")
    print(f"\nAvailable atlases:")
    print("\n".join(installed))
    print(f"\nEndpoints:")
    print(f"  Health:        GET  http://{args.host}:{args.port}/health")
    print(f"  List atlases:  GET  http://{args.host}:{args.port}/api/atlases")
    print(f"  Lookup region: POST http://{args.host}:{args.port}/api/lookup_region")
    print(f"  List regions:  POST http://{args.host}:{args.port}/api/list_regions")
    print(f"  Generate mask: POST http://{args.host}:{args.port}/api/generate_mask")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
