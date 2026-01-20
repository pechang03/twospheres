# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

TwoSphere MCP Server provides optical physics simulations and MRI spherical geometry analysis, integrating pyoptools with the MRISpheres/twospheres research project.

## Environment Setup

```bash
# Create and activate conda environment
conda create -n twosphere python=3.11
conda activate twosphere

# Install dependencies
pip install -r requirements.txt

# Link MRISpheres research data (optional)
ln -s ~/MRISpheres/twospheres data/twospheres

# Copy environment file
cp .env_twosphere.example .env_twosphere
```

Or use the setup script: `./setup.sh`

## Running the Server

```bash
# MCP server (stdio mode for LLM integration)
python bin/twosphere_mcp.py

# HTTP REST API server
python bin/twosphere_http_server.py --port 8006
```

## Testing

```bash
pytest tests/
pytest tests/test_specific.py::test_function -v
```

## Architecture

### Entry Points (`bin/`)
- `twosphere_mcp.py` - MCP stdio server; defines tools via `@server.list_tools()` and handlers via `@server.call_tool()`
- `twosphere_http_server.py` - Starlette HTTP wrapper that imports handlers from the MCP server

### Core Modules (`src/`)

**MRI Analysis (`src/mri_analysis/`)**
- `two_sphere.py` - `TwoSphereModel` class for paired brain region geometry (mesh generation, distance/overlap calculation)
- `vortex_ring.py` - `VortexRing` class for trefoil knot neural pathway modeling using Frenet-Serret frames
- `fft_correlation.py` - FFT-based signal correlation (`compute_fft_correlation`, `coherence`, `phase_correlation`)

**Optical Simulations (`src/simulations/`)**
- `ray_tracing.py` - `RayTracer` class wrapping pyoptools (gracefully degrades if pyoptools unavailable)
- `wavefront.py` - `WavefrontAnalyzer` using Zernike polynomials for aberration analysis

### MCP Tools Exposed
1. `two_sphere_model` - Create/visualize paired brain region spheres
2. `vortex_ring` - Generate trefoil knot vortex structures
3. `fft_correlation` - Frequency-domain signal correlation
4. `ray_trace` - Optical ray tracing
5. `wavefront_analysis` - Zernike wavefront analysis
6. `list_twosphere_files` - List MRISpheres research files

## Configuration

Environment variables in `.env_twosphere`:
- `TWOSPHERES_PATH` - Path to MRISpheres research data (default: `~/MRISpheres/twospheres`)
- `TWOSPHERE_PORT` - HTTP server port (default: 8006)
- `OPENAI_API_KEY` - Optional for advanced analysis

## Ecosystem Integration

This project is part of the merge2docs ecosystem, providing physics simulation expertise:
- **Task tracking**: Uses `../merge2docs/yada-work.db` with PHY task-ids (e.g., `PHY.1.1`, `PHY.2.3`)
- **Similar to**: `../biosearch/` which provides bioinformatics expertise
- **Connects with**: `../merge2docs/` for documentation and knowledge graph integration

See `AGENTS.md` for task management workflow.

## Key Dependencies
- `pyoptools` >= 0.3.7 - Optical simulations (optional, graceful fallback)
- `mcp` >= 1.0.0 - Model Context Protocol
- `scipy` - Spline interpolation and signal processing
