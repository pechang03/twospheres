# MCP Tools Enhancement Design

**Version**: 0.1.0  
**Date**: 2026-01-21  
**Status**: Draft

## Purpose

Define new MCP tools that would improve the quality of answers to neurology/physics research questions like those in this design phase. These tools fill gaps identified during the glymphatic modeling research.

## Gap Analysis

During the fractal cortex + glymphatic synthesis, ernie2_swarm queries would have benefited from:

| Gap | Current State | Proposed Solution |
|-----|---------------|-------------------|
| No PDE solver access | Text-only equations | `solve_pde_on_surface` tool |
| No mesh generation | Describe algorithms only | `generate_icosphere_mesh` tool |
| No MRI preprocessing | Manual FSL/ANTs | `preprocess_4d_mri` tool |
| No graph-on-surface ops | NetworkX only | `embed_graph_on_manifold` tool |
| No phase analysis | Describe Hilbert transform | `extract_phase_coherence` tool |
| No literature integration | Manual citations | `search_pubmed_neurology` tool |

## Proposed New Tools

### 1. Geometry Tools

#### `generate_fractal_surface`
```json
{
  "name": "generate_fractal_surface",
  "description": "Generate a fractal-perturbed sphere using Julia sets or L-systems",
  "inputSchema": {
    "type": "object",
    "properties": {
      "method": {"enum": ["julia", "mandelbrot", "lsystem", "perlin"]},
      "epsilon": {"type": "number", "description": "Perturbation amplitude (0.01-0.2)"},
      "julia_c_real": {"type": "number", "default": -0.7},
      "julia_c_imag": {"type": "number", "default": 0.27},
      "resolution": {"type": "integer", "default": 100},
      "compute_safety_bound": {"type": "boolean", "default": true}
    },
    "required": ["method", "epsilon"]
  },
  "returns": {
    "vertices": "array of [x,y,z]",
    "faces": "array of triangle indices",
    "f_values": "fractal displacement per vertex",
    "epsilon_max": "maximum safe epsilon before self-intersection",
    "fractal_dimension": "estimated D"
  }
}
```

#### `embed_graph_on_manifold`
```json
{
  "name": "embed_graph_on_manifold",
  "description": "Embed a graph onto a surface mesh with geodesic edge weights",
  "inputSchema": {
    "type": "object",
    "properties": {
      "graph_type": {"enum": ["neural_pathways", "vascular_tree", "pvs_network"]},
      "surface_id": {"type": "string", "description": "Reference to previously generated surface"},
      "seed_points": {"type": "array", "items": {"type": "array"}, "description": "[[θ,φ], ...] for node placement"},
      "connectivity": {"enum": ["delaunay", "knn", "arterial_tree", "custom"]},
      "k": {"type": "integer", "default": 6, "description": "k for knn connectivity"}
    }
  },
  "returns": {
    "nodes": "array with positions and properties",
    "edges": "array with geodesic lengths and conductances",
    "graph_id": "reference for later use"
  }
}
```

### 2. Simulation Tools

#### `solve_shallow_water_on_surface`
```json
{
  "name": "solve_shallow_water_on_surface",
  "description": "Solve shallow-water equations on a curved surface with sources",
  "inputSchema": {
    "type": "object",
    "properties": {
      "surface_id": {"type": "string"},
      "initial_h": {"type": "string", "description": "Expression or 'uniform:value'"},
      "source_graph_id": {"type": "string", "description": "PVS network providing S_v"},
      "mu": {"type": "number", "default": 0.001, "description": "Viscosity Pa·s"},
      "dt": {"type": "number", "default": 0.0005, "description": "Time step (s)"},
      "t_end": {"type": "number", "default": 1.0},
      "output_interval": {"type": "number", "default": 0.1}
    }
  },
  "returns": {
    "time_series": "array of {t, h_field, u_field}",
    "total_flux": "integrated flow through boundaries",
    "steady_state_reached": "boolean"
  }
}
```

#### `simulate_sleep_transition`
```json
{
  "name": "simulate_sleep_transition",
  "description": "Simulate awake→sleep transition with δ(t) modulation",
  "inputSchema": {
    "type": "object", 
    "properties": {
      "model_id": {"type": "string"},
      "delta_awake": {"type": "number", "default": 7e-6, "description": "Gap in meters"},
      "delta_sleep": {"type": "number", "default": 9e-6},
      "transition_tau": {"type": "number", "default": 180, "description": "Time constant (s)"},
      "simulate_duration": {"type": "number", "default": 600}
    }
  },
  "returns": {
    "clearance_rate_vs_time": "array",
    "interstitial_volume_vs_time": "array",
    "steady_state_ratio": "sleep/awake clearance"
  }
}
```

### 3. MRI Analysis Tools

#### `preprocess_4d_mri`
```json
{
  "name": "preprocess_4d_mri",
  "description": "Preprocess 4D MRI water content data for glymphatic analysis",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string"},
      "steps": {
        "type": "array",
        "items": {"enum": ["motion_correct", "b0_unwarp", "bias_correct", "register_to_template"]},
        "default": ["motion_correct", "bias_correct"]
      },
      "template": {"type": "string", "description": "Path to template if registering"},
      "output_dir": {"type": "string"}
    }
  },
  "returns": {
    "preprocessed_nifti": "path",
    "motion_parameters": "path to .par file",
    "qc_report": "path to HTML"
  }
}
```

#### `extract_phase_coherence`
```json
{
  "name": "extract_phase_coherence",
  "description": "Extract phase relationships between brain regions in frequency bands",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string"},
      "roi_arterial": {"type": "string", "description": "NIfTI mask for arterial region"},
      "roi_venous": {"type": "string", "description": "NIfTI mask for venous region"},
      "bands": {
        "type": "object",
        "default": {"cardiac": [0.5, 2.0], "respiratory": [0.1, 0.3], "ultraslow": [0.001, 0.01]}
      },
      "compute_directionality": {"type": "boolean", "default": true}
    }
  },
  "returns": {
    "phase_difference_by_band": {"cardiac": "Δφ", "respiratory": "Δφ", "ultraslow": "Δφ"},
    "coherence_by_band": {"cardiac": "C", ...},
    "flow_direction_indicator": "+1 (inward) or -1 (outward)"
  }
}
```

#### `extract_pvs_from_mri`
```json
{
  "name": "extract_pvs_from_mri",
  "description": "Segment perivascular spaces from T2-weighted or water content MRI",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string"},
      "method": {"enum": ["frangi", "threshold", "unet"], "default": "frangi"},
      "frangi_scales": {"type": "array", "default": [0.2, 0.4, 0.6]},
      "vessel_mask": {"type": "string", "description": "Optional: exclude known vessels"}
    }
  },
  "returns": {
    "pvs_mask": "path to binary NIfTI",
    "pvs_volume_mm3": "total volume",
    "pvs_centerlines": "path to VTK polydata",
    "estimated_delta_um": "median gap width"
  }
}
```

### 4. Literature & Knowledge Tools

#### `search_glymphatic_literature`
```json
{
  "name": "search_glymphatic_literature",
  "description": "Search PubMed/bioRxiv for glymphatic and perivascular space research",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "date_range": {"type": "string", "default": "2020-2026"},
      "max_results": {"type": "integer", "default": 20},
      "include_preprints": {"type": "boolean", "default": true},
      "focus": {"enum": ["methods", "clinical", "basic_science", "computational"]}
    }
  },
  "returns": {
    "papers": "array of {title, authors, doi, abstract_snippet, relevance_score}",
    "key_findings": "synthesized bullet points from top 5"
  }
}
```

#### `lookup_brain_atlas`
```json
{
  "name": "lookup_brain_atlas",
  "description": "Query brain atlases for region coordinates and connectivity",
  "inputSchema": {
    "type": "object",
    "properties": {
      "atlas": {"enum": ["AAL", "Desikan-Killiany", "HCP-MMP1", "Julich", "primate_D99"]},
      "query": {"type": "string", "description": "Region name or coordinate"},
      "return_neighbors": {"type": "boolean", "default": false},
      "return_connectivity": {"type": "boolean", "default": false}
    }
  },
  "returns": {
    "region": "matched region info",
    "coordinates_mni": "[x, y, z]",
    "volume_mm3": "region volume",
    "neighbors": "if requested",
    "structural_connectivity": "if requested, list of connected regions"
  }
}
```

### 5. Export/Integration Tools

*(See `03_export_tools.md` for full specs)*

- `export_to_nifti`
- `export_to_gifti`  
- `export_to_vtk`
- `export_graph`

## Tool Synergies

These tools work together in pipelines:

```
┌──────────────────────────────────────────────────────────────────┐
│ Research Query: "Model glymphatic flow in primate with 4D MRI"  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│ Literature    │   │ Geometry      │   │ MRI Analysis      │
│ search        │   │ generation    │   │ preprocessing     │
└───────┬───────┘   └───────┬───────┘   └─────────┬─────────┘
        │                   │                     │
        │           ┌───────┴───────┐             │
        │           ▼               ▼             │
        │   ┌───────────┐   ┌───────────┐         │
        │   │ fractal   │   │ embed     │         │
        │   │ surface   │   │ graph     │         │
        │   └─────┬─────┘   └─────┬─────┘         │
        │         │               │               │
        │         └───────┬───────┘               │
        │                 ▼                       ▼
        │         ┌───────────────┐       ┌───────────────┐
        │         │ shallow-water │       │ extract_phase │
        │         │ solver        │◄──────┤ coherence     │
        │         └───────┬───────┘       └───────────────┘
        │                 │
        │                 ▼
        │         ┌───────────────┐
        └────────►│ synthesize    │
                  │ answer        │
                  └───────┬───────┘
                          ▼
                  ┌───────────────┐
                  │ export to     │
                  │ VTK/GIFTI     │
                  └───────────────┘
```

## Implementation Priority

| Priority | Tool | Rationale |
|----------|------|-----------|
| P0 | `generate_fractal_surface` | Core geometry for all models |
| P0 | `export_to_gifti` | Enable FreeSurfer integration |
| P1 | `embed_graph_on_manifold` | Required for G + G_pvs |
| P1 | `extract_pvs_from_mri` | Validate δ parameter |
| P1 | `extract_phase_coherence` | Validate flow direction |
| P2 | `solve_shallow_water_on_surface` | Full simulation capability |
| P2 | `preprocess_4d_mri` | Streamline data pipeline |
| P3 | `simulate_sleep_transition` | Advanced dynamics |
| P3 | `search_glymphatic_literature` | Knowledge augmentation |

## Resource Requirements

| Tool Category | Compute | Memory | Dependencies |
|---------------|---------|--------|--------------|
| Geometry | CPU | 1-4 GB | numpy, numba |
| Simulation | CPU/GPU | 4-16 GB | scipy, petsc (optional) |
| MRI Analysis | CPU | 8-32 GB | nibabel, FSL, ANTs |
| Literature | Network | <1 GB | requests, biopython |
