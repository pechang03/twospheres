# Export Tools Design

**Version**: 0.1.0  
**Date**: 2026-01-21  
**Status**: Draft

## Purpose

Provide standardized export formats for twosphere-mcp models to integrate with:
- FreeSurfer / FSL / ANTs neuroimaging pipelines
- ParaView / 3D Slicer visualization
- Connectome Workbench (HCP tools)
- Custom graph analysis tools

## Export Formats

### 1. NIfTI Export (`export_nifti.py`)

**Use case**: Volumetric data for FSL/SPM/ANTs

```python
class NIfTIExporter:
    """Export volumetric representations to NIfTI format."""
    
    def export_water_content_map(
        self,
        water_field: np.ndarray,  # 4D: (x, y, z, t)
        affine: np.ndarray,       # 4x4 voxel-to-world
        output_path: str,
        compress: bool = True
    ) -> str:
        """Export time-series water content to NIfTI-1."""
        
    def export_sulcal_depth_map(
        self,
        depth_field: np.ndarray,  # 3D
        affine: np.ndarray,
        output_path: str
    ) -> str:
        """Export signed distance to surface (+ in sulci)."""
        
    def export_pvs_mask(
        self,
        pvs_binary: np.ndarray,   # 3D binary
        affine: np.ndarray,
        output_path: str
    ) -> str:
        """Export perivascular space segmentation."""
```

**MCP Tool**:
```json
{
  "name": "export_to_nifti",
  "description": "Export volumetric brain data to NIfTI format",
  "inputSchema": {
    "type": "object",
    "properties": {
      "data_type": {"enum": ["water_content", "sulcal_depth", "pvs_mask", "flow_magnitude"]},
      "output_path": {"type": "string"},
      "time_range": {"type": "array", "items": {"type": "number"}, "description": "[start_s, end_s]"}
    }
  }
}
```

### 2. GIFTI Export (`export_gifti.py`)

**Use case**: Surface data for FreeSurfer/Connectome Workbench

```python
class GIFTIExporter:
    """Export surface meshes and overlays to GIFTI format."""
    
    def export_cortical_surface(
        self,
        vertices: np.ndarray,     # (N, 3)
        faces: np.ndarray,        # (M, 3)
        output_path: str,
        surface_type: str = "pial"  # "pial", "white", "midthickness"
    ) -> str:
        """Export triangulated surface mesh."""
        
    def export_scalar_overlay(
        self,
        values: np.ndarray,       # (N,) per-vertex
        output_path: str,
        intent: str = "NIFTI_INTENT_SHAPE"
    ) -> str:
        """Export per-vertex scalar data (curvature, flow, etc.)."""
        
    def export_fractal_surface(
        self,
        model: TwoSphereModel,
        epsilon: float,
        julia_c: complex,
        resolution: int,
        output_dir: str
    ) -> dict:
        """Export both white and pial surfaces with correspondence."""
        return {
            "white": "lh.white.gii",
            "pial": "lh.pial.gii",
            "sulc": "lh.sulc.shape.gii",
            "fractal_f": "lh.fractal.shape.gii"
        }
```

**MCP Tool**:
```json
{
  "name": "export_to_gifti",
  "description": "Export cortical surfaces to GIFTI for FreeSurfer/Workbench",
  "inputSchema": {
    "type": "object",
    "properties": {
      "surface_type": {"enum": ["white", "pial", "midthickness", "inflated"]},
      "epsilon": {"type": "number", "description": "Fractal perturbation amplitude"},
      "julia_c_real": {"type": "number"},
      "julia_c_imag": {"type": "number"},
      "hemisphere": {"enum": ["left", "right", "both"]}
    }
  }
}
```

### 3. VTK Export (`export_vtk.py`)

**Use case**: ParaView / 3D Slicer / PyVista visualization

```python
class VTKExporter:
    """Export meshes with field data to VTK formats."""
    
    def export_polydata(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        point_data: dict[str, np.ndarray],  # {"flow": ..., "pressure": ...}
        cell_data: dict[str, np.ndarray],
        output_path: str,
        file_format: str = "vtp"  # "vtp" (XML), "vtk" (legacy)
    ) -> str:
        """Export surface with attached field data."""
        
    def export_unstructured_grid(
        self,
        vertices: np.ndarray,
        cells: np.ndarray,        # tetrahedral mesh
        cell_types: np.ndarray,
        point_data: dict[str, np.ndarray],
        output_path: str
    ) -> str:
        """Export volumetric mesh (for FEM solvers)."""
        
    def export_time_series(
        self,
        base_mesh: pv.PolyData,
        time_data: list[dict[str, np.ndarray]],  # list of point_data per timestep
        output_dir: str,
        prefix: str = "glymphatic"
    ) -> list[str]:
        """Export PVD collection for time-series animation."""
```

**MCP Tool**:
```json
{
  "name": "export_to_vtk",
  "description": "Export mesh with flow/pressure fields for ParaView visualization",
  "inputSchema": {
    "type": "object",
    "properties": {
      "fields": {"type": "array", "items": {"enum": ["flow", "pressure", "delta", "metabolic_demand"]}},
      "format": {"enum": ["vtp", "vtk", "pvd"]},
      "time_steps": {"type": "integer", "description": "Number of timesteps for animation"}
    }
  }
}
```

### 4. Graph Export (`export_graph.py`)

**Use case**: Network analysis in NetworkX / igraph / Gephi

```python
class GraphExporter:
    """Export neural and glymphatic graphs."""
    
    def export_graphml(
        self,
        graph: nx.Graph,
        output_path: str,
        node_attrs: list[str] = ["position", "metabolic_demand"],
        edge_attrs: list[str] = ["conductance", "flow"]
    ) -> str:
        """Export to GraphML with full attributes."""
        
    def export_gexf(
        self,
        graph: nx.Graph,
        output_path: str,
        dynamic: bool = False,
        time_slices: list[float] = None
    ) -> str:
        """Export to GEXF for Gephi (supports dynamics)."""
        
    def export_adjacency_csv(
        self,
        graph: nx.Graph,
        output_dir: str
    ) -> dict:
        """Export nodes.csv and edges.csv for simple tools."""
        return {
            "nodes": "nodes.csv",
            "edges": "edges.csv"
        }
        
    def export_connectome(
        self,
        neural_graph: nx.Graph,
        pvs_graph: nx.Graph,
        output_path: str
    ) -> str:
        """Export combined connectome in CIFTI-like format."""
```

**MCP Tool**:
```json
{
  "name": "export_graph",
  "description": "Export neural/glymphatic networks to graph formats",
  "inputSchema": {
    "type": "object",
    "properties": {
      "graph_type": {"enum": ["neural", "glymphatic", "combined"]},
      "format": {"enum": ["graphml", "gexf", "csv", "json"]},
      "include_positions": {"type": "boolean"},
      "include_flow_state": {"type": "boolean"}
    }
  }
}
```

## Combined Export Pipeline

```python
class TwoSphereExporter:
    """Unified export interface for all formats."""
    
    def __init__(self, model: TwoSphereModel):
        self.model = model
        self.nifti = NIfTIExporter()
        self.gifti = GIFTIExporter()
        self.vtk = VTKExporter()
        self.graph = GraphExporter()
        
    def export_full_model(
        self,
        output_dir: str,
        formats: list[str] = ["gifti", "vtk", "graphml"],
        include_simulation: bool = True
    ) -> dict:
        """Export complete model state in multiple formats."""
        
    def export_for_freesurfer(self, output_dir: str) -> dict:
        """Export in FreeSurfer-compatible directory structure."""
        
    def export_for_workbench(self, output_dir: str) -> dict:
        """Export CIFTI-2 compatible files for Connectome Workbench."""
```

## File Naming Convention

```
{subject_id}/
├── surf/
│   ├── lh.white.gii
│   ├── lh.pial.gii
│   ├── rh.white.gii
│   └── rh.pial.gii
├── func/
│   ├── water_content.nii.gz
│   ├── flow_magnitude.nii.gz
│   └── pvs_mask.nii.gz
├── graph/
│   ├── neural_connectome.graphml
│   └── glymphatic_network.graphml
└── viz/
    ├── glymphatic_t0000.vtp
    ├── glymphatic_t0001.vtp
    └── glymphatic.pvd
```

## Integration Points

| External Tool | Format | Notes |
|---------------|--------|-------|
| FreeSurfer | GIFTI + NIfTI | Use `mris_convert` for legacy formats |
| FSL | NIfTI | Ensure RAS orientation |
| ANTs | NIfTI | Include affine in header |
| Connectome Workbench | GIFTI + CIFTI | Pair surfaces with `wb_command` |
| ParaView | VTK/PVD | Time series via PVD collection |
| 3D Slicer | VTK + NIfTI | Load as model + volume |
| Gephi | GEXF | Enable dynamics for flow animation |
| Cytoscape | GraphML | Requires position attributes |
