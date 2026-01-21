# Brain Atlas Tools Specification

## Overview

MCP interface for querying brain atlases across species (macaque, rodent). Enables merge2docs/ernie2_swarm to:
- Look up region coordinates and labels
- Generate ROI masks for glymphatic analysis
- Query structural connectivity
- Support cross-species comparisons

## Supported Atlases

### Macaque
| Atlas | Regions | Description |
|-------|---------|-------------|
| **D99 v2.0** | 368 (149 cortical + 219 subcortical) | MRI-histology based, Saleem & Logothetis |
| CHARM | 6 levels hierarchy | Hierarchical cortical parcellation |
| SARM | Subcortical | Subcortical segmentation |

### Rodent (Mouse/Rat)
| Atlas | Regions | Description |
|-------|---------|-------------|
| **Allen CCF v3** | 800+ | Allen Mouse Brain Common Coordinate Framework |
| **Waxholm Rat** | 222 | Sprague-Dawley rat atlas |
| Paxinos-Watson | ~600 | Classic rat stereotaxic atlas |

---

## Tool 1: `lookup_brain_region`

### Purpose
Query brain atlas for region information by name, abbreviation, or coordinate.

### MCP Schema
```json
{
  "name": "lookup_brain_region",
  "description": "Look up brain region information from species-specific atlases. Returns coordinates, hierarchy, and neighboring regions.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "species": {
        "type": "string",
        "enum": ["macaque", "mouse", "rat"],
        "description": "Target species"
      },
      "atlas": {
        "type": "string",
        "enum": ["D99", "CHARM", "Allen_CCF", "Waxholm", "Paxinos"],
        "default": "D99",
        "description": "Atlas to query"
      },
      "query": {
        "type": "string",
        "description": "Region name, abbreviation, or 'coord:x,y,z' for coordinate lookup"
      },
      "query_type": {
        "type": "string",
        "enum": ["name", "abbreviation", "coordinate", "parent", "children"],
        "default": "name"
      },
      "return_neighbors": {
        "type": "boolean",
        "default": false,
        "description": "Include neighboring regions"
      },
      "return_hierarchy": {
        "type": "boolean",
        "default": false,
        "description": "Include parent/child hierarchy"
      },
      "coordinate_space": {
        "type": "string",
        "enum": ["native", "MNI", "stereotaxic"],
        "default": "native"
      }
    },
    "required": ["species", "query"]
  }
}
```

### Response Schema
```json
{
  "region": {
    "id": "integer",
    "name": "string",
    "abbreviation": "string",
    "hemisphere": "left|right|bilateral",
    "type": "cortical|subcortical|cerebellar|white_matter"
  },
  "coordinates": {
    "centroid": [x, y, z],
    "bounding_box": [[x_min, y_min, z_min], [x_max, y_max, z_max]],
    "space": "native|MNI|stereotaxic"
  },
  "volume_mm3": "float",
  "hierarchy": {
    "parent": {"id": "int", "name": "string"},
    "children": [{"id": "int", "name": "string"}, ...],
    "level": "int (0=whole brain)"
  },
  "neighbors": [
    {"id": "int", "name": "string", "contact_area_mm2": "float"}
  ],
  "functional_systems": ["visual", "motor", "limbic", ...],
  "atlas_info": {
    "atlas": "D99|Allen_CCF|...",
    "version": "string",
    "citation": "string"
  }
}
```

---

## Tool 2: `generate_roi_mask`

### Purpose
Generate NIfTI ROI mask for specified regions.

### MCP Schema
```json
{
  "name": "generate_roi_mask",
  "description": "Generate NIfTI ROI mask for one or more brain regions. Useful for extracting time series from glymphatic analysis.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "species": {"type": "string", "enum": ["macaque", "mouse", "rat"]},
      "atlas": {"type": "string", "default": "D99"},
      "regions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of region names or abbreviations"
      },
      "combine_mode": {
        "type": "string",
        "enum": ["separate", "union", "labeled"],
        "default": "union",
        "description": "How to combine multiple regions"
      },
      "dilate_mm": {
        "type": "number",
        "default": 0,
        "description": "Dilate mask by N mm"
      },
      "hemisphere": {
        "type": "string",
        "enum": ["left", "right", "both"],
        "default": "both"
      },
      "output_path": {"type": "string"},
      "reference_volume": {
        "type": "string",
        "description": "Optional: resample to match this volume's grid"
      }
    },
    "required": ["species", "regions", "output_path"]
  }
}
```

### Response Schema
```json
{
  "mask_path": "string",
  "regions_included": [
    {"name": "string", "id": "int", "voxel_count": "int"}
  ],
  "total_volume_mm3": "float",
  "dimensions": [x, y, z],
  "voxel_size_mm": [dx, dy, dz]
}
```

---

## Tool 3: `get_vascular_rois`

### Purpose
Get predefined ROIs for glymphatic analysis (arterial, venous territories).

### MCP Schema
```json
{
  "name": "get_vascular_rois",
  "description": "Get predefined vascular ROIs for glymphatic flow analysis. Returns masks for arterial (MCA, ACA, PCA territories) and venous (SSS, transverse sinus) regions.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "species": {"type": "string", "enum": ["macaque", "mouse", "rat"]},
      "roi_type": {
        "type": "string",
        "enum": ["arterial", "venous", "perivascular", "csf", "all"],
        "default": "all"
      },
      "arterial_territory": {
        "type": "string",
        "enum": ["MCA", "ACA", "PCA", "all"],
        "default": "MCA",
        "description": "For arterial ROI: which territory"
      },
      "depth_from_surface_mm": {
        "type": "number",
        "default": 2.0,
        "description": "For perivascular: distance from pial surface"
      },
      "output_dir": {"type": "string"}
    },
    "required": ["species", "output_dir"]
  }
}
```

### Response Schema
```json
{
  "rois": {
    "arterial": {"path": "string", "volume_mm3": "float", "regions": ["list"]},
    "venous": {"path": "string", "volume_mm3": "float", "regions": ["list"]},
    "perivascular": {"path": "string", "volume_mm3": "float"},
    "csf": {"path": "string", "volume_mm3": "float"}
  },
  "suggested_use": {
    "phase_coherence": "Use arterial + venous for extract_phase_coherence tool",
    "pvs_analysis": "Use perivascular for extract_pvs_from_mri tool"
  }
}
```

---

## Tool 4: `query_connectivity`

### Purpose
Query structural/functional connectivity between regions.

### MCP Schema
```json
{
  "name": "query_connectivity",
  "description": "Query known structural connectivity between brain regions from tract-tracing or diffusion MRI databases.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "species": {"type": "string", "enum": ["macaque", "mouse", "rat"]},
      "source_region": {"type": "string"},
      "target_region": {"type": "string", "description": "Optional: if omitted, return all targets"},
      "connectivity_type": {
        "type": "string",
        "enum": ["structural", "functional", "both"],
        "default": "structural"
      },
      "data_source": {
        "type": "string",
        "enum": ["CoCoMac", "Allen_connectivity", "tracer_studies", "diffusion"],
        "default": "CoCoMac"
      },
      "min_strength": {
        "type": "number",
        "default": 0,
        "description": "Minimum connection strength (0-1)"
      }
    },
    "required": ["species", "source_region"]
  }
}
```

### Response Schema
```json
{
  "source": {"name": "string", "id": "int"},
  "connections": [
    {
      "target": {"name": "string", "id": "int"},
      "strength": "float 0-1",
      "direction": "afferent|efferent|bidirectional",
      "pathway": "string (if known)",
      "evidence": "tracer|diffusion|functional",
      "citations": ["doi:..."]
    }
  ],
  "total_connections": "int",
  "data_source": "string"
}
```

---

## Tool 5: `list_atlas_regions`

### Purpose
List all regions in an atlas with filtering.

### MCP Schema
```json
{
  "name": "list_atlas_regions",
  "description": "List all regions in a brain atlas with optional filtering by type, hierarchy level, or functional system.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "species": {"type": "string", "enum": ["macaque", "mouse", "rat"]},
      "atlas": {"type": "string", "default": "D99"},
      "filter_type": {
        "type": "string",
        "enum": ["all", "cortical", "subcortical", "cerebellar", "white_matter"],
        "default": "all"
      },
      "filter_system": {
        "type": "string",
        "enum": ["all", "visual", "auditory", "motor", "somatosensory", "limbic", "prefrontal"],
        "default": "all"
      },
      "hierarchy_level": {
        "type": "integer",
        "description": "0=whole brain, higher=finer parcellation"
      },
      "search_pattern": {
        "type": "string",
        "description": "Regex pattern to filter region names"
      }
    },
    "required": ["species"]
  }
}
```

### Response Schema
```json
{
  "atlas": "string",
  "total_regions": "int",
  "filtered_count": "int",
  "regions": [
    {
      "id": "int",
      "name": "string",
      "abbreviation": "string",
      "type": "cortical|subcortical|...",
      "hierarchy_level": "int",
      "parent_id": "int|null"
    }
  ]
}
```

---

## Implementation

### File Structure
```
src/
├── atlases/
│   ├── __init__.py
│   ├── base_atlas.py        # Abstract base class
│   ├── d99_atlas.py         # Macaque D99 v2.0
│   ├── allen_ccf_atlas.py   # Mouse Allen CCF
│   ├── waxholm_atlas.py     # Rat Waxholm
│   └── vascular_rois.py     # Predefined vascular territories
├── mri_analysis/
│   └── atlas_integration.py # Integration with MRI tools
```

### Base Class
```python
# src/atlases/base_atlas.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class BrainRegion:
    id: int
    name: str
    abbreviation: str
    hemisphere: str  # 'left', 'right', 'bilateral'
    region_type: str  # 'cortical', 'subcortical', etc.
    parent_id: Optional[int] = None
    centroid: Optional[Tuple[float, float, float]] = None
    volume_mm3: Optional[float] = None


class BrainAtlas(ABC):
    """Abstract base class for brain atlases."""
    
    @property
    @abstractmethod
    def species(self) -> str:
        """Return species name."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return atlas name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Return atlas version."""
        pass
    
    @abstractmethod
    def get_region(self, query: str, query_type: str = "name") -> Optional[BrainRegion]:
        """Look up a region by name, abbreviation, or ID."""
        pass
    
    @abstractmethod
    def get_region_at_coordinate(self, x: float, y: float, z: float) -> Optional[BrainRegion]:
        """Get region at specified coordinate."""
        pass
    
    @abstractmethod
    def list_regions(
        self, 
        region_type: Optional[str] = None,
        hierarchy_level: Optional[int] = None
    ) -> List[BrainRegion]:
        """List all regions with optional filtering."""
        pass
    
    @abstractmethod
    def generate_mask(
        self, 
        region_ids: List[int],
        output_path: str,
        reference_volume: Optional[str] = None
    ) -> str:
        """Generate NIfTI mask for specified regions."""
        pass
    
    @abstractmethod
    def get_neighbors(self, region_id: int) -> List[BrainRegion]:
        """Get neighboring regions."""
        pass
```

### D99 Implementation
```python
# src/atlases/d99_atlas.py

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from .base_atlas import BrainAtlas, BrainRegion

class D99Atlas(BrainAtlas):
    """D99 v2.0 Macaque Atlas implementation."""
    
    ATLAS_DIR = Path(__file__).parent.parent.parent / "data" / "atlases" / "D99_v2.0_dist"
    
    def __init__(self):
        self._load_atlas()
    
    def _load_atlas(self):
        """Load atlas volume and labels."""
        # Load atlas volume
        atlas_path = self.ATLAS_DIR / "D99_atlas_v2.0.nii.gz"
        self._atlas_img = nib.load(atlas_path)
        self._atlas_data = self._atlas_img.get_fdata().astype(np.int32)
        
        # Load template
        template_path = self.ATLAS_DIR / "D99_template.nii.gz"
        self._template_img = nib.load(template_path)
        
        # Load labels
        labels_path = self.ATLAS_DIR / "D99_v2.0_labels_semicolon.txt"
        self._labels = self._parse_labels(labels_path)
        
        # Build lookup tables
        self._id_to_region = {}
        self._name_to_id = {}
        self._abbrev_to_id = {}
        
        for region in self._labels:
            self._id_to_region[region.id] = region
            self._name_to_id[region.name.lower()] = region.id
            self._abbrev_to_id[region.abbreviation.lower()] = region.id
    
    def _parse_labels(self, labels_path: Path) -> List[BrainRegion]:
        """Parse D99 labels file."""
        regions = []
        with open(labels_path, 'r') as f:
            for line in f:
                if ';' in line:
                    parts = line.strip().split(';')
                    if len(parts) >= 2:
                        try:
                            region_id = int(parts[0])
                            abbrev = parts[1].strip()
                            name = parts[2].strip() if len(parts) > 2 else abbrev
                            
                            # Determine type from ID ranges (approximate)
                            if region_id <= 149:
                                region_type = "cortical"
                            else:
                                region_type = "subcortical"
                            
                            regions.append(BrainRegion(
                                id=region_id,
                                name=name,
                                abbreviation=abbrev,
                                hemisphere="bilateral",  # D99 is symmetric
                                region_type=region_type
                            ))
                        except ValueError:
                            continue
        return regions
    
    @property
    def species(self) -> str:
        return "macaque"
    
    @property
    def name(self) -> str:
        return "D99"
    
    @property
    def version(self) -> str:
        return "2.0"
    
    def get_region(self, query: str, query_type: str = "name") -> Optional[BrainRegion]:
        """Look up region by name, abbreviation, or ID."""
        query_lower = query.lower()
        
        if query_type == "name":
            region_id = self._name_to_id.get(query_lower)
        elif query_type == "abbreviation":
            region_id = self._abbrev_to_id.get(query_lower)
        elif query_type == "id":
            region_id = int(query)
        else:
            # Try all
            region_id = (self._name_to_id.get(query_lower) or 
                        self._abbrev_to_id.get(query_lower))
            if region_id is None:
                try:
                    region_id = int(query)
                except ValueError:
                    pass
        
        if region_id is not None:
            region = self._id_to_region.get(region_id)
            if region:
                # Compute centroid if not cached
                if region.centroid is None:
                    region.centroid = self._compute_centroid(region_id)
                    region.volume_mm3 = self._compute_volume(region_id)
                return region
        return None
    
    def get_region_at_coordinate(self, x: float, y: float, z: float) -> Optional[BrainRegion]:
        """Get region at voxel or world coordinate."""
        # Convert world to voxel coordinates
        affine_inv = np.linalg.inv(self._atlas_img.affine)
        voxel = affine_inv @ np.array([x, y, z, 1])
        i, j, k = int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
        
        # Bounds check
        if (0 <= i < self._atlas_data.shape[0] and
            0 <= j < self._atlas_data.shape[1] and
            0 <= k < self._atlas_data.shape[2]):
            
            region_id = self._atlas_data[i, j, k]
            if region_id > 0:
                return self._id_to_region.get(region_id)
        return None
    
    def _compute_centroid(self, region_id: int) -> Tuple[float, float, float]:
        """Compute centroid of region in world coordinates."""
        mask = self._atlas_data == region_id
        indices = np.argwhere(mask)
        if len(indices) == 0:
            return (0, 0, 0)
        
        centroid_voxel = indices.mean(axis=0)
        centroid_world = self._atlas_img.affine @ np.append(centroid_voxel, 1)
        return tuple(centroid_world[:3])
    
    def _compute_volume(self, region_id: int) -> float:
        """Compute volume in mm³."""
        mask = self._atlas_data == region_id
        voxel_count = np.sum(mask)
        voxel_vol = np.abs(np.linalg.det(self._atlas_img.affine[:3, :3]))
        return voxel_count * voxel_vol
    
    def list_regions(
        self,
        region_type: Optional[str] = None,
        hierarchy_level: Optional[int] = None
    ) -> List[BrainRegion]:
        """List all regions with optional filtering."""
        regions = list(self._id_to_region.values())
        
        if region_type:
            regions = [r for r in regions if r.region_type == region_type]
        
        return regions
    
    def generate_mask(
        self,
        region_ids: List[int],
        output_path: str,
        reference_volume: Optional[str] = None
    ) -> str:
        """Generate NIfTI mask for specified regions."""
        mask = np.isin(self._atlas_data, region_ids).astype(np.uint8)
        
        if reference_volume:
            # Resample to reference grid
            ref_img = nib.load(reference_volume)
            from scipy.ndimage import map_coordinates
            # ... resampling logic
        
        mask_img = nib.Nifti1Image(mask, self._atlas_img.affine)
        nib.save(mask_img, output_path)
        return output_path
    
    def get_neighbors(self, region_id: int) -> List[BrainRegion]:
        """Get neighboring regions (share boundary voxels)."""
        from scipy.ndimage import binary_dilation
        
        mask = self._atlas_data == region_id
        dilated = binary_dilation(mask)
        boundary = dilated & ~mask
        
        neighbor_ids = np.unique(self._atlas_data[boundary])
        neighbor_ids = neighbor_ids[neighbor_ids > 0]
        neighbor_ids = neighbor_ids[neighbor_ids != region_id]
        
        return [self._id_to_region[nid] for nid in neighbor_ids 
                if nid in self._id_to_region]


# Factory function
def get_atlas(species: str, atlas_name: str = None) -> BrainAtlas:
    """Get atlas instance for species."""
    if species == "macaque":
        if atlas_name is None or atlas_name == "D99":
            return D99Atlas()
    elif species == "mouse":
        # return AllenCCFAtlas()
        raise NotImplementedError("Allen CCF atlas not yet implemented")
    elif species == "rat":
        # return WaxholmAtlas()
        raise NotImplementedError("Waxholm atlas not yet implemented")
    
    raise ValueError(f"Unknown species/atlas: {species}/{atlas_name}")
```

### MCP Handler
```python
# bin/twosphere_mcp.py (additions)

from src.atlases import get_atlas, D99Atlas

@server.list_tools()
async def handle_list_tools():
    return [
        # ... existing tools ...
        Tool(
            name="lookup_brain_region",
            description="Look up brain region from species-specific atlas",
            inputSchema={...}  # As defined above
        ),
        Tool(
            name="generate_roi_mask",
            description="Generate NIfTI ROI mask for brain regions",
            inputSchema={...}
        ),
        Tool(
            name="get_vascular_rois",
            description="Get predefined vascular ROIs for glymphatic analysis",
            inputSchema={...}
        ),
        Tool(
            name="list_atlas_regions",
            description="List all regions in a brain atlas",
            inputSchema={...}
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == "lookup_brain_region":
        atlas = get_atlas(arguments["species"], arguments.get("atlas"))
        region = atlas.get_region(
            arguments["query"],
            arguments.get("query_type", "name")
        )
        if region:
            result = {
                "region": {
                    "id": region.id,
                    "name": region.name,
                    "abbreviation": region.abbreviation,
                    "hemisphere": region.hemisphere,
                    "type": region.region_type
                },
                "coordinates": {
                    "centroid": region.centroid,
                    "space": "native"
                },
                "volume_mm3": region.volume_mm3
            }
            
            if arguments.get("return_neighbors"):
                neighbors = atlas.get_neighbors(region.id)
                result["neighbors"] = [
                    {"id": n.id, "name": n.name} for n in neighbors
                ]
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            return [TextContent(type="text", text=f"Region not found: {arguments['query']}")]
    
    elif name == "generate_roi_mask":
        atlas = get_atlas(arguments["species"], arguments.get("atlas"))
        
        # Resolve region names to IDs
        region_ids = []
        for region_name in arguments["regions"]:
            region = atlas.get_region(region_name)
            if region:
                region_ids.append(region.id)
        
        if not region_ids:
            return [TextContent(type="text", text="No valid regions found")]
        
        output_path = atlas.generate_mask(
            region_ids,
            arguments["output_path"],
            arguments.get("reference_volume")
        )
        
        return [TextContent(type="text", text=json.dumps({
            "mask_path": output_path,
            "regions_included": region_ids,
            "total_regions": len(region_ids)
        }))]
    
    # ... other handlers
```

---

## Usage Examples

### From merge2docs (via ernie2_swarm)
```bash
# Look up a region
python bin/ernie2_swarm_mcp_e.py -q "What are the coordinates of V1 in macaque D99 atlas?" --auto-route

# Generate arterial ROI for glymphatic analysis
# (calls twosphere-mcp via MCP)
```

### Direct MCP call
```json
{
  "name": "lookup_brain_region",
  "arguments": {
    "species": "macaque",
    "atlas": "D99",
    "query": "V1",
    "return_neighbors": true
  }
}
```

### Glymphatic pipeline integration
```json
{
  "name": "get_vascular_rois",
  "arguments": {
    "species": "macaque",
    "roi_type": "all",
    "arterial_territory": "MCA",
    "output_dir": "/path/to/output"
  }
}
```

---

## Data Sources to Add

### Rodent Atlases (Future)
```bash
# Allen Mouse Brain CCF v3
curl -O http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_25.nii.gz

# Waxholm Rat
curl -O https://www.nitrc.org/frs/download.php/9423/WHS_SD_rat_atlas_v4.nii.gz
```

---

## Citations

**D99 v2.0**:
> Saleem KS, Logothetis NK (2012). A Combined MRI and Histology Atlas of the Rhesus Monkey Brain in Stereotaxic Coordinates. Academic Press.
> 
> Reveley C et al. (2017). Three-Dimensional Digital Template Atlas of the Macaque Brain. Cereb Cortex.

**Allen CCF**:
> Wang Q et al. (2020). The Allen Mouse Brain Common Coordinate Framework. Cell.

**Waxholm**:
> Papp EA et al. (2014). Waxholm Space atlas of the Sprague Dawley rat brain. NeuroImage.
