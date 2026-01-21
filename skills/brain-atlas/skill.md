---
name: brain-atlas
description: Multi-species brain atlas queries for neuroimaging ROI analysis
---

# Brain Atlas Skill

**Category:** Neuroimaging & Brain Region Analysis  
**MCP Port:** 8007  
**Audience:** Neuroscientists, MRI Researchers, Computational Neuroanatomists  
**Dependencies:** NIfTI volumes, species-specific atlases

---

## Overview

Brain Atlas provides unified access to multi-species brain atlases for neuroimaging analysis. Lookup brain regions by name, abbreviation, or MRI coordinate. Generate ROI masks for fMRI/DTI analysis. Identify anatomical neighbors for connectivity studies.

**Supported Atlases:**
- **Macaque**: D99 v2.0 (368 regions) - Saleem & Logothetis MRI-histology atlas
- **Mouse**: Allen CCF v3 (800+ regions) - Allen Institute Common Coordinate Framework
- **Rat**: Waxholm Space v4 (222 regions) - Sprague-Dawley reference atlas

## 🎯 When to Use

- Look up brain region by name ("V1", "hippocampus", "amygdala")
- Find region at MRI coordinate (x, y, z in native space)
- Generate NIfTI ROI masks for fMRI/DTI analysis
- Identify neighboring regions for connectivity analysis
- Cross-species region comparison (macaque ↔ rodent homologs)
- Validate stereotaxic injection coordinates

## 🔌 Essential Commands/APIs

### 1. Region Lookup by Name

```bash
# HTTP API - lookup V1 in macaque
curl -X POST http://localhost:8007/api/lookup_region \
  -H "Content-Type: application/json" \
  -d '{"species": "macaque", "query": "V1"}'

# Response:
# {
#   "found": true,
#   "region": {
#     "id": 332,
#     "name": "visual area 1 (primary visual cortex)",
#     "abbreviation": "V1",
#     "centroid": [0.0, -37.18, 8.20],
#     "volume_mm3": 5031.81,
#     "functional_systems": ["visual"]
#   }
# }
```

**Python API**:
```python
import requests

def lookup_region(species: str, query: str, return_neighbors: bool = False):
    """Look up brain region by name or abbreviation."""
    resp = requests.post(
        "http://localhost:8007/api/lookup_region",
        json={
            "species": species,
            "query": query,
            "return_neighbors": return_neighbors
        }
    )
    return resp.json()

# Example: Find hippocampus in mouse
result = lookup_region("mouse", "hippocampus")
print(f"Region: {result['region']['name']}")
print(f"Volume: {result['region']['volume_mm3']:.1f} mm³")
```

### 2. Coordinate-Based Lookup

```bash
# Find region at MRI coordinate (x=10, y=-5, z=3 mm)
curl -X POST http://localhost:8007/api/lookup_region \
  -H "Content-Type: application/json" \
  -d '{
    "species": "macaque",
    "query": "coord: 10, -5, 3",
    "query_type": "coordinate"
  }'

# Response:
# {
#   "found": true,
#   "region": {
#     "id": 45,
#     "name": "caudate nucleus",
#     "abbreviation": "cd"
#   }
# }
```

**Python API**:
```python
def region_at_coordinate(species: str, x: float, y: float, z: float):
    """Get brain region at MRI coordinate."""
    resp = requests.post(
        "http://localhost:8007/api/lookup_region",
        json={
            "species": species,
            "query": f"coord: {x}, {y}, {z}",
            "query_type": "coordinate"
        }
    )
    return resp.json()

# Example: What's at stereotaxic coord AP=-3, ML=2, DV=5?
result = region_at_coordinate("rat", -3, 2, 5)
print(f"Region at (-3, 2, 5): {result['region']['name']}")
```

### 3. List Regions with Filtering

```bash
# List all visual cortex regions in macaque
curl -X POST http://localhost:8007/api/list_regions \
  -H "Content-Type: application/json" \
  -d '{
    "species": "macaque",
    "search_pattern": "V[0-9]",
    "filter_system": "visual"
  }'

# Response:
# {
#   "total_regions": 12,
#   "regions": [
#     {"id": 332, "abbreviation": "V1", "name": "visual area 1"},
#     {"id": 333, "abbreviation": "V2", "name": "visual area 2"},
#     ...
#   ]
# }
```

**Python API**:
```python
def list_regions(species: str, search_pattern: str = None, filter_type: str = None):
    """List brain regions with optional filtering."""
    resp = requests.post(
        "http://localhost:8007/api/list_regions",
        json={
            "species": species,
            "search_pattern": search_pattern,
            "filter_type": filter_type
        }
    )
    return resp.json()

# Example: Find all cortical regions in rat
cortical = list_regions("rat", filter_type="cortical")
print(f"Found {cortical['total_regions']} cortical regions")
```

### 4. Generate ROI Mask

```bash
# Generate NIfTI mask for V1+V2 in macaque
curl -X POST http://localhost:8007/api/generate_mask \
  -H "Content-Type: application/json" \
  -d '{
    "species": "macaque",
    "regions": ["V1", "V2"],
    "output_path": "/tmp/visual_cortex_mask.nii.gz",
    "hemisphere": "both",
    "dilate_mm": 1
  }'

# Response:
# {
#   "mask_path": "/tmp/visual_cortex_mask.nii.gz",
#   "regions_included": [
#     {"name": "V1", "id": 332},
#     {"name": "V2", "id": 333}
#   ]
# }
```

**Python API**:
```python
def generate_roi_mask(
    species: str,
    regions: list,
    output_path: str,
    hemisphere: str = "both",
    dilate_mm: float = 0
):
    """Generate NIfTI ROI mask for specified regions."""
    resp = requests.post(
        "http://localhost:8007/api/generate_mask",
        json={
            "species": species,
            "regions": regions,
            "output_path": output_path,
            "hemisphere": hemisphere,
            "dilate_mm": dilate_mm
        }
    )
    return resp.json()

# Example: Create hippocampus mask for mouse fMRI
result = generate_roi_mask(
    species="mouse",
    regions=["hippocampus", "dentate gyrus", "CA1", "CA3"],
    output_path="/data/mouse_hippo_mask.nii.gz",
    dilate_mm=0.5  # Slight dilation for partial volume
)
print(f"Saved mask: {result['mask_path']}")
```

### 5. Get Neighboring Regions

```bash
# Find neighbors of V1 (for connectivity analysis)
curl -X POST http://localhost:8007/api/get_neighbors \
  -H "Content-Type: application/json" \
  -d '{"species": "macaque", "region": "V1"}'

# Response:
# {
#   "region": {"id": 332, "name": "V1"},
#   "neighbors": [
#     {"id": 333, "name": "V2", "abbreviation": "V2"},
#     {"id": 334, "name": "V3", "abbreviation": "V3"},
#     {"id": 340, "name": "V4", "abbreviation": "V4"}
#   ],
#   "neighbor_count": 8
# }
```

---

## 🔄 Common Workflows

### Workflow 1: fMRI ROI Analysis

**Scenario**: Extract BOLD signal from visual cortex regions

```python
import requests
import nibabel as nib
import numpy as np

# 1. Generate visual cortex mask
mask_result = requests.post(
    "http://localhost:8007/api/generate_mask",
    json={
        "species": "macaque",
        "regions": ["V1", "V2", "V4", "MT"],
        "output_path": "/tmp/visual_mask.nii.gz"
    }
).json()

# 2. Load mask and fMRI data
mask = nib.load(mask_result['mask_path']).get_fdata()
fmri = nib.load('/data/monkey_fmri.nii.gz').get_fdata()

# 3. Extract mean timeseries from ROI
roi_indices = np.where(mask > 0)
timeseries = fmri[roi_indices].mean(axis=0)

print(f"Extracted {len(timeseries)} timepoints from visual cortex")
```

### Workflow 2: Stereotaxic Coordinate Validation

**Scenario**: Verify injection site hits target structure

```python
# Planned injection coordinates (rat stereotaxic)
injection_coords = [
    {"name": "VTA", "AP": -5.3, "ML": 0.8, "DV": 8.0},
    {"name": "NAc", "AP": 1.7, "ML": 1.5, "DV": 7.0},
]

# Validate each coordinate
for site in injection_coords:
    result = requests.post(
        "http://localhost:8007/api/lookup_region",
        json={
            "species": "rat",
            "query": f"coord: {site['AP']}, {site['ML']}, {site['DV']}",
            "query_type": "coordinate"
        }
    ).json()
    
    actual = result['region']['name'] if result['found'] else "Unknown"
    match = "✅" if site['name'].lower() in actual.lower() else "⚠️"
    print(f"{match} {site['name']}: coordinates hit '{actual}'")

# Output:
# ✅ VTA: coordinates hit 'ventral tegmental area'
# ✅ NAc: coordinates hit 'nucleus accumbens'
```

### Workflow 3: Cross-Species Homolog Mapping

**Scenario**: Find mouse equivalent of macaque region

```python
def find_homolog(source_species: str, target_species: str, region_name: str):
    """Find potential homolog in another species."""
    # Get source region info
    source = requests.post(
        "http://localhost:8007/api/lookup_region",
        json={"species": source_species, "query": region_name}
    ).json()
    
    if not source['found']:
        return None
    
    # Search for similar name in target species
    target_search = requests.post(
        "http://localhost:8007/api/list_regions",
        json={
            "species": target_species,
            "search_pattern": region_name.split()[0]  # First word
        }
    ).json()
    
    return {
        "source": source['region'],
        "candidates": target_search['regions'][:5]
    }

# Example: Find mouse homolog of macaque "hippocampus"
result = find_homolog("macaque", "mouse", "hippocampus")
print(f"Macaque: {result['source']['name']}")
print(f"Mouse candidates: {[r['name'] for r in result['candidates']]}")
```

### Workflow 4: Connectivity Seed Generation

**Scenario**: Generate seeds for DTI tractography

```python
# Get V1 and its neighbors for connectivity analysis
v1_info = requests.post(
    "http://localhost:8007/api/get_neighbors",
    json={"species": "macaque", "region": "V1"}
).json()

# Generate mask for V1 (seed)
seed_mask = requests.post(
    "http://localhost:8007/api/generate_mask",
    json={
        "species": "macaque",
        "regions": ["V1"],
        "output_path": "/tmp/v1_seed.nii.gz"
    }
).json()

# Generate masks for each neighbor (targets)
neighbor_names = [n['abbreviation'] for n in v1_info['neighbors']]
target_mask = requests.post(
    "http://localhost:8007/api/generate_mask",
    json={
        "species": "macaque",
        "regions": neighbor_names,
        "output_path": "/tmp/v1_targets.nii.gz"
    }
).json()

print(f"Seed: {seed_mask['mask_path']}")
print(f"Targets ({len(neighbor_names)}): {target_mask['mask_path']}")
```

---

## 💡 Best Practices

### 1. Species-Specific Coordinate Systems

| Species | Coordinate System | Origin |
|---------|------------------|--------|
| Macaque | D99 native | AC-PC aligned |
| Mouse | Allen CCF | Bregma (0,0,0) |
| Rat | Waxholm | Interaural line |

**Important**: Coordinates are atlas-native. Transform your data to atlas space first.

### 2. ROI Mask Dilation

```python
# For fMRI (low resolution): dilate 1-2mm to capture partial volume
generate_roi_mask(regions=["V1"], dilate_mm=1.5)

# For high-res structural: no dilation
generate_roi_mask(regions=["V1"], dilate_mm=0)

# For DTI tractography seeds: slight dilation (0.5mm)
generate_roi_mask(regions=["V1"], dilate_mm=0.5)
```

### 3. Hemisphere Selection

```python
# Both hemispheres (default)
generate_roi_mask(regions=["hippocampus"], hemisphere="both")

# Left only (for lateralization studies)
generate_roi_mask(regions=["hippocampus"], hemisphere="left")

# Right only
generate_roi_mask(regions=["hippocampus"], hemisphere="right")
```

### 4. Region Name Matching

The atlas supports flexible matching:
- **Exact abbreviation**: "V1", "CA1", "VTA"
- **Full name**: "primary visual cortex", "hippocampus"
- **Partial match**: "visual" matches all visual areas
- **Regex patterns**: "V[0-9]" matches V1, V2, V3, V4

---

## 🚨 Troubleshooting

### Issue: Region not found

**Diagnosis**: Name doesn't match atlas nomenclature

**Solution**:
```python
# List all regions and search
all_regions = requests.post(
    "http://localhost:8007/api/list_regions",
    json={"species": "macaque"}
).json()

# Search for partial match
query = "visual"
matches = [r for r in all_regions['regions'] 
           if query.lower() in r['name'].lower()]
print(f"Found {len(matches)} regions containing '{query}'")
```

### Issue: Coordinate returns unexpected region

**Diagnosis**: Coordinate may be at region boundary or in wrong space

**Solution**:
```python
# Check neighbors to understand boundary
result = requests.post(
    "http://localhost:8007/api/lookup_region",
    json={
        "species": "macaque",
        "query": "coord: 10, -5, 3",
        "query_type": "coordinate",
        "return_neighbors": True  # Get boundary context
    }
).json()

if result['found']:
    print(f"Region: {result['region']['name']}")
    print(f"Neighbors: {[n['name'] for n in result.get('neighbors', [])]}")
```

### Issue: Mask generation fails

**Diagnosis**: Region IDs not found or output path not writable

**Solution**:
```python
# Verify regions exist first
for region in ["V1", "V2", "V99"]:  # V99 doesn't exist
    result = requests.post(
        "http://localhost:8007/api/lookup_region",
        json={"species": "macaque", "query": region}
    ).json()
    status = "✅" if result['found'] else "❌"
    print(f"{status} {region}: {'found' if result['found'] else 'NOT FOUND'}")
```

---

## 📊 Atlas Specifications

### D99 Macaque (v2.0)

| Property | Value |
|----------|-------|
| Species | Macaca mulatta |
| Regions | 368 |
| Resolution | 0.25mm isotropic |
| Space | Native T1 |
| Citation | Saleem & Logothetis (2012) |

### Allen CCF Mouse (v3)

| Property | Value |
|----------|-------|
| Species | Mus musculus (C57BL/6J) |
| Regions | 800+ (hierarchical) |
| Resolution | 25µm isotropic |
| Space | CCFv3 (Bregma-centered) |
| Citation | Wang et al. (2020) |

### Waxholm Rat (v4)

| Property | Value |
|----------|-------|
| Species | Rattus norvegicus (SD) |
| Regions | 222 |
| Resolution | 39µm isotropic |
| Space | Waxholm Space |
| Citation | Papp et al. (2014) |

---

## 🔗 Related Documentation

- **TwoSphere MCP**: `bin/twosphere_http_server.py` (optical/MRI physics)
- **Atlas Module**: `src/atlases/` (Python implementation)
- **D99 Atlas**: `src/atlases/d99_atlas.py`
- **Download Script**: `scripts/download_atlases.py`
- **Service Startup**: `scripts/start_twosphere_services.sh`

---

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Region lookup (name) | 5ms | Hash table lookup |
| Region lookup (coordinate) | 50ms | Volume indexing |
| List all regions | 20ms | 368-800 regions |
| Generate mask (single region) | 200ms | NIfTI I/O |
| Generate mask (10 regions) | 500ms | Combined mask |
| Get neighbors | 100ms | Boundary voxel analysis |

---

**Last Updated**: 2026-01-21  
**Version**: 1.0  
**Status**: Production Ready  
**MCP Port**: 8007
