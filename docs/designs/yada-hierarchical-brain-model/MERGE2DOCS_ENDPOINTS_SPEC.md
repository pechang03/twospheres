# merge2docs QEC Tensor HTTP Endpoints Specification

## Overview

This document specifies HTTP endpoints that merge2docs needs to expose for twosphere-mcp to bootstrap its brain tensor.

**Architecture**: One-time bootstrap + independent computation

```
twosphere (brain) ←──[HTTP/MCP]──→ merge2docs (documents)

Step 1: Bootstrap (one-time)
  twosphere ←── GET /qec/tensor/corpus/download ──── merge2docs
  twosphere ←── GET /qec/tensor/cells ──── merge2docs

Step 2: Independent computation
  twosphere: Build 380-region brain tensor
  merge2docs: Continue with document tensor

Step 3: Expose back (later)
  twosphere ──→ MCP tool: get_brain_tensor() ──→ merge2docs
```

---

## Required Endpoints in merge2docs

### 1. Download Full Tensor Corpus

**Endpoint**: `GET /qec/tensor/corpus/download`

**Purpose**: One-time download of full merge2docs tensor for bootstrap

**Response**:
- **Format**: `application/octet-stream` (pickle binary)
- **Size**: ~56 MB
- **Content**: Complete tensor corpus with all cells + metadata

**Implementation** (merge2docs):

```python
# File: src/backend/services/v4_tensor_router.py or new qec_tensor_endpoints.py

from starlette.responses import FileResponse
from pathlib import Path

@router.get("/qec/tensor/corpus/download")
async def download_tensor_corpus():
    """Download full QEC tensor corpus (56MB).

    Returns pickled corpus with:
    - cells: Dict of all populated tensor cells
    - functors: List of F_i functors
    - domains: List of domains
    - levels: List of levels
    - fi_config: F_i hierarchy configuration
    - rids_config: r-IDS parameters
    - cross_training_config: Training parameters
    - syndrome_config: Syndrome thresholds
    - version: merge2docs version
    """
    import pickle
    from src.config.tensor_matrix import tensor_config

    corpus = {
        "version": "2.0",
        "cells": load_all_tensor_cells(),  # Helper function
        "functors": ["wisdom", "papers", "code", "testing", "git"],
        "domains": tensor_config.domains,
        "levels": tensor_config.levels,
        "fi_config": {
            "levels": tensor_config.fi_levels,
            "teaching_rules": get_fi_teaching_rules(),
            "direction_aware": tensor_config.direction_aware
        },
        "rids_config": {
            "r": tensor_config.r,
            "method": "greedy",
            "coverage_threshold": 0.95
        },
        "cross_training_config": {
            "rounds": tensor_config.cross_training_rounds,
            "lr": tensor_config.learning_rate,
            "bridge_threshold": tensor_config.bridge_threshold
        },
        "syndrome_config": {
            "threshold": 0.1,
            "types": ["cross_functor", "cross_domain", "cross_level"]
        }
    }

    # Save to temp file
    temp_path = Path("/tmp/merge2docs_tensor_corpus.pkl")
    with open(temp_path, 'wb') as f:
        pickle.dump(corpus, f)

    return FileResponse(
        path=temp_path,
        media_type="application/octet-stream",
        filename="merge2docs_tensor_corpus.pkl"
    )


def load_all_tensor_cells() -> Dict:
    """Load all populated tensor cells from disk.

    Returns:
        cells: {cell_name: cell_data}
    """
    from src.config.tensor_matrix import tensor_config
    from src.backend.models.tensor_loader import TensorLoader

    loader = TensorLoader(tensor_config)
    cells = {}

    # Iterate through all possible cells
    for functor in ["wisdom", "papers", "code", "testing", "git"]:
        for domain in tensor_config.domains:
            for level in tensor_config.levels:
                cell_name = f"{functor}_{domain}_{level}"
                cell_path = tensor_config.get_cell_model_path(cell_name)

                if cell_path.exists():
                    # Load cell
                    cell_model = loader.load_cell(cell_name)

                    cells[cell_name] = {
                        "functor": functor,
                        "domain": domain,
                        "level": level,
                        "features": extract_features(cell_model),
                        "rids_connections": extract_rids(cell_model),
                        "training_history": extract_training_history(cell_model)
                    }

    return cells


def get_fi_teaching_rules() -> Dict:
    """Get F_i hierarchy teaching rules.

    Returns:
        rules: {(source_fi, target_fi): should_teach}
    """
    from src.config.tensor_matrix import tensor_config

    rules = {}
    fi_levels = tensor_config.fi_levels

    for source in fi_levels:
        for target in fi_levels:
            # Higher abstraction (lower index) teaches lower
            rules[(source, target)] = tensor_config.fi_teaches(source, target)

    return rules
```

---

### 2. List Available Cells

**Endpoint**: `GET /qec/tensor/cells`

**Purpose**: List all populated tensor cells with metadata (fast, no download)

**Response**:
```json
{
  "total_cells": 28,
  "cells": [
    {
      "name": "wisdom_mathematics_document",
      "functor": "wisdom",
      "domain": "mathematics",
      "level": "document",
      "size_kb": 1.2,
      "last_trained": "2026-01-20T10:30:00Z",
      "rids_count": 30
    },
    {
      "name": "papers_molecular_bio_section",
      "functor": "papers",
      "domain": "molecular_bio",
      "level": "section",
      "size_kb": 0.8,
      "last_trained": "2026-01-20T09:15:00Z",
      "rids_count": 25
    }
    // ... more cells
  ]
}
```

**Implementation**:

```python
@router.get("/qec/tensor/cells")
async def list_tensor_cells():
    """List all populated tensor cells with metadata."""
    from src.config.tensor_matrix import tensor_config
    from pathlib import Path

    cells = []

    for functor in ["wisdom", "papers", "code", "testing", "git"]:
        for domain in tensor_config.domains:
            for level in tensor_config.levels:
                cell_name = f"{functor}_{domain}_{level}"
                cell_path = tensor_config.get_cell_model_path(cell_name)

                if cell_path.exists():
                    stat = cell_path.stat()

                    cells.append({
                        "name": cell_name,
                        "functor": functor,
                        "domain": domain,
                        "level": level,
                        "size_kb": stat.st_size / 1024,
                        "last_trained": stat.st_mtime,
                        "rids_count": 30  # Could extract from model
                    })

    return {
        "total_cells": len(cells),
        "cells": cells
    }
```

---

### 3. Brain Region Mapping (Optional, Future)

**Endpoint**: `GET /qec/brain_regions/mapping`

**Purpose**: Suggest mapping from merge2docs domains → brain regions

**Response**:
```json
{
  "mapping": {
    "mathematics": "PFC",
    "molecular_bio": "hippocampus",
    "combinatorial": "PFC",
    "physics": "parietal_cortex",
    "machine_learning": "temporal_cortex"
  },
  "method": "semantic_similarity",
  "confidence": 0.75
}
```

**Implementation** (can be added later):

```python
@router.get("/qec/brain_regions/mapping")
async def get_brain_region_mapping():
    """Get suggested domain → brain region mapping.

    This uses semantic similarity between domain descriptions
    and brain region functions.
    """
    # This can be populated gradually
    # For now, return empty or basic mapping
    mapping = {
        "mathematics": "PFC",  # Executive function
        "molecular_bio": "hippocampus",  # Memory
        "combinatorial": "PFC",  # Problem solving
        "machine_learning": "temporal_cortex",  # Pattern recognition
    }

    return {
        "mapping": mapping,
        "method": "manual_annotation",
        "confidence": 0.5
    }
```

---

## Integration Points

### Where to Add in merge2docs

**Option 1: Extend existing `v4_tensor_router.py`**

```python
# File: src/backend/services/v4_tensor_router.py

from starlette.routing import Route

# Add new routes
routes = [
    # ... existing routes
    Route("/qec/tensor/corpus/download", download_tensor_corpus, methods=["GET"]),
    Route("/qec/tensor/cells", list_tensor_cells, methods=["GET"]),
    Route("/qec/brain_regions/mapping", get_brain_region_mapping, methods=["GET"]),
]
```

**Option 2: Create new `qec_tensor_endpoints.py`**

```python
# File: src/backend/services/qec_tensor_endpoints.py

from starlette.applications import Starlette
from starlette.routing import Route

# New service specifically for QEC tensor HTTP access

routes = [
    Route("/qec/tensor/corpus/download", download_tensor_corpus, methods=["GET"]),
    Route("/qec/tensor/cells", list_tensor_cells, methods=["GET"]),
    Route("/qec/brain_regions/mapping", get_brain_region_mapping, methods=["GET"]),
]

app = Starlette(routes=routes)
```

Then mount in main app:

```python
# File: src/main.py or wherever Starlette app is configured

from src.backend.services.qec_tensor_endpoints import app as qec_app

# Mount QEC endpoints
main_app.mount("/qec", qec_app)
```

---

## Usage from twosphere-mcp

Once endpoints are available, twosphere can bootstrap:

```python
from backend.services.qec_tensor_service import bootstrap_brain_tensor

# One-time bootstrap
brain_tensor = await bootstrap_brain_tensor()

# Result:
# - Downloads 56MB corpus from merge2docs
# - Extracts learned patterns (F_i, r-IDS, cross-training)
# - Adapts to brain: 6 functors × 100 regions × 3 scales
# - Saves to local cache
```

---

## Later: twosphere → merge2docs Integration

After twosphere builds its brain tensor, expose back via MCP:

```python
# File: twosphere-mcp/src/backend/mcp/brain_tensor_tools.py

@mcp_tool
def get_brain_tensor(region: str, functor: str) -> Dict:
    """Get brain tensor cell for region and functor.

    Args:
        region: Brain region name (e.g., "V1", "PFC")
        functor: Functor name (e.g., "anatomy", "function")

    Returns:
        cell: Brain tensor cell data
    """
    # Load from twosphere brain tensor
    cell = load_brain_cell(region, functor)

    return {
        "region": region,
        "functor": functor,
        "features": cell.features,
        "rids_connections": cell.rids_connections,
        "syndrome": cell.syndrome_mean
    }


@mcp_tool
def list_brain_regions() -> List[str]:
    """List all available brain regions in tensor.

    Returns:
        regions: List of region names
    """
    # Query D99 atlas
    from backend.services.brain_atlas_client import BrainAtlasClient

    client = BrainAtlasClient("http://localhost:8007")
    regions = client.list_regions(species="macaque", atlas="D99")

    return [r["name"] for r in regions]
```

merge2docs can then call:

```python
# In merge2docs
from mcp import ClientSession

async with ClientSession("twosphere-mcp") as session:
    # Get V1 function data
    v1_function = await session.call_tool(
        "get_brain_tensor",
        region="V1",
        functor="function"
    )

    # Use in document analysis
    # (e.g., map "vision research papers" → V1 function tensor)
```

---

## Summary

**What merge2docs needs to add**:

1. ✅ **Endpoint**: `GET /qec/tensor/corpus/download`
   - Returns: 56MB pickled corpus
   - One-time download for bootstrap

2. ✅ **Endpoint**: `GET /qec/tensor/cells`
   - Returns: JSON list of cells
   - Fast metadata query

3. ⏳ **Endpoint**: `GET /qec/brain_regions/mapping` (optional, later)
   - Returns: Domain → region suggestions
   - Can start with empty/basic mapping

**Where to add**:
- Option A: Extend `src/backend/services/v4_tensor_router.py`
- Option B: Create new `src/backend/services/qec_tensor_endpoints.py`

**Port**: 8091 (merge2docs default)

**Timeline**:
- Week 1: Add endpoints 1-2 to merge2docs
- Week 2: twosphere bootstrap implementation
- Week 3: Build brain-specific tensor
- Week 4: Expose back via MCP (optional)

---

**Files to modify in merge2docs**:
1. `src/backend/services/v4_tensor_router.py` (or new file)
2. `src/main.py` (mount new routes)

**Files created in twosphere-mcp**:
1. ✅ `src/backend/services/qec_tensor_service.py`
2. ⏳ `src/backend/services/amem_e_service.py` (routing point, later)
3. ⏳ `src/backend/mcp/brain_tensor_tools.py` (MCP tools, later)
