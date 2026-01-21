# Beads: QEC Tensor Brain Integration

## 🟢 BEAD-QEC-1: Bootstrap Service Implementation
**Priority**: High
**Dependencies**: merge2docs tensor matrix
**Estimated Complexity**: Medium
**Status**: ✅ COMPLETED (2026-01-21)

### Objective
Implement HTTP bootstrap service to download merge2docs' existing 20×5 tensor matrix and adapt it for brain-specific 380-region QEC tensor.

### Implementation Tasks

#### 1. QEC Tensor Service
**File**: `src/backend/services/qec_tensor_service.py` (434 lines)

**Key Components**:
- `QECTensorConfig` - Configuration dataclass
- `QECTensorClient` - HTTP client for merge2docs endpoints
- `bootstrap_brain_tensor()` - High-level bootstrap API

**Functor Mapping**:
```python
functor_mapping = {
    "wisdom": "behavior",    # F0: High-level → Task relevance
    "papers": "function",    # F1: Research → What computes
    "code": "anatomy",       # F2: Implementation → Structure
    "testing": "electro",    # F3: Validation → Dynamics
    "git": "genetics",       # F5: Version control → Heritage
    "pathology": "pathology" # New: Disease markers
}
```

**Bootstrap Strategy**:
1. ONE-TIME download from merge2docs (56MB corpus)
2. Extract learned patterns (F_i, r-IDS, cross-training)
3. Adapt to brain: 6 functors × 100 regions × 3 scales
4. Save to local cache
5. Build independently (no continuous sync)

#### 2. merge2docs Endpoints Specification
**File**: `docs/designs/yada-hierarchical-brain-model/MERGE2DOCS_ENDPOINTS_SPEC.md` (456 lines)

**Required Endpoints** (to be implemented by merge2docs team):
- `GET /qec/tensor/corpus/download` - Download 56MB pickled corpus
- `GET /qec/tensor/cells` - List populated cells with metadata
- `GET /qec/brain_regions/mapping` - Domain → region suggestions (optional)

**Integration Points**:
- Port: 8091 (merge2docs default)
- Location: `src/backend/services/v4_tensor_router.py` or new `qec_tensor_endpoints.py`
- Helper functions: `load_all_tensor_cells()`, `get_fi_teaching_rules()`

### Success Criteria
- ✅ Bootstrap service implemented with full HTTP client
- ✅ Functor mapping defined (merge2docs → brain)
- ✅ Pattern extraction logic implemented
- ✅ Specification document for merge2docs team complete
- ✅ Cache system for one-time bootstrap

### Files Created
- `src/backend/services/qec_tensor_service.py` - Bootstrap service
- `docs/designs/yada-hierarchical-brain-model/MERGE2DOCS_ENDPOINTS_SPEC.md` - API spec

### Next Steps
Ready for BEAD-QEC-2 (merge2docs endpoint implementation) before testing.

---

## 🟢 BEAD-QEC-2: Functor Hierarchies Catalog
**Priority**: High
**Dependencies**: None
**Estimated Complexity**: Low
**Status**: ✅ COMPLETED (2026-01-21)

### Objective
Create central registry for all F_i functor hierarchies across domains, enabling multiple hierarchies per domain.

### Implementation

**File**: `docs/designs/yada-hierarchical-brain-model/FUNCTOR_HIERARCHIES_CATALOG.md` (450 lines)

**Hierarchies Registered**:

#### Math Domain
- `math-theoretical-v1`: proof, construction, theorem, lemma, corollary, example
- `math-lean-validation-v1`: statement, proof_skeleton, tactic_search, tactic_apply, verification, validation

#### Brain Domain
- `brain-research-v1`: anatomy, function, electro, genetics, behavior, pathology
- `brain-clinical-v1`: symptoms, diagnosis, imaging, treatment, prognosis, longitudinal
- `brain-computational-v1`: structure, dynamics, learning, prediction, optimization, validation

#### Software Domain
- `software-merge2docs-v1`: wisdom, papers, code, testing, git

#### Chess Domain (example)
- `chess-analysis-v1`: opening, middlegame, tactics, strategy, endgame, evaluation

**Database Schema**:
```sql
CREATE TABLE functor_hierarchies (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    domain TEXT NOT NULL,
    purpose TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    functors JSONB NOT NULL,
    cross_functor_syndromes JSONB,
    integration_with JSONB,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Key Features**:
- Version tracking for evolution
- Cross-functor syndrome configuration
- Integration mapping between hierarchies
- Template for adding new hierarchies

### Success Criteria
- ✅ Central catalog with all existing hierarchies
- ✅ Database schema for functor registry
- ✅ Template for adding new hierarchies
- ✅ Integration examples (Math ↔ Brain, LEAN ↔ Brain)

### Files Created
- `docs/designs/yada-hierarchical-brain-model/FUNCTOR_HIERARCHIES_CATALOG.md`

### Impact
**User quote**: "its very important we record our F_i hierarchies somewhere we can find them"

This catalog fulfills that requirement - the F_i vertical dimension is the critical architectural component.

---

## 🟢 BEAD-QEC-3: Advanced Cache and Cross-Training
**Priority**: Medium
**Dependencies**: BEAD-QEC-1
**Estimated Complexity**: High
**Status**: ✅ COMPLETED (2026-01-21)

### Objective
Design LRU cache with smart prefetching for 380-region brain tensor (only 20 regions in RAM) plus recursive cross-training architecture.

### Implementation

**File**: `docs/designs/yada-hierarchical-brain-model/BRAIN_QEC_CACHE_CROSSTRAINING.md` (800+ lines)

**Key Components**:

#### 1. BrainRegionCache
```python
class BrainRegionCache:
    """LRU cache for brain region tensors with smart prefetching."""

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.cache = OrderedDict()  # LRU cache
        self.prefetch_queue = asyncio.Queue()

    async def get_or_load(self, region_name: str) -> RegionTensor:
        # Check cache first
        if region_name in self.cache:
            return self.cache[region_name]  # Cache HIT

        # Load from database
        tensor = await self._load_from_database(region_name)

        # Prefetch r-IDS neighbors
        if tensor.rids_connections:
            await self._schedule_prefetch(tensor.rids_connections)

        return tensor
```

**Cache Strategy**:
- LRU eviction when capacity reached (20 regions)
- Smart prefetching: Load r-IDS neighbors in background
- Anticipates future needs based on graph connectivity
- Async loading for non-blocking operations

#### 2. RecursiveCrossTrainer
```python
class RecursiveCrossTrainer:
    """Iteratively train brain regions based on r-IDS bridges."""

    async def train_iteration(self, region_name: str, iteration: int):
        # Load region tensor
        region = await self.cache.get_or_load(region_name)

        # Gather teaching signals from r-IDS neighbors
        teaching_signals = []
        for neighbor in region.rids_connections:
            neighbor_tensor = await self.cache.get_or_load(neighbor)

            # Apply F_i hierarchy teaching rules
            if self.can_teach(neighbor_tensor.functor, region.functor):
                teaching_signals.append(neighbor_tensor.features)

        # Update region with aggregated signal
        updated_features = self.aggregate_teaching(
            region.features,
            teaching_signals
        )

        # Check for cross-functor syndromes
        syndrome = self.detect_syndrome(region, updated_features)

        return updated_features, syndrome
```

**Cross-Training Strategy**:
- Recursive: Regions train each other via r-IDS bridges
- F_i hierarchy aware: Higher functors teach lower
- Syndrome detection: Identify inconsistencies across functors
- Convergence: Monitor until stable

#### 3. BrainQECSystem
Complete integration system combining cache + cross-training.

### Success Criteria
- ✅ LRU cache design with 20-region capacity
- ✅ Smart prefetching based on r-IDS connections
- ✅ Recursive cross-training algorithm
- ✅ Syndrome detection across functors
- ✅ Complete system integration architecture

### Files Created
- `docs/designs/yada-hierarchical-brain-model/BRAIN_QEC_CACHE_CROSSTRAINING.md`

### Impact
Enables 380-region brain tensor with only 20 in RAM - scalable architecture for future expansion.

---

## 🔵 BEAD-QEC-4: merge2docs Endpoint Implementation
**Priority**: CRITICAL (blocking testing)
**Dependencies**: BEAD-QEC-1
**Estimated Complexity**: Medium
**Status**: 🚧 PENDING (waiting on merge2docs team)

### Objective
Implement HTTP endpoints in merge2docs to expose tensor corpus for twosphere bootstrap.

### Implementation Tasks

**Location**: `merge2docs/src/backend/services/`

**Option A**: Extend existing `v4_tensor_router.py`
**Option B**: Create new `qec_tensor_endpoints.py`

#### 1. Download Endpoint
```python
# File: src/backend/services/v4_tensor_router.py

@router.get("/qec/tensor/corpus/download")
async def download_tensor_corpus():
    """Download full QEC tensor corpus (56MB)."""
    corpus = {
        "version": "2.0",
        "cells": load_all_tensor_cells(),
        "functors": ["wisdom", "papers", "code", "testing", "git"],
        "domains": tensor_config.domains,
        "levels": tensor_config.levels,
        "fi_config": {
            "levels": tensor_config.fi_levels,
            "teaching_rules": get_fi_teaching_rules(),
            "direction_aware": tensor_config.direction_aware
        },
        "rids_config": {
            "r": 4,  # r=4 optimal for brain LID≈4-7
            "method": "greedy",
            "coverage_threshold": 0.95
        },
        "cross_training_config": {...},
        "syndrome_config": {...}
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
```

#### 2. List Cells Endpoint
```python
@router.get("/qec/tensor/cells")
async def list_tensor_cells():
    """List all populated tensor cells with metadata."""
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
                        "rids_count": 30
                    })

    return {
        "total_cells": len(cells),
        "cells": cells
    }
```

#### 3. Helper Functions
```python
def load_all_tensor_cells() -> Dict:
    """Load all populated tensor cells from disk."""
    loader = TensorLoader(tensor_config)
    cells = {}

    for functor in ["wisdom", "papers", "code", "testing", "git"]:
        for domain in tensor_config.domains:
            for level in tensor_config.levels:
                cell_name = f"{functor}_{domain}_{level}"
                cell_path = tensor_config.get_cell_model_path(cell_name)

                if cell_path.exists():
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
    """Get F_i hierarchy teaching rules."""
    rules = {}
    fi_levels = tensor_config.fi_levels

    for source in fi_levels:
        for target in fi_levels:
            # Higher abstraction (lower index) teaches lower
            rules[(source, target)] = tensor_config.fi_teaches(source, target)

    return rules
```

### Test Coverage
- Test endpoint returns valid pickle file
- Test corpus structure has all required keys
- Test cell listing returns correct metadata
- Integration test: twosphere downloads successfully

### Success Criteria
- ✅ Endpoints accessible at `http://localhost:8091/qec/*`
- ✅ Download returns 56MB corpus pickle
- ✅ List returns JSON with cell metadata
- ✅ twosphere can bootstrap successfully

### Blocking
This bead blocks BEAD-QEC-5 (testing). Coordinate with merge2docs team for implementation.

**Specification**: See `MERGE2DOCS_ENDPOINTS_SPEC.md` for complete implementation guide.

---

## 🔵 BEAD-QEC-5: Bootstrap Testing and Validation
**Priority**: High
**Dependencies**: BEAD-QEC-4
**Estimated Complexity**: Medium
**Status**: 🚧 NOT STARTED (blocked by BEAD-QEC-4)

### Objective
Test end-to-end bootstrap flow from merge2docs to brain tensor construction.

### Implementation Tasks

#### 1. Integration Tests
**File**: `tests/backend/services/test_qec_tensor_service.py`

```python
@pytest.mark.asyncio
async def test_bootstrap_from_merge2docs():
    """Test full bootstrap flow."""
    # Mock merge2docs endpoints
    with aioresponses() as mocked:
        # Mock corpus download
        corpus_data = create_mock_corpus()
        mocked.get(
            "http://localhost:8091/qec/tensor/corpus/download",
            body=pickle.dumps(corpus_data),
            status=200
        )

        # Run bootstrap
        brain_tensor = await bootstrap_brain_tensor(force_download=True)

        # Validate structure
        assert "regions" in brain_tensor
        assert "functors" in brain_tensor
        assert len(brain_tensor["functors"]) == 6  # Brain functors

        # Validate functor mapping applied
        assert "behavior" in brain_tensor["functors"]  # Mapped from wisdom
        assert "anatomy" in brain_tensor["functors"]    # Mapped from code

@pytest.mark.asyncio
async def test_pattern_extraction():
    """Test extraction of learned patterns from corpus."""
    corpus = create_mock_corpus()
    client = QECTensorClient()

    patterns = await client.extract_learned_patterns(corpus)

    assert "fi_teaching_rules" in patterns
    assert "rids_connections" in patterns
    assert "cross_training_params" in patterns

@pytest.mark.asyncio
async def test_brain_adaptation():
    """Test adaptation from merge2docs to brain tensor."""
    corpus = create_mock_corpus()
    patterns = {
        "fi_teaching_rules": {...},
        "rids_connections": {...}
    }

    client = QECTensorClient()
    brain_tensor = await client.adapt_to_brain(corpus, patterns)

    # Validate brain-specific structure
    assert brain_tensor["dimensions"] == [6, 100, 3]  # 6 functors, 100 regions, 3 scales
    assert brain_tensor["r"] == 4  # r-IDS parameter
```

#### 2. Validation Tests
```python
def test_functor_mapping():
    """Test merge2docs → brain functor mapping."""
    mapping = {
        "wisdom": "behavior",
        "papers": "function",
        "code": "anatomy",
        "testing": "electro",
        "git": "genetics"
    }

    for source, target in mapping.items():
        assert map_functor(source) == target

def test_r_parameter():
    """Test r=4 used consistently."""
    corpus = load_corpus()
    assert corpus["rids_config"]["r"] == 4

    brain_tensor = load_brain_tensor()
    assert brain_tensor["r"] == 4
```

#### 3. Performance Tests
```python
@pytest.mark.benchmark
def test_bootstrap_performance():
    """Bootstrap should complete in <60 seconds."""
    start = time.time()
    brain_tensor = await bootstrap_brain_tensor()
    elapsed = time.time() - start

    assert elapsed < 60.0  # Should be fast (local download)
```

### Test Coverage
- Unit tests: Pattern extraction, adaptation logic
- Integration tests: End-to-end bootstrap
- Validation tests: Functor mapping, r-IDS parameter
- Performance tests: Bootstrap timing

### Success Criteria
- ✅ Bootstrap completes successfully
- ✅ Brain tensor has correct structure (6×100×3)
- ✅ Functor mapping applied correctly
- ✅ r=4 used throughout
- ✅ Cache system works (no duplicate downloads)
- ✅ All tests pass

### Blocking
Blocked by BEAD-QEC-4 (merge2docs endpoints must be implemented first).

---

## 🔵 BEAD-QEC-6: PRIME-DE MRI Data Processing
**Priority**: High
**Dependencies**: BEAD-QEC-5
**Estimated Complexity**: High
**Status**: 🚧 NOT STARTED (waiting on download completion)

### Objective
Process PRIME-DE macaque fMRI dataset and integrate with brain tensor for validation.

### Implementation Tasks

#### 1. Data Loading Pipeline
**File**: `src/backend/data/prime_de_loader.py`

```python
class PRIMEDELoader:
    """Loader for PRIME-DE macaque fMRI dataset."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.atlas = D99Atlas()  # 368 regions

    async def load_subject(self, subject_id: str) -> Dict:
        """Load fMRI data for one subject.

        Returns:
            {
                "subject_id": str,
                "timeseries": np.ndarray,  # (T, N) time points × regions
                "regions": List[str],       # D99 region names
                "tr": float,                # Repetition time (seconds)
                "metadata": Dict
            }
        """
        nifti_path = self.data_dir / f"{subject_id}.nii.gz"
        img = nib.load(nifti_path)

        # Extract ROI timeseries using D99 atlas
        timeseries = self.atlas.extract_timeseries(img)

        return {
            "subject_id": subject_id,
            "timeseries": timeseries,
            "regions": self.atlas.region_names,
            "tr": img.header.get_zooms()[-1],
            "metadata": self._load_metadata(subject_id)
        }
```

#### 2. Connectivity Analysis
```python
async def compute_functional_connectivity(subject_data: Dict) -> np.ndarray:
    """Compute functional connectivity matrix for subject.

    Uses distance correlation (Phase 1 implementation).
    """
    from backend.mri.mri_signal_processing import compute_distance_correlation

    timeseries = subject_data["timeseries"]
    n_regions = timeseries.shape[1]

    # Compute pairwise distance correlations
    conn_matrix = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            dCor = await compute_distance_correlation(
                timeseries[:, i],
                timeseries[:, j]
            )
            conn_matrix[i, j] = dCor
            conn_matrix[j, i] = dCor  # Symmetric

    return conn_matrix
```

#### 3. Brain Tensor Integration
```python
async def populate_brain_tensor_from_prime_de(
    subject_ids: List[str],
    brain_tensor: Dict
) -> Dict:
    """Populate brain tensor with PRIME-DE connectivity data.

    Strategy:
    1. Compute connectivity for each subject
    2. Average across subjects for each region pair
    3. Store in brain_tensor["function"] functor
    """
    loader = PRIMEDELoader(data_dir=PRIME_DE_PATH)

    all_conn_matrices = []

    for subject_id in subject_ids:
        subject_data = await loader.load_subject(subject_id)
        conn_matrix = await compute_functional_connectivity(subject_data)
        all_conn_matrices.append(conn_matrix)

    # Average across subjects
    avg_conn = np.mean(all_conn_matrices, axis=0)

    # Store in brain tensor (function functor)
    for i, region_i in enumerate(subject_data["regions"]):
        for j, region_j in enumerate(subject_data["regions"]):
            if i < j:  # Upper triangle only
                brain_tensor["function"][region_i][region_j] = {
                    "connectivity": avg_conn[i, j],
                    "n_subjects": len(subject_ids),
                    "source": "PRIME-DE"
                }

    return brain_tensor
```

### Test Coverage
- Test PRIME-DE data loading (NIfTI format)
- Test D99 atlas ROI extraction
- Test connectivity computation
- Test brain tensor population
- Validation: Compare to published PRIME-DE papers

### Success Criteria
- ✅ Load PRIME-DE data successfully
- ✅ Extract timeseries for D99 regions
- ✅ Compute connectivity matrices
- ✅ Populate brain tensor function functor
- ✅ Results match published literature

### Notes
**User quote**: "the prime data is almost down just need to service it"

Waiting for download completion before starting implementation.

---

## 🔵 BEAD-QEC-7: AMEM-E Service Routing Point
**Priority**: Medium
**Dependencies**: BEAD-QEC-5
**Estimated Complexity**: Medium
**Status**: 🚧 NOT STARTED

### Objective
Add routing point to `amem_e_service.py` for QEC tensor access within twosphere system.

### Implementation Tasks

#### 1. Routing Point Addition
**File**: `src/backend/services/amem_e_service.py`

```python
class AMEMEService:
    """Adaptive Memory Engine - Enhanced (AMEM-E) Service."""

    def __init__(self):
        self.qec_client = QECTensorClient()
        self.brain_tensor = None

    async def initialize(self):
        """Initialize AMEM-E with brain tensor."""
        # Load cached brain tensor
        if not self.brain_tensor:
            cache_path = QECTensorConfig().corpus_path.parent / "brain_tensor.pkl"

            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.brain_tensor = pickle.load(f)
            else:
                # Bootstrap if not cached
                self.brain_tensor = await bootstrap_brain_tensor()

    async def query_region_tensor(
        self,
        region: str,
        functor: str,
        scale: str = "macro"
    ) -> Dict:
        """Query brain tensor for specific region/functor/scale.

        Args:
            region: Brain region name (D99 atlas)
            functor: Functor name (anatomy, function, electro, etc.)
            scale: Scale name (macro, meso, micro)

        Returns:
            Cell data with features, r-IDS connections, syndrome
        """
        cell_key = f"{functor}_{region}_{scale}"

        if cell_key in self.brain_tensor["cells"]:
            return self.brain_tensor["cells"][cell_key]
        else:
            # Cell not populated - return empty structure
            return {
                "region": region,
                "functor": functor,
                "scale": scale,
                "status": "unpopulated"
            }

    async def get_rids_neighbors(
        self,
        region: str,
        r: int = 4
    ) -> List[str]:
        """Get r-IDS neighbors for a region.

        Returns list of region names within radius r.
        """
        if region in self.brain_tensor["rids_map"]:
            return self.brain_tensor["rids_map"][region]
        else:
            return []

    async def detect_cross_functor_syndrome(
        self,
        region: str
    ) -> Optional[Dict]:
        """Detect inconsistencies across functors for a region.

        Example syndrome:
        - Anatomy says V1→V2 connected
        - Function shows weak correlation
        → Syndrome detected!
        """
        functors = ["anatomy", "function", "electro", "genetics", "behavior"]

        region_data = {}
        for functor in functors:
            cell = await self.query_region_tensor(region, functor)
            region_data[functor] = cell

        # Check for inconsistencies
        syndrome = self._check_functor_consistency(region_data)

        return syndrome if syndrome else None
```

#### 2. MCP Tool Integration
```python
# File: src/backend/mcp/amem_e_tools.py

@mcp_tool
async def query_brain_tensor(
    region: str,
    functor: str,
    scale: str = "macro"
) -> Dict:
    """Query brain QEC tensor for region/functor/scale.

    Args:
        region: Brain region (e.g., "V1", "PFC")
        functor: anatomy, function, electro, genetics, behavior, pathology
        scale: macro, meso, micro

    Returns:
        Cell data with features and r-IDS connections
    """
    amem = AMEMEService()
    await amem.initialize()

    return await amem.query_region_tensor(region, functor, scale)

@mcp_tool
async def list_brain_regions() -> List[str]:
    """List all brain regions in tensor."""
    from backend.services.brain_atlas_client import BrainAtlasClient

    client = BrainAtlasClient("http://localhost:8007")
    regions = client.list_regions(species="macaque", atlas="D99")

    return [r["name"] for r in regions]
```

### Test Coverage
- Test AMEM-E initialization with brain tensor
- Test region tensor queries
- Test r-IDS neighbor lookup
- Test cross-functor syndrome detection
- Integration test: MCP tool invocation

### Success Criteria
- ✅ AMEM-E loads brain tensor on initialization
- ✅ Query interface works for all functors/scales
- ✅ r-IDS neighbor lookup accurate
- ✅ Syndrome detection identifies inconsistencies
- ✅ MCP tools expose functionality

### Integration Points
- Brain tensor: Loaded from cache
- Atlas service: Region name validation
- MCP: Natural language queries

---

## 🟢 BEAD-QEC-8: MCP Tools for Brain Tensor
**Priority**: Medium
**Dependencies**: BEAD-QEC-7
**Estimated Complexity**: Low
**Status**: 🚧 NOT STARTED

### Objective
Expose brain tensor back to merge2docs and other services via MCP tools.

### Implementation Tasks

**File**: `src/backend/mcp/brain_tensor_tools.py`

```python
@mcp_tool
def get_brain_tensor_cell(region: str, functor: str) -> Dict:
    """Get brain tensor cell for region and functor.

    Args:
        region: Brain region name (e.g., "V1", "PFC")
        functor: Functor name (e.g., "anatomy", "function")

    Returns:
        cell: Brain tensor cell data
    """
    amem = AMEMEService()
    cell = await amem.query_region_tensor(region, functor)

    return {
        "region": region,
        "functor": functor,
        "features": cell.features,
        "rids_connections": cell.rids_connections,
        "syndrome": cell.syndrome_mean
    }

@mcp_tool
def list_brain_functors() -> List[str]:
    """List all brain functors."""
    return ["anatomy", "function", "electro", "genetics", "behavior", "pathology"]

@mcp_tool
def get_brain_connectivity_matrix(functor: str = "function") -> np.ndarray:
    """Get full connectivity matrix for a functor.

    Args:
        functor: Functor to query (default: function)

    Returns:
        Connectivity matrix (380×380 for full atlas)
    """
    brain_tensor = load_brain_tensor()

    regions = list(brain_tensor["regions"].keys())
    n = len(regions)
    conn_matrix = np.zeros((n, n))

    for i, region_i in enumerate(regions):
        for j, region_j in enumerate(regions):
            cell_i = brain_tensor["cells"].get(f"{functor}_{region_i}_macro")

            if cell_i and region_j in cell_i["rids_connections"]:
                conn_matrix[i, j] = cell_i["rids_connections"][region_j]["weight"]

    return conn_matrix
```

### Usage from merge2docs
```python
# In merge2docs
from mcp import ClientSession

async with ClientSession("twosphere-mcp") as session:
    # Get V1 function data
    v1_function = await session.call_tool(
        "get_brain_tensor_cell",
        region="V1",
        functor="function"
    )

    # Use in document analysis
    # Map "vision research papers" → V1 function tensor
```

### Test Coverage
- Test MCP tool registration
- Test tool invocation from external client
- Test return data structure
- Integration test: merge2docs queries twosphere

### Success Criteria
- ✅ All MCP tools registered and accessible
- ✅ merge2docs can query brain tensor
- ✅ Response time <1 second for single cell query
- ✅ Natural language interface works

### Use Cases
- merge2docs queries V1 function tensor for vision papers
- Research tool queries pathology functor for disease analysis
- Visualization tool queries full connectivity matrix

---

## 🔵 BEAD-QEC-9: D99 Atlas Integration Enhancement
**Priority**: High
**Dependencies**: BEAD-QEC-5
**Estimated Complexity**: Medium
**Status**: 🚧 NOT STARTED

### Objective
Enhance D99 atlas integration to use r-IDS with r=4 for backbone hub computation.

### Implementation Tasks

**File**: `examples/demo_atlas_hierarchical_brain.py`

**Current Status**: ✅ Already updated to use r-IDS (Phase 7)

**Remaining Tasks**:
1. Validate r-IDS implementation with real D99 graph
2. Compare r-IDS vs. betweenness centrality for hub selection
3. Benchmark performance for 368-region atlas
4. Document hub selection rationale

#### Enhanced Hub Selection
```python
async def compute_backbone_hubs_rids(
    G: nx.Graph,
    r: int = 4,
    use_gpu: bool = False
) -> Set[int]:
    """Compute backbone hubs using r-IDS.

    Args:
        G: Brain connectivity graph
        r: Radius for r-IDS (default: 4, optimal for brain LID≈4-7)
        use_gpu: Use GPU-accelerated algorithm

    Returns:
        Set of hub node IDs
    """
    from backend.integration.merge2docs_bridge import call_algorithm_service

    result = await call_algorithm_service(
        algorithm_name="ids",
        graph_data=G,
        r=r,
        use_gpu=use_gpu
    )

    # Parse result (multiple possible keys)
    for key in ['independent_set', 'dominating_set', 'ids', 'result']:
        if result and key in result:
            return set(result[key])

    # Fallback: betweenness centrality
    logger.warning("r-IDS service unavailable, using betweenness centrality")
    return await compute_backbone_hubs_betweenness(G, fraction=0.10)
```

### Validation Plan
1. **Mathematical validation**: Verify r-IDS properties
   - Independence: No two hubs within distance r
   - Domination: Every node within distance r of some hub

2. **Biological validation**: Compare to known brain hubs
   - PFC, V1, hippocampus should be selected
   - Match literature on macaque brain architecture

3. **Performance validation**: Benchmark 368-region graph
   - r-IDS computation time
   - GPU acceleration benefit

### Test Coverage
```python
def test_rids_hubs_d99():
    """Test r-IDS hub selection on D99 atlas."""
    # Load D99 connectivity
    G = load_d99_connectivity()

    # Compute r-IDS hubs
    hubs = await compute_backbone_hubs_rids(G, r=4)

    # Validate r-IDS properties
    assert verify_independence(G, hubs, r=4)
    assert verify_domination(G, hubs, r=4)

    # Check biological plausibility
    known_hubs = ["PFC", "V1", "hippocampus"]
    for hub in known_hubs:
        assert hub in hubs or adjacent_to_hub(hub, hubs, G)
```

### Success Criteria
- ✅ r-IDS implemented and tested
- ✅ Fallback to betweenness centrality works
- ✅ Hub selection matches biological expectations
- ✅ Performance acceptable for 368 regions

### Files Modified
- `examples/demo_atlas_hierarchical_brain.py` - Already updated ✅
- Tests to be added

---

## Priority Summary

**CRITICAL (Blocking Progress)**:
- 🔵 BEAD-QEC-4: merge2docs Endpoint Implementation - **BLOCKS ALL TESTING**

**High Priority (Core Functionality)**:
- 🔵 BEAD-QEC-5: Bootstrap Testing and Validation
- 🔵 BEAD-QEC-6: PRIME-DE MRI Data Processing
- 🔵 BEAD-QEC-7: AMEM-E Service Routing Point
- 🔵 BEAD-QEC-9: D99 Atlas Integration Enhancement

**Medium Priority (Integration)**:
- 🟢 BEAD-QEC-8: MCP Tools for Brain Tensor

**Completed** ✅:
- 🟢 BEAD-QEC-1: Bootstrap Service Implementation
- 🟢 BEAD-QEC-2: Functor Hierarchies Catalog
- 🟢 BEAD-QEC-3: Advanced Cache and Cross-Training

## Recommended Implementation Order

1. **Week 1**: ~~BEAD-QEC-1, 2, 3~~ ✅ **COMPLETED**
2. **Week 2**: BEAD-QEC-4 (coordinate with merge2docs team)
3. **Week 3**: BEAD-QEC-5 (testing once endpoints live)
4. **Week 4**: BEAD-QEC-6 (PRIME-DE data) + BEAD-QEC-9 (D99 enhancement)
5. **Week 5**: BEAD-QEC-7 (AMEM-E routing) + BEAD-QEC-8 (MCP tools)

## Dependency Graph

```
BEAD-QEC-1 (Bootstrap Service) ✅ COMPLETED
         ↓
BEAD-QEC-4 (merge2docs Endpoints) 🚧 CRITICAL
         ↓
BEAD-QEC-5 (Testing) ──┬──→ BEAD-QEC-6 (PRIME-DE Data)
                       ├──→ BEAD-QEC-7 (AMEM-E Routing)
                       └──→ BEAD-QEC-9 (D99 Enhancement)
                                ↓
                       BEAD-QEC-8 (MCP Tools)

BEAD-QEC-2 (Functor Catalog) ✅ COMPLETED (independent)
BEAD-QEC-3 (Cache Design) ✅ COMPLETED (independent)
```

## Key Technical Decisions

### 1. One-Time Bootstrap Strategy
**Decision**: Download corpus once, build independently
**Rationale**:
- No continuous sync overhead
- Each system optimized for its domain
- Reduces coupling between systems

### 2. Functor Mapping
**Decision**: Map merge2docs functors → brain functors
**Mapping**:
- wisdom → behavior (high-level understanding)
- papers → function (computational role)
- code → anatomy (structure)
- testing → electro (dynamics/validation)
- git → genetics (heritage/evolution)
- New: pathology (disease markers)

**Rationale**: Principled semantic correspondence

### 3. r=4 Parameter
**Decision**: Use r=4 for r-IDS throughout
**Rationale**:
- Matches brain LID (Local Intrinsic Dimension) ≈ 4-7
- Optimal for macaque cortical connectivity
- Consistent with merge2docs tensor_r4

### 4. Cache Strategy
**Decision**: LRU cache with 20-region capacity + smart prefetching
**Rationale**:
- 380 regions too large for RAM (each region has 6 functors × 3 scales)
- r-IDS neighbors likely to be queried together
- Prefetching reduces latency

### 5. Syndrome Detection
**Decision**: Cross-functor syndrome detection at region level
**Types**:
- Cross-functor: Anatomy vs. function mismatch
- Cross-domain: One region inconsistent with neighbors
- Cross-level: Macro/meso/micro scale mismatch

**Rationale**: QEC error detection requires multi-view validation

## Research Applications

### 1. Brain Disease Modeling
- Populate pathology functor with disease markers
- Detect syndromes (inconsistencies across functors)
- Example: Alzheimer's - anatomy intact but function degraded

### 2. Drug Discovery
- Test drug effects on brain tensor
- Cross-validate with merge2docs document tensor
- Identify compounds with consistent multi-scale effects

### 3. Connectomics
- PRIME-DE data → function functor
- Diffusion MRI → anatomy functor
- Cross-validate functional vs. structural connectivity

### 4. Multi-Scale Integration
- Macro: Whole-brain networks
- Meso: Local circuits
- Micro: Cellular connectivity
- Detect scale-crossing patterns

## References

### QEC and Tensor Networks
- Kitaev, A. (2003). "Fault-tolerant quantum computation by anyons"
- Dennis, E. et al. (2002). "Topological quantum memory"

### Brain Connectivity
- Glasser, M. F. et al. (2016). "The Human Connectome Project"
- Reveley, C. et al. (2017). "D99 atlas of macaque brain"

### r-IDS and Graph Theory
- Garey, M. R. & Johnson, D. S. (1979). "Computers and Intractability"
- LID (Local Intrinsic Dimension) in brain networks

### merge2docs Integration
- See `MERGE2DOCS_ENDPOINTS_SPEC.md` for API details
- tensor_matrix.py configuration (r=4, F_i hierarchy)

---

**Next Critical Action**: Coordinate with merge2docs team to implement BEAD-QEC-4 endpoints! This unblocks all testing and validation work.
