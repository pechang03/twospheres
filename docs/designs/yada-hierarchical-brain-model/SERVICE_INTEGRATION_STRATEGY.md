# Service Integration Strategy: merge2docs vs. Brain-Specific Tensor

## Executive Summary

The QEC tensor array in `merge2docs` is powerful but **requires training**. This document outlines when to:
1. **Use merge2docs services** (graph algorithms, r-IDS, QEC)
2. **Build brain-specific components** (fMRI features, anatomical embeddings)
3. **Avoid reimplementation** (leverage existing infrastructure)

## Available merge2docs Services

### Currently Registered Algorithms

From earlier error message:
```
Available: adaptive_selector, bi_cluster_editing_vs, bi_cluster_editing_vs_gpu,
bipartite_builders, bipartite_matching, cluster_editing, cluster_editing_vs,
diff_based_bipartite, dominating_set, enhanced_monte_carlo, enhanced_sat,
feature_selection, feedback_vertex_set, fpt_complexity_analyzer, ids,
nash_equilibrium, no_free_lunch, rb_domination, sat, treewidth,
tripartite_p3_cover, turbo_charging, vertex_cover, gnn_euler_embedding
```

### What We're Already Using ✅

| Algorithm | Purpose | Status |
|-----------|---------|--------|
| `cluster_editing_vs` | Community detection | ✅ Working (Phase 7) |
| `cluster_editing_vs_gpu` | GPU-accelerated clustering | ✅ Available |
| `dominating_set` | Basic dominating set | ✅ Available |
| `ids` | Independent Dominating Set | ✅ Available |

### What We Need but Isn't Registered ⚠️

| Algorithm | Purpose | File Location | Status |
|-----------|---------|---------------|--------|
| `struction_rids` | r-IDS with Struction optimization | `algorithms/struction_rids.py` | ⚠️ **Exists but not registered** |
| `qec_syndrome` | QEC error detection | `algorithms/quantum_fpt_compressor.py` | ⚠️ Needs investigation |
| `yada_gate` | Bio-inspired gating | Part of QMamba | ⚠️ Integrated in model |

## Recommended Strategy

### Option 1: Use Existing Services (RECOMMENDED)

**Principle**: Leverage merge2docs for domain-agnostic graph algorithms, build brain-specific features locally.

```python
# ✅ USE merge2docs services for:
# 1. Graph clustering
communities = await call_algorithm_service(
    algorithm_name="cluster_editing_vs",
    graph_data=G_anatomical,
    k=k_budget,
    use_gpu=True
)

# 2. r-IDS (using 'ids' algorithm with r parameter)
rids = await call_algorithm_service(
    algorithm_name="ids",
    graph_data=G_anatomical,
    r=4  # Radius parameter
)

# 3. Feature selection
selected_features = await call_algorithm_service(
    algorithm_name="feature_selection",
    graph_data=G_features,
    k=30  # Top 30 features
)

# ❌ BUILD locally for brain-specific:
# 1. fMRI preprocessing
timeseries = preprocess_fmri(fmri_path, atlas_mask)  # Our code

# 2. Diffusion CNN features
features = extract_diffusion_features(timeseries)  # Our model

# 3. Brain atlas queries
regions = atlas_client.list_regions(species="macaque")  # Our service
```

**Benefits**:
- ✅ Leverage trained models (cluster-editing-vs has tuned parameters)
- ✅ GPU acceleration automatically handled
- ✅ Minimal maintenance (merge2docs handles updates)
- ✅ Fast development (no reimplementation)

**Limitations**:
- ⚠️ `struction_rids` not registered (need to request registration)
- ⚠️ QEC syndrome requires understanding of V₄ stabilizer implementation

### Option 2: Build Brain-Specific Tensor (Only If Needed)

**When to build our own**:
1. Need brain-specific priors (e.g., anatomical constraints)
2. Multi-modal fusion requires domain knowledge (fMRI + DTI + EEG)
3. Real-time inference (can't afford service latency)
4. Custom training on macaque data

**What to build**:
```python
class BrainTensor:
    """Lightweight brain-specific tensor using merge2docs services."""

    def __init__(self):
        # Axes
        self.modalities = ["anatomy", "fmri", "dti", "eeg"]
        self.systems = ["visual", "motor", "executive", "limbic"]
        self.scales = ["region", "column", "neuron"]

        # ✅ Use merge2docs for graph operations
        self.algorithm_service = AlgorithmServiceClient()

        # ❌ Brain-specific components (build locally)
        self.fmri_encoder = fMRIDiffusionEncoder()
        self.atlas_client = BrainAtlasClient()

    async def compute_rids(self, G: nx.Graph, r=4) -> Set[int]:
        """Compute r-IDS via merge2docs service."""
        # ✅ Delegate to merge2docs
        result = await self.algorithm_service.call(
            algorithm_name="ids",  # Use 'ids' until 'struction_rids' registered
            graph_data=G,
            r=r
        )
        return result.rids

    def extract_features(self, timeseries: np.ndarray) -> np.ndarray:
        """Extract brain-specific features."""
        # ❌ Brain-specific: implement locally
        return self.fmri_encoder(timeseries)
```

### Option 3: Hybrid Approach (BEST FOR NOW)

**Strategy**: Use merge2docs services + lightweight brain adaptations

```python
async def phase7_with_service_integration():
    """Phase 7: Syntactic hierarchy using merge2docs services."""

    # 1. ✅ Get anatomical data (brain-specific)
    atlas_client = BrainAtlasClient("http://localhost:8007")
    regions = atlas_client.list_regions(species="macaque", atlas="D99")
    G_anatomical, positions = build_anatomical_graph(regions, atlas_client)

    # 2. ✅ Cluster via merge2docs (domain-agnostic)
    communities = await call_algorithm_service(
        algorithm_name="cluster_editing_vs",
        graph_data=G_anatomical,
        k=auto_tune_k(G_anatomical),
        use_gpu=True
    )

    # 3. ⚠️ Compute backbone (workaround until struction_rids registered)
    # Option A: Use 'ids' with r=4
    rids_result = await call_algorithm_service(
        algorithm_name="ids",
        graph_data=G_anatomical,
        r=4
    )
    backbone = rids_result.independent_set

    # Option B: Use betweenness centrality (fallback)
    if not backbone:
        centrality = nx.betweenness_centrality(G_anatomical)
        n_hubs = max(5, int(len(G_anatomical.nodes()) * 0.10))
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        backbone = {node for node, _ in sorted_nodes[:n_hubs]}

    return G_anatomical, communities, backbone


async def phase8_with_service_integration(fmri_path: str):
    """Phase 8: Semantic hierarchy using hybrid approach."""

    # 1. ❌ Preprocess fMRI (brain-specific)
    timeseries = preprocess_fmri(fmri_path, atlas_mask="D99")

    # 2. ❌ Extract features (brain-specific, needs trained diffusion CNN)
    # Option A: Use pre-trained diffusion CNN (if we have it)
    # features = diffusion_cnn_encoder(timeseries)

    # Option B: PCA baseline (simple, no training needed)
    from sklearn.decomposition import PCA
    correlation = np.corrcoef(timeseries)
    pca = PCA(n_components=50)
    features = pca.fit_transform(correlation)

    # 3. ✅ Build functional graph (domain-agnostic)
    G_functional = build_graph_from_features(features, threshold=0.3)

    # 4. ✅ Cluster via merge2docs (domain-agnostic)
    communities = await call_algorithm_service(
        algorithm_name="cluster_editing_vs",
        graph_data=G_functional,
        k=auto_tune_k(G_functional),
        use_gpu=True
    )

    # 5. ✅ Feature selection via merge2docs (domain-agnostic)
    selected_features = await call_algorithm_service(
        algorithm_name="feature_selection",
        graph_data=G_functional,
        k=30  # Top 30 features (instead of 100)
    )

    return G_functional, communities, selected_features
```

## What to Request from merge2docs Team

### Priority 1: Register `struction_rids`

**File**: `/Users/petershaw/code/aider/merge2docs/src/backend/algorithms/struction_rids.py`

**Request**:
```python
# Add to factory.py
from src.backend.algorithms.struction_rids import StructionRIDSAlgorithm

ALGORITHMS = {
    # ... existing algorithms ...
    "struction_rids": StructionRIDSAlgorithm,
}
```

**Why**: Provides optimal r-IDS (r=4) with GPU acceleration and Struction optimization

**Alternative**: Use `ids` algorithm with `r` parameter until registration

### Priority 2: Access to QEC Syndrome Detection

**Files**:
- `algorithms/quantum_fpt_compressor.py`
- `design-qec-comorag-yadamamba.md`

**Request**: Understand V₄ stabilizer interface:
```python
# What we need
syndrome = await call_algorithm_service(
    algorithm_name="qec_syndrome",  # Does this exist?
    state=neural_state,
    stabilizer="V4"
)
```

**Why**: Prediction error detection for iterative fMRI correction

**Alternative**: Implement simple prediction error locally:
```python
def compute_syndrome_simple(predicted, actual):
    """Simple prediction error as syndrome proxy."""
    return np.abs(predicted - actual)
```

### Priority 3: QMamba Integration

**Status**: QMamba appears to be integrated into larger model (not standalone algorithm)

**Question**: Can we access YadaGate component separately?

**Alternative**: Implement simple winner-take-all as proxy:
```python
def yada_gate_simple(predictions, threshold=0.7):
    """Simple winner-take-all as Yada gate proxy."""
    max_prob = predictions.max(axis=-1, keepdims=True)
    mask = (predictions >= threshold * max_prob).astype(float)
    return predictions * mask
```

## Decision Matrix

| Component | Use merge2docs? | Build Locally? | Rationale |
|-----------|----------------|----------------|-----------|
| **Graph Clustering** | ✅ Yes (`cluster_editing_vs`) | ❌ No | Trained, GPU-accelerated |
| **r-IDS Backbone** | ⚠️ Request `struction_rids` | ✅ Fallback: betweenness | Exists but not registered |
| **QEC Syndrome** | ❌ Complex integration | ✅ Simple proxy | Need to understand V₄ interface |
| **fMRI Preprocessing** | ❌ Domain-specific | ✅ Yes (nilearn) | Brain imaging specific |
| **Diffusion CNN** | ❌ Needs training | ✅ PCA baseline first | Start simple, upgrade later |
| **Brain Atlas** | ❌ Already have MCP | ✅ Yes (local service) | Custom for D99/Allen/Waxholm |
| **Feature Selection** | ✅ Yes | ❌ No | Generic algorithm |
| **Yada Gate** | ❌ Model-internal | ✅ Simple proxy | Winner-take-all sufficient |

## Recommended Action Plan

### Immediate (Phase 7 Update)
1. ✅ Continue using `cluster_editing_vs` for communities
2. ⚠️ Use `ids` algorithm with `r=4` OR betweenness centrality for backbone
3. ✅ Keep brain atlas MCP service

### Short-term (Phase 8)
1. ❌ Implement fMRI preprocessing with nilearn (brain-specific)
2. ✅ Use PCA for baseline features (no training needed)
3. ✅ Use merge2docs `feature_selection` for dimensionality reduction
4. ❌ Simple prediction error as syndrome proxy

### Long-term (Phase 9-10)
1. ⚠️ Request `struction_rids` registration in merge2docs
2. ⚠️ Investigate QEC syndrome interface
3. ❌ Train diffusion CNN on macaque fMRI (research project)
4. ✅ Use merge2docs for all graph operations

## Conclusion

**Recommended Approach**: **Hybrid (Option 3)**

- ✅ **Use merge2docs services** for: Graph clustering, feature selection, r-IDS (when registered)
- ❌ **Build locally** for: fMRI preprocessing, brain atlas, diffusion CNN features
- ⚠️ **Request** from merge2docs: `struction_rids` registration, QEC syndrome interface

**Why Hybrid**:
- Leverages existing trained models (cluster-editing-vs)
- Avoids reimplementing complex graph algorithms
- Allows brain-specific customization where needed
- Fast development (can start immediately with PCA baseline)
- Upgrade path clear (replace PCA with diffusion CNN later)

**Do NOT build**:
- ❌ Full QEC tensor array (too complex, requires training)
- ❌ Graph clustering algorithms (merge2docs has better versions)
- ❌ r-IDS from scratch (use service or simple fallback)

**Key Insight**: The QEC tensor architecture is **framework**, not **code to copy**. Use the *principles* (fractal self-similarity, r-IDS, syndrome detection) with *services* (merge2docs) + *domain data* (brain imaging).

---

**Status**: Hybrid strategy recommended, using merge2docs services for graph operations
**Next**: Update Phase 7 to use `ids` service with r=4, implement Phase 8 with PCA baseline
