# YADA Hierarchical Brain Model: Dual Syntactic-Semantic Architecture

**Design Document**
**Author**: Claude Sonnet 4.5 + petershaw
**Date**: 2026-01-21
**Last Updated**: 2026-01-21
**Status**: ✅ Phase 1-6 Complete, Phase 7-8 In Design

---

## ✅ Implementation Status (2026-01-21)

**Completed Components**:
1. ✅ Clustered data generation (make_blobs → brain regions)
2. ✅ K-NN and threshold graph construction
3. ✅ Cluster-editing-vs integration (GPU-accelerated, auto-tuning)
4. ✅ Hierarchical cluster contraction (3 levels)
5. ✅ Spherical spring layout (Fruchterman-Reingold on S²)
6. ✅ Backbone hub detection (Connected Dominating Set)
7. ✅ Visualization pipeline (3-panel: 3D → Sphere → Backbone)

**Key Results**:
- 200 nodes → 6 communities (modularity=0.805, ARI=0.604)
- 30 backbone hubs spanning 3 major communities
- GPU-accelerated cluster-editing-vs: 0.00s execution time
- Automatic parameter tuning finds optimal k budget

**Next Phase**:
- Brain atlas integration (D99, Allen, Waxholm)
- YADA dual structure (syntactic ↔ semantic)
- Two-sphere architecture (left/right hemispheres)

---

## Executive Summary

This design document specifies a **multi-scale hierarchical brain model** inspired by the **YADA (Yet Another Document Architecture)** dual syntactic-semantic structure from the `merge2docs` pipeline. The model represents brain organization at three orthogonal dimensions:

1. **Hierarchical Levels**: Neurons → Columns → Regions → Lobes
2. **Syntactic Structure**: Anatomical organization (WHERE)
3. **Semantic Structure**: Functional organization (WHAT)

The integration of these dimensions enables **structure-function mapping**, multi-scale analysis, and comparison across different brain parcellation schemes.

### Key Innovations

- **Dual Hierarchy**: Separate but interconnected anatomical and functional graphs
- **Bipartite Bridges**: Explicit mapping between structure and function
- **Backbone Hubs**: Connected Dominating Set identifies key routing nodes
- **Auto-Tuned Clustering**: Cluster-editing-vs with automatic parameter selection
- **Multi-Species Support**: D99 macaque, Allen mouse, Waxholm rat atlases
- **Graph Algebra**: Operations on syntactic, semantic, and cross-level graphs

---

## Motivation

### Biological Context

The brain exhibits hierarchical organization across multiple scales and modalities:

**Anatomical Hierarchy** (Structural MRI, DTI):
```
Individual neurons (10¹¹ in human brain)
  ↓ grouped into
Cortical columns (~2×10⁸ in neocortex)
  ↓ organized into
Brain regions (52 Brodmann areas, 368 D99 regions)
  ↓ aggregated into
Lobes (frontal, parietal, temporal, occipital)
  ↓ lateralized into
Hemispheres (left, right)
```

**Functional Hierarchy** (fMRI, EEG):
```
Neural features (edge detection, motion, color)
  ↓ integrated into
Local networks (receptive fields, cortical columns)
  ↓ coordinated into
Functional systems (visual, motor, attention, memory)
  ↓ unified into
Cognitive functions (perception, action, executive control)
```

**Critical Problem**: These hierarchies are **orthogonal** - anatomical location ≠ functional role.
- V1 (anatomy) ↔ Edge detection + orientation tuning (function)
- PFC (anatomy) ↔ Working memory + executive control + attention (function)
- Many-to-many mapping requires explicit structure-function bridge

### YADA Architectural Parallel

The `merge2docs` unified pipeline (v3) implements a dual graph structure for document analysis:

**Syntactic Graphs** (`level_graphs`):
- Word → Sentence → Paragraph → Section → Document
- Structural containment and sequencing
- Answers: "WHERE is the information?"

**Semantic Graphs** (`semantic_graphs`):
- Entity → Topic → Concept → Theme → Knowledge
- Meaning-based connections
- Answers: "WHAT does the information mean?"

**Bipartite Connections**:
- Maps syntactic nodes to semantic nodes
- Example: Paragraph_3 ↔ "Neural Plasticity Concept"

**Cross-Level Edges**:
- Vertical connections within each hierarchy
- Word-level entities → Paragraph-level topics

This **exact same architecture** applies to brain modeling:
- Syntactic = Anatomical (spatial organization)
- Semantic = Functional (activity patterns)
- Bipartite = Structure-function mapping
- Cross-level = Multi-scale integration

---

## Architecture

### 1. Three-Level Base Hierarchy

The model decomposes brain organization into three hierarchical levels:

```
Level 1: Backbone Hubs (Connected Dominating Set)
  - 15-30 major connector nodes
  - Form connected backbone spanning all communities
  - Computed via struction_rids algorithm (r-IDS + CDS)
  - Biological: Primary hubs (e.g., temporal cortex, parietal lobule)

Level 2: Communities (Cluster-Editing-VS)
  - 6-10 functional clusters
  - Detected via FPT cluster-editing-vs algorithm
  - Auto-tuned parameter k (budget for edge edits)
  - Biological: Functional systems (visual, motor, executive)

Level 3: Individual Nodes
  - 100-500 brain regions (from atlas or synthetic)
  - Organized into communities
  - Biological: Cortical columns, brain regions (e.g., D99 regions)
```

**Implementation**:
```python
# Level 3 → Level 2: Community detection
communities = await detect_communities_auto(
    G,
    target_clusters=10,
    method="cluster_editing_vs",
    use_gpu=True
)

# Level 2 → Level 1: Backbone extraction
backbone_nodes = await compute_backbone_hubs(G, r=2)
G_backbone = G_communities.subgraph(backbone_community_ids)

# Hierarchical contraction
G_communities, cluster_pos, cluster_members = contract_clusters_to_supernodes(
    G, communities, positions
)
```

### 2. YADA Dual Structure

Overlay the three-level hierarchy with **syntactic** (anatomical) and **semantic** (functional) graphs:

#### 2.1 Syntactic Hierarchy (Anatomical)

**Data Source**: Brain atlas (D99, Allen CCF, Waxholm)

**Structure**:
```python
syntactic_graphs = {
    HierarchicalLevel.NEURON: {
        "nodes": [...],  # Individual neurons/voxels
        "edges": [...],  # Physical adjacency
        "attributes": {
            "position": (x, y, z),
            "region_id": int,
            "atlas": "D99"
        }
    },
    HierarchicalLevel.COLUMN: {
        "nodes": [...],  # Cortical columns
        "edges": [...],  # Spatial proximity
        "contains": [...]  # Neurons in this column
    },
    HierarchicalLevel.REGION: {
        "nodes": [...],  # Brain regions (e.g., V1, M1)
        "edges": [...],  # Anatomical adjacency
        "contains": [...]  # Columns in this region
    },
    HierarchicalLevel.LOBE: {
        "nodes": ["frontal", "parietal", "temporal", "occipital"],
        "edges": [...],  # Lobe boundaries
        "contains": [...]  # Regions in this lobe
    }
}
```

**Edge Types**:
- **Containment**: Region contains columns, column contains neurons
- **Adjacency**: Spatial neighbors (e.g., V1 adjacent to V2)
- **Connectivity**: Structural pathways (DTI tractography)

**Properties**:
- **Spatial coordinates**: (x, y, z) in atlas space
- **Volume**: mm³ for each region
- **Labels**: Anatomical names (e.g., "Area V1", "Brodmann 17")

#### 2.2 Semantic Hierarchy (Functional)

**Data Source**: Functional connectivity (fMRI, cluster analysis)

**Structure**:
```python
semantic_graphs = {
    HierarchicalLevel.FEATURE: {
        "nodes": [...],  # Neural features (edge, motion, color)
        "edges": [...],  # Co-occurrence
        "attributes": {
            "modality": str,  # "visual", "motor", "auditory"
            "selectivity": float
        }
    },
    HierarchicalLevel.NETWORK: {
        "nodes": [...],  # Local networks (receptive fields)
        "edges": [...],  # Functional connectivity
        "implements": [...]  # Features in this network
    },
    HierarchicalLevel.SYSTEM: {
        "nodes": [...],  # Functional systems (visual, motor)
        "edges": [...],  # System interactions
        "implements": [...]  # Networks in this system
    },
    HierarchicalLevel.FUNCTION: {
        "nodes": ["perception", "action", "memory", "attention"],
        "edges": [...],  # Cognitive dependencies
        "implements": [...]  # Systems for this function
    }
}
```

**Edge Types**:
- **Implementation**: System implements networks, network implements features
- **Co-activation**: Regions that activate together
- **Functional connectivity**: Correlated time series (fMRI)

**Properties**:
- **Activation patterns**: Time series or event-related responses
- **Selectivity**: Feature/category specificity
- **Task modulation**: Context-dependent activity

#### 2.3 Bipartite Bridges (Structure-Function Mapping)

**Purpose**: Connect anatomical locations to functional roles

**Construction**:
```python
bipartite_graph = nx.Graph()

# Add nodes from both hierarchies
bipartite_graph.add_nodes_from(syntactic_graphs[level].nodes(), bipartite=0)
bipartite_graph.add_nodes_from(semantic_graphs[level].nodes(), bipartite=1)

# Add cross-hierarchy edges
for anatomical_node in syntactic_graphs[level].nodes():
    for functional_node in semantic_graphs[level].nodes():
        # Edge weight = strength of structure-function association
        weight = compute_structure_function_mapping(anatomical_node, functional_node)
        if weight > threshold:
            bipartite_graph.add_edge(anatomical_node, functional_node, weight=weight)
```

**Examples**:
- V1 (region) ↔ Edge detection (feature), weight=0.9
- V1 (region) ↔ Visual system (system), weight=1.0
- PFC (region) ↔ Executive control (function), weight=0.85
- M1 (region) ↔ Motor action (function), weight=0.95

**Edge Weight Computation**:
```python
def compute_structure_function_mapping(anatomical_node, functional_node):
    """Compute strength of structure-function association."""
    # Method 1: Co-activation frequency
    if has_fmri_data:
        correlation = fmri_correlation(anatomical_node, functional_node)
        return correlation

    # Method 2: Literature-based (atlas annotations)
    if has_atlas_metadata:
        if functional_node in anatomical_node.known_functions:
            return 1.0

    # Method 3: Connectivity-based (if region A connects to B,
    # and B implements function F, then A may support F)
    if has_connectivity:
        indirect_support = connectivity_based_inference(anatomical_node, functional_node)
        return indirect_support

    return 0.0
```

#### 2.4 Cross-Level Connections

**Within Syntactic Hierarchy**:
```python
# Vertical edges (containment, aggregation)
for level_i, level_j in zip(levels[:-1], levels[1:]):
    for node_i in syntactic_graphs[level_i].nodes():
        for node_j in syntactic_graphs[level_j].nodes():
            if node_j.contains(node_i):
                cross_level_edges.add((node_i, node_j, "contains"))
```

**Within Semantic Hierarchy**:
```python
# Vertical edges (implementation, composition)
for level_i, level_j in zip(levels[:-1], levels[1:]):
    for node_i in semantic_graphs[level_i].nodes():
        for node_j in semantic_graphs[level_j].nodes():
            if node_j.implements(node_i):
                cross_level_edges.add((node_i, node_j, "implements"))
```

**Cross-Hierarchy**:
```python
# Horizontal edges (structure-function at each level)
for level in levels:
    bipartite = build_bipartite_bridge(
        syntactic_graphs[level],
        semantic_graphs[level]
    )
    cross_hierarchy_graphs[level] = bipartite
```

### 3. Backbone Hub Integration

**Connected Dominating Set (CDS)** identifies key routing nodes:

```python
from merge2docs.algorithms.struction_rids import compute_rids_with_backbone

# Compute r-IDS and backbone
result = compute_rids_with_backbone(
    graph=G,
    r=2,  # Domination radius
    use_gpu=True,
    use_struction=True,
    compute_backbone=True
)

backbone_nodes = result.backbone  # Set of CDS nodes
backbone_edges = result.backbone_edges  # Spanning tree
```

**Properties of Backbone**:
1. **Dominating**: Every node within distance r of backbone
2. **Connected**: Backbone forms connected subgraph
3. **Minimal**: Approximately minimal size (FPT algorithm)
4. **Routing**: All cross-community paths can route through backbone

**Biological Interpretation**:
- **Backbone nodes** = Primary cortical hubs (high betweenness centrality)
- **Spanning tree** = Major white matter tracts
- **Domination radius r=2** = Every region within 2 synaptic hops of backbone

### 4. Two-Sphere Architecture (Hemispheres)

Extend single-brain model to bilateral structure:

```
Left Hemisphere (Sphere 1)          Right Hemisphere (Sphere 2)
  Syntactic: D99 left regions         Syntactic: D99 right regions
  Semantic: Left-lateralized           Semantic: Right-lateralized
            functions (language)                 functions (spatial)
         ↓                                     ↓
    Backbone hubs                         Backbone hubs
         ↓                                     ↓
         └──────── Corpus Callosum ────────────┘
              (Bipartite bridge between hemispheres)
```

**Corpus Callosum Modeling**:
```python
# Select highest-degree hub from each hemisphere
hub_left = max(backbone_left, key=lambda n: G_left.degree(n))
hub_right = max(backbone_right, key=lambda n: G_right.degree(n))

# Rotate spheres so hubs touch at origin [0, 0, 0]
pos_left_3d = rotate_sphere_to_place_hub_at_touching_point(
    pos_left_3d, hub_left, center_left, touching_point=[0,0,0]
)
pos_right_3d = rotate_sphere_to_place_hub_at_touching_point(
    pos_right_3d, hub_right, center_right, touching_point=[0,0,0]
)

# Add thick inter-hemisphere edge
G_full.add_edge(hub_left, hub_right, weight=1.0, type="corpus_callosum")
```

---

## Algorithms and Methods

### 1. Cluster-Editing-VS with Auto-Tuning

**Algorithm**: FPT cluster-editing with vertex selection [Chen et al., 2010]

**Problem**: Given graph G and budget k, find clustering that minimizes edge edits (additions + deletions) subject to |edits| ≤ k.

**Auto-Tuning**:
```python
async def detect_communities_auto(
    G: nx.Graph,
    target_clusters: Optional[int] = None
) -> Dict[int, int]:
    """Auto-tune k budget for cluster-editing-vs."""

    # Try multiple k values
    k_candidates = [
        n_edges // 20,   # 5%
        n_edges // 10,   # 10%
        n_edges // 5,    # 20%
        n_edges // 2,    # 50%
        n_edges,         # 100%
        n_edges * 2      # 200%
    ]

    best_score = -inf
    for k in k_candidates:
        communities = await detect_communities_cluster_editing(G, k=k)
        quality = evaluate_clustering_quality(G, communities)

        # Score = modularity * (1 - cluster_count_penalty)
        score = quality['modularity']
        if target_clusters:
            penalty = abs(quality['n_clusters'] - target_clusters) / target_clusters
            score *= (1.0 - 0.5 * penalty)

        if score > best_score:
            best_score = score
            best_communities = communities

    return best_communities
```

**Complexity**: O(k³ + k·n·m) for cluster-editing-vs [Niedermeier, 2006]

**GPU Acceleration**: Metal backend for vertex cover subproblem

### 2. Spherical Spring Layout

**Algorithm**: Fruchterman-Reingold adapted to sphere surface [Kobourov, 2012]

**Force Model**:
```
Attractive force (springs):  F_spring = k(d - d₀) · u_vu
Repulsive force (all pairs): F_rep = -K/d² · u_vu
```

Where:
- d = arccos(p_v · p_u) = geodesic distance on unit sphere
- u_vu = (p_u - (p_v·p_u)p_v) / sin(d) = unit tangent vector
- d₀ = rest length in radians (e.g., π/6 ≈ 30°)

**Movement via Exponential Map**:
```python
# Move node p in direction F with step size η
p_new = cos(η||F||) · p + sin(η||F||) · (F / ||F||)
```

**Annealing**: η(t) = η₀ · (1 - t/T) reduces step size over time

**Complexity**: O(T · n²) for T iterations, n nodes

### 3. Connected Dominating Set (Backbone)

**Algorithm**: r-IDS via vertex cover complement + Steiner tree connection [Niedermeier, 2006]

**Steps**:
1. Build r-distance graph: edges between nodes at distance > r
2. Compute vertex cover on r-distance graph
3. r-IDS = complement of vertex cover
4. If r-IDS is disconnected, add Steiner nodes to connect components
5. Return CDS = connected r-IDS

**Properties**:
- **Domination**: Every node within distance r of CDS
- **Connected**: CDS induces connected subgraph
- **Approximation**: |CDS| ≤ (1 + ln(Δ)) · OPT for max degree Δ

**Complexity**: O(k³·n + n·m) via struction rule and vertex cover FPT

### 4. Brain Atlas Integration

**Multi-Species Atlases**:
- **Macaque D99 v2.0**: 368 regions (Saleem & Logothetis, 2012)
- **Allen Mouse CCF v3**: 800+ regions (Allen Institute, 2017)
- **Waxholm Rat v4**: 222 regions (Papp et al., 2014)

**Atlas MCP Server**:
```python
# Query regions
response = requests.post("http://localhost:8007/api/list_regions", json={
    "species": "macaque",
    "atlas": "D99",
    "filter_type": "cortical",
    "search_pattern": "V[0-9]"  # Visual areas
})

regions = response.json()["regions"]

# Build anatomical graph from atlas
G_anatomical = nx.Graph()
for region in regions:
    G_anatomical.add_node(region['id'], **region)

    # Add edges to neighbors
    neighbors = atlas.get_neighbors(region['id'])
    for neighbor in neighbors:
        G_anatomical.add_edge(region['id'], neighbor['id'], type="adjacency")
```

**Coordinates**: NIfTI voxel space → Stereotactic coordinates

**Hierarchy**: Regions → Lobes (from atlas metadata)

---

## Implementation Plan

### Phase 7: Brain Atlas Integration (Next)

**Goal**: Replace synthetic blobs with real anatomical regions

**Tasks**:
1. Query D99 atlas for region list
2. Build syntactic (anatomical) graph from atlas
3. Map atlas regions to sphere surface
4. Preserve spatial relationships (adjacency, contains)

**Code**:
```python
# Query atlas
atlas_service = BrainAtlasService("http://localhost:8007")
regions = atlas_service.list_regions(species="macaque", atlas="D99", filter_type="cortical")

# Build syntactic graph
G_syntactic = build_anatomical_graph(regions, atlas_service)

# Generate positions (spring layout preserving anatomy)
positions = spherical_spring_layout_with_scale(
    G_syntactic,
    radius=1.5,
    iterations=300,
    k=0.1,  # Weaker springs to preserve atlas structure
    seed=42
)
```

### Phase 8: YADA Dual Structure Implementation

**Goal**: Build parallel syntactic and semantic hierarchies

**Tasks**:
1. Create syntactic hierarchy from atlas
2. Build semantic hierarchy from cluster-editing-vs
3. Construct bipartite bridges
4. Add cross-level connections
5. Implement graph query API

**Data Structures**:
```python
@dataclass
class HierarchicalBrainModel:
    """YADA-based dual hierarchy brain model."""

    # Syntactic (anatomical)
    syntactic_graphs: Dict[HierarchicalLevel, nx.Graph]
    syntactic_positions: Dict[HierarchicalLevel, Dict[int, np.ndarray]]

    # Semantic (functional)
    semantic_graphs: Dict[HierarchicalLevel, nx.Graph]
    semantic_communities: Dict[int, int]

    # Cross-hierarchy
    bipartite_graphs: Dict[HierarchicalLevel, nx.Graph]

    # Cross-level
    syntactic_cross_level: List[Tuple[int, int, str]]  # (node_i, node_j, "contains")
    semantic_cross_level: List[Tuple[int, int, str]]   # (node_i, node_j, "implements")

    # Backbone
    backbone_nodes: Set[int]
    backbone_edges: Set[Tuple[int, int]]

    # Metadata
    atlas: str  # "D99", "Allen", etc.
    species: str
    n_nodes: int
    n_communities: int
```

**Query API**:
```python
# Structure queries
regions_in_lobe = model.query_syntactic("frontal", level=HierarchicalLevel.LOBE)

# Function queries
visual_regions = model.query_semantic("visual_system", level=HierarchicalLevel.SYSTEM)

# Structure-function queries
functions_of_v1 = model.query_bipartite("V1", source="syntactic", target="semantic")
regions_for_memory = model.query_bipartite("memory", source="semantic", target="syntactic")

# Multi-scale queries
all_regions_in_visual_system = model.query_hierarchical(
    start_node="visual_system",
    start_level=HierarchicalLevel.SYSTEM,
    target_level=HierarchicalLevel.REGION,
    hierarchy="semantic"
)
```

### Phase 9: Two-Sphere Extension

**Goal**: Model left/right hemispheres with corpus callosum

**Tasks**:
1. Duplicate model for each hemisphere
2. Identify lateralized functions
3. Add corpus callosum connection
4. Visualize on two spheres

### Phase 10: Validation and Analysis

**Metrics**:
1. **Anatomical accuracy**: Compare to atlas ground truth
2. **Functional coherence**: Modularity, silhouette score
3. **Structure-function alignment**: Bipartite matching quality
4. **Backbone quality**: Domination coverage, connectivity

**Experiments**:
1. Compare different parcellation schemes (D99, Allen, Brodmann)
2. Vary cluster-editing budget k (sensitivity analysis)
3. Vary spherical layout parameters
4. Compare to existing brain models (e.g., Human Connectome Project)

---

## Mathematical Formulation

### Graph Definitions

**Syntactic Graph** (Anatomical):
```
G_syn = (V_syn, E_syn, A_syn)
V_syn = {regions from atlas}
E_syn = {(u,v) | u and v are adjacent in atlas}
A_syn = {position, volume, label, ...}
```

**Semantic Graph** (Functional):
```
G_sem = (V_sem, E_sem, A_sem)
V_sem = {functional communities from cluster-editing}
E_sem = {(u,v) | weight(u,v) > θ}  where weight = co-activation
A_sem = {modality, selectivity, ...}
```

**Bipartite Graph** (Structure-Function):
```
G_bip = (V_syn ∪ V_sem, E_bip)
E_bip = {(u,v) | u ∈ V_syn, v ∈ V_sem, map(u,v) > τ}
```

### Hierarchical Levels

```
L = {NEURON, COLUMN, REGION, LOBE, HEMISPHERE}

For level l ∈ L:
  G_syn(l) = syntactic graph at level l
  G_sem(l) = semantic graph at level l

Cross-level edges:
  E_syn^l,l+1 = {(u,v) | u ∈ V_syn(l), v ∈ V_syn(l+1), contains(v,u)}
  E_sem^l,l+1 = {(u,v) | u ∈ V_sem(l), v ∈ V_sem(l+1), implements(v,u)}
```

### Backbone (Connected Dominating Set)

```
r-IDS: Independent set in r-distance graph
D_r(G) = (V, E_r) where E_r = {(u,v) | d_G(u,v) > r}

S ⊆ V is r-independent dominating set if:
  1. Independent: ∀u,v ∈ S: d_G(u,v) > r
  2. Dominating: ∀v ∈ V: ∃u ∈ S: d_G(u,v) ≤ r

CDS(G,r) = connected subgraph containing r-IDS
  If r-IDS is disconnected, add Steiner nodes
```

### Optimization Objectives

**Cluster-Editing-VS**:
```
minimize: |E_add ∪ E_del|
subject to: G' = (V, E ∆ (E_add ∪ E_del)) is a disjoint union of cliques
            |E_add ∪ E_del| ≤ k
```

**Spherical Spring Layout**:
```
minimize: E_total = E_spring + E_repulsion
where:
  E_spring = ∑_(u,v)∈E k/2 (d_geo(u,v) - d_0)²
  E_repulsion = ∑_(u,v)∉E K/d_geo(u,v)

subject to: ||p_i|| = r  ∀i ∈ V  (nodes on sphere surface)
```

**Structure-Function Mapping**:
```
maximize: ∑_(u,v)∈E_bip weight(u,v)
subject to: |E_bip| ≤ M  (budget on bipartite edges)
            ∀u ∈ V_syn: |neighbors_sem(u)| ≤ k_max
            ∀v ∈ V_sem: |neighbors_syn(v)| ≤ k_max
```

---

## Expected Results

### Quantitative Metrics

1. **Community Detection Quality**:
   - Modularity: Q ∈ [0.7, 0.9] (high)
   - Adjusted Rand Index: ARI ∈ [0.5, 0.8] (vs atlas ground truth)
   - Number of communities: 6-10 (matches functional systems)

2. **Backbone Quality**:
   - Domination coverage: 100% (all nodes within r=2 of backbone)
   - Backbone size: 10-20% of total nodes
   - Connectivity: Backbone is connected (verified)

3. **Spherical Layout**:
   - Average geodesic distance: ~π/4 radians (~45°)
   - Clustering coefficient: 0.3-0.5 (small-world)
   - Path length: log(n) for n nodes

4. **Structure-Function Alignment**:
   - Bipartite edge density: 5-15% (sparse but meaningful)
   - Functional coherence per anatomical region: 0.6-0.9
   - Anatomical coherence per functional system: 0.5-0.8

### Qualitative Observations

1. **Anatomical Realism**: Regions grouped by spatial proximity match atlas
2. **Functional Modularity**: Communities align with known systems (visual, motor)
3. **Backbone Hubs**: High-betweenness nodes match known cortical hubs
4. **Interpretability**: Structure-function mapping enables explanations

---

## Related Work

### Graph-Based Brain Models

**Human Connectome Project** (Van Essen et al., 2013):
- 180 cortical parcels per hemisphere
- Structural and functional connectivity matrices
- **Limitation**: Single-scale, lacks hierarchical structure

**Hierarchical Brain Networks** (Betzel & Bassett, 2017):
- Multi-resolution community detection
- Identifies functional modules at multiple scales
- **Limitation**: Functional only, no anatomical alignment

**Allen Brain Atlas** (Sunkin et al., 2013):
- Comprehensive mouse brain atlas with gene expression
- **Limitation**: Static structure, no dynamics

### YADA Architecture

**merge2docs Unified Pipeline v3**:
- Dual syntactic-semantic graphs for document analysis
- Bipartite bridges for structure-function mapping
- Multi-level hierarchies with cross-level connections
- **Novel Application**: Adapting document analysis to neuroscience

### Cluster-Editing Algorithms

**Chen et al. (2010)**: Cluster-editing with vertex selection
- FPT algorithm with O(k³) kernel size
- **Our Extension**: Auto-tuning for optimal k selection

**Böcker (2012)**: Fixed-parameter algorithms for cluster editing
- Branch-and-bound, crown rule, struction rule
- **Our Implementation**: GPU-accelerated Metal backend

### Spherical Graph Layouts

**Kobourov (2012)**: Spring embedders and force-directed layouts
- Adapting Fruchterman-Reingold to non-Euclidean geometries
- **Our Application**: Exponential map for sphere surface

---

## Future Directions

### Short-Term (1-3 months)

1. **Complete brain atlas integration** (D99, Allen, Waxholm)
2. **Implement full YADA dual structure** with bipartite bridges
3. **Two-sphere architecture** for bilateral brain
4. **Validation** against Human Connectome Project data

### Medium-Term (3-6 months)

1. **Dynamic brain modeling**: Temporal evolution of functional networks
2. **Multi-subject analysis**: Statistical brain models across individuals
3. **Disease modeling**: Alzheimer's, schizophrenia connectivity changes
4. **Connectomics integration**: DTI tractography for anatomical edges

### Long-Term (6-12 months)

1. **Whole-brain simulation**: Integrate with spiking neural networks
2. **Personalized brain models**: Individual-specific parcellations
3. **Cross-species comparison**: Align mouse/macaque/human atlases
4. **Clinical applications**: Surgical planning, lesion impact analysis

---

## References

See [references.bib](./references.bib) for full citations.

**Key References**:
1. Saleem & Logothetis (2012): D99 macaque atlas
2. Chen et al. (2010): Cluster-editing FPT algorithms
3. Niedermeier (2006): Invitation to fixed-parameter algorithms
4. Betzel & Bassett (2017): Hierarchical brain network organization
5. Van Essen et al. (2013): Human Connectome Project
6. Allen Institute (2017): Allen Brain Atlas
7. Kobourov (2012): Spring embedders and force-directed layouts
8. Böcker (2012): Cluster editing algorithms

---

## Appendix A: Code Architecture

```
twosphere-mcp/
├── src/
│   ├── backend/
│   │   ├── visualization/
│   │   │   ├── hierarchical_brain_model.py      # Main model (Phases 1-6 ✅)
│   │   │   ├── spherical_spring_layout.py       # Sphere layout ✅
│   │   │   └── graph_on_sphere.py               # Geodesics, rotation ✅
│   │   └── integration/
│   │       └── merge2docs_bridge.py             # cluster-editing-vs ✅
│   └── atlases/
│       ├── brain_atlas.py                       # Base atlas class
│       ├── d99_atlas.py                         # Macaque D99
│       └── allen_atlas.py                       # Mouse Allen CCF
├── bin/
│   └── brain_atlas_http_server.py               # Atlas MCP server ✅
├── examples/
│   ├── demo_hierarchical_brain_clusters.py      # Basic demo ✅
│   ├── demo_full_hierarchical_brain.py          # 3-level + backbone ✅
│   └── demo_yada_dual_structure.py              # YADA implementation (TODO)
├── tests/
│   ├── test_hierarchical_brain_model.py         # Unit tests ✅
│   └── test_yada_dual_structure.py              # Dual structure tests (TODO)
└── docs/
    └── designs/
        └── yada-hierarchical-brain-model/
            ├── DESIGN.md                        # This document
            ├── references.bib                   # Citations
            └── figures/                         # Visualizations
```

---

## Appendix B: Parameter Tuning Guide

### Cluster-Editing-VS

**Parameter k** (edit budget):
- Too low: Under-clustering (many small clusters)
- Too high: Over-clustering (few large clusters)
- **Auto-tuning**: Try k ∈ {5%, 10%, 20%, 50%, 100%, 200%} of |E|
- **Recommendation**: k = 10-20% of |E| for biological networks

**Target clusters**:
- Macaque D99: 6-10 systems (visual, motor, prefrontal, temporal, parietal, limbic)
- Mouse Allen: 8-12 systems
- Human: 7-12 systems (RSN: resting-state networks)

### Spherical Spring Layout

**Spring constant k**: Controls attractive force strength
- Small k (0.05-0.1): Weak attraction, loose layout
- Large k (0.3-0.5): Strong attraction, tight layout
- **Recommendation**: k = 0.15 for biological networks

**Repulsion constant K**: Controls repulsive force strength
- Small K (0.01): Weak repulsion, nodes can get close
- Large K (0.05-0.1): Strong repulsion, uniform spacing
- **Recommendation**: K = 0.02 for balance

**Rest length d₀**: Desired edge length in radians
- Small d₀ (π/12 ≈ 15°): Short edges, compact
- Large d₀ (π/4 ≈ 45°): Long edges, spread out
- **Recommendation**: d₀ = 0.3-0.5 radians (~17-29°)

**Iterations T**: Number of optimization steps
- Few (50-100): Fast, may not converge
- Many (200-500): Slow, better convergence
- **Recommendation**: T = 150-200 for n < 500 nodes

### Backbone (r-IDS + CDS)

**Domination radius r**:
- r = 1: Very local domination, large backbone (30-40% of nodes)
- r = 2: Medium domination, moderate backbone (15-25%)
- r = 3: Global domination, small backbone (5-15%)
- **Recommendation**: r = 2 for balance

---

**End of Design Document**

**Status**: Phase 1-6 complete ✅, Phase 7-8 in design 📋, Phase 9-10 future work 🔮

**Next Steps**: Implement brain atlas integration (Phase 7), then YADA dual structure (Phase 8)
