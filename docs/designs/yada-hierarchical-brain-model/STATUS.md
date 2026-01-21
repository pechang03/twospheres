# YADA Hierarchical Brain Model - Implementation Status

## Quick Reference

### YADA Dual Structure

The **YADA (Yet Another Document Analyzer)** architecture from `merge2docs` provides the theoretical foundation for our hierarchical brain model:

```
MERGE2DOCS (Document Analysis)          BRAIN MODEL (Neural Systems)
═══════════════════════════════════     ═══════════════════════════════════

SYNTACTIC HIERARCHY                     SYNTACTIC HIERARCHY (Anatomical)
Word → Sentence → Paragraph             Neuron → Column → Region → Lobe
↓                                       ↓
WHERE is information?                   WHERE are neurons?
Structural containment                  Spatial organization

SEMANTIC HIERARCHY                      SEMANTIC HIERARCHY (Functional)
Entity → Topic → Concept                Feature → Network → System → Function
↓                                       ↓
WHAT does it mean?                      WHAT does it compute?
Meaning-based connections               Activity patterns

BIPARTITE MAPPING                       BIPARTITE MAPPING
Paragraph ↔ Concept                     V1 (anatomy) ↔ Edge Detection (function)
Document ↔ Theme                        PFC (anatomy) ↔ Working Memory (function)

CROSS-LEVEL CONNECTIONS                 CROSS-LEVEL CONNECTIONS
Word-entity → Paragraph-topic           Neuron-feature → Region-system
Vertical within hierarchies             Multi-scale integration
```

## Implementation Status

### ✅ Phase 1-6: Base Hierarchical Model (Synthetic Data)

**Status**: Complete
**Files**:
- `src/backend/visualization/hierarchical_brain_model.py`
- `examples/demo_hierarchical_brain_clusters.py`
- `examples/demo_full_hierarchical_brain.py`

**Features**:
- ✅ Synthetic clustered data (make_blobs)
- ✅ k-NN graph construction
- ✅ Auto-tuned cluster-editing-vs
- ✅ 3-level hierarchy (Backbone → Communities → Nodes)
- ✅ Spherical spring layout
- ✅ Visualization pipeline

**Results**:
- 200 nodes, 10 ground truth clusters
- Detected 6 communities (modularity=0.805, ARI=0.604)
- 30 backbone hubs spanning 3 communities

---

### ✅ Phase 7: Syntactic (Anatomical) Hierarchy

**Status**: Complete
**Files**:
- `examples/demo_atlas_hierarchical_brain.py`

**Features**:
- ✅ D99 macaque atlas integration (368 regions)
- ✅ Real anatomical regions via HTTP API
- ✅ Neighbor-based connectivity graph
- ✅ Hemisphere-aware positioning
- ✅ 3-level anatomical hierarchy

**Results**:
- 100 cortical regions from D99 atlas
- 189 anatomical edges
- 5 functional communities detected
- 15 backbone hubs

**YADA Mapping**:
```python
syntactic_graphs = {
    "regions": G_anatomical,      # Level 3: Individual brain regions
    "communities": G_communities, # Level 2: Anatomical clusters
    "backbone": G_backbone        # Level 1: Major hubs
}
```

---

### 🚧 Phase 8: Semantic (Functional) Hierarchy

**Status**: Designed (awaiting implementation)
**Files**:
- `docs/designs/yada-hierarchical-brain-model/PHASE8_MRI_INTEGRATION.md`

**Planned Features**:
- 🔲 4D fMRI time-series preprocessing
- 🔲 Diffusion CNN feature extraction
- 🔲 Functional connectivity graph
- 🔲 Cluster-editing-vs on functional data
- 🔲 3-level functional hierarchy

**Data Requirements**:
- 4D macaque fMRI (PRIME-DE dataset recommended)
- D99 atlas mask for ROI extraction
- Pre-trained diffusion CNN model (or PCA baseline)

**Expected Results**:
- Features: 100 regions × latent_dim
- Networks: 10-15 functional systems (visual, motor, DMN)
- Systems: 3-5 integration hubs

**YADA Mapping**:
```python
semantic_graphs = {
    "features": G_functional,     # Level 3: Neural features
    "networks": G_networks,       # Level 2: Functional systems
    "systems": backbone_functional # Level 1: Integration zones
}
```

---

### 🔲 Phase 9: Bipartite Structure-Function Mapping

**Status**: Designed
**File**: `docs/designs/yada-hierarchical-brain-model/DESIGN.md` (lines 253-292)

**Planned Features**:
- 🔲 Bipartite graph connecting syntactic ↔ semantic
- 🔲 V1 (anatomy) ↔ Edge detection (function)
- 🔲 PFC (anatomy) ↔ Working memory (function)
- 🔲 Many-to-many mapping validation

**YADA Mapping**:
```python
bipartite_graph = create_structure_function_mapping(
    G_anatomical=syntactic_graphs["regions"],
    G_functional=semantic_graphs["features"],
    region_ids=shared_regions
)

# Result: Bipartite graph with two node sets
# - Left partition: Anatomical regions (WHERE)
# - Right partition: Functional features (WHAT)
# - Edges: Structure ↔ Function mapping
```

**Example Mappings**:
| Syntactic (Anatomy) | Semantic (Function) | Strength |
|---------------------|---------------------|----------|
| V1 (primary visual) | Edge detection      | 0.95     |
| V1                  | Orientation tuning  | 0.88     |
| MT (motion area)    | Motion detection    | 0.92     |
| PFC (prefrontal)    | Working memory      | 0.85     |
| PFC                 | Executive control   | 0.79     |

---

### 🔲 Phase 10: Cross-Level Connections

**Status**: Designed
**File**: `docs/designs/yada-hierarchical-brain-model/DESIGN.md` (lines 293-344)

**Planned Features**:
- 🔲 Vertical edges within syntactic hierarchy
- 🔲 Vertical edges within semantic hierarchy
- 🔲 Multi-scale integration

**YADA Mapping**:
```python
# Syntactic cross-level (containment)
G_syntactic_cross = nx.DiGraph()
G_syntactic_cross.add_edge("neuron_123", "column_45")  # Neuron in column
G_syntactic_cross.add_edge("column_45", "region_V1")   # Column in region
G_syntactic_cross.add_edge("region_V1", "lobe_occipital") # Region in lobe

# Semantic cross-level (implementation)
G_semantic_cross = nx.DiGraph()
G_semantic_cross.add_edge("feature_edge", "network_V1")  # Feature in network
G_semantic_cross.add_edge("network_V1", "system_visual") # Network in system
G_semantic_cross.add_edge("system_visual", "function_perception") # System implements function
```

---

## Comparison with merge2docs Implementation

### unified_pipeline_workflow_v3.py

**Lines 466, 473**: `semantic_graphs` extraction from repair matrix

```python
# merge2docs
semantic_graphs = {
    level: graph
    for level, graph in enumerate(graphs)
    if level in semantic_levels
}
```

**Our Brain Model Equivalent**:
```python
# twosphere-mcp
semantic_graphs = {
    "features": extract_from_fmri(timeseries),
    "networks": detect_communities(G_functional),
    "systems": compute_backbone(G_networks)
}
```

### Lines 890-931: Repair matrix construction

```python
# merge2docs: Build repair matrix from semantic graphs
def build_repair_matrix(semantic_graphs, syntactic_graphs):
    # Extract inconsistencies between hierarchies
    repair_edges = []
    for syn_node, sem_node in bipartite_mapping:
        if not consistent(syn_node, sem_node):
            repair_edges.append((syn_node, sem_node, "repair"))
    return repair_matrix
```

**Our Brain Model Equivalent**:
```python
# twosphere-mcp: Validate structure-function consistency
def validate_structure_function_mapping(
    anatomical_regions: List[int],
    functional_systems: List[int],
    bipartite_mapping: nx.Graph
) -> Dict[str, List]:
    """Find inconsistencies between anatomy and function."""
    inconsistencies = []

    for anat_node, func_node in bipartite_mapping.edges():
        # Check if functional activity matches anatomical connectivity
        if not functionally_connected(anat_node, func_node):
            inconsistencies.append({
                "anatomical": anat_node,
                "functional": func_node,
                "type": "missing_connection"
            })

    return {"repairs_needed": inconsistencies}
```

---

## Next Steps

### Immediate (Phase 8)
1. **Acquire macaque fMRI data**: Download from PRIME-DE
2. **Implement preprocessing**: `preprocess_fmri()` function
3. **Feature extraction**: Start with PCA baseline, then diffusion CNN
4. **Functional graph**: Build from connectivity matrix
5. **Community detection**: Apply cluster-editing-vs to functional data

### Short-term (Phase 9)
1. **Bipartite mapping**: Connect anatomical regions to functional features
2. **Validate mappings**: Compare with known structure-function relationships
3. **Visualize dual structure**: 3-panel figure (syntactic | bipartite | semantic)

### Long-term (Phase 10)
1. **Cross-level edges**: Implement containment and implementation hierarchies
2. **Multi-scale analysis**: Study how features aggregate into systems
3. **Paper writing**: Use design docs for methods section

---

## Key References

**YADA Architecture**:
- `merge2docs/src/backend/graphs/workflows/unified_pipeline_workflow_v3.py`
- `merge2docs/specs/yada.md`

**Brain Model Implementation**:
- `docs/designs/yada-hierarchical-brain-model/DESIGN.md`
- `docs/designs/yada-hierarchical-brain-model/PHASE8_MRI_INTEGRATION.md`
- `docs/designs/yada-hierarchical-brain-model/references.bib`

**Code**:
- `src/backend/visualization/hierarchical_brain_model.py`
- `examples/demo_atlas_hierarchical_brain.py`

---

**Last Updated**: 2026-01-21
**Status**: Phase 7 complete, Phase 8 designed and ready for implementation
