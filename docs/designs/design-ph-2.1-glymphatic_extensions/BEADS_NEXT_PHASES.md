# Beads: Next Implementation Phases

## 🔵 BEAD-P4: Visualization and 3D Plotting
**Priority**: High
**Dependencies**: Phase 1, Phase 2, Phase 3
**Estimated Complexity**: Medium

### Objective
Create matplotlib-based visualization tools for brain network overlays on sphere surfaces, including 3D interactive plots and 2D projections.

### Implementation Tasks
1. **3D Sphere Visualization** (`src/backend/mri/visualization.py`)
   - `plot_sphere_surface()` - Render sphere with wireframe/surface
   - `plot_network_on_sphere()` - Overlay nodes and edges on 3D sphere
   - `plot_interhemispheric_connections()` - Highlight corpus callosum edges
   - Color-coding by connectivity strength, edge thickness by weight

2. **2D Projections**
   - `plot_azimuthal_projection()` - Flatten sphere to 2D (stereographic/orthographic)
   - `plot_connectivity_matrix_heatmap()` - Distance correlation matrix visualization
   - `plot_degree_distribution()` - Node degree histogram

3. **Interactive Widgets** (if using plotly/ipywidgets)
   - Rotate sphere view
   - Toggle interhemispheric edges
   - Filter by connectivity threshold

4. **Export Formats**
   - PNG/SVG static images
   - Interactive HTML (plotly)
   - Animation frames for time-varying connectivity

### Test Coverage
- Test sphere rendering with mock data
- Test node/edge positioning accuracy
- Test projection correctness (compare to known transformations)
- Smoke tests for all plot functions (no crashes)

### Integration Points
- Phase 3: `overlay_network_on_sphere()` output → visualization input
- Example notebooks: Demonstrate visualizations with real MRI data

### Success Criteria
- Generate publication-quality 3D network overlays
- Render 100+ node networks in <5 seconds
- Export to standard formats (PNG, SVG, HTML)

---

## 🔵 BEAD-P5: MCP Server Tools
**Priority**: High
**Dependencies**: Phase 1, Phase 2, Phase 3
**Estimated Complexity**: Medium

### Objective
Expose brain network analysis functions as MCP tools for Claude API integration, enabling natural language queries about brain connectivity.

### Implementation Tasks
1. **MCP Tool Definitions** (`src/backend/mcp/brain_network_tools.py`)
   - `analyze_brain_connectivity` - End-to-end pipeline tool
     - Input: Connectivity matrix file path, node locations JSON
     - Output: Network metrics, interhemispheric stats, geodesic distances

   - `compute_functional_connectivity` - Phase 1 tool
     - Input: MRI time-series data files
     - Output: Distance correlation matrix, PLV matrix

   - `map_to_sphere` - Phase 2 tool
     - Input: Node locations (spherical coords)
     - Output: Cartesian positions, geodesic distances

   - `identify_network_modules` - Phase 3 tool
     - Input: Connectivity graph
     - Output: Community detection results, modularity score

2. **MCP Server Integration** (`src/backend/mcp/server.py`)
   - Register tools with MCP server
   - Add input validation schemas
   - Error handling for malformed inputs
   - Logging for tool invocations

3. **Tool Documentation**
   - JSON schemas for each tool
   - Example usage in MCP client
   - Parameter descriptions and constraints

### Test Coverage
- Test tool registration with MCP server
- Test input validation (reject invalid data)
- Test output format consistency
- Integration test: Claude API query → tool invocation → result

### Integration Points
- Ernie2 integration: Query collections for parameter suggestions
- Visualization: Tools can return plot URLs/base64 images

### Success Criteria
- All Phase 1-3 functions accessible via MCP
- Natural language queries like "Analyze brain connectivity for patient X"
- Response time <10s for typical datasets

---

## 🔵 BEAD-P6: Optical LOC Cross-Validation
**Priority**: High
**Dependencies**: Phase 1, Phase 2, Phase 3
**Estimated Complexity**: High

### Objective
Implement cross-validation framework between MRI brain-level measurements and optical LOC molecular-level measurements using identical mathematical operations.

### Implementation Tasks
1. **Unified Signal Processing** (`src/backend/integration/signal_alignment.py`)
   - `align_mri_with_optical()` - Temporal alignment between MRI (seconds) and optical (microseconds)
   - `resample_signals()` - Match sampling rates via interpolation
   - `synchronize_timeseries()` - Cross-correlation based sync

2. **Cross-Validation Metrics** (`src/backend/integration/cross_validation.py`)
   - `compute_correlation_equivalence()` - Compare MRI dCor with optical FFT correlation
   - `compute_phase_equivalence()` - Compare MRI PLV with optical lock-in phase
   - `validate_spatial_consistency()` - Check if brain-level and molecular-level patterns match

3. **Mathematical Equivalence Table**
   ```python
   equivalence_map = {
       "distance_correlation": {
           "mri_method": "compute_distance_correlation()",
           "optical_method": "fft_correlation()",
           "validation": "pearson_r > 0.8"
       },
       "phase_locking": {
           "mri_method": "compute_phase_locking_value()",
           "optical_method": "lock_in_phase()",
           "validation": "circular_correlation > 0.7"
       }
   }
   ```

4. **Drug Effect Analysis** (`src/backend/integration/drug_analysis.py`)
   - `compare_drug_effects()` - Brain vs. molecular response
   - `identify_discrepancies()` - Where MRI and optical disagree
   - `rank_drug_candidates()` - Score drugs by multi-scale consistency

### Test Coverage
- Test signal alignment with synthetic data
- Test correlation equivalence (known ground truth)
- Test phase equivalence (synthetic oscillators)
- Integration test: Full MRI + optical pipeline

### Integration Points
- Phase 1: MRI signal processing
- Optical LOC: Existing optical simulation code
- Ernie2: Query neuroscience_MRI and bioengineering_LOC collections

### Success Criteria
- Achieve >0.8 correlation between MRI and optical measurements for validation dataset
- Identify drug candidates with consistent effects at both scales
- Generate cross-validation reports with confidence intervals

---

## 🔵 BEAD-P7: Example Notebooks and Documentation
**Priority**: Medium
**Dependencies**: Phase 1, Phase 2, Phase 3, Phase 4
**Estimated Complexity**: Medium

### Objective
Create comprehensive Jupyter notebooks demonstrating end-to-end workflows for brain network analysis, plus user documentation.

### Implementation Tasks
1. **Example Notebooks** (`examples/notebooks/`)
   - `01_basic_connectivity_analysis.ipynb`
     - Load MRI data
     - Compute distance correlation matrix
     - Visualize connectivity network

   - `02_sphere_overlay.ipynb`
     - Map brain regions to sphere coordinates
     - Compute geodesic distances
     - Identify interhemispheric connections

   - `03_network_topology.ipynb`
     - Convert connectivity to graph
     - Compute network metrics
     - Community detection

   - `04_drug_effect_analysis.ipynb`
     - Compare pre/post drug connectivity
     - Statistical significance testing
     - Visualize changes

   - `05_mri_optical_cross_validation.ipynb`
     - Load both MRI and optical data
     - Cross-validate measurements
     - Generate equivalence report

2. **User Documentation** (`docs/user_guide/`)
   - Installation instructions
   - Quick start guide
   - API reference
   - Troubleshooting guide
   - FAQ

3. **Synthetic Datasets** (`examples/data/`)
   - `synthetic_connectivity.npz` - 100-node connectivity matrix
   - `brain_regions.json` - Anatomical locations (spherical coords)
   - `mri_timeseries.npz` - Multi-region time series
   - README explaining data format

### Test Coverage
- Smoke test: Run all notebooks end-to-end (no errors)
- Test synthetic data loads correctly
- Test example outputs match expected values

### Integration Points
- All phases: Demonstrate complete pipeline
- Visualization: Include plots in notebooks
- MCP tools: Show how to query from Claude API

### Success Criteria
- 5 working notebooks with clear explanations
- <10 minute runtime for all notebooks
- Documentation covers 90% of common use cases

---

## 🔵 BEAD-P8: Advanced Network Analysis
**Priority**: Low
**Dependencies**: Phase 3
**Estimated Complexity**: High

### Objective
Implement advanced graph-theoretic analysis methods for brain connectivity networks.

### Implementation Tasks
1. **Community Detection** (`src/backend/mri/network_advanced.py`)
   - `detect_communities_louvain()` - Louvain modularity optimization
   - `detect_communities_infomap()` - Information-theoretic method
   - `compute_modularity()` - Q-score for community structure

2. **Centrality Measures**
   - `compute_betweenness_centrality()` - Bridge nodes
   - `compute_eigenvector_centrality()` - Hub nodes
   - `compute_pagerank()` - Importance ranking

3. **Motif Analysis**
   - `count_triangles()` - Three-node cliques
   - `count_four_cycles()` - Square motifs
   - `motif_enrichment()` - Compare to random networks

4. **Network Comparison**
   - `compare_networks()` - Pre/post drug, patient/control
   - `network_distance()` - Graph edit distance, spectral distance
   - `permutation_test()` - Statistical significance

5. **Small-World Properties**
   - `compute_small_worldness()` - σ = (C/C_rand) / (L/L_rand)
   - `compute_rich_club_coefficient()` - Hub connectivity

### Test Coverage
- Test community detection on known networks (e.g., karate club)
- Test centrality measures (compare to NetworkX reference)
- Test motif counting accuracy
- Test network distance metrics (triangle inequality, symmetry)

### Integration Points
- Phase 3: Extend network_analysis.py
- Visualization: Plot community structure with colors
- Drug analysis: Compare network topology changes

### Success Criteria
- Detect communities in <30s for 1000-node networks
- Accurate motif counting (validated against ground truth)
- Publishable network comparison metrics

---

## 🔵 BEAD-P9: Performance Optimization
**Priority**: Low
**Dependencies**: All phases
**Estimated Complexity**: Medium

### Objective
Optimize computational performance for large-scale brain networks (>1000 nodes, long time series).

### Implementation Tasks
1. **Profiling**
   - Profile Phase 1-3 functions with cProfile
   - Identify bottlenecks (likely: distance correlation, geodesic distances)

2. **NumPy Vectorization**
   - Replace Python loops with vectorized operations
   - Use `np.einsum()` for tensor operations
   - Batch geodesic distance computation

3. **Parallel Processing**
   - Use `multiprocessing.Pool` for embarrassingly parallel tasks
   - Parallelize pairwise distance correlation computation
   - Parallelize edge geodesic length computation

4. **Caching**
   - Cache geodesic distance matrices (LRU cache)
   - Cache connectivity graph conversions
   - Persistent cache to disk for large datasets

5. **Optional C Extensions**
   - Numba JIT compilation for hot loops
   - Cython for distance correlation (if needed)

### Test Coverage
- Benchmark tests: Measure runtime for different network sizes
- Regression tests: Ensure optimized code produces identical results
- Stress tests: 10,000-node networks, 10,000-timepoint series

### Integration Points
- All phases: Drop-in replacements for slow functions

### Success Criteria
- 10x speedup for distance correlation on large datasets
- Handle 1000-node networks in real-time (<1 minute)
- Memory usage scales linearly (no memory leaks)

---

## 🔵 BEAD-P10: Clinical Validation Dataset
**Priority**: Medium
**Dependencies**: Phase 1, Phase 2, Phase 3, Phase 7
**Estimated Complexity**: High

### Objective
Acquire or generate realistic clinical validation dataset for Alzheimer's disease brain connectivity analysis.

### Implementation Tasks
1. **Dataset Acquisition**
   - Option A: Use public MRI datasets (ADNI, UK Biobank)
   - Option B: Generate synthetic data mimicking clinical distributions
   - Option C: Collaborate with research institution

2. **Data Preprocessing** (`src/backend/data/preprocessing.py`)
   - `load_nifti_data()` - Load MRI NIfTI files
   - `extract_roi_timeseries()` - Extract ROI time series from fMRI
   - `motion_correction()` - Correct for head motion artifacts
   - `spatial_normalization()` - Register to MNI template

3. **Clinical Metadata**
   - Patient demographics (age, sex, diagnosis)
   - Cognitive scores (MMSE, CDR)
   - Drug treatment status
   - Scan parameters

4. **Ground Truth Labels**
   - Healthy control vs. Alzheimer's
   - Disease severity (mild, moderate, severe)
   - Drug response (responder vs. non-responder)

5. **Validation Metrics** (`src/backend/validation/clinical_metrics.py`)
   - `compute_classification_accuracy()` - Predict diagnosis from connectivity
   - `compute_drug_response_auc()` - ROC curve for drug response prediction
   - `compare_to_literature()` - Compare metrics to published studies

### Test Coverage
- Test data loading for all supported formats
- Test preprocessing pipeline (visual QC)
- Test validation metrics (known ground truth)

### Integration Points
- Phase 1-3: Apply full pipeline to clinical data
- Phase 4: Visualize clinical network patterns
- Phase 6: Cross-validate with optical measurements (if available)

### Success Criteria
- Load and process >50 patient datasets
- Achieve >70% classification accuracy (healthy vs. AD)
- Reproduce key findings from published literature

---

## Priority Summary

**Immediate (High Priority)**:
- BEAD-P4: Visualization (enables interpretation)
- BEAD-P5: MCP Tools (core product feature)
- BEAD-P6: Optical LOC Cross-Validation (unique value proposition)

**Near-term (Medium Priority)**:
- BEAD-P7: Example Notebooks (usability)
- BEAD-P10: Clinical Validation (credibility)

**Future (Low Priority)**:
- BEAD-P8: Advanced Network Analysis (research features)
- BEAD-P9: Performance Optimization (polish)

## Dependency Graph

```
Phase 1 ─┬─→ Phase 4 (Visualization) ─→ Phase 7 (Notebooks)
         ├─→ Phase 5 (MCP Tools)
         └─→ Phase 6 (LOC Cross-Val) ─→ Phase 10 (Clinical)

Phase 2 ─┴─→ [same as above]

Phase 3 ─┬─→ [same as above]
         └─→ Phase 8 (Advanced Analysis)

Phase 9 (Optimization) can be applied to any phase
```

## Recommended Implementation Order

1. **BEAD-P4** (Visualization) - Enables visual verification of Phases 1-3
2. **BEAD-P5** (MCP Tools) - Makes system accessible via Claude API
3. **BEAD-P7** (Notebooks) - Documents how to use everything
4. **BEAD-P6** (LOC Cross-Val) - Core scientific contribution
5. **BEAD-P10** (Clinical) - Real-world validation
6. **BEAD-P8** (Advanced) - Research extensions
7. **BEAD-P9** (Optimization) - Performance tuning
