# Ernie2 Query Guide: Disc Dimension Analysis

**Purpose**: Query ernie2_swarm_mcp_e for theoretical insights about disc dimension obstructions WITHOUT empirically finding them first

---

## Quick Start

### Run All Queries

```bash
cd /Users/petershaw/code/aider/twosphere-mcp
./bin/query_ernie2_disc_dimension_mcp.sh
```

**Duration**: ~10-15 minutes for all 8 queries (with caching)

**Output**: `docs/papers/ernie2_q*.md` (markdown with diagrams)

### Run Single Query

```bash
python /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm_mcp_e.py \
  --question "How does K5 forbidden minor affect Local Intrinsic Dimension?" \
  --collection docs_library_mathematics \
  --collection docs_library_neuroscience_MRI \
  --style technical \
  --enable-math-glue-enhanced \
  -o docs/papers/ernie2_lid_analysis.md
```

---

## 8 Key Questions

### Q1: LID and Disc Dimension
**File**: `ernie2_q1_lid_disc_dimension.md`
**Question**: Theoretical bounds between disc dimension and Local Intrinsic Dimension

**Expected Output**:
- Formula: `disc(G) ≤ k ⟹ LID(G) ≤ f(k, n)`
- Prediction: Can we infer disc-dim from LID distribution?
- Brain networks: LID ≈ 5 (signal), LID ≈ 12 (lymphatic) → disc-dim?

### Q2: VC Dimension
**File**: `ernie2_q2_vc_dimension.md`
**Question**: VC dimension for brain graphs and sample complexity

**Expected Output**:
- VC-dim for 368-node brain graph with small-world topology
- Sample size requirements for learning graph structure
- Multiplex effect on VC dimension

### Q3: Graph Curvature
**File**: `ernie2_q3_graph_curvature.md` (+ diagram)
**Question**: Ricci/Forman curvature signatures of K₅, K₃,₃

**Expected Output**:
- Curvature near K₅ (negative/hyperbolic)
- Curvature discontinuities in multiplex networks
- Ricci flow for obstruction detection

### Q4: Treewidth Bounds
**File**: `ernie2_q4_treewidth_bounds.md`
**Question**: Relationship between treewidth and disc dimension

**Expected Output**:
- Bounds: `treewidth(G) ≈ 2 × disc(G)`
- Brain networks: tw ≈ 6 → disc-dim ≈ 3
- K₅ increases treewidth by +1 to +2

### Q5: r-IDS Sampling
**File**: `ernie2_q5_rids_sampling.md`
**Question**: Coverage probability for r-IDS obstruction sampling

**Expected Output**:
- Probability that r-IDS (r=4, k=50) captures all K₅, K₃,₃
- Complexity: O(n log n) vs O(n⁵) exhaustive
- Theoretical guarantees from domination property

### Q6: Multiplex Dimension Formula
**File**: `ernie2_q6_multiplex_dimension.md` (+ diagram)
**Question**: De Domenico's multiplex formula application

**Expected Output**:
- `d_eff = d_layer + log₂(L) + δ_coupling ≈ 3.5`
- Neurovascular coupling term δ
- Does d_eff > 3 require 3D embedding?

### Q7: Property-Based Prediction
**File**: `ernie2_q7_property_prediction.md` (+ diagram)
**Question**: Predict disc-dim from measurable properties

**Expected Output**:
- Formula: `disc(G) = f(LID, VC, curvature, clustering, ...)`
- Decision boundary for disc-dim ∈ {2, 3, 4}
- Expected accuracy for brain networks

### Q8: Obstruction Catalog
**File**: `ernie2_q8_obstruction_catalog.md` (+ diagram)
**Question**: Characterize multiplex obstruction set Obs_M(2,2)

**Expected Output**:
- Minimal forbidden multiplex minors
- Brain-specific: neurovascular star, VCG, CCB, DMN-core
- Finite basis for generating all Obs_M(2,2)

---

## MCP-Enhanced Features

### Engram O(1) Lookup
- Fast retrieval from 36 domain collections
- Collections used:
  - `docs_library_mathematics` (graph theory, FPT)
  - `docs_library_neuroscience_MRI` (brain networks)
  - `docs_library_physics_differential_geometry` (curvature)

### Tensor Routing
- Automatic routing to best model/agent for question type
- Math questions → math-enhanced agents
- Biological questions → neuroscience experts

### Math Glue Enhanced
- `--enable-math-glue-enhanced`: LaTeX formulas, proofs
- Symbolic manipulation for bounds and inequalities
- Theorem citation and proof sketches

### Diagram Generation
- `--diagram --diagram-format mermaid`
- Visual representations of:
  - Graph obstructions (K₅, K₃,₃)
  - Multiplex layer structure
  - Decision boundaries

### Caching
- `--use-cache`: Speed up repeated queries
- Cached results in merge2docs engram database

---

## Expected Results Structure

Each query result will have:

### 1. Direct Answer
Concise answer to the specific question

### 2. Theoretical Framework
- Definitions
- Theorems/bounds
- Formulas

### 3. Brain Network Application
- Specific values for D99 atlas (368 regions)
- Signal layer properties
- Lymphatic layer properties

### 4. Computational Complexity
- Algorithm complexity
- FPT analysis when applicable

### 5. Citations
- Papers referenced
- Theorems used

### 6. Diagrams (where applicable)
- Mermaid/GraphViz visualizations
- Network structures
- Decision trees

---

## Synthesis After Queries

### Step 1: Extract Formulas

Collect all formulas from 8 results:
```python
formulas = {
    'disc_from_LID': 'extracted from Q1',
    'disc_from_treewidth': 'extracted from Q4',
    'multiplex_dimension': 'extracted from Q6',
    'prediction_function': 'extracted from Q7'
}
```

### Step 2: Build Unified Model

Combine into single prediction framework:
```
disc_dimension = argmax_{d ∈ {2,3,4}} P(d | props)

where props = {LID, VC, curvature, clustering, betweenness, treewidth}
```

### Step 3: Validate on Synthetic

Test predictions on:
- Watts-Strogatz (small-world)
- Random geometric (Euclidean)
- Multiplex combinations

### Step 4: Apply to PRIME-DE

Use predictions on real brain data:
- Load BORDEAUX24 subjects
- Compute all properties
- Predict disc dimension
- Validate with r-IDS obstruction sampling

---

## File Organization

```
docs/papers/
├── ernie2_q1_lid_disc_dimension.md          # Q1 results
├── ernie2_q2_vc_dimension.md                # Q2 results
├── ernie2_q3_graph_curvature.md             # Q3 results + diagram
├── ernie2_q4_treewidth_bounds.md            # Q4 results
├── ernie2_q5_rids_sampling.md               # Q5 results
├── ernie2_q6_multiplex_dimension.md         # Q6 results + diagram
├── ernie2_q7_property_prediction.md         # Q7 results + diagram
├── ernie2_q8_obstruction_catalog.md         # Q8 results + diagram
├── ernie2_synthesis_unified_framework.md    # Combined insights
└── ERNIE2_QUERY_GUIDE.md                    # This file
```

---

## Troubleshooting

### Issue: "ernie2_swarm_mcp_e.py not found"
**Solution**: Check path to merge2docs
```bash
ls /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm_mcp_e.py
```

### Issue: "Collection not found"
**Solution**: Check available collections
```bash
python /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm_mcp_e.py --help
```

### Issue: Queries too slow
**Solution**: Use `--fast` mode
```bash
--fast --synthesis-mode fast
```

### Issue: Need more detail
**Solution**: Use adaptive synthesis
```bash
--synthesis-mode adaptive --num-minions 10
```

---

## Quick Commands

### List available collections
```bash
python /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm.py --show-collections
```

### Test single query (fast)
```bash
python /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm_mcp_e.py \
  -q "What is disc dimension?" \
  -c docs_library_mathematics \
  --fast \
  -o test_output.md
```

### With diagram output
```bash
python /Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm_mcp_e.py \
  -q "Visualize K5 graph obstruction" \
  -c docs_library_mathematics \
  --diagram \
  --diagram-format mermaid \
  -o k5_diagram.md
```

---

## Next Steps After Queries

1. **Review all 8 result files**
   - Extract key formulas
   - Note theoretical bounds
   - Identify prediction strategies

2. **Synthesize unified framework**
   - Combine formulas into single model
   - Create decision boundary
   - Validate consistency

3. **Implement in code**
   - `src/backend/mri/disc_dimension_analysis.py`
   - Property extraction functions
   - Prediction from properties

4. **Validate empirically**
   - Synthetic graphs first
   - PRIME-DE brain data
   - Compare predicted vs actual

5. **Write paper sections**
   - Methods: Use formulas from ernie2
   - Theory: Cite bounds and theorems
   - Results: Empirical validation

---

**Status**: Ready to run
**Command**: `./bin/query_ernie2_disc_dimension_mcp.sh`
**Duration**: ~10-15 minutes
**Output**: 8 markdown files with formulas, bounds, and diagrams
