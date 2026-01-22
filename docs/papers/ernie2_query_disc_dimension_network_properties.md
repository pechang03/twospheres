# Ernie2 Swarm Query: Disc Dimension Obstructions and Network Properties

**Date**: 2026-01-22
**Purpose**: Theoretical exploration of how disc dimension obstructions (K₅, K₃,₃, multiplex minors) affect network properties
**Target**: Query ernie2_swarm for insights before empirical analysis

---

## Query Context

We're investigating multiplex brain networks (signal + lymphatic layers) and their disc dimension. Before empirically detecting obstructions in PRIME-DE data, we want to understand:

**How do disc dimension obstructions relate to:**
1. LID (Local Intrinsic Dimension)
2. VC dimension (Vapnik-Chervonenkis dimension)
3. Graph curvature (Ollivier-Ricci, Forman)
4. Clustering coefficient
5. Betweenness centrality
6. Treewidth
7. Pathwidth
8. Chromatic number

---

## Specific Questions for Ernie2

### Q1: LID and Disc Dimension

**Query**:
```
How does the presence of forbidden minors (K₅, K₃,₃) in a graph relate to its
Local Intrinsic Dimension (LID)?

Specifically:
- If a graph G contains K₅ as a minor, does this imply LID(G) ≥ 4 locally around K₅?
- For multiplex networks with signal (LID ≈ 4-7) and lymphatic (LID ≈ 10-15) layers,
  how does disc dimension obstruction manifest in LID distribution?
- Can we use LID estimation to predict disc dimension without explicit minor detection?

Context: Brain networks with D99 parcellation (368 regions)
Signal: fMRI functional connectivity (distance correlation)
Lymphatic: Glymphatic CSF flow (estimated from anatomy)
```

**Expected Insight**:
- LID measures local dimensionality of data manifold
- K₅ creates high-dimensional "bulge" in graph metric space
- Hypothesis: Regions near obstructions have higher LID

### Q2: VC Dimension and Embeddability

**Query**:
```
What is the relationship between VC dimension and disc dimension for graphs?

Specifically:
- Does a graph with disc dimension d have bounded VC dimension?
- For brain connectivity graphs with small-world topology (C ≈ 0.5, L ≈ 2-3),
  what is the expected VC dimension?
- How do multiplex obstructions (cross-layer K₅ × Star) affect VC dimension?

Context: Using VC dimension for statistical learning on brain graphs
Need to understand sample complexity for learning graph structure
```

**Expected Insight**:
- VC dimension bounds generalization in graph learning
- Higher disc dimension → higher VC dimension → more samples needed
- Multiplex adds ~1 to VC dimension

### Q3: Graph Curvature Near Obstructions

**Query**:
```
How do forbidden graph minors affect local graph curvature (Ollivier-Ricci, Forman)?

Specifically:
- Does K₅ create negative curvature regions (hyperbolic geometry)?
- For K₃,₃ bipartite obstructions, what is the curvature signature?
- In multiplex networks, do cross-layer obstructions create curvature discontinuities?

Context: Brain networks may have mixed curvature (hyperbolic hubs, Euclidean periphery)
Lymphatic layer expected to be near-flat (Euclidean-like)
Signal layer expected to have negative curvature (small-world hubs)
```

**Expected Insight**:
- K₅ creates negatively curved regions (hyperbolic)
- Obstructions = curvature singularities
- Can detect via Ricci flow

### Q4: Clustering and Disc Dimension

**Query**:
```
How does clustering coefficient relate to disc dimension obstructions?

Specifically:
- If a graph has high clustering (C > 0.4), does it contain K₅ minors?
- For brain networks with small-world property (high C, low L),
  what is the expected obstruction count?
- Do multiplex networks have higher clustering than single-layer?

Context: PRIME-DE macaque fMRI expected to have C ≈ 0.5 (signal), C ≈ 0.3 (lymph)
```

**Expected Insight**:
- High clustering → triangles → potential K₅ minors
- Small-world → disc-dim ≥ 3 almost certainly
- Clustering coefficient as proxy for obstruction density

### Q5: Betweenness and Hub Structure

**Query**:
```
How do high-betweenness hubs relate to disc dimension obstructions?

Specifically:
- Are high-betweenness nodes typically in K₅ minors?
- For multiplex networks, do neurovascular hubs (high betweenness in both layers)
  create cross-layer obstructions?
- Can betweenness centrality predict disc dimension?

Context: Brain has ~5-10 hub regions (thalamus, PFC, default mode core)
These likely participate in many K₅ configurations
```

**Expected Insight**:
- Hubs = high betweenness → participate in many cliques → in K₅ minors
- Neurovascular hubs create multiplex obstructions
- Top 10% betweenness nodes likely in 80%+ obstructions

### Q6: Treewidth and Disc Dimension

**Query**:
```
What is the relationship between treewidth and disc dimension?

Specifically:
- If disc(G) = d, what bounds exist on treewidth(G)?
- For brain networks with expected treewidth ~5-8, what disc dimension?
- Do obstructions (K₅, K₃,₃) increase treewidth?

Context: Already using treewidth for FPT algorithms (cluster-editing, r-IDS)
Want to relate to disc dimension
```

**Expected Insight**:
- disc(G) ≤ d ⟹ treewidth(G) ≤ O(d × n^(1/d))
- K₅ increases treewidth by +1 to +2
- Brain networks: treewidth ≈ 2 × disc_dim

### Q7: Pathwidth and Layered Structure

**Query**:
```
How does pathwidth relate to disc dimension in multiplex networks?

Specifically:
- For layered graphs (signal + lymphatic), pathwidth vs disc dimension?
- Does adding inter-layer edges increase pathwidth more than disc dimension?
- Can pathwidth decomposition help find obstructions?

Context: Multiplex networks have natural layer structure
Pathwidth might be more tractable than disc dimension
```

**Expected Insight**:
- Pathwidth ≤ disc dimension (usually)
- Multiplex layers increase pathwidth by ~1 per layer
- Pathwidth decomposition reveals bottleneck obstructions

### Q8: Chromatic Number and Planar Obstructions

**Query**:
```
How does chromatic number relate to disc dimension obstructions?

Specifically:
- K₅ has chromatic number 5, K₃,₃ has chromatic number 2
- For brain networks, what is expected chromatic number?
- Does chromatic number increase with disc dimension?

Context: Graph coloring for brain region segmentation
Functional connectivity might allow 4-coloring (planar-like locally)
Anatomical connectivity might need 5+ colors
```

**Expected Insight**:
- Chromatic number ≥ 5 → contains K₅ → non-planar → disc-dim ≥ 3
- Brain networks likely need 4-5 colors
- Multiplex needs max(χ_signal, χ_lymph)

---

## Synthetic Example for Ernie2

To help ernie2_swarm ground the analysis, provide this concrete example:

```python
# Synthetic brain-like graph for analysis
N = 368  # D99 regions

# Signal layer (small-world)
G_signal = nx.watts_strogatz_graph(N, k=18, p=0.1)
# Properties: C ≈ 0.5, L ≈ 3, degree ≈ 18

# Lymphatic layer (Euclidean-like)
G_lymph = nx.random_geometric_graph(N, radius=0.15, dim=3)
# Properties: C ≈ 0.3, L ≈ 5, follows 3D geometry

# Multiplex union
G_multiplex = nx.compose(G_signal, G_lymph)

# Expected properties to query about:
{
    "signal": {
        "nodes": 368,
        "edges": ~3300,
        "clustering": 0.5,
        "path_length": 3,
        "degree_avg": 18,
        "expected_k5_count": "15-25 (high clustering)",
        "expected_LID": "4-7 (small-world embedding)",
        "expected_disc_dim": "2-3"
    },
    "lymphatic": {
        "nodes": 368,
        "edges": ~4000,
        "clustering": 0.3,
        "path_length": 5,
        "degree_avg": 22,
        "expected_k5_count": "5-10 (lower clustering)",
        "expected_LID": "10-15 (Euclidean-like)",
        "expected_disc_dim": "3-4"
    },
    "multiplex": {
        "nodes": 368,
        "edges": ~7300,
        "clustering": "max(C_s, C_l) ≈ 0.5",
        "expected_k5_count": "30-40 (union)",
        "expected_cross_layer_obstructions": "10-20",
        "expected_LID": "varies 5-15 (heterogeneous)",
        "expected_disc_dim": "3-4"
    }
}
```

---

## Ernie2 Swarm Invocation Format

```bash
# Query ernie2_swarm with specific focus

bin/ernie2_swarm_mcp_e.py \
  --query "How do K5 and K33 forbidden minors affect Local Intrinsic Dimension (LID) in brain connectivity graphs? Specifically for D99 atlas (368 regions) with small-world signal layer (C=0.5, L=3) and Euclidean lymphatic layer (C=0.3, L=5). Include analysis of VC dimension, graph curvature, and treewidth relationships." \
  --context docs/papers/disc_dimension_obstructions_brain_networks.md \
  --focus "theoretical relationships, no empirical detection needed yet" \
  --output docs/papers/ernie2_analysis_disc_dim_properties.md
```

---

## Expected Ernie2 Output Structure

### Section 1: LID Analysis
- Theoretical bounds: disc-dim → LID
- Local vs global LID in multiplex networks
- Obstruction detection via LID anomalies

### Section 2: VC Dimension
- Sample complexity for learning graph structure
- Relationship to disc dimension
- Multiplex VC dimension formula

### Section 3: Graph Curvature
- Ricci curvature near K₅ (negative/hyperbolic)
- Forman curvature for K₃,₃ (saddle points)
- Curvature-based obstruction detection

### Section 4: Clustering & Hubs
- Clustering coefficient as obstruction proxy
- Hub participation in K₅ minors
- Betweenness distribution

### Section 5: Width Parameters
- Treewidth bounds from disc dimension
- Pathwidth in layered graphs
- FPT implications

### Section 6: Chromatic Properties
- Chromatic number bounds
- Graph coloring for layer separation
- 4-color theorem violations

### Section 7: Unified Framework
- All properties as obstruction indicators
- Feature vector for ML-based obstruction prediction
- Computational complexity comparison

---

## Follow-Up Queries (After Initial Analysis)

### Q9: Prediction Without Detection

```
Given network properties (LID, VC-dim, curvature, clustering, betweenness, treewidth),
can we predict disc dimension without explicitly finding forbidden minors?

Build a feature-based classifier:
- Input: [LID_mean, LID_std, clustering, betweenness_max, treewidth, ...]
- Output: disc_dimension ∈ {2, 3, 4}

Expected accuracy for brain networks?
```

### Q10: r-IDS and Obstruction Sampling

```
If we use r-IDS (r=4, target_size=50) to sample 50 representative regions from 368,
what is the probability that the induced subgraph contains all obstruction types
present in the full graph?

Coverage analysis:
- r-IDS guarantees all nodes within distance r of sample
- Obstructions (K₅, K₃,₃) have diameter ≤ 4
- Should capture all obstructions with high probability

Prove or bound this probability.
```

### Q11: Multiplex Obstruction Algebra

```
Define an algebra of multiplex obstructions:
- Union: obs₁ ∪ obs₂
- Intersection: obs₁ ∩ obs₂
- Complement: ¬obs (graphs without obstruction)

What are the closure properties?
Can we generate all Obs_M(2,2) from a finite basis?
```

---

## Implementation Plan

### Step 1: Query Ernie2
Run queries Q1-Q8 to get theoretical framework

### Step 2: Synthesize Results
Combine ernie2 insights into unified model of:
```
disc_dimension = f(LID, VC_dim, curvature, clustering, betweenness, treewidth, ...)
```

### Step 3: Validate on Synthetic Graphs
Test predictions on Watts-Strogatz + random geometric graphs

### Step 4: Apply to PRIME-DE
Compute all properties on real brain data
Compare predicted vs empirical disc dimension

### Step 5: Paper Section
Write "Theoretical Framework" section using ernie2 analysis

---

## Success Metrics

**Ernie2 analysis is successful if:**
1. ✅ Provides theoretical bounds: disc-dim → LID, VC-dim, etc.
2. ✅ Identifies at least 3 proxy measures for obstruction detection
3. ✅ Suggests efficient algorithms (better than O(n⁵) exhaustive)
4. ✅ Generates testable predictions for PRIME-DE validation
5. ✅ Produces equations/formulas for paper "Methods" section

---

## Questions for Initial Ernie2 Session

**Primary Question**:
> "Given a multiplex brain network with signal layer (small-world: C=0.5, L=3, LID≈5) and lymphatic layer (Euclidean: C=0.3, L=5, LID≈12), derive theoretical relationships between disc dimension obstructions (K₅, K₃,₃, multiplex minors) and measurable network properties (LID, VC-dimension, Ricci curvature, clustering, betweenness, treewidth). Provide bounds, formulas, and efficient detection algorithms without requiring explicit minor enumeration."

**Secondary Questions**:
1. Can LID distribution predict disc dimension within ±1?
2. What is VC dimension of typical brain connectivity graph?
3. How does Ollivier-Ricci curvature behave near K₅?
4. Can we use r-IDS (r=4, k=50) to sample all obstructions with ≥95% probability?
5. What is the minimum property set to distinguish disc-dim 2 vs 3 vs 4?

---

**Status**: Ready for ernie2_swarm query
**Next Step**: Run swarm analysis, synthesize results into paper framework
**Expected Duration**: 30-60 minutes for comprehensive theoretical analysis
