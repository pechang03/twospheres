# Ernie2 Swarm Integration: 36 Collections + twosphere-mcp

**Date:** 2026-01-20
**Status:** Planning
**Priority:** High

> **⚠️ NOTE:** This document focuses on ernie2_swarm query integration patterns. For complete brain communication integration analysis (MRI functional connectivity ↔ optical sensing), see **[BRAIN_COMMUNICATION_INTEGRATION.md](./BRAIN_COMMUNICATION_INTEGRATION.md)**.

## Overview

The merge2docs **ernie2_swarm** has **36 domain-specific collections** with trained QTRM (Quantum Theory Router Models) and domain experts. Two collections directly align with twosphere-mcp's **brain communication focus**:

1. **`docs_library_neuroscience_MRI`** - fMRI functional connectivity, neural circuits, brain communication patterns
2. **`docs_library_bioengineering_LOC`** - Lab-on-Chip biosensing, tissue engineering, optical detection

These collections contain domain-specific research papers on:
- **Brain communication** measurement via MRI (distance correlation, phase-locking)
- **Optical sensing** methods for molecular detection (interferometry, spectroscopy)
- **Cross-domain bridges** connecting MRI biomarkers to optical measurements

The collections can **augment our MCP tools with expert knowledge** by:
- Suggesting optimal parameter ranges (e.g., refractive index sensitivity for neurotransmitter detection)
- Identifying relevant frequency bands (e.g., alpha/gamma oscillations in brain communication)
- Providing cross-validation strategies (e.g., correlating MRI connectivity with optical binding kinetics)

---

## The 36 Collections Architecture

### System Components

**From:** `merge2docs/docs/tasks/TASK_STRUT_SCALING_36_COLLECTIONS.md`

#### 1. **QTRM Models** (Quantum Theory Router Models)
Per collection, 15 models trained:
- 7 F_i level models (F₀-F₆) - Functor hierarchy routing
- 1 main router - Collection-level classifier
- 1 domain/style classifier - Tone and approach
- 6 QEC bridges - Cross-level transitions (F₀↔F₁, ..., F₅↔F₆)

**Example: Art Collection (✅ Complete)**
- Routes questions to appropriate abstraction level
- Bridges between quantum/primitive (F₀) and meta/planning (F₅)
- Falls back gracefully when uncertainty is high

#### 2. **CTL Sampling** (Central Limit Theorem)
- Selects r-1 key papers per collection (typically 2-4 papers)
- Uses centrality metrics: degree, betweenness, eigenvector centrality
- Scores by: citation count + semantic centrality + syntactic links
- Ensures diversity across subclusters

#### 3. **YADA Structures** (Yet Another Directed Acyclic)
Neo4j graph structure for each key paper:
```
Paper → Concepts → Analogies → Cross-Domain Bridges
  ↓         ↓           ↓              ↓
Labels  Relations  Hypotheses    Interdisciplinary
```

---

## Relevant Collections for twosphere-mcp

### 1. `docs_library_neuroscience_MRI` 🧠

**Domain Expert:** NeuroscienceExpert
**Key Topics:** fMRI, neural circuits, DTI, functional connectivity
**Status:** Planning (Phase 2)

**Direct Integration Points:**

| MRISpheres/twospheres Feature | Collection Resource | Integration |
|------------------------------|---------------------|-------------|
| 4D MRI time-series analysis | fMRI papers + embeddings | Query for FFT methods |
| Two-sphere brain model | Neural circuit papers | Geodesic distance refs |
| Phase-locking value (PLV) | Functional connectivity | Mathematical frameworks |
| DTI fractional anisotropy | White matter papers | FA calculation methods |
| Cross-frequency coupling | Neural oscillations | Phase-amplitude coupling |

**MCP Tool Enhancement:**
```python
# NEW: Query ernie2_swarm before processing
async def compute_fft_correlation_enhanced(signals, question=None):
    # 1. Query neuroscience_MRI collection for relevant context
    context = await query_ernie2_swarm(
        question="How to compute phase-locking value between fMRI signals?",
        collections=["docs_library_neuroscience_MRI"]
    )

    # 2. Use domain expert knowledge to guide analysis
    fft_method = extract_best_practice(context)

    # 3. Apply our implementation with expert-informed parameters
    return compute_fft_correlation(signals, method=fft_method)
```

**Key Papers (CTL Sampled):**
- Friston et al. - Dynamic causal modeling
- Bullmore & Sporns - Complex brain networks
- Poldrack - fMRI preprocessing pipelines

---

### 2. `docs_library_bioengineering_LOC` 🔬

**Domain Expert:** BioengineeringExpert
**Key Topics:** Lab-on-Chip, tissue engineering, biomaterials, microfluidics
**Status:** Planning (Phase 2)
**Document Count:** 3 papers (needs expansion)

**Direct Integration Points:**

| twosphere-mcp Feature | Collection Resource | Integration |
|----------------------|---------------------|-------------|
| InterferometricSensor | LOC biosensing papers | Refractive index sensing |
| Ring resonator feedback | Photonic biosensors | Resonance tracking methods |
| Absorption spectroscopy | Beer-Lambert applications | Concentration measurement |
| Microfluidic channels | CFD simulations (planned) | Flow dynamics |
| Fabrication constraints | Tolerance analysis papers | Alignment sensitivity |

**MCP Tool Enhancement:**
```python
# NEW: LOC design with expert knowledge
async def design_loc_biosensor_enhanced(
    target_analyte: str,
    sensitivity_requirement: float
):
    # 1. Query LOC collection for similar designs
    context = await query_ernie2_swarm(
        question=f"Design LOC sensor for {target_analyte} detection",
        collections=["docs_library_bioengineering_LOC"]
    )

    # 2. Extract design patterns from expert papers
    design_params = extract_design_parameters(context)

    # 3. Use our OpticalSystem with informed parameters
    return design_loc_system(
        wavelength=design_params.optimal_wavelength,
        resonator_q=design_params.quality_factor,
        sensitivity=sensitivity_requirement
    )
```

**Integration with OOC Platform:**
From `design-6.1-biological-analysis-integration/biological_analysis_framework.md`:
- OOC sensors → real-time data
- pH, lactate monitoring at Days 7, 14, 21
- DMR modules → OOC sensor data

---

### 3. `docs_library_physics_optics` 🔬

**Domain Expert:** OpticsExpert
**Key Topics:** Photonics, lasers, optical systems, ray tracing
**Status:** Planning (Phase 2)

**Direct Integration Points:**

| twosphere-mcp Feature | Collection Resource | Integration |
|----------------------|---------------------|-------------|
| pyoptools ray tracing | Optical design papers | Lens optimization |
| Grating spectroscopy | Diffraction theory | Wavelength separation |
| MTF analysis | Image quality metrics | System optimization |
| Fiber coupling | Beam propagation | Coupling efficiency |
| Polarization | Jones/Mueller calculus | Quantum bridge |

**MCP Tool Enhancement:**
```python
# NEW: Optical design with expert knowledge
async def optimize_resonator_enhanced(
    target_q_factor: float,
    wavelength_nm: float
):
    # 1. Query optics collection for resonator designs
    context = await query_ernie2_swarm(
        question=f"Design ring resonator with Q={target_q_factor}",
        collections=["docs_library_physics_optics"]
    )

    # 2. Extract optimization strategies
    strategies = extract_optimization_methods(context)

    # 3. Apply merit function from domain knowledge
    return optimize_with_merit_function(
        merit_func=strategies.strehl_ratio_mtf,
        constraints=strategies.fabrication_limits
    )
```

---

### 4. Supporting Collections

#### `docs_library_physics` ⚛️
- Quantum mechanics (for quantum optics, F₀ level)
- Field theory (for electromagnetic propagation)
- Statistical mechanics (for thermal noise)

#### `docs_library_statistics` 📊
- Bayesian inference (for emcee integration)
- Error propagation (for uncertainty analysis)
- Central Limit Theorem (for signal averaging)

#### `docs_library_MachineLearning` 🤖
- Deep learning for MRI segmentation
- CNN for image analysis
- Predictive modeling (drug response)

#### `docs_library_SoftwareEngineering` 💻
- Architecture patterns (for MCP server)
- Testing strategies (for TDD approach)
- API design (for service layer)

---

## Integration Architecture

### Level 1: Query Augmentation

**Before:**
```python
# User asks question
result = await interferometric_sensing(position, intensity)
```

**After (Enhanced with Ernie2):**
```python
# User asks question
async def interferometric_sensing_enhanced(position, intensity, question=None):
    # 1. Query relevant collections for context
    if question:
        context = await query_ernie2_swarm(
            question=question,
            collections=[
                "docs_library_neuroscience_MRI",
                "docs_library_bioengineering_LOC",
                "docs_library_physics_optics"
            ]
        )

        # 2. Extract domain-specific parameters
        params = extract_parameters_from_context(context)
    else:
        params = default_params

    # 3. Run our implementation with informed params
    return await fit_visibility(position, intensity, **params)
```

### Level 2: Domain Expert Routing

**F_i Hierarchy Alignment:**

| twosphere-mcp Level | merge2docs F_i | Collection Routing |
|--------------------|--------------|--------------------|
| Quantum optics (lab6) | F₀ Quantum/Primitive | physics, optics |
| Material properties | F₁ Physics/Chemistry | physics_optics, bioengineering |
| LOC system design | F₂ Composition | bioengineering_LOC |
| Service layer | F₃ Service | software_engineering |
| Integration (MRI+LOC) | F₄ Integration | neuroscience_MRI, bioengineering |
| Meta/Planning (optimization) | F₅ Meta/Planning | mathematics, statistics |
| Deployment (clinical) | F₆ Deployment | policies, meta_analysis |

**Routing Example:**
```python
# User question determines F_i level and collection
question = "How to optimize ring resonator Q factor?"

# Ernie2 routes to:
# - F_i level: F₅ (optimization/meta-planning)
# - Collections: physics_optics, mathematics
# - Expert: OpticsExpert + MathematicsExpert

# Returns: Multi-objective optimization strategies
```

### Level 3: Cross-Domain Bridges (YADA)

**YADA Structure Integration:**

```cypher
// Example: Bridge between MRI and LOC domains
MATCH (mri:Paper {collection: "neuroscience_MRI"})-[:HAS_CONCEPT]->(c1:LABEL_CONCEPT {name: "phase_locking"})
MATCH (loc:Paper {collection: "bioengineering_LOC"})-[:HAS_CONCEPT]->(c2:LABEL_CONCEPT {name: "optical_phase_detection"})
MATCH (c1)-[:ANALOGIZES]->(c2)
RETURN mri, loc, c1, c2

// Result: MRI phase-locking ↔ LOC optical phase detection
// Common framework: Lock-in amplification!
```

**Applications:**
- **Cancer drug testing:** MRI brain changes ↔ LOC molecular binding
- **Biomarker discovery:** fMRI patterns ↔ Protein expression
- **Multi-modal validation:** In vivo (MRI) ↔ In vitro (LOC)

---

## Implementation Roadmap

### Phase 1: Basic Query Integration (Week 1)

**Task:** Add ernie2_swarm queries to MCP tools

1. **Create integration module:**
   ```python
   # src/backend/services/ernie2_integration.py
   class Ernie2SwarmClient:
       async def query(self, question: str, collections: List[str]) -> str:
           """Query ernie2_swarm via subprocess or API."""
           pass
   ```

2. **Enhance 4 MCP tools:**
   - `interferometric_sensing` → neuroscience_MRI, physics_optics
   - `lock_in_detection` → physics_optics, statistics
   - `absorption_spectroscopy` → bioengineering_LOC, physics_optics
   - `optimize_resonator` → physics_optics, mathematics

3. **Add collection parameter:**
   ```python
   Tool(
       name="interferometric_sensing",
       inputSchema={
           # ... existing params ...
           "query_expert_collections": {
               "type": "array",
               "items": {"type": "string"},
               "description": "Optional: Query ernie2_swarm collections for context"
           }
       }
   )
   ```

**Bead:** `twosphere-mcp-XXX` - "Integrate ernie2_swarm query capability"

### Phase 2: F_i Level Routing (Week 2-3)

**Task:** Align twosphere-mcp with F_i hierarchy

1. **Map existing tools to F_i levels:**
   ```python
   FI_LEVEL_MAPPING = {
       "quantum_optics": "F0",  # Quantum/Primitive
       "material_properties": "F1",  # Physics
       "loc_system_design": "F2",  # Composition
       "service_layer": "F3",  # Service
       "mri_loc_integration": "F4",  # Integration
       "optimization": "F5",  # Meta/Planning
       "clinical_deployment": "F6",  # Deployment
   }
   ```

2. **Add F_i routing to MCP server:**
   ```python
   async def route_to_fi_level(question: str) -> Tuple[str, List[str]]:
       """Route question to F_i level and relevant collections."""
       fi_level = classify_fi_level(question)
       collections = get_collections_for_fi_level(fi_level)
       return fi_level, collections
   ```

3. **Create domain expert prompts:**
   - NeuroscienceExpert: "Focus on fMRI analysis, neural circuits"
   - BioengineeringExpert: "Focus on LOC, microfluidics, biosensors"
   - OpticsExpert: "Focus on photonics, resonators, spectroscopy"

**Bead:** `twosphere-mcp-YYY` - "Add F_i level routing to MCP tools"

### Phase 3: YADA Cross-Domain Bridges (Week 4-5)

**Task:** Connect MRI and LOC knowledge via YADA structures

1. **Identify bridge concepts:**
   ```python
   BRIDGE_CONCEPTS = {
       "phase_locking_value": ["neuroscience_MRI", "physics_optics"],
       "refractive_index_sensing": ["bioengineering_LOC", "physics_optics"],
       "spectroscopy": ["neuroscience_MRI", "bioengineering_LOC", "physics_optics"],
       "drug_response_monitoring": ["neuroscience_MRI", "bioengineering_LOC"]
   }
   ```

2. **Query YADA structures:**
   ```python
   async def find_cross_domain_analogies(
       concept: str,
       source_collection: str,
       target_collection: str
   ) -> List[Dict]:
       """Find analogies between domains via YADA graph."""
       query = f"""
       MATCH (src:Paper {{collection: '{source_collection}'}})-[:HAS_CONCEPT]->(c1:LABEL_CONCEPT {{name: '{concept}'}})
       MATCH (c1)-[:ANALOGIZES]->(c2:LABEL_CONCEPT)
       MATCH (tgt:Paper {{collection: '{target_collection}'}})-[:HAS_CONCEPT]->(c2)
       RETURN src, tgt, c1, c2
       """
       return await neo4j_query(query)
   ```

3. **Add cross-domain MCP tool:**
   ```python
   Tool(
       name="find_interdisciplinary_insights",
       description="Find cross-domain insights by querying YADA structures",
       inputSchema={
           "concept": "phase_locking",
           "domains": ["neuroscience_MRI", "physics_optics"]
       }
   )
   ```

**Bead:** `twosphere-mcp-ZZZ` - "Integrate YADA cross-domain bridges"

### Phase 4: CTL Key Paper Sampling (Week 6)

**Task:** Sample key papers for twosphere-mcp domain

1. **Create twosphere-mcp collection:**
   ```bash
   # Add our docs to merge2docs ChromaDB
   python bin/add_collection.py \
     --name docs_library_twosphere_mcp \
     --docs docs/designs/*.md \
     --docs docs/SPECTROSCOPY_IMPLEMENTATION_SUMMARY.md \
     --docs docs/MRI_TWOSPHERES_INTEGRATION.md
   ```

2. **Run CTL sampling:**
   ```bash
   python bin/sample_key_papers_clt.py \
     --collection docs_library_twosphere_mcp \
     --r-value 3 \
     --output data/twosphere_mcp_key_papers.json
   ```

3. **Build YADA structures:**
   ```bash
   python bin/build_yada_structure.py \
     --paper-ids-file data/twosphere_mcp_key_papers.json \
     --neo4j-uri bolt://localhost:7687
   ```

**Bead:** `twosphere-mcp-AAA` - "Create twosphere-mcp collection in ernie2_swarm"

---

## Collection Statistics

### Current Status (from TASK_STRUT_SCALING_36_COLLECTIONS.md)

| Category | Collections | Status | Relevant to twosphere-mcp |
|----------|-------------|--------|--------------------------|
| **Mathematical Sciences** | 5 | Planning | Statistics, Mathematics |
| **Life Sciences** | 4 | Planning | **MRI, LOC** ✅ |
| **Computer Science & AI** | 3 | Planning | ML, Software Eng |
| **Physical Sciences** | - | Planning | **Physics, Optics** ✅ |
| **Interdisciplinary** | 4 | Planning | Meta-Analysis |
| **Total** | **36** | 1 complete (art) | **4 high-priority** |

### Priority for twosphere-mcp

**High Priority (P1):**
1. `docs_library_neuroscience_MRI` - Direct MRI analysis integration
2. `docs_library_bioengineering_LOC` - Core LOC biosensing domain
3. `docs_library_physics_optics` - Optical system design
4. `docs_library_statistics` - Bayesian analysis, error propagation

**Medium Priority (P2):**
5. `docs_library_physics` - Quantum optics, field theory
6. `docs_library_MachineLearning` - MRI segmentation, predictive models
7. `docs_library_mathematics` - Optimization, differential geometry

**Low Priority (P3):**
8. `docs_library_SoftwareEngineering` - Architecture patterns
9. `docs_library_policies` - Clinical validation, FDA/CE

---

## Benefits of Integration

### 1. **Expert-Informed Parameters**
Instead of guessing optimal parameters, query domain experts:
- Ring resonator Q factor → OpticsExpert
- Bayesian MCMC settings → StatisticsExpert
- MRI preprocessing → NeuroscienceExpert

### 2. **Cross-Domain Discovery**
YADA structures reveal unexpected connections:
- MRI phase-locking ↔ Optical lock-in detection
- fMRI connectivity ↔ LOC sensor networks
- Drug response (MRI) ↔ Molecular binding (LOC)

### 3. **Fallback Robustness**
If local implementation fails, query experts for alternatives:
```python
try:
    result = await fit_visibility_bayesian(...)
except ConvergenceError:
    # Query expert for alternative methods
    alternative = await query_ernie2_swarm(
        "What are alternative methods for Bayesian visibility fitting?",
        collections=["statistics", "physics_optics"]
    )
    result = apply_alternative_method(alternative)
```

### 4. **Documentation Augmentation**
Automatically enrich docstrings with expert references:
```python
async def fit_visibility(self, ...):
    """
    Fit interference pattern and compute visibility.

    Expert References:
    - Friston (2003): Bayesian methods for fMRI (neuroscience_MRI)
    - Goodman (2007): Statistical optics (physics_optics)
    - Gelman (2013): Bayesian Data Analysis (statistics)

    [Automatically populated from ernie2_swarm collections]
    """
```

---

## Dependencies

### External Services
- **Neo4j:** bolt://localhost:7687 (YADA structures)
- **PostgreSQL:** merge2docs_dev database
- **ChromaDB:** Document embeddings
- **Nomic:** Embedding service (port 8765)

### Python Packages
```python
# requirements.txt additions for ernie2 integration
neo4j>=5.0.0          # YADA graph queries
psycopg2-binary>=2.9  # PostgreSQL connection
chromadb>=0.4.0       # Vector database
```

### merge2docs Modules
```python
# Import from merge2docs
from merge2docs.src.backend.services.ernie2net import (
    Ernie2SwarmWorkflow,
    DomainExpertRegistry,
    QTRMRouter
)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/backend/integration/test_ernie2_integration.py

async def test_query_neuroscience_mri_collection():
    """Test querying MRI collection for fMRI methods."""
    client = Ernie2SwarmClient()
    result = await client.query(
        question="How to compute phase-locking value?",
        collections=["docs_library_neuroscience_MRI"]
    )
    assert "PLV" in result
    assert "phase synchronization" in result

async def test_query_loc_collection():
    """Test querying LOC collection for biosensor design."""
    client = Ernie2SwarmClient()
    result = await client.query(
        question="Design ring resonator biosensor for protein detection",
        collections=["docs_library_bioengineering_LOC"]
    )
    assert "refractive index" in result
    assert "sensitivity" in result
```

### Integration Tests
```python
async def test_interferometric_sensing_with_expert_knowledge():
    """Test enhanced interferometric sensing with ernie2."""
    # Generate synthetic interference pattern
    position = np.linspace(0, 10, 100)
    intensity = generate_interference_pattern(position)

    # Query with expert knowledge
    result = await interferometric_sensing_enhanced(
        position, intensity,
        query_collections=["neuroscience_MRI", "physics_optics"]
    )

    # Should have better accuracy with expert-informed params
    assert result["visibility_uncertainty"] < baseline_uncertainty
```

---

## Documentation

### Files to Create
1. `docs/designs/ERNIE2_SWARM_INTEGRATION.md` (THIS FILE)
2. `src/backend/services/ernie2_integration.py` (integration module)
3. `tests/backend/integration/test_ernie2_integration.py` (tests)
4. `docs/guides/USING_ERNIE2_COLLECTIONS.md` (user guide)

### Files to Update
1. `bin/twosphere_mcp.py` - Add collection query parameter to tools
2. `requirements.txt` - Add neo4j, chromadb dependencies
3. `README.md` - Document ernie2_swarm integration

---

## Next Steps

**Immediate (this week):**
1. Create ernie2_integration.py module
2. Add collection query to interferometric_sensing tool
3. Test basic query to neuroscience_MRI collection
4. Document integration architecture

**Short-term (next 2 weeks):**
1. Align all MCP tools with F_i hierarchy
2. Create domain expert routing logic
3. Add YADA cross-domain bridge queries
4. Integration testing

**Medium-term (next 4-6 weeks):**
1. Create twosphere-mcp collection in ernie2_swarm
2. CTL sample key papers from our docs
3. Build YADA structures for twosphere-mcp
4. Cross-domain validation (MRI ↔ LOC)

---

## Related Beads

**To Create:**
- `twosphere-mcp-XXX` - "Integrate ernie2_swarm query capability" (P1)
- `twosphere-mcp-YYY` - "Add F_i level routing to MCP tools" (P1)
- `twosphere-mcp-ZZZ` - "Integrate YADA cross-domain bridges" (P2)
- `twosphere-mcp-AAA` - "Create twosphere-mcp collection in ernie2_swarm" (P2)

**Dependencies:**
- merge2docs QTRM training (Phase 2: neuroscience, bioengineering, optics)
- Neo4j YADA structures
- CTL key paper sampling

---

**End of Integration Analysis**
