# F_i Functor Hierarchies Catalog

**Purpose**: Registry of all F_i functor hierarchies used across domains.

**Last Updated**: 2026-01-21

---

## What is a Functor Hierarchy (F_i)?

The **vertical dimension** of the tensor array. Each functor represents a different **lens/view** of the same entity.

**merge2docs example** (documents):
```
F_i functors for analyzing a codebase:
├─ wisdom   → High-level architectural insights
├─ papers   → Research papers citing this code
├─ code     → Implementation details
├─ testing  → Test coverage and validation
└─ git      → Version control history
```

**Brain example** (regions):
```
F_i functors for analyzing V1:
├─ Anatomy    → Physical structure (D99 atlas)
├─ Function   → What it computes (fMRI)
├─ Electro    → Neural dynamics (EEG)
├─ Genetics   → Gene expression (Allen)
└─ Behavior   → Task relevance (experiments)
```

---

## Registry: Domain-Specific F_i Hierarchies

### 1. Mathematics Domain

#### 1.1 Theoretical Math F_i

**ID**: `math-theoretical-v1`
**Created**: 2026-01-21
**Purpose**: Standard mathematical proof analysis

```yaml
functors:
  - name: axioms
    description: Foundational axioms used
    data_source: Manual annotation
    rids_method: Logic graph (axiom dependencies)

  - name: lemmas
    description: Supporting lemmas
    data_source: Proof structure extraction
    rids_method: Proof dependency graph (r=4)

  - name: theorems
    description: Main theorems
    data_source: Theorem database
    rids_method: Citation graph

  - name: applications
    description: Where theorem is used
    data_source: Citation analysis
    rids_method: Application domain clustering

  - name: proofs
    description: Proof techniques
    data_source: Proof mining
    rids_method: Technique similarity (r=4)

cross_functor_syndromes:
  - axioms vs proofs: "Proof uses axiom not declared"
  - theorems vs applications: "Theorem cited in wrong domain"
  - lemmas vs theorems: "Lemma not used in any theorem"
```

#### 1.2 LEAN Validation F_i

**ID**: `math-lean-validation-v1`
**Created**: 2026-01-21
**Purpose**: Formal proof verification via LEAN

```yaml
functors:
  - name: lean_types
    description: Type definitions in LEAN
    data_source: LEAN type system
    rids_method: Type dependency graph (r=3)

  - name: lean_tactics
    description: Proof tactics used
    data_source: LEAN tactic library
    rids_method: Tactic co-occurrence (r=4)

  - name: lean_theorems
    description: Formalized theorems
    data_source: LEAN mathlib
    rids_method: Theorem dependency (r=4)

  - name: lean_sorry
    description: Unproven steps (sorry markers)
    data_source: LEAN compiler warnings
    rids_method: Gap analysis (r=2)

  - name: lean_verification
    description: Verification status
    data_source: LEAN proof checker
    rids_method: Validation graph (r=1)

cross_functor_syndromes:
  - lean_theorems vs math_theorems: "Formalized ≠ informal statement"
  - lean_sorry vs lean_verification: "Unproven step claimed verified"
  - lean_types vs axioms: "LEAN type missing for axiom"

integration_with_theoretical:
  - Map lean_theorems → theorems (validation)
  - Map lean_tactics → proofs (technique verification)
  - Syndrome when informal proof valid but LEAN fails
```

### 2. Brain/Neuroscience Domain

#### 2.1 Research Brain F_i

**ID**: `brain-research-v1`
**Created**: 2026-01-21
**Purpose**: Multi-modal brain research data

```yaml
functors:
  - name: anatomy
    description: Structural organization
    data_sources:
      - D99 atlas (macaque, 368 regions)
      - Allen CCF (mouse, 800+ regions)
      - Waxholm (rat, 222 regions)
      - HCP (human, 180 regions)
    rids_method: Anatomical adjacency (r=4)
    features:
      - volume_mm3
      - centroid_xyz
      - layer_structure
      - neighbors

  - name: function
    description: Functional connectivity
    data_sources:
      - fMRI BOLD (PRIME-DE, HCP)
      - Task-based fMRI
      - Resting-state fMRI
    rids_method: Correlation-based (r=4)
    features:
      - response_selectivity
      - functional_networks
      - task_modulation

  - name: electro
    description: Neural dynamics
    data_sources:
      - EEG (scalp)
      - LFP (local field potential)
      - Single-unit recording
      - Calcium imaging
    rids_method: Oscillation coherence (r=4)
    features:
      - firing_rate_hz
      - oscillations (alpha, beta, gamma)
      - spike_timing

  - name: genetics
    description: Gene expression
    data_sources:
      - Allen Brain Atlas
      - Brain Span Atlas
      - GTEx brain samples
    rids_method: Gene expression similarity (r=4)
    features:
      - expressed_genes
      - cell_type_markers
      - transcription_factors

  - name: behavior
    description: Task-related activity
    data_sources:
      - Behavioral experiments
      - Lesion studies
      - Optogenetics
    rids_method: Behavioral coupling (r=4)
    features:
      - task_selectivity
      - attention_effects
      - learning_curves

  - name: pathology
    description: Disease markers
    data_sources:
      - Clinical neuroimaging
      - Lesion maps
      - Atrophy measures
    rids_method: Disease co-occurrence (r=3)
    features:
      - atrophy_percentage
      - lesion_presence
      - pathology_type

cross_functor_syndromes:
  - anatomy vs function: "Anatomical connection without functional correlation"
  - function vs electro: "fMRI BOLD without neural firing"
  - genetics vs pathology: "Gene mutation without pathology"
  - behavior vs function: "Behavioral deficit without BOLD change"

example_usage:
  - V1_anatomy: Structure of primary visual cortex
  - V1_function: fMRI response to visual stimuli
  - V1_electro: Gamma oscillations during attention
  - V1_genetics: PVALB+ interneuron expression
  - V1_behavior: Orientation discrimination performance
```

#### 2.2 Clinical Brain F_i

**ID**: `brain-clinical-v1`
**Created**: 2026-01-21
**Purpose**: Clinical diagnosis and treatment

```yaml
functors:
  - name: symptoms
    description: Clinical symptoms by region
    data_sources:
      - Patient reports
      - Clinical assessments
      - Diagnostic interviews
    rids_method: Symptom co-occurrence (r=3)

  - name: imaging
    description: Clinical scans
    data_sources:
      - Clinical MRI
      - CT scans
      - PET scans
    rids_method: Imaging abnormality (r=4)

  - name: diagnosis
    description: Clinical diagnoses
    data_sources:
      - ICD-10 codes
      - DSM-5 criteria
      - Clinical notes
    rids_method: Diagnosis co-occurrence (r=3)

  - name: treatment
    description: Treatment interventions
    data_sources:
      - Medication records
      - TMS protocols
      - Surgery reports
    rids_method: Treatment pathway (r=2)

  - name: outcomes
    description: Patient outcomes
    data_sources:
      - Follow-up assessments
      - Recovery metrics
      - Quality of life
    rids_method: Outcome correlation (r=3)

cross_functor_syndromes:
  - symptoms vs imaging: "Symptoms without imaging correlate"
  - diagnosis vs treatment: "Diagnosis without standard treatment"
  - treatment vs outcomes: "Treatment without expected outcome"

integration_with_research:
  - Map imaging → anatomy (localization)
  - Map symptoms → function (behavioral deficits)
  - Syndrome when research predicts outcome but clinical differs
```

#### 2.3 Computational Brain F_i

**ID**: `brain-computational-v1`
**Created**: 2026-01-21
**Purpose**: Computational models and simulations

```yaml
functors:
  - name: models
    description: Computational models
    data_sources:
      - Published models
      - ModelDB database
      - GitHub repositories
    rids_method: Model architecture similarity (r=4)

  - name: simulations
    description: Simulation results
    data_sources:
      - NEURON/NEST simulations
      - Brian2 simulations
      - Custom simulators
    rids_method: Parameter space (r=4)

  - name: predictions
    description: Model predictions
    data_sources:
      - Predicted activity patterns
      - Predicted connectivity
      - Predicted behavior
    rids_method: Prediction similarity (r=4)

  - name: validation
    description: Experimental validation
    data_sources:
      - Published experiments
      - Replication studies
      - Meta-analyses
    rids_method: Validation graph (r=2)

  - name: parameters
    description: Model parameters
    data_sources:
      - Literature estimates
      - Fitted parameters
      - Physiological constraints
    rids_method: Parameter correlation (r=4)

cross_functor_syndromes:
  - predictions vs validation: "Prediction fails experimental test"
  - models vs simulations: "Simulation doesn't match model spec"
  - parameters vs validation: "Parameters outside physiological range"

integration_with_research:
  - Map predictions → function (test against fMRI)
  - Map parameters → genetics (constrain by expression)
  - Map validation → electro (validate with recordings)
```

### 3. Software Engineering Domain

#### 3.1 merge2docs F_i (Standard)

**ID**: `software-merge2docs-v1`
**Created**: Original merge2docs design
**Purpose**: Document/codebase analysis

```yaml
functors:
  - name: wisdom
    description: High-level architectural insights
    data_source: Extracted wisdom connections
    rids_method: Semantic similarity (r=4)

  - name: papers
    description: Research papers
    data_source: Citation graph
    rids_method: Citation network (r=4)

  - name: code
    description: Implementation
    data_source: Source code parsing
    rids_method: Call graph + imports (r=4)

  - name: testing
    description: Test coverage
    data_source: Test files + coverage
    rids_method: Test dependency (r=3)

  - name: git
    description: Version control
    data_source: Git history
    rids_method: Commit co-occurrence (r=5)

cross_functor_syndromes:
  - wisdom vs code: "Architecture doesn't match implementation"
  - papers vs code: "Code doesn't implement cited algorithm"
  - code vs testing: "Code without test coverage"
  - testing vs git: "Tests failing in commit history"
```

---

## How to Add a New F_i Hierarchy

### Template

```yaml
functor_hierarchy:
  id: "{domain}-{purpose}-v{version}"
  created: "YYYY-MM-DD"
  domain: "{domain_name}"
  purpose: "{short_description}"

  functors:
    - name: "{functor_name}"
      description: "{what_this_functor_represents}"
      data_sources:
        - "{source_1}"
        - "{source_2}"
      rids_method: "{how_to_compute_rids}"
      features:
        - "{feature_1}"
        - "{feature_2}"

  cross_functor_syndromes:
    - "{functor_A} vs {functor_B}: {what_mismatch_means}"

  integration_with:
    - "{other_hierarchy_id}": "{how_they_integrate}"
```

### Example: Adding Chess Domain F_i

```yaml
functor_hierarchy:
  id: "chess-analysis-v1"
  created: "2026-01-21"
  domain: "chess"
  purpose: "Chess position and game analysis"

  functors:
    - name: positions
      description: Board positions
      data_sources:
        - PGN game databases
        - Lichess database
      rids_method: Position similarity (r=4 moves)
      features:
        - material_balance
        - king_safety
        - center_control

    - name: tactics
      description: Tactical motifs
      data_sources:
        - Puzzle databases
        - Tactical pattern recognition
      rids_method: Motif co-occurrence (r=3)
      features:
        - fork
        - pin
        - skewer

    - name: openings
      description: Opening theory
      data_sources:
        - ECO classification
        - Master game database
      rids_method: Transposition graph (r=5)
      features:
        - opening_name
        - frequency
        - win_rate

    - name: engines
      description: Engine evaluation
      data_sources:
        - Stockfish
        - Leela Chess Zero
      rids_method: Evaluation similarity (r=4)
      features:
        - centipawn_loss
        - best_move
        - evaluation

    - name: human_games
      description: Human play patterns
      data_sources:
        - Master games
        - Player statistics
      rids_method: Player style (r=4)
      features:
        - blunder_rate
        - time_usage
        - style_category

  cross_functor_syndromes:
    - positions vs tactics: "Position has tactic but not recognized"
    - engines vs human_games: "Engine prefers different move than humans"
    - openings vs engines: "Opening refuted by engine"
```

---

## Cross-Hierarchy Integration

### Example: Math ↔ Brain Integration

**Use case**: Understanding mathematical cognition

```yaml
integration:
  primary: math-theoretical-v1
  secondary: brain-research-v1

  mappings:
    - math_theorems → brain_function:
        description: "Which brain regions activate for this theorem type?"
        method: "fMRI during theorem comprehension"

    - math_proofs → brain_electro:
        description: "Neural dynamics during proof reading"
        method: "EEG during proof verification"

    - math_axioms → brain_genetics:
        description: "Genetic basis for mathematical ability"
        method: "GWAS for math performance"

  cross_integration_syndromes:
    - "Theorem understood (behavior) but no PFC activation (function)"
    - "Proof error detected (math) but no error-related negativity (electro)"
```

### Example: LEAN ↔ Brain Integration

**Use case**: Neural basis of formal reasoning

```yaml
integration:
  primary: math-lean-validation-v1
  secondary: brain-research-v1

  mappings:
    - lean_tactics → brain_function:
        description: "Which brain regions for different tactics?"
        method: "fMRI during LEAN proof construction"

    - lean_sorry → brain_electro:
        description: "Neural signature of proof gaps"
        method: "EEG when encountering 'sorry'"

  cross_integration_syndromes:
    - "LEAN proof valid but subject reports difficulty (behavior mismatch)"
```

---

## Database Schema for F_i Registry

```sql
CREATE TABLE functor_hierarchies (
    id TEXT PRIMARY KEY,  -- e.g., "brain-research-v1"
    created_at TIMESTAMP DEFAULT NOW(),
    domain TEXT NOT NULL,  -- e.g., "neuroscience"
    purpose TEXT NOT NULL,
    version INTEGER DEFAULT 1,

    functors JSONB NOT NULL,  -- Array of functor definitions
    cross_functor_syndromes JSONB,  -- Syndrome definitions
    integration_with JSONB,  -- Links to other hierarchies

    is_active BOOLEAN DEFAULT TRUE,
    deprecated_by TEXT REFERENCES functor_hierarchies(id)
);

CREATE TABLE functor_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hierarchy_id TEXT REFERENCES functor_hierarchies(id),
    entity_id TEXT NOT NULL,  -- e.g., "V1" (region name)
    functor_name TEXT NOT NULL,  -- e.g., "anatomy"

    features JSONB,  -- Actual feature values
    rids_connections INTEGER[],  -- r-IDS connections
    syndrome_history JSONB,  -- Syndrome values over time

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(hierarchy_id, entity_id, functor_name)
);

CREATE INDEX idx_functor_instances_entity
    ON functor_instances(entity_id);
CREATE INDEX idx_functor_instances_hierarchy
    ON functor_instances(hierarchy_id);
```

---

## Usage Examples

### 1. Load a Functor Hierarchy

```python
def load_functor_hierarchy(hierarchy_id: str) -> Dict:
    """Load functor hierarchy definition from registry."""
    result = db.query(
        "SELECT * FROM functor_hierarchies WHERE id = %s",
        (hierarchy_id,)
    ).first()

    return {
        "id": result.id,
        "domain": result.domain,
        "functors": result.functors,
        "syndromes": result.cross_functor_syndromes
    }

# Example
brain_fi = load_functor_hierarchy("brain-research-v1")
print(f"Brain F_i has {len(brain_fi['functors'])} functors")
# "Brain F_i has 6 functors"
```

### 2. Compute Cross-Functor Syndrome

```python
def compute_cross_functor_syndrome(
    entity_id: str,  # e.g., "V1"
    hierarchy_id: str,  # e.g., "brain-research-v1"
    functor_A: str,  # e.g., "anatomy"
    functor_B: str   # e.g., "function"
) -> float:
    """Compute syndrome between two functors for same entity."""

    # Load both functor instances
    instance_A = db.query(FunctorInstance).filter_by(
        hierarchy_id=hierarchy_id,
        entity_id=entity_id,
        functor_name=functor_A
    ).first()

    instance_B = db.query(FunctorInstance).filter_by(
        hierarchy_id=hierarchy_id,
        entity_id=entity_id,
        functor_name=functor_B
    ).first()

    # Get r-IDS connections
    rids_A = set(instance_A.rids_connections)
    rids_B = set(instance_B.rids_connections)

    # Syndrome = Jaccard distance
    intersection = len(rids_A & rids_B)
    union = len(rids_A | rids_B)

    syndrome = 1.0 - (intersection / union) if union > 0 else 0.0

    return syndrome

# Example
syndrome = compute_cross_functor_syndrome(
    entity_id="V1",
    hierarchy_id="brain-research-v1",
    functor_A="anatomy",
    functor_B="function"
)

print(f"V1 anatomy-function syndrome: {syndrome:.3f}")
# "V1 anatomy-function syndrome: 0.234"
# (Anatomy and function somewhat agree on connections)
```

### 3. Register New F_i Hierarchy

```python
def register_functor_hierarchy(definition: Dict):
    """Register new F_i hierarchy in database."""

    hierarchy = FunctorHierarchy(
        id=definition["id"],
        domain=definition["domain"],
        purpose=definition["purpose"],
        functors=definition["functors"],
        cross_functor_syndromes=definition.get("cross_functor_syndromes", [])
    )

    db.session.add(hierarchy)
    db.session.commit()

    print(f"✅ Registered F_i hierarchy: {definition['id']}")

# Example: Register chess F_i
chess_fi = {
    "id": "chess-analysis-v1",
    "domain": "chess",
    "purpose": "Chess position and game analysis",
    "functors": [
        {
            "name": "positions",
            "description": "Board positions",
            "data_sources": ["PGN databases"],
            "rids_method": "Position similarity (r=4)"
        },
        # ... more functors
    ]
}

register_functor_hierarchy(chess_fi)
```

---

## Migration Guide

### From Single-Functor to Multi-Functor

If you have existing region tensors with only `function` functor:

```python
async def migrate_to_multi_functor(session_id: str):
    """Add anatomy functor to existing function-only tensors."""

    # Get all existing tensors
    existing = db.query(BrainRegionTensor).filter_by(
        fmri_session_id=session_id
    ).all()

    for tensor in existing:
        # Existing data becomes 'function' functor
        function_instance = FunctorInstance(
            hierarchy_id="brain-research-v1",
            entity_id=tensor.region_name,
            functor_name="function",
            features=tensor.features,
            rids_connections=tensor.rids_connections
        )
        db.session.add(function_instance)

        # Add anatomy functor from atlas
        atlas_data = query_atlas(tensor.region_name, atlas="D99")
        anatomy_instance = FunctorInstance(
            hierarchy_id="brain-research-v1",
            entity_id=tensor.region_name,
            functor_name="anatomy",
            features={
                "volume_mm3": atlas_data.volume,
                "centroid_xyz": atlas_data.centroid
            },
            rids_connections=atlas_data.neighbors
        )
        db.session.add(anatomy_instance)

    db.session.commit()
    print(f"✅ Migrated {len(existing)} tensors to multi-functor")
```

---

## Summary

**What we've cataloged**:
1. ✅ Math (theoretical + LEAN validation)
2. ✅ Brain (research + clinical + computational)
3. ✅ Software (merge2docs standard)
4. ✅ Chess (example for games domain)

**How to use**:
- Look up F_i hierarchy by ID
- Load functor definitions
- Compute cross-functor syndromes
- Integrate multiple hierarchies

**Where to find**:
- **This file**: `/docs/designs/yada-hierarchical-brain-model/FUNCTOR_HIERARCHIES_CATALOG.md`
- **Database**: `functor_hierarchies` table
- **Code**: `src/backend/qec/functor_registry.py`

---

**Last Updated**: 2026-01-21
**Maintainer**: twosphere-mcp team
**Related**: `BRAIN_QEC_CACHE_CROSSTRAINING.md`, `QEC_TENSOR_BRAIN_MAPPING.md`
