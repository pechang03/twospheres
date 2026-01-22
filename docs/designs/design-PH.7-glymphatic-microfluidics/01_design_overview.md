# PH-7: Glymphatic-Microfluidics Integration Design

## Overview

This design documents the cross-domain integration between:
1. **PHLoC microfluidics** (organoid culture, CFD simulation)
2. **Brain glymphatic system** (CSF flow in perivascular spaces)
3. **Disc dimension analysis** (brain network topology complexity)

**Key Insight**: The same Stokes flow physics (Re << 1) governs both artificial microfluidic channels and biological perivascular spaces, enabling validated PHLoC designs to inform glymphatic flow models.

## Task Reference

- **Task ID**: PH-7
- **Status**: Open
- **Priority**: 1 (High)
- **Labels**: cross-domain, glymphatic, microfluidics, disc-dimension

## Problem Statement

### The Glymphatic System

The brain's glymphatic system clears metabolic waste (amyloid-β, tau) through:
- Perivascular spaces around arteries (3-50 µm gaps)
- CSF flow driven by arterial pulsations
- ~60% increased clearance during sleep

**Clinical Relevance**: Impaired glymphatic function correlates with Alzheimer's disease and other neurodegenerative conditions.

### The Modeling Challenge

Current fMRI research lacks tools to:
1. Simulate CSF flow at the perivascular scale
2. Connect network topology (disc dimension) to clearance efficiency
3. Validate models with controllable experimental systems

## Solution: Cross-Domain Physics Bridge

### Shared Physics: Stokes Flow Regime

| Parameter | PHLoC Organoid | Glymphatic System |
|-----------|---------------|-------------------|
| Channel size | 10-100 µm | 3-50 µm |
| Flow velocity | 0.1-10 mm/s | 10-20 µm/s |
| Reynolds number | 0.001-0.1 | 0.0001-0.001 |
| Viscosity | ~1 cP (media) | ~1.2 cP (CSF) |
| Wall shear limit | <0.5 Pa | ~0.1-0.5 Pa |

Both systems obey the **Hagen-Poiseuille equation**:
```
Q = (πδ⁴/128μL) ΔP
```

### Disc Dimension Connection

The **disc dimension** measures the minimum topological dimension to embed a brain network without edge crossings.

**Hypothesis**: Networks with lower disc dimension provide more efficient waste clearance pathways.

| Disc Dimension | Network Topology | Predicted Clearance |
|----------------|-----------------|---------------------|
| disc ≤ 2 | Planar (tree-like) | High efficiency |
| disc = 3-4 | Moderately complex | Medium efficiency |
| disc ≥ 5 | Highly non-planar | Low efficiency |

**Biological Observation**: Real brain networks have disc ≈ 5 with sparse cross-layer coupling, maintaining FPT-tractability.

## Architecture

### Three-Layer Model

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL LAYER (G_signal)                  │
│  Neural connectivity graph, disc ≈ 5                        │
│  Embedded on inner sphere (white matter)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ metabolic coupling
                              │ ∑q_e = α·m_v
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LYMPHATIC LAYER (G_lymph)                  │
│  Perivascular + sulcal CSF network, disc ≈ 5               │
│  CFD simulation via Stokes flow                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ validation
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  MICROFLUIDIC LAYER (PHLoC)                 │
│  Experimental validation chip                               │
│  Same physics, controllable parameters                      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PHLoC CFD Simulation (validated)
        │
        ▼
Scale to glymphatic dimensions
        │
        ▼
Apply to brain network topology
        │
        ▼
Predict clearance efficiency from disc dimension
        │
        ▼
Validate against fMRI glymphatic imaging
```

## Implementation Components

### 1. Glymphatic CFD Simulator

**File**: `prototypes/glymphatic_cfd_simulator.py`

Extends `handle_cfd_microfluidics()` with:
- Perivascular space geometry (annular gap)
- Cardiac pulse-driven flow (pulsatile boundary conditions)
- Sleep/wake state transitions (60% flow increase)
- Amyloid-β clearance rate estimation

### 2. Disc-Dimension Clearance Model

**File**: `prototypes/disc_dimension_clearance.py`

Connects network topology to waste clearance:
- Input: Brain graph G with edge weights (connectivity strength)
- Compute disc dimension via obstruction detection
- Model clearance efficiency as f(disc, treewidth)
- Output: Predicted regional clearance rates

### 3. Brain-on-Chip Design Generator

**File**: `prototypes/brain_on_chip_designer.py`

Generates PHLoC designs for glymphatic validation:
- Network topology from brain atlas (D99, AAL)
- Scale to microfluidic dimensions
- Add optical monitoring (absorption spectroscopy)
- Export to FreeCAD MCP for fabrication

## Key Equations

### Perivascular Flow (Annular Gap)

For flow between two concentric cylinders (vessel wall + tissue):
```
Q = (πΔP/8μL) × [R_out⁴ - R_in⁴ - (R_out² - R_in²)²/ln(R_out/R_in)]
```

Simplified for thin gap (δ = R_out - R_in << R_in):
```
Q ≈ (πR·δ³/6μL) ΔP
```

### Metabolic Coupling Constraint

At each node v with metabolic rate m_v:
```
∑_{e ∈ edges(v)} q_e = α · m_v
```

Where:
- q_e = flow rate on edge e
- α = clearance coefficient (~0.1 for awake, ~0.16 for sleep)

### Clearance Efficiency Model

Proposed model relating disc dimension to clearance:
```
η_clearance = η_max × exp(-β × (disc - disc_opt)²)
```

Where:
- η_max = maximum clearance efficiency
- disc_opt ≈ 2-3 (optimal disc dimension)
- β = sensitivity parameter

## Validation Pathway

### Phase 1: In-Vitro (PHLoC)

1. Fabricate microfluidic chip with brain-inspired topology
2. Compare straight channels vs. low-disc network patterns
3. Measure fluorescent tracer clearance rates
4. Validate CFD predictions

### Phase 2: Ex-Vivo (Brain Tissue)

1. Apply CFD model to mouse brain atlas
2. Compare predicted vs. measured clearance (tracer studies)
3. Correlate disc dimension with regional clearance

### Phase 3: In-Vivo (fMRI)

1. Use glymphatic fMRI sequences (diffusion tensor imaging)
2. Map perivascular space dimensions
3. Validate disc dimension → clearance hypothesis

## Dependencies

### Internal
- `src/backend/mri/disc_dimension_analysis.py` - Disc dimension computation
- `src/backend/mri/fast_obstruction_detection.py` - K₅/K₃,₃ detection
- `bin/twosphere_mcp.py` - `handle_cfd_microfluidics()` function

### External
- OpenFOAM/CfdOF for full 3D CFD (optional)
- NetworkX for graph operations
- NumPy/SciPy for numerical computation

## Files in This Design

```
docs/designs/design-PH.7-glymphatic-microfluidics/
├── 01_design_overview.md              # This file
├── 02_theory_stokes_flow.md           # Stokes flow physics derivations
├── 03_disc_clearance_model.md         # Disc dimension → clearance theory
└── prototypes/
    ├── glymphatic_cfd_simulator.py    # CFD for perivascular flow
    ├── disc_dimension_clearance.py    # Topology → efficiency model
    └── brain_on_chip_designer.py      # PHLoC design generator
```

## References

1. **Nedergaard & Goldman (2020)** - "Glymphatic failure as a final common pathway to dementia"
2. **Paul, Protopapas, Thilikos (2023)** - "Graph Parameters, Universal Obstructions, and WQO" arXiv:2304.03688
3. **Iliff et al. (2012)** - "A paravascular pathway facilitates CSF flow through the brain parenchyma"

## Related Tasks

- PH-2: Complete composition level (CFD integration)
- PH-4: Integration systems (brain network analysis)
- design-ph-2.1-glymphatic_extensions: Existing glymphatic model framework
