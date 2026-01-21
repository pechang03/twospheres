# MRISpheres/twospheres Integration Analysis

**Date:** 2026-01-20
**Project:** twosphere-mcp ↔ MRISpheres/twospheres
**Purpose:** Integrate brain connectivity analysis with optical LOC biosensing

> **⚠️ NOTE:** This document provides technical implementation details. For comprehensive brain communication integration analysis, see **[BRAIN_COMMUNICATION_INTEGRATION.md](./BRAIN_COMMUNICATION_INTEGRATION.md)** which connects MRISpheres, merge2docs neuroscience research, and twosphere-mcp optical sensing.

## Overview

The MRISpheres/twospheres project models the human brain as **two touching spheres** representing the left and right hemispheres. This project has **two main functionalities**:

1. **Brain Communication Pattern Analysis** - FFT-based functional connectivity measurement for drug response monitoring (focus on interhemispheric communication, not just drug testing)
2. **4D MRI Analysis** - Temporal analysis of brain structure and function using frequency-domain methods

### Key Insight: 2D Surface Representation

The folded brain cortex can be accurately represented as the **surface of two spheres** rather than a full 3D volume. This is because:
- The cerebral cortex is approximately 2-3mm thick
- Most functional activity occurs at the cortical surface
- Geodesic (surface) distances better represent functional connectivity than Euclidean distances
- Surface-based analysis accounts for cortical folding patterns

---

## Core Concepts from MRISpheres/twospheres

### 1. Two-Sphere Brain Model

**File:** `two_spheres.py`

```python
# Two touching spheres representing left/right hemispheres
radius = 1.0
center1 = [0, radius, 0]   # Right hemisphere
center2 = [0, -radius, 0]  # Left hemisphere

# Parametric sphere surface: (θ, φ) → (x, y, z)
X = center[0] + radius * sin(φ) * cos(θ)
Y = center[1] + radius * sin(φ) * sin(θ)
Z = center[2] + radius * cos(φ)
```

**Key Properties:**
- **Geodesic distance** on sphere surface: Great circle arc length
- **Haversine formula** for surface distance calculation
- **Equator** represents mid-sagittal plane (interhemispheric boundary)

**From main.tex (lines 299-303):**
```latex
d = arccos(sin(φ₁)sin(φ₂) + cos(φ₁)cos(φ₂)cos(λ₂ - λ₁))
```
where (φ₁, λ₁) and (φ₂, λ₂) are latitude/longitude coordinates on the sphere.

---

### 2. Spectroscopy: FFT-Based Signal Analysis

**File:** `signal_processing.py`

**Three Key Functions:**

#### A. Signal Alignment (Cross-Correlation)
```python
def align_signals(signals, peak_index):
    """Align multiple signals using cross-correlation."""
    for i in range(1, len(signals)):
        cross_corr = np.correlate(signals[0], signals[i], mode="full")
        shift = np.argmax(cross_corr) - (len(signals[0]) - 1)
        # Apply shift to align signals
```
**Application:** Align MRI time-series from different brain regions before FFT analysis.

#### B. Pairwise FFT Correlation
```python
def compute_pairwise_correlation_fft(signals):
    """Compute pairwise correlation between FFTs of signals."""
    fft_signals = [np.fft.fft(signal) for signal in signals]
    # Compute correlation matrix between frequency-domain signals
    correlations = np.corrcoef(fft_i, fft_j)[0, 1]
```
**Application:** Measure functional connectivity in frequency domain.

#### C. Statistical Analysis
```python
def compute_stats(vectors):
    """Compute mean, std dev, and phase shifts."""
    average_vector = np.mean(vectors, axis=0)
    std_dev = np.std(vectors, axis=0, ddof=1)
```

---

### 3. Network Overlay on Sphere Surface

**File:** `overlay_graph.py`

**Workflow:**
1. Create random geometric graph (brain connectivity network)
2. Map 2D graph coordinates to spherical coordinates (θ, φ)
3. Use **quaternions** for rotation on sphere surface
4. Overlay network edges as great circle arcs

```python
# Mapping (u, v) in [0,1]² to sphere
theta, phi, r = spherical_coordinates(u, v)

# Convert to Cartesian
x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

# Quaternion rotation
q_rot = qx * qy
rotated_point = q_rot * q1 * q_rot.conj()
```

**Key Dependencies:**
- `networkx` - Graph creation and analysis
- `quaternion` - Rotation on sphere surface
- Custom `quaternions.spherical_mapping` module

---

### 4. Mathematical Framework (from main.tex)

#### A. Diffusion Tensor Imaging (DTI)
**Fractional Anisotropy (FA)** - White matter tract integrity:
```
FA = √(1/3 · Σᵢ(λᵢ - ⟨λ⟩)²)
```
where λᵢ are eigenvalues of diffusion tensor.

#### B. Functional Connectivity
**Correlation coefficient** between time-series:
```
r = Σᵢ(xᵢ - x̄)(yᵢ - ȳ) / √(Σᵢ(xᵢ - x̄)²) · √(Σᵢ(yᵢ - ȳ)²)
```

#### C. FFT Analysis
**Discrete Fourier Transform:**
```
X(k) = Σₙ₌₀^(N-1) x(n) · e^(-2πikn/N)
```

#### D. Phase-Locking Value (PLV)
**Phase synchronization** between signals:
```
PLV = (1/N) |Σₙ₌₁^N e^(i(φ₁(n) - φ₂(n)))|
```

#### E. Cross-Frequency Coupling (CFC)
**Phase-amplitude coupling:**
```
PAC(f₁, f₂) = Σₜ x₁(t, f₁) · sgn[x₂(t, f₂)]
```

---

## Integration Points with twosphere-mcp

### 1. **Spectroscopy Integration** ✅ COMPLETE

Our newly implemented spectroscopy functionality directly complements MRISpheres:

| MRISpheres Function | twosphere-mcp Implementation | Status |
|-------------------|----------------------------|--------|
| FFT-based signal analysis | `signal_processing.py` → Already in MCP | ✅ Complete |
| Cross-correlation alignment | `InterferometricSensor` visibility fitting | ✅ Analogous |
| Pairwise FFT correlation | `lock_in_detection` I/Q demodulation | ✅ Frequency-domain |
| Phase-locking value | Lock-in phase measurement | ✅ Phase-sensitive |

**Key Connection:**
- MRISpheres: FFT of **MRI time-series** → frequency-domain correlation
- twosphere-mcp: FFT of **optical signals** → lock-in amplification
- **Same mathematical framework, different physical domain!**

---

### 2. **Two-Sphere Geometry** → MCP Tools

We already have `two_sphere_model` MCP tool in `bin/twosphere_mcp.py`!

**Current Implementation (lines 45-78):**
```python
Tool(
    name="two_sphere_model",
    description="Create and visualize a two-sphere model for paired brain regions.",
    inputSchema={
        "radius": 1.0,
        "center1": [0, 1, 0],
        "center2": [0, -1, 0],
        "resolution": 100
    }
)
```

**Enhancement Needed:**
- Add geodesic distance calculation (haversine formula)
- Add spherical coordinate mapping (θ, φ) → (x, y, z)
- Integrate with signal_processing.py FFT correlation

---

### 3. **Network Overlay** → Quaternion Rotation

**Current Implementation:** `vortex_ring` tool (lines 80-118)

**MRISpheres Pattern:**
```python
# Quaternion rotation on sphere
q_rot = qx * qy
rotated_point = q_rot * q1 * q_rot.conj()
```

**Needed:**
- Add `networkx` graph overlay on sphere surface
- Integrate quaternion-based rotation
- Map connectivity matrices to great circle arcs

---

### 4. **4D MRI Analysis** → New Service Layer

**Proposed Architecture:**

```
src/backend/mri/
├── mri_signal_processing.py   # FFT, alignment, correlation
├── sphere_mapping.py           # Geodesic distances, spherical coords
├── network_analysis.py         # Graph overlay, quaternion rotation
└── dti_analysis.py             # Fractional anisotropy, white matter
```

**New MCP Tools:**
1. `compute_fft_correlation` - Pairwise FFT correlation (from signal_processing.py)
2. `map_to_sphere_surface` - (u,v) → spherical coordinates
3. `geodesic_distance` - Great circle distance between points
4. `overlay_connectivity_network` - Graph on sphere surface
5. `compute_phase_locking_value` - PLV between signals
6. `dti_fractional_anisotropy` - White matter integrity

---

## Cancer Drug Testing Connection

**From main.tex (lines 222-223):**
> "By analyzing serial 4D+ MRI scans, DL models can track the progression of disease or the **effects of therapeutic interventions** in real-time, providing valuable insights into the **efficacy and mechanisms of action of new drugs**."

### Integration with LOC Biosensing

**Dual-Modality Approach:**

1. **In Vivo (MRI):**
   - 4D MRI monitors brain changes over time in mouse models
   - FFT analysis identifies frequency-domain biomarkers
   - Drug efficacy tracked via functional connectivity changes

2. **In Vitro (LOC):**
   - Lab-on-chip spectroscopy measures drug-protein interactions
   - Interferometric sensing detects molecular binding
   - Ring resonator tracks concentration changes

**Unified Framework:**
```
Drug → Brain (4D MRI) → FFT Analysis → Connectivity Changes
  ↓
Drug → LOC Sensor → Spectroscopy → Molecular Changes
  ↓
Correlation: MRI biomarkers ↔ Molecular signatures
```

---

## Recommended Implementation Roadmap

### Phase 1: Core Signal Processing (Week 1-2)

**Task:** Port signal_processing.py to twosphere-mcp

1. Create `src/backend/mri/mri_signal_processing.py`
2. Add MCP tools:
   - `align_mri_signals` - Cross-correlation alignment
   - `compute_fft_correlation` - Pairwise FFT correlation
   - `compute_mri_stats` - Statistical summaries

**Bead:** `twosphere-mcp-XXX` - "Port MRISpheres signal processing"

### Phase 2: Sphere Geometry (Week 2-3)

**Task:** Enhanced two-sphere model with geodesics

1. Create `src/backend/mri/sphere_mapping.py`
2. Implement:
   - Haversine formula for geodesic distance
   - Spherical coordinate transformations
   - Quaternion-based rotation (from overlay_graph.py)
3. Update `two_sphere_model` MCP tool

**Bead:** `twosphere-mcp-YYY` - "Add geodesic distance and quaternion rotation"

### Phase 3: Network Overlay (Week 3-4)

**Task:** Graph connectivity on sphere surface

1. Create `src/backend/mri/network_analysis.py`
2. Add dependencies: `networkx`, `quaternion`
3. Implement:
   - Random geometric graph generation
   - Spherical mapping with quaternion rotation
   - Great circle edge rendering
4. New MCP tool: `overlay_connectivity_network`

**Bead:** `twosphere-mcp-ZZZ` - "Overlay connectivity graphs on two-sphere model"

### Phase 4: DTI Analysis (Week 4-5)

**Task:** White matter tract analysis

1. Create `src/backend/mri/dti_analysis.py`
2. Implement:
   - Fractional anisotropy calculation
   - Mean diffusivity
   - Streamline tractography (basic)
3. New MCP tool: `compute_dti_metrics`

**Bead:** `twosphere-mcp-AAA` - "Add DTI white matter analysis"

### Phase 5: Integration Testing (Week 5-6)

**Task:** End-to-end validation

1. Create test dataset (synthetic 4D MRI time-series)
2. Test FFT correlation workflow
3. Validate geodesic distance calculations
4. Test network overlay visualization
5. Document complete pipeline

**Bead:** `twosphere-mcp-BBB` - "Integration testing for MRI analysis pipeline"

---

## Dependencies to Add

```python
# requirements.txt additions
networkx>=3.0        # Graph analysis
quaternion>=2023.0.0 # Quaternion rotations
nibabel>=5.0.0       # NIfTI file reading (for real MRI data)
nilearn>=0.10.0      # MRI data preprocessing
scikit-image>=0.21.0 # Image processing for MRI
```

---

## File Mapping: MRISpheres → twosphere-mcp

| MRISpheres File | Function | twosphere-mcp Target |
|----------------|----------|---------------------|
| `signal_processing.py` | FFT, correlation, alignment | `src/backend/mri/mri_signal_processing.py` |
| `two_spheres.py` | Two-sphere geometry | Enhance existing `two_sphere_model` tool |
| `overlay_graph.py` | Network on sphere | `src/backend/mri/network_analysis.py` |
| `VortexRing*.py` | Trefoil knots | Already in `vortex_ring` tool ✅ |
| `main.tex` (equations) | Mathematical framework | Docstrings + docs/designs/ |

---

## Test Structure (TDD)

Following your TDD preference:

```
tests/backend/mri/
├── __init__.py
├── test_mri_signal_processing.py  # FFT, correlation, alignment
├── test_sphere_mapping.py          # Geodesic distance, quaternions
├── test_network_analysis.py        # Graph overlay
└── test_dti_analysis.py            # DTI metrics
```

**Test Coverage Goals:**
- Signal alignment with known phase shifts
- FFT correlation with synthetic signals
- Geodesic distance validation (known sphere geometry)
- Quaternion rotation correctness
- Network overlay edge accuracy

---

## Research Applications

### 1. **Alzheimer's Disease Progression Monitoring**
- Track functional connectivity changes over time
- Identify early biomarkers via frequency-domain patterns
- Correlate MRI changes with molecular signatures (LOC)

### 2. **Cancer Drug Efficacy Testing**
- Serial 4D MRI in mouse models
- FFT-based change detection
- Correlate brain changes with drug concentration (LOC)

### 3. **Multi-Modal Biomarker Discovery**
- MRI: Functional connectivity patterns
- LOC: Protein binding signatures
- Machine learning: Predict drug response from early biomarkers

---

## Key Innovations

1. **Unified Spectroscopy Framework**
   - Same FFT/correlation methods for MRI and optical signals
   - Phase-sensitive detection in both domains
   - Cross-modal validation

2. **2D Surface Representation**
   - Reduces 3D brain volume to 2D sphere surfaces
   - Geodesic distances better represent functional connectivity
   - Computational efficiency (O(N²) instead of O(N³))

3. **Quaternion-Based Rotation**
   - Elegant rotation on sphere surface without gimbal lock
   - Preserves geodesic distances
   - Essential for interhemispheric connectivity analysis

4. **Real-Time Drug Monitoring**
   - 4D MRI provides temporal dynamics
   - FFT reveals frequency-domain changes
   - LOC provides molecular validation

---

## References

### MRISpheres/twospheres
- `main.tex` - Theoretical framework (Alzheimer's disease, 4D MRI)
- `signal_processing.py` - FFT correlation implementation
- `overlay_graph.py` - Network mapping with quaternions

### twosphere-mcp (Current Implementation)
- `src/backend/services/sensing_service.py` - InterferometricSensor (FFT-based)
- `src/backend/optics/feedback_control.py` - DigitalLockIn (frequency-domain)
- `bin/twosphere_mcp.py` - two_sphere_model, vortex_ring tools

### Key Papers (from main.tex)
- FreeSurfer - Cortical surface reconstruction
- CIVET - Brain segmentation pipeline
- DTI tractography algorithms

---

## Next Steps

1. **Immediate:** Create bead for Phase 1 (signal processing port)
2. **Review:** MRISpheres quaternion module structure
3. **Design:** Test cases for geodesic distance calculation
4. **Prototype:** FFT correlation MCP tool with synthetic data

**Priority:** P1 (High) - Aligns with existing spectroscopy work
**Dependencies:** None (standalone module)
**Estimated Effort:** 2-3 weeks for complete integration

---

**End of Integration Analysis**
