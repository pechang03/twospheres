# Brain Communication Integration: MRI ↔ Optical Sensing

**Date:** 2026-01-20
**Projects:** MRISpheres/twospheres + merge2docs neuroscience + twosphere-mcp
**Focus:** Brain Communication Patterns via Functional Connectivity Analysis

## Executive Summary

This document integrates three key projects to enable **brain communication pattern analysis** using shared mathematical frameworks between MRI functional connectivity and optical interferometric sensing:

1. **MRISpheres/twospheres** - Two-sphere brain model + FFT-based spectroscopy
2. **merge2docs neuroscience/MRI** - Research on functional connectivity and brain communication
3. **twosphere-mcp** - Optical LOC biosensing with interferometry

**Key Insight:** MRI functional connectivity measurement and optical lock-in detection use **identical mathematical frameworks** (FFT correlation, phase-locking, frequency-domain analysis). This enables:
- Cross-validation between MRI and optical biomarkers
- Unified analysis pipeline for brain communication patterns
- Optical LOC as validation tool for MRI-detected drug effects

---

## Part 1: Brain Communication via Functional Connectivity

### 1.1 What is Brain Communication?

**From Lu et al. (2019)** - *Abnormal intra-network architecture in extra-striate cortices in amblyopia*:

> "The brain is an extraordinarily complex and highly organized network in which dysfunction can spread easily between linked cortices. Brain regions showing synchronized fluctuations during rs-fMRI form the intrinsic connectivity networks (ICNs), which provide the physiological basis for cortical information processing."

**Key Concept:** Brain communication = synchronized oscillations between regions

**Three Visual Networks Analyzed:**
1. **Primary Visual Network (PVN)** - V1, early visual processing
2. **Higher Visual Network (HVN)** - V2, V3v, V4 (extra-striate cortices)
3. **Visuospatial Network (VSN)** - hIP3, PFt, BA7p (spatial processing, attention)

**Communication Metrics:**
- **Intra-network connectivity** - Synchronization within a network
- **Inter-network connectivity** - Communication between networks
- **Local efficiency** - Information processing efficiency at each node

---

### 1.2 Mathematical Framework: Distance Correlation

**From Lu et al. (2019), Section "MRI data analysis":**

The **distance correlation (dCor)** is a multivariate measure of dependence between high-dimensional brain region time-series:

```
For brain regions A and B with voxels v_A and v_B:

1. Z-transform each voxel time course
2. Compute Euclidean distance between time points:
   dA(t1, t2) = ||voxels_A(t1) - voxels_A(t2)||
   dB(t1, t2) = ||voxels_B(t1) - voxels_B(t2)||

3. U-center distance matrices (set row/column means to zero)

4. Distance correlation:
   dCor(A, B) = dCov(A, B) / √(dVar(A) · dVar(B))
```

**Why Distance Correlation?**
- Detects **non-linear** dependencies (unlike Pearson correlation)
- Uses **multi-voxel patterns** (not averaged BOLD signals)
- More robust than univariate methods
- Encodes associations lost by signal averaging

**Fisher Transform for Network Analysis:**
```
z = 0.5 · ln[(1 + dCor) / (1 - dCor)]
```
Produces 19×19 functional connectivity matrix for graph theory analysis.

---

### 1.3 Network Efficiency: Local and Global

**Local Efficiency (LE)** - Measures fault tolerance of information processing:

```
LE_i = (1 / |N_Gi|) · Σ_(j,k ∈ Gi, j≠k) 1/l_jk
```

Where:
- `N_Gi` = immediate neighbors of node i (subgraph)
- `l_jk` = shortest path length between neighbors j and k
- High LE = efficient processing even if node i is removed

**From Lu et al. (2019) Results:**

> "Amblyopic patients exhibited disruptions in local efficiency in the V3v and V4 of the HVN, as well as in the PFt, hIP3, and BA7p of the VSN. This suggests that these regions are intrinsically less fault tolerant in amblyopes and can be interpreted to have a more fragile visual system."

**Key Finding:** Brain communication deficits manifest as **reduced local efficiency** in extra-striate networks.

---

### 1.4 Drugs and Brain Communication

**From Wang et al. (2019)** - *Decoding and mapping task states via deep learning*:

Drug testing in MRI focuses on **how therapeutic interventions alter functional connectivity patterns**:

1. **Baseline MRI** - Measure functional connectivity in disease state
2. **Drug Administration** - Apply therapeutic intervention
3. **Follow-up MRI** - Measure connectivity changes
4. **FFT Analysis** - Identify frequency-domain biomarkers of drug efficacy

**Example: Autism Spectrum Disorder (ASD)**

From "Dynamic Functional Connectivity in Autism" paper (found in directory):
- ASD patients show **hyper-connected patterns** and **abnormal variability**
- Drugs that modulate inhibitory neurotransmitters (GABA) aim to restore normal connectivity
- FFT correlation reveals **frequency-specific changes** in brain communication

**Clinical Application:**
- Serial 4D MRI scans track drug effects over time
- Functional connectivity changes = biomarker of drug efficacy
- Frequency-domain analysis reveals mechanisms of action

---

## Part 2: Two-Sphere Model for Interhemispheric Communication

### 2.1 Why Two Spheres?

**From MRISpheres/twospheres main.tex:**

> "The cerebral cortex is approximately 2-3mm thick, and most functional activity occurs at the cortical surface. Surface-based analysis accounts for cortical folding patterns. Geodesic (surface) distances better represent functional connectivity than Euclidean distances."

**2D Surface Representation Benefits:**
1. **Computational efficiency** - O(N²) instead of O(N³)
2. **Accurate functional connectivity** - Geodesic distance matches axonal path lengths
3. **Cortical folding awareness** - Surface mapping preserves gyri/sulci structure
4. **Interhemispheric analysis** - Equator represents corpus callosum (communication bridge)

**Two-Sphere Geometry:**
```python
radius = 1.0
center_right = [0, radius, 0]   # Right hemisphere
center_left = [0, -radius, 0]   # Left hemisphere

# Parametric surface: (θ, φ) → (x, y, z)
X = center[0] + radius · sin(φ) · cos(θ)
Y = center[1] + radius · sin(φ) · sin(θ)
Z = center[2] + radius · cos(φ)
```

---

### 2.2 Geodesic Distance = Communication Path Length

**Haversine Formula** (great circle distance on sphere):

```
d = arccos(sin(φ₁)·sin(φ₂) + cos(φ₁)·cos(φ₂)·cos(λ₂ - λ₁))
```

Where (φ, λ) are latitude/longitude on sphere surface.

**Why Geodesic Distance Matters:**
- **White matter tracts** follow cortical surface (not straight lines through brain)
- **Functional connectivity** correlates with geodesic distance, not Euclidean
- **Interhemispheric communication** via corpus callosum at equator (shortest path)

**From Lu et al. (2019):**
> "Through diffusion tensor imaging (DTI), amblyopic brain structural connectivity studies have found decreased fractional anisotropy (FA) in the optic radiation, inferior longitudinal fasciculus, and superior longitudinal fasciculus."

**Interpretation:** Brain communication deficits appear as both:
- **Structural** - Reduced white matter integrity (DTI)
- **Functional** - Reduced synchronization (rs-fMRI)

---

### 2.3 Interhemispheric Communication = Equator Analysis

**Key Regions at Equator (Mid-Sagittal Plane):**
- **Corpus callosum** - 200+ million axons connecting hemispheres
- **V1 equator** - Primary visual cortex interhemispheric boundary
- **Motor cortex equator** - Left/right motor coordination

**MRISpheres Pattern:**
```python
# Overlay connectivity network on sphere
import networkx as nx
import quaternion

# Create graph (brain regions)
G = nx.random_geometric_graph(n=100, radius=0.3)

# Map to sphere surface with quaternion rotation
for node in G.nodes():
    theta, phi, r = spherical_coordinates(node.x, node.y)
    q1 = quaternion.from_spherical_coords(theta, phi)

    # Rotate to position on sphere
    q_rot = qx * qy
    rotated = q_rot * q1 * q_rot.conj()
```

**Edges = Communication Pathways:**
- **Great circle arcs** between nodes (shortest geodesic paths)
- **Interhemispheric edges** cross equator (corpus callosum)
- **Intrahemispheric edges** remain on same sphere

---

## Part 3: FFT-Based Analysis = Shared Framework

### 3.1 MRISpheres: FFT Correlation for Brain Signals

**From signal_processing.py:**

```python
def compute_pairwise_correlation_fft(signals):
    """Compute pairwise correlation between FFTs of signals."""
    fft_signals = [np.fft.fft(signal) for signal in signals]

    # Compute correlation matrix in frequency domain
    correlations = np.corrcoef([fft_i.real for fft_i in fft_signals])
    return correlations
```

**Application:**
- Input: MRI time-series from multiple brain regions
- FFT: Transform to frequency domain
- Correlation: Measure synchronization across frequencies
- Output: Functional connectivity matrix

---

### 3.2 twosphere-mcp: FFT for Optical Signals

**From src/backend/optics/feedback_control.py (DigitalLockIn):**

```python
def demodulate(signal_in, time):
    """Lock-in amplifier with FFT-based phase detection."""
    # Multiply by reference signals
    i_comp = signal_in * np.cos(2 * np.pi * reference_freq * time)
    q_comp = signal_in * np.sin(2 * np.pi * reference_freq * time)

    # Low-pass filter (removes 2ω component)
    i_filtered = butter_lowpass_filter(i_comp)
    q_filtered = butter_lowpass_filter(q_comp)

    # Extract amplitude and phase
    amplitude = np.sqrt(i_filtered**2 + q_filtered**2)
    phase = np.arctan2(q_filtered, i_filtered)

    return i_filtered, q_filtered, amplitude, phase
```

**Application:**
- Input: Optical signal from ring resonator
- FFT: Frequency-domain decomposition
- Phase extraction: Lock-in detection
- Output: Phase error for feedback control

---

### 3.3 Mathematical Equivalence

| **MRI (Brain Communication)** | **Optical (LOC Sensing)** | **Mathematical Form** |
|-------------------------------|---------------------------|-----------------------|
| Distance correlation | FFT correlation | `np.fft.fft(signal)` → `np.corrcoef()` |
| Phase-Locking Value (PLV) | Lock-in phase detection | `PLV = (1/N)|Σ e^(i·Δφ)|` |
| Cross-Frequency Coupling | I/Q demodulation | Multiply by cos(ωt), sin(ωt) |
| BOLD time-series alignment | Signal cross-correlation | `np.correlate(s1, s2, mode='full')` |
| Local efficiency | Network fault tolerance | Graph theory shortest paths |

**Key Insight:**

> **The same FFT-based correlation methods used to measure brain communication in MRI can be applied to optical interferometric sensing. This enables cross-validation: MRI detects brain-level drug effects, optical LOC detects molecular-level drug-protein interactions, and the unified mathematical framework links them.**

---

## Part 4: Cross-Domain Integration Architecture

### 4.1 Dual-Modality Drug Testing Workflow

```
Drug Candidate
    ↓
┌───────────────────────────────────────────────────────────┐
│ In Vivo (Brain Communication)                             │
├───────────────────────────────────────────────────────────┤
│ • 4D MRI (serial scans in mouse models)                   │
│ • rs-fMRI → functional connectivity matrices              │
│ • FFT correlation → frequency-domain biomarkers           │
│ • Graph theory → local efficiency changes                 │
│                                                            │
│ Biomarker: Δ(connectivity) and Δ(local efficiency)        │
└───────────────────────────────────────────────────────────┘
    ↓ Correlation Analysis
┌───────────────────────────────────────────────────────────┐
│ In Vitro (Molecular Interactions)                         │
├───────────────────────────────────────────────────────────┤
│ • Lab-on-chip interferometric sensing                     │
│ • Drug-protein binding → Δn (refractive index shift)      │
│ • Lock-in detection → phase-sensitive measurement         │
│ • Visibility fitting (lmfit/emcee) → binding kinetics     │
│                                                            │
│ Biomarker: Δn and binding affinity (K_d)                  │
└───────────────────────────────────────────────────────────┘
    ↓
Unified Framework: MRI biomarkers ↔ Molecular signatures
```

---

### 4.2 Ernie2 Swarm Integration for Brain Communication

**Query Strategy:** Use domain expert collections to inform parameter selection

**Example 1: Interferometric Sensing for Neurotransmitter Detection**

```python
# Query neuroscience + optics collections
context = await ernie2_client.query(
    question="What refractive index changes are expected for GABA binding?",
    collections=["docs_library_neuroscience_MRI", "docs_library_physics_optics"],
    expert_agents=["NeuroscienceExpert", "OpticsExpert"]
)

# Extract parameters from research
expected_delta_n = context.extract_parameter("refractive_index_shift")
path_length = context.suggest_parameter("interferometer_path_length")

# Apply to sensing
result = await interferometric_sensing(
    position=position_data,
    intensity=intensity_data,
    wavelength_nm=633,
    path_length_mm=path_length,
    visibility_baseline=context.baseline
)
```

**Example 2: Phase-Locking Analysis (MRI → Optical)**

```python
# Query for phase-locking patterns
context = await ernie2_client.query(
    question="What phase-locking frequencies indicate altered connectivity in ASD?",
    collections=["docs_library_neuroscience_MRI"],
    expert_agents=["NeuroscienceExpert"]
)

# Extract frequency bands
freq_bands = context.extract_frequencies()  # e.g., [8-12 Hz (alpha), 30-80 Hz (gamma)]

# Apply to lock-in detection
for freq in freq_bands:
    lock_in = DigitalLockIn(reference_frequency=freq, ...)
    i_comp, q_comp, amp, phase = lock_in.demodulate(optical_signal, time)

    # Compare optical phase-locking to MRI PLV
    optical_PLV = compute_PLV(phase)
    if abs(optical_PLV - context.MRI_PLV) < threshold:
        print(f"Optical biomarker matches MRI at {freq} Hz")
```

---

### 4.3 YADA Cross-Domain Bridges

**From merge2docs ernie2_swarm:**

YADA (Yet Another Directed Acyclic) structures in Neo4j connect concepts across domains:

```cypher
// Example: Connect MRI phase-locking to optical lock-in detection
MATCH (mri:LABEL_CONCEPT {name: "Phase-Locking Value"})
MATCH (opt:LABEL_CONCEPT {name: "Lock-In Amplifier"})
CREATE (mri)-[:ANALOGIZES {
  mathematical_form: "PLV = (1/N)|Σ e^(i·Δφ)|",
  shared_principle: "Phase synchronization measurement",
  domains: ["neuroscience_MRI", "physics_optics"]
}]->(opt)

// Query cross-domain analogies
MATCH (c1:LABEL_CONCEPT)-[r:ANALOGIZES]->(c2:LABEL_CONCEPT)
WHERE c1.collection = "neuroscience_MRI"
  AND c2.collection = "physics_optics"
RETURN c1.name, r.shared_principle, c2.name
```

**Result:**
```
Phase-Locking Value → "Phase synchronization measurement" → Lock-In Amplifier
Distance Correlation → "Frequency-domain correlation" → FFT Pairwise Correlation
Local Efficiency → "Network fault tolerance" → Resonator Q-Factor
```

---

## Part 5: Implementation Roadmap

### Phase 1: MRI Signal Processing Integration (Week 1-2)

**Bead:** `twosphere-mcp-3ge` (P1)

**Tasks:**
1. Port `signal_processing.py` from MRISpheres to `src/backend/mri/`
2. Implement MCP tools:
   - `align_mri_signals` - Cross-correlation alignment
   - `compute_fft_correlation` - Pairwise FFT correlation
   - `compute_distance_correlation` - Multivariate distance correlation
3. Add tests mirroring Lu et al. (2019) methods

**Expected Input/Output:**
```python
# Input: MRI time-series from brain regions
brain_region_A = np.load("V1_timeseries.npy")  # Shape: (voxels, timepoints)
brain_region_B = np.load("V4_timeseries.npy")

# Align signals
aligned_A, aligned_B = await align_mri_signals(
    signals=[brain_region_A, brain_region_B],
    peak_index=50
)

# Compute functional connectivity
dCor = await compute_distance_correlation(aligned_A, aligned_B)
print(f"Functional connectivity (dCor): {dCor:.3f}")
```

---

### Phase 2: Two-Sphere Geometry + Geodesics (Week 2-3)

**Bead:** `twosphere-mcp-9jc` (P1)

**Tasks:**
1. Enhance `two_sphere_model` MCP tool with geodesic distance
2. Implement haversine formula for great circle distances
3. Add spherical coordinate transformations: (θ, φ) ↔ (x, y, z)
4. Quaternion-based rotation for network overlay

**Expected Input/Output:**
```python
# Create two-sphere brain model
model = await two_sphere_model(
    radius=1.0,
    center1=[0, 1, 0],
    center2=[0, -1, 0],
    resolution=100
)

# Compute geodesic distance between brain regions
region1 = {"theta": np.pi/4, "phi": np.pi/3}  # V1 location
region2 = {"theta": np.pi/2, "phi": np.pi/4}  # V4 location

geodesic_dist = await compute_geodesic_distance(
    sphere_model=model,
    point1=region1,
    point2=region2
)

print(f"Geodesic distance (cortical surface): {geodesic_dist:.2f} mm")
print(f"Euclidean distance (straight line): {np.linalg.norm(p1 - p2):.2f} mm")
```

---

### Phase 3: Network Overlay with Quaternions (Week 3-4)

**Bead:** `twosphere-mcp-i8c` (P2)

**Tasks:**
1. Port `overlay_graph.py` quaternion rotation logic
2. Integrate `networkx` for brain connectivity graphs
3. Map functional connectivity matrices to sphere surfaces
4. Render interhemispheric edges (crossing equator)

**Expected Input/Output:**
```python
# Connectivity matrix from rs-fMRI
connectivity_matrix = np.load("functional_connectivity.npy")  # 19×19 from Lu et al.

# Overlay on two-sphere model
network_viz = await overlay_connectivity_network(
    sphere_model=model,
    connectivity_matrix=connectivity_matrix,
    node_labels=["V1", "V2", "V3v", "V4", ...],  # 19 visual ICN nodes
    threshold=0.3  # Only show strong connections
)

# Identify interhemispheric edges
interhemispheric = [e for e in network_viz.edges if crosses_equator(e)]
print(f"Found {len(interhemispheric)} interhemispheric connections")
```

---

### Phase 4: Ernie2 Swarm Query Integration (Week 4-5)

**Bead:** `twosphere-mcp-3sg` (P1)

**Tasks:**
1. Create `Ernie2SwarmClient` in `src/backend/services/ernie2_integration.py`
2. Add `query_expert_collections` parameter to 4 MCP tools:
   - `interferometric_sensing`
   - `lock_in_detection`
   - `absorption_spectroscopy`
   - `optimize_resonator`
3. Query `docs_library_neuroscience_MRI` and `docs_library_bioengineering_LOC` collections

**Expected Input/Output:**
```python
# Query neuroscience research for parameter guidance
result = await interferometric_sensing(
    position=position_data,
    intensity=intensity_data,
    wavelength_nm=633,
    query_expert_collections={
        "enabled": True,
        "question": "What refractive index sensitivity is needed for neurotransmitter detection?",
        "collections": ["neuroscience_MRI", "bioengineering_LOC"],
        "use_context_for": ["refractive_index_sensitivity", "path_length_mm"]
    }
)

# Returns visibility + expert-informed parameters
print(f"Visibility: {result['visibility']:.3f}")
print(f"Recommended sensitivity (from research): {result['expert_params']['sensitivity']}")
```

---

### Phase 5: End-to-End Validation (Week 5-6)

**Bead:** `twosphere-mcp-BBB` (Integration testing)

**Test Cases:**
1. **Synthetic 4D MRI** - Generate time-series with known phase-locking
2. **Distance Correlation Validation** - Compare to Lu et al. (2019) results
3. **Geodesic Distance Accuracy** - Test against analytical sphere geometry
4. **Network Overlay Correctness** - Verify quaternion rotations preserve geodesics
5. **Cross-Domain Validation** - MRI PLV vs. optical lock-in phase

**Test Dataset:**
```python
# Generate synthetic brain communication patterns
synthetic_mri = generate_synthetic_fmri(
    n_regions=19,
    timepoints=400,
    plv_target=0.75,  # Strong phase-locking
    frequency_band=(8, 12)  # Alpha band
)

# Test complete pipeline
connectivity = await compute_distance_correlation(synthetic_mri)
network = await overlay_connectivity_network(connectivity)
local_eff = await compute_local_efficiency(network)

# Validate against ground truth
assert abs(connectivity.mean() - expected_connectivity) < tolerance
assert abs(local_eff["V3v"] - expected_eff_V3v) < tolerance
```

---

## Part 6: Research Applications

### 6.1 Alzheimer's Disease Progression

**From MRISpheres main.tex:**

> "By analyzing serial 4D+ MRI scans, DL models can track the progression of disease or the effects of therapeutic interventions in real-time."

**Workflow:**
1. **Baseline MRI** - Measure functional connectivity in early-stage AD
2. **Drug intervention** - Administer amyloid-targeting therapy
3. **Follow-up MRI** (every 3 months) - Track connectivity changes
4. **FFT correlation** - Identify frequency-domain biomarkers
5. **Local efficiency** - Monitor network resilience
6. **Optical validation** - LOC detects amyloid-β binding to antibodies

**Expected Biomarkers:**
- Increased **intra-network connectivity** in HVN (indicates restoration)
- Improved **local efficiency** in V3v, V4 (less fragile network)
- Optical Δn correlates with MRI connectivity changes

---

### 6.2 Autism Spectrum Disorder (ASD)

**From "Dynamic Functional Connectivity in ASD" paper:**

Key findings:
- **Hyper-connectivity** in some networks
- **Abnormal variability** in connectivity over time
- **Reduced local efficiency** in social cognition regions

**Drug Testing Focus:**
- GABA modulators (restore inhibitory balance)
- Oxytocin (improve social communication)
- FFT reveals frequency-specific alterations

**Cross-Validation:**
- MRI: Measure functional connectivity changes
- Optical LOC: Detect GABA receptor binding kinetics
- Correlate MRI Δ(connectivity) with optical Δn

---

### 6.3 Multi-Modal Biomarker Discovery

**Goal:** Predict drug response from early biomarkers

**Machine Learning Pipeline:**
```
Features (Week 1 post-drug):
  - MRI functional connectivity changes
  - MRI local efficiency Δ
  - Optical Δn (drug-protein binding)
  - FFT power spectrum shifts

Training Data:
  - 100 patients, 6-month outcomes

Model:
  - Random forest classifier
  - Predict: Responder vs. Non-responder

Validation:
  - 80% accuracy at Week 1 (vs. 50% chance)
  - Early termination for non-responders
```

---

## Part 7: Dependencies and Tools

### 7.1 New Python Dependencies

Add to `requirements.txt`:
```
networkx>=3.0        # Graph analysis (connectivity networks)
quaternion>=2023.0.0 # Quaternion rotations on sphere
nibabel>=5.0.0       # NIfTI file reading (MRI data format)
nilearn>=0.10.0      # MRI preprocessing and analysis
scikit-image>=0.21.0 # Image processing for MRI
```

### 7.2 External Data Sources

**MRI Data Formats:**
- **NIfTI** (.nii, .nii.gz) - Standard MRI format, use `nibabel`
- **AFNI** (.BRIK, .HEAD) - Analysis of Functional NeuroImages format
- **CIFTI** (.dtseries.nii) - Surface-based fMRI (HCP format)

**Example: Load NIfTI MRI Data**
```python
import nibabel as nib

# Load 4D fMRI (x, y, z, time)
img = nib.load("sub-01_task-rest_bold.nii.gz")
data = img.get_fdata()  # Shape: (91, 109, 91, 400)

# Extract time-series from V1 ROI
v1_mask = nib.load("V1_mask.nii.gz").get_fdata()
v1_timeseries = data[v1_mask > 0, :].mean(axis=0)  # Average over voxels
```

### 7.3 Integration with merge2docs

**Ernie2 Swarm CLI:**
```bash
# Query neuroscience collection
python ../merge2docs/bin/ernie2_swarm.py \
  --collection docs_library_neuroscience_MRI \
  --question "What phase-locking frequencies indicate drug efficacy in AD?" \
  --local

# Query multiple collections for cross-domain insights
python ../merge2docs/bin/ernie2_swarm.py \
  --collection docs_library_neuroscience_MRI,docs_library_physics_optics \
  --question "How to measure phase synchronization optically?" \
  --cloud  # Use Groq for fast response
```

---

## Part 8: Test Structure (TDD)

### 8.1 MRI Signal Processing Tests

**File:** `tests/backend/mri/test_mri_signal_processing.py`

```python
import numpy as np
import pytest
from src.backend.mri.mri_signal_processing import (
    align_signals,
    compute_fft_correlation,
    compute_distance_correlation
)

class TestMRISignalProcessing:
    def test_align_signals_with_known_shift(self):
        """Test cross-correlation alignment with known time shift."""
        # Generate signal with known 10-sample shift
        t = np.linspace(0, 10, 100)
        signal1 = np.sin(2 * np.pi * t)
        signal2 = np.sin(2 * np.pi * (t - 1.0))  # 1-second delay

        aligned = align_signals([signal1, signal2], peak_index=50)

        # After alignment, signals should be correlated
        corr = np.corrcoef(aligned[0], aligned[1])[0, 1]
        assert corr > 0.95

    def test_fft_correlation_pure_sine(self):
        """Test FFT correlation on pure sinusoidal signals."""
        t = np.linspace(0, 10, 1000)
        freq = 5.0  # Hz

        # Two signals at same frequency
        signal1 = np.sin(2 * np.pi * freq * t)
        signal2 = np.sin(2 * np.pi * freq * t + np.pi/4)

        fft_corr = compute_fft_correlation([signal1, signal2])

        # Should have high correlation in frequency domain
        assert fft_corr > 0.8

    def test_distance_correlation_multivariate(self):
        """Test distance correlation on multivariate brain data."""
        # Simulate V1 and V4 voxel time-series
        n_voxels_v1 = 100
        n_voxels_v4 = 80
        n_timepoints = 400

        v1_data = np.random.randn(n_voxels_v1, n_timepoints)
        v4_data = np.random.randn(n_voxels_v4, n_timepoints)

        # Add correlation (simulate functional connectivity)
        shared_signal = np.random.randn(n_timepoints)
        v1_data += 0.5 * shared_signal
        v4_data += 0.5 * shared_signal

        dCor = compute_distance_correlation(v1_data, v4_data)

        # Should detect correlation
        assert dCor > 0.3
```

### 8.2 Two-Sphere Geometry Tests

**File:** `tests/backend/mri/test_sphere_mapping.py`

```python
import numpy as np
import pytest
from src.backend.mri.sphere_mapping import (
    compute_geodesic_distance,
    spherical_to_cartesian,
    cartesian_to_spherical,
    quaternion_rotate
)

class TestSphereMathematics:
    def test_geodesic_distance_equator(self):
        """Test geodesic distance along equator."""
        radius = 1.0

        # Two points on equator, 90° apart
        p1 = {"theta": 0, "phi": np.pi/2}
        p2 = {"theta": np.pi/2, "phi": np.pi/2}

        dist = compute_geodesic_distance(p1, p2, radius)
        expected = radius * np.pi / 2  # Quarter circle

        assert abs(dist - expected) < 1e-6

    def test_geodesic_vs_euclidean(self):
        """Geodesic distance should be >= Euclidean distance."""
        radius = 1.0
        p1 = {"theta": 0, "phi": np.pi/4}
        p2 = {"theta": np.pi/3, "phi": np.pi/3}

        geodesic = compute_geodesic_distance(p1, p2, radius)

        # Convert to Cartesian for Euclidean distance
        cart1 = spherical_to_cartesian(p1, radius)
        cart2 = spherical_to_cartesian(p2, radius)
        euclidean = np.linalg.norm(cart1 - cart2)

        assert geodesic >= euclidean

    def test_quaternion_rotation_preserves_distance(self):
        """Quaternion rotation should preserve geodesic distances."""
        radius = 1.0
        p1 = {"theta": 0, "phi": np.pi/4}
        p2 = {"theta": np.pi/6, "phi": np.pi/3}

        # Compute original distance
        dist_before = compute_geodesic_distance(p1, p2, radius)

        # Rotate both points by same quaternion
        angle = np.pi / 4
        axis = [0, 0, 1]
        p1_rot = quaternion_rotate(p1, angle, axis)
        p2_rot = quaternion_rotate(p2, angle, axis)

        # Distance should be unchanged
        dist_after = compute_geodesic_distance(p1_rot, p2_rot, radius)

        assert abs(dist_before - dist_after) < 1e-6
```

---

## Part 9: Key References

### 9.1 Research Papers (merge2docs neuroscience/MRI)

1. **Lu et al. (2019)** - *Abnormal intra-network architecture in extra-striate cortices in amblyopia*
   - Distance correlation method for functional connectivity
   - Local efficiency analysis in visual networks
   - Graph theory application to brain communication

2. **Wang et al. (2019)** - *Decoding and mapping task states via deep learning*
   - 4D MRI analysis with DNNs
   - Transfer learning for small fMRI datasets
   - Brain task state classification (93.7% accuracy)

3. **Dynamic Functional Connectivity in ASD** (found in directory)
   - Hyper-connected patterns in autism
   - Abnormal variability in brain communication
   - Drug targets for connectivity restoration

### 9.2 MRISpheres/twospheres

- `main.tex` - Mathematical framework (Alzheimer's disease, 4D MRI)
- `signal_processing.py` - FFT correlation, signal alignment
- `two_spheres.py` - Two-sphere brain model
- `overlay_graph.py` - NetworkX + quaternion rotation

### 9.3 twosphere-mcp (Current Implementation)

- `src/backend/services/sensing_service.py` - InterferometricSensor (lmfit/emcee)
- `src/backend/optics/feedback_control.py` - DigitalLockIn (FFT-based)
- `bin/twosphere_mcp.py` - MCP tools for spectroscopy
- `docs/SPECTROSCOPY_IMPLEMENTATION_SUMMARY.md` - Complete spectroscopy integration

---

## Part 10: Next Steps

### Immediate Actions (This Week)

1. ✅ **Document brain communication focus** (THIS FILE)
2. **Create Phase 1 bead** - MRI signal processing integration
3. **Port signal_processing.py** from MRISpheres
4. **Add distance correlation method** (Lu et al. 2019 implementation)

### Short-Term (Next 2 Weeks)

1. **Two-sphere geodesic distances** (haversine formula)
2. **Quaternion rotation** for network overlay
3. **Test with synthetic fMRI data** (validate against known results)
4. **Ernie2 swarm query integration** (neuroscience collection)

### Medium-Term (Next 4-6 Weeks)

1. **Network overlay visualization** (connectivity graphs on sphere)
2. **Cross-domain YADA bridges** (MRI ↔ optical analogies)
3. **End-to-end testing** with real MRI datasets
4. **Performance benchmarking** (compare to standard tools: FSL, SPM, AFNI)

---

## Appendix A: Mathematical Glossary

### MRI Terms

- **BOLD** - Blood-Oxygenation-Level Dependent signal (fMRI basis)
- **rs-fMRI** - Resting-state functional MRI (no task, intrinsic connectivity)
- **ICN** - Intrinsic Connectivity Network (synchronized brain regions)
- **DTI** - Diffusion Tensor Imaging (white matter tract mapping)
- **FA** - Fractional Anisotropy (white matter integrity measure)
- **dCor** - Distance Correlation (multivariate dependence measure)
- **PLV** - Phase-Locking Value (phase synchronization metric)

### Optical Terms

- **Lock-in amplifier** - Phase-sensitive detection at reference frequency
- **I/Q demodulation** - In-phase and quadrature signal extraction
- **Visibility** - Fringe contrast V = (I_max - I_min)/(I_max + I_min)
- **Δn** - Refractive index shift (biosensing output)
- **Q-factor** - Quality factor of resonator (storage time measure)

### Network Terms

- **Geodesic distance** - Shortest path on curved surface (great circle)
- **Local efficiency** - Fault tolerance of node's neighborhood
- **Quaternion** - 4D rotation representation (no gimbal lock)
- **Haversine formula** - Great circle distance on sphere

---

**End of Brain Communication Integration Document**
