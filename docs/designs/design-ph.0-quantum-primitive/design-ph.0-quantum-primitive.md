# Design: Quantum/Primitive Level (F₀)

**Task ID**: PH-0
**Functor Level**: F₀ (Quantum/Primitive)
**Biological Level**: Level 0-1 (Quantum fields → Molecular)
**Priority**: Medium
**Status**: Not Started

---

## Overview

This design covers the **F₀ functor level** - the foundational quantum and primitive operations layer. This includes quantum optics, material properties, physical constants, and geometric primitives that form the basis for all higher-level functors.

---

## Scope

### Level 0: Quantum/Axion Field Level

#### Optics/Photonics
- Quantum coherence modeling (photon antibunching, entanglement)
- Single-photon sources (quantum dots, NV centers)
- Coherence time and phase relationships
- **FPT Parameter**: Coherence time (τ_c), photon number (n)

#### MRI/Neural Systems
- Quantum brain hypothesis (microtubule coherence)
- Nuclear spin dynamics, quantum entanglement in molecules
- Decoherence timescales
- **FPT Parameter**: Decoherence time (T₂), spin system size

#### OOC/Fabrication
- Molecular interactions (Van der Waals forces)
- Quantum tunneling in nanostructures
- Surface chemistry at PDMS/glass interfaces
- **FPT Parameter**: Interaction range (r_vdw), surface area

### Level 1: Molecular/Chemical Level (Syntactic)

#### Optics/Photonics
- ✅ **IMPLEMENTED**: Material library (MATERIAL_LIBRARY, MATERIAL_LIBRARY_NIR)
- ✅ **IMPLEMENTED**: Fiber specifications (FiberSpec, FIBER_TYPES)
- Dispersion relations: n(λ) via Cauchy/Sellmeier equations
- Absorption coefficients: α(λ) for materials

#### MRI/Brain Geometry
- ✅ **IMPLEMENTED**: TwoSphereModel (sphere centers, radii)
- Voxel data structures (raw MRI intensities)
- Tissue properties (T1/T2 relaxation, proton density)
- **FPT Parameter**: Voxel count (N_voxels), resolution

#### OOC/Fabrication
- Material property database (PDMS, glass, silicon)
- Geometric feature specifications (channel width, height, aspect ratio)
- Surface properties (contact angle θ_c, surface energy γ)
- **FPT Parameter**: Feature count, material count

---

## Implementation Status

### Completed (F₀ Level 1 - Syntactic)
- ✅ `src/backend/optics/ray_tracing.py` - MATERIAL_LIBRARY
- ✅ `src/backend/optics/fiber_optics.py` - FIBER_TYPES, FiberSpec
- ✅ `src/backend/mri/two_sphere.py` - TwoSphereModel geometric primitives

### Not Started (F₀ Level 0 - Quantum)
- ❌ Quantum optics module (`src/backend/optics/quantum_optics.py`)
- ❌ SymPy quantum operator framework
- ❌ Entangled photon pair modeling
- ❌ Quantum state evolution

### Gaps Identified
- **Quantum coherence**: No quantum optics capabilities
- **Dispersion modeling**: Material library lacks n(λ) functions
- **Thermal properties**: No temperature-dependent refractive indices
- **Surface chemistry**: No molecular interaction models for OOC

---

## Proposed Implementation

### Phase 1: Quantum Optics Foundation

**File**: `src/backend/optics/quantum_optics.py` (NEW)

```python
"""
Quantum optics module for entangled-photon sensing.

Uses SymPy for symbolic quantum mechanics, enabling:
- Quantum state vector representation
- Operator algebra (beamsplitters, phase shifters, polarizers)
- Expectation value calculations
- Entangled photon pair correlations
"""
from sympy import Matrix, sqrt, exp, I, symbols, cos, sin
from sympy.physics.quantum import TensorProduct

TP = TensorProduct

# Basis states
H = Matrix([1, 0])  # Horizontal polarization
V = Matrix([0, 1])  # Vertical polarization

def beamsplitter_operator() -> Matrix:
    """50:50 beamsplitter transformation."""
    return Matrix([[1, I], [I, 1]]) / sqrt(2)

def phase_delay_operator(delta) -> Matrix:
    """Phase delay transformation δ (radians)."""
    return Matrix([[1, 0], [0, exp(I * delta)]])

def hwp_operator(theta) -> Matrix:
    """Half-wave plate at angle θ."""
    return Matrix([[cos(2*theta), sin(2*theta)],
                   [sin(2*theta), -cos(2*theta)]])

class QuantumOpticsSimulator:
    """Symbolic quantum optics simulator using SymPy."""

    def __init__(self):
        self.delta = symbols('delta', real=True)
        self.theta = symbols('theta', real=True)

    def apply_mzi(self, input_state: Matrix) -> Matrix:
        """
        Apply Mach-Zehnder interferometer to input state.

        Returns
        -------
        Matrix
            Output state after MZI
        """
        # Beamsplitter 1
        # Phase delay in upper arm
        # Beamsplitter 2
        pass

    def compute_visibility(self, output_state: Matrix) -> float:
        """Compute interference visibility from output state."""
        pass
```

**Adaptation Source**: `../entangled-pair-quantum-eraser/lab6entangled.py`

---

### Phase 2: Extended Material Library

**File**: `src/backend/optics/materials.py` (NEW)

```python
"""
Extended material property database with dispersion, absorption, thermal effects.
"""
import numpy as np
from typing import Dict, Callable

class MaterialDatabase:
    """Database of optical material properties."""

    @staticmethod
    def sellmeier_bk7(wavelength_nm: float) -> float:
        """
        Refractive index of BK7 glass via Sellmeier equation.

        Parameters
        ----------
        wavelength_nm : float
            Vacuum wavelength (nm)

        Returns
        -------
        float
            Refractive index n(λ)
        """
        # Sellmeier coefficients for BK7
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 103.560653

        lam_um = wavelength_nm / 1000
        n_squared = 1 + (B1*lam_um**2)/(lam_um**2 - C1) + \
                        (B2*lam_um**2)/(lam_um**2 - C2) + \
                        (B3*lam_um**2)/(lam_um**2 - C3)
        return np.sqrt(n_squared)

    @staticmethod
    def pdms_thermal(temperature_c: float) -> float:
        """
        PDMS refractive index with thermal dependence.

        dn/dT ≈ -4.5e-4 / °C
        """
        n_20C = 1.412  # at 20°C, 633 nm
        dn_dT = -4.5e-4
        return n_20C + dn_dT * (temperature_c - 20)
```

---

### Phase 3: Surface Chemistry Module (OOC)

**File**: `src/backend/fabrication/surface_chemistry.py` (NEW)

```python
"""
Surface chemistry and molecular interactions for OOC devices.
"""

class SurfaceInteractionModel:
    """Model molecular interactions at material interfaces."""

    @staticmethod
    def van_der_waals_force(
        distance_nm: float,
        hamaker_constant: float = 1e-19  # J for PDMS-water
    ) -> float:
        """
        Van der Waals force between surfaces.

        F_vdw = -A/(6πd³)

        Parameters
        ----------
        distance_nm : float
            Surface separation (nm)
        hamaker_constant : float
            Hamaker constant A (J)

        Returns
        -------
        float
            Force per unit area (N/m²)
        """
        d_m = distance_nm * 1e-9
        return -hamaker_constant / (6 * np.pi * d_m**3)

    @staticmethod
    def contact_angle(
        surface_energy_solid: float,
        surface_tension_liquid: float
    ) -> float:
        """
        Young's equation for contact angle.

        cos(θ) = (γ_sv - γ_sl) / γ_lv
        """
        pass
```

---

## Integration with Quantum Eraser Project

### Direct Code Adaptations

From `../entangled-pair-quantum-eraser/`:

1. **`lab6entangled.py`** → `src/backend/optics/quantum_optics.py`
   - Quantum state vectors (H, V, entangled pairs)
   - Operator algebra (beamsplitters, phase delays, HWP)
   - Tensor product operations

2. **`plot_utils.py`** → `src/backend/services/sensing_service.py`
   - Visibility calculation: V = A/(A + 2*C0)
   - Poisson shot-noise: σ = √N
   - Curve fitting for interference patterns

3. **`plot_heatmap.py`** → `src/backend/services/loc_simulator.py`
   - Misalignment sensitivity analysis
   - 2D/3D heatmaps of performance vs. errors
   - Monte Carlo tolerance analysis

---

## FPT Tractability

### Divine Parameters at F₀ Level

**Quantum (Level 0)**:
- **Coherence time** (τ_c): Bounds quantum simulation depth
- **Photon number** (n): Bounds Hilbert space dimension
- **Heisenberg uncertainty**: ΔE·Δt ≥ ℏ/2 (ultimate FPT bound)

**Molecular (Level 1)**:
- **Material count** (N_mat): Bounds search space for optimization
- **Wavelength samples** (N_λ): Bounds dispersion calculations
- **Voxel count** (N_voxels): Bounds MRI resolution

### Tractability Theorem

**Theorem**: Quantum optical simulations are FPT when parameterized by photon number n and coherence time τ_c.

**Proof Sketch**:
1. Hilbert space dimension bounded by photon number: dim(H) = O(n^d)
2. Decoherence truncates evolution after time τ_c
3. Operator algebra composes via functors (Higman's preservation)
4. **Therefore**: NP-hard quantum simulation → FPT(n, τ_c) with divine bounds

---

## Dependencies

### External Libraries
- **SymPy** >= 1.12: Symbolic quantum mechanics
- **NumPy** >= 1.24: Numerical arrays
- **SciPy** >= 1.10: Interpolation, curve fitting

### Internal Dependencies
- `src/backend/optics/ray_tracing.py` (extends MATERIAL_LIBRARY)
- `src/backend/optics/fiber_optics.py` (extends FIBER_TYPES)
- `src/backend/services/sensing_service.py` (uses quantum operators)

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_quantum_optics.py`

```python
"""Unit tests for quantum optics module."""

def test_beamsplitter_unitary():
    """Verify beamsplitter is unitary: B†B = I."""
    B = beamsplitter_operator()
    assert (B.H * B).simplify() == eye(2)

def test_phase_delay_commutation():
    """Verify phase operators commute."""
    A1 = phase_delay_operator(symbols('d1'))
    A2 = phase_delay_operator(symbols('d2'))
    assert (A1 * A2 - A2 * A1).simplify() == zeros(2, 2)

def test_visibility_bounds():
    """Visibility must be in [0, 1]."""
    sim = QuantumOpticsSimulator()
    # Test various input states
    pass
```

### Integration Tests

**File**: `tests/test_quantum_sensing.py`

```python
"""Integration tests for quantum-enhanced sensing."""

async def test_interferometric_visibility():
    """Test visibility analysis pipeline."""
    sensor = InterferometricSensor(config)
    # Generate synthetic interference data
    # Fit visibility
    # Verify V = A/(A + 2*C0)
    pass
```

---

## Documentation Requirements

1. **API Reference**: Docstrings for all quantum operators
2. **Examples**: Jupyter notebook demonstrating MZI simulation
3. **Theory**: Quantum optics background (interference, entanglement)
4. **Fabrication**: Surface chemistry models for OOC devices

---

## Deliverables

### Phase 1 (Quantum Optics)
- [ ] `src/backend/optics/quantum_optics.py` - SymPy quantum operators
- [ ] `tests/test_quantum_optics.py` - Unit tests
- [ ] Jupyter notebook: MZI simulation demo

### Phase 2 (Extended Materials)
- [ ] `src/backend/optics/materials.py` - Dispersion, thermal properties
- [ ] `tests/test_materials.py` - Sellmeier validation
- [ ] Update MATERIAL_LIBRARY with n(λ) functions

### Phase 3 (Surface Chemistry)
- [ ] `src/backend/fabrication/surface_chemistry.py` - Molecular interactions
- [ ] `tests/test_surface_chemistry.py` - Van der Waals validation
- [ ] Contact angle calculator for OOC design

---

## Timeline

- **Phase 1**: 2-4 weeks (quantum optics foundation)
- **Phase 2**: 1-2 weeks (extended materials)
- **Phase 3**: 2-3 weeks (surface chemistry for OOC)
- **Total**: 5-9 weeks

---

## References

- **Quantum Eraser**: `../entangled-pair-quantum-eraser/` (Paul Gauthier)
- **Quantum Optics**: Gerry & Knight, "Introductory Quantum Optics"
- **Material Properties**: Polyanskiy, RefractiveIndex.INFO database
- **Surface Chemistry**: Israelachvili, "Intermolecular and Surface Forces"
- **Biological Mapping**: `../design-ph.1-physics_architecture/functor_hierarchy_biological_mapping.md`

---

**Soli Deo Gloria** - Quantum mechanics reveals divine order at the most fundamental level. Every photon, every quantum state, every interference pattern points to the transcendent Source of all coherence.
