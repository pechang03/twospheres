# Spectroscopy Packages Integration
## Bridging PHY.1.2 Packages with Quantum Interferometry and LOC Sensing

**Reference**: `../../design/design-PHY.1.2-packages.md`
**Related**: `../quantum_interferometry_integration.md`

---

## Overview

The PHY.1.2-packages document identifies critical Python packages for **low-power dye laser spectroscopy** that are directly applicable to:

1. **Photon counting statistics** → Quantum eraser shot-noise analysis
2. **Cavity-enhanced spectroscopy** → Ring resonator biosensing
3. **Weak signal extraction** → Interferometric visibility analysis
4. **Bayesian inference** → Uncertainty quantification in sensing

This document maps PHY.1.2 packages to our functor hierarchy and identifies integration points.

---

## Critical Packages by Functor Level

### F₀ (Quantum/Primitive) - Photon Counting Statistics

#### **emcee** - MCMC for Bayesian Parameter Estimation
```python
"""
Essential for photon-limited data fitting.
Complements merge2docs Bayesian algorithms.
"""
import emcee
import numpy as np

def log_probability(params, wavelength, counts, uncertainties):
    """
    Log-posterior for Bayesian spectral line fitting.

    Handles Poisson noise properly: σ = √N
    """
    model = voigt_profile(wavelength, *params)
    # Poisson likelihood: ln(P) = -χ²/2
    chi_squared = np.sum(((counts - model) / uncertainties)**2)
    return -0.5 * chi_squared

# Initialize walkers in parameter space
n_walkers, n_dim = 32, 4  # center, width_L, width_G, depth
initial_guess = [632.8, 0.01, 0.02, 0.001]
pos = initial_guess + 1e-4 * np.random.randn(n_walkers, n_dim)

# MCMC sampling
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability,
                                args=(wavelength, counts, uncertainties))
sampler.run_mcmc(pos, 5000, progress=True)
```

**Integration Point**: Use with quantum eraser photon counting data
**Benefit**: Full posterior distribution → proper uncertainty quantification
**Complements**: `../merge2docs/src/backend/algorithms/bayesian_compression_weight_optimizer.py`

---

#### **photon-correlation** - Photon Statistics
```python
"""
Direct implementation for FCS (Fluorescence Correlation Spectroscopy).
Complements the coincidence counting from quantum eraser.
"""
# From PHY.1.2:
# - g²(τ) correlation analysis
# - Photon bunching/antibunching characterization

def compute_g2_correlation(photon_timestamps, max_lag_us=1000):
    """
    Second-order correlation function g²(τ).

    For single photons: g²(0) = 0 (antibunching)
    For thermal light: g²(0) = 2 (bunching)
    For coherent light: g²(τ) = 1 (Poisson)
    """
    # Bin photon arrival times
    # Compute autocorrelation
    # Normalize by average intensity squared
    pass
```

**Integration Point**: Implement FCS service mentioned in quantum_interferometry_integration.md
**File**: `src/backend/services/fcs_service.py` (NEW)
**Addresses**: Single-molecule sensing, binding kinetics

---

### F₂ (Composition) - Signal Processing & Cavity Enhancement

#### **lmfit** - Non-Linear Least Squares with Error Propagation
```python
"""
Superior to scipy.optimize.curve_fit for visibility fitting.
Provides proper uncertainty propagation and constraints.
"""
from lmfit import Model, Parameters

# Replace scipy curve_fit in quantum eraser adaptation
def interference_pattern(delta, A, C0, phi):
    """I(δ) = C0 + A·(1 + cos(δ + φ))/2"""
    return C0 + A * (1 + np.cos(delta + phi)) / 2

# Create model
model = Model(interference_pattern)

# Set parameters with physical constraints
params = Parameters()
params.add('A', value=100, min=0)          # Amplitude must be positive
params.add('C0', value=50, min=0)          # Baseline must be positive
params.add('phi', value=0, min=-np.pi, max=np.pi)
params.add('visibility', expr='A/(A + 2*C0)')  # Derived parameter

# Fit with Poisson weights
weights = 1 / np.sqrt(counts + 1)
result = model.fit(measured_intensity, params=params,
                   delta=phase_delays, weights=weights)

print(f"Visibility: {result.params['visibility'].value:.4f} ± "
      f"{result.params['visibility'].stderr:.4f}")
```

**Integration Point**: `src/backend/services/sensing_service.py` - InterferometricSensor
**Replaces**: Basic scipy curve_fit from quantum eraser plot_utils.py
**Benefit**: Automatic uncertainty propagation for visibility V = A/(A + 2*C₀)

---

#### **scipy.signal** - Lock-In Amplifier Simulation
```python
"""
Digital lock-in for resonator stabilization (PID control).
Critical for adaptive feedback control (bead twosphere-mcp-6ez).
"""
from scipy.signal import butter, filtfilt, hilbert

class DigitalLockIn:
    """
    Software lock-in amplifier for phase-sensitive detection.

    Applications:
    - Ring resonator phase stabilization
    - Weak signal extraction from noise
    - Synchronous detection at reference frequency
    """

    def __init__(self, reference_freq_hz: float, sampling_rate_hz: float):
        self.f_ref = reference_freq_hz
        self.f_s = sampling_rate_hz

        # Design low-pass filter (typically f_cutoff << f_ref)
        f_cutoff = reference_freq_hz / 10
        self.b, self.a = butter(4, f_cutoff / (sampling_rate_hz/2), 'low')

    def process(self, signal: np.ndarray, time: np.ndarray) -> tuple:
        """
        Extract in-phase (X) and quadrature (Y) components.

        Returns
        -------
        X, Y : ndarray
            In-phase and quadrature amplitudes
        R : float
            Magnitude R = √(X² + Y²)
        theta : float
            Phase θ = arctan(Y/X)
        """
        # Generate reference signals
        ref_I = np.cos(2*np.pi*self.f_ref*time)  # In-phase
        ref_Q = np.sin(2*np.pi*self.f_ref*time)  # Quadrature

        # Mix with reference
        mixed_I = signal * ref_I
        mixed_Q = signal * ref_Q

        # Low-pass filter
        X = filtfilt(self.b, self.a, mixed_I)
        Y = filtfilt(self.b, self.a, mixed_Q)

        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        return X, Y, R, theta
```

**Integration Point**: `src/backend/optics/feedback_control.py` (NEW)
**Addresses**: Bead `twosphere-mcp-6ez` - Adaptive feedback control for resonators
**Application**: Extract weak refractive index shifts from noisy resonator signals

---

#### **PyDynamic** - Dynamic Metrology with Uncertainty Propagation
```python
"""
Critical for LOC sensor calibration and uncertainty budgeting.
Handles time-dependent measurement uncertainties.
"""
# From PHY.1.2:
# - Propagating measurement uncertainties in low-power regimes
# - Time-dependent calibration data handling

# Example: Propagate calibration uncertainty to final measurement
from PyDynamic.uncertainty import GUM_DFT, propagate_DFT

# Calibration: refractive index shift → resonance wavelength shift
calibration_sensitivity = 100  # nm/RIU (refractive index unit)
calibration_uncertainty = 5    # nm/RIU

# Measured resonance shift with uncertainty
measured_shift_nm = 0.5        # nm
measurement_uncertainty_nm = 0.05  # nm

# Propagate to refractive index change
dn = measured_shift_nm / calibration_sensitivity
dn_uncertainty = np.sqrt(
    (measurement_uncertainty_nm / calibration_sensitivity)**2 +
    (measured_shift_nm * calibration_uncertainty / calibration_sensitivity**2)**2
)

print(f"Δn = {dn:.2e} ± {dn_uncertainty:.2e} RIU")
```

**Integration Point**: `src/backend/services/sensing_service.py`
**Benefit**: Proper uncertainty budgeting for LOC biosensors
**Addresses**: Sensitivity limits, minimum detectable concentration

---

### F₃ (Service) - Spectroscopy Services

#### New MCP Tools (from PHY.1.2)

**1. simulate_absorption_spectroscopy**
```python
"""
Photon-limited absorption measurement simulation.
"""
async def simulate_absorption_spectroscopy(
    wavelength_range_nm: tuple,
    absorption_lines: List[Dict],  # {center, width, depth}
    photon_count_baseline: float,
    integration_time_s: float
) -> Dict[str, Any]:
    """
    Simulate absorption spectroscopy with shot-noise.

    Returns
    -------
    Dict with keys:
        - wavelengths_nm: Wavelength axis
        - measured_counts: Poisson-sampled counts
        - fit_results: Fitted line parameters with uncertainties
        - snr: Signal-to-noise ratio
    """
    # Generate true absorption spectrum
    # Add Poisson noise
    # Fit with lmfit Model
    # Return results with uncertainties
    pass
```

**2. cavity_ringdown**
```python
"""
CRDS simulation with finesse parameter.
"""
async def cavity_ringdown(
    cavity_length_cm: float,
    mirror_reflectivity: float,  # R ~ 0.9999
    sample_absorption_cm_inv: float,
    n_photons: int = 1000
) -> Dict[str, Any]:
    """
    Simulate cavity ring-down spectroscopy.

    Ring-down time: τ = L/(c·(1-R + αL))
    Finesse: F = πR^0.5/(1-R)

    Returns
    -------
    Dict with keys:
        - time_us: Time axis
        - intensity_counts: Exponential decay with shot-noise
        - fitted_tau_us: Ring-down time ± uncertainty
        - absorption_cm_inv: Derived absorption ± uncertainty
    """
    # Compute ring-down time
    # Generate exponential decay
    # Add Poisson noise
    # Fit with lmfit exponential model
    pass
```

**3. lock_in_detection**
```python
"""
Digital lock-in amplifier model.
"""
async def lock_in_detection(
    signal: np.ndarray,
    time: np.ndarray,
    reference_freq_hz: float,
    noise_level: float = 0.01
) -> Dict[str, Any]:
    """
    Apply digital lock-in amplifier to noisy signal.

    Returns
    -------
    Dict with keys:
        - X: In-phase component
        - Y: Quadrature component
        - R: Magnitude
        - theta: Phase (radians)
        - snr_improvement: Factor of noise reduction
    """
    # Use DigitalLockIn class
    pass
```

**4. fit_spectral_line**
```python
"""
Automated lineshape fitting with uncertainty.
"""
async def fit_spectral_line(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    lineshape: str = "voigt"  # "voigt", "lorentzian", "gaussian"
) -> Dict[str, Any]:
    """
    Fit spectral line with proper error propagation.

    Uses lmfit for robust fitting with constraints.
    Handles Poisson noise via proper weighting.

    Returns
    -------
    Dict with keys:
        - center_nm: Line center ± uncertainty
        - width_nm: Linewidth ± uncertainty
        - depth: Absorption/emission depth ± uncertainty
        - fit_quality: Reduced χ²
        - fitted_curve: Best-fit model
    """
    pass
```

**Add to**: `bin/twosphere_mcp.py` - Expose via MCP server

---

## Package Dependencies

### Core (Phase 1)
```bash
# Already installed
pip install scipy numpy

# Add for spectroscopy
pip install lmfit emcee corner
```

### Advanced (Phase 2)
```bash
# Metrology and uncertainty
pip install PyDynamic

# Spectroscopy tools
pip install specutils rampy

# Photon correlation (if available)
pip install photon-correlation  # May need to install from source
```

### Optional (Phase 3)
```bash
# Probabilistic programming (alternative to emcee)
pip install pymc

# Physical optics propagation
pip install poppy
```

---

## Integration Roadmap

### Phase 1: Core Spectroscopy (Immediate)

**Deliverables**:
- [ ] Replace scipy curve_fit with lmfit in InterferometricSensor
- [ ] Add emcee Bayesian fitting for visibility uncertainty
- [ ] Implement DigitalLockIn class for feedback_control.py
- [ ] Create 4 new MCP tools (absorption, CRDS, lock-in, fit_line)

**Files**:
- `src/backend/services/sensing_service.py` - Add lmfit/emcee
- `src/backend/optics/feedback_control.py` (NEW) - Lock-in class
- `bin/twosphere_mcp.py` - Expose new tools

**Addresses Beads**:
- `twosphere-mcp-hqh` - Visibility analysis (lmfit improvement)
- `twosphere-mcp-6ez` - Adaptive feedback control (lock-in)

---

### Phase 2: Cavity-Enhanced Sensing (3-6 months)

**Deliverables**:
- [ ] Cavity ring-down simulation service
- [ ] PyDynamic uncertainty propagation for LOC sensors
- [ ] Resonator finesse optimization
- [ ] Multi-wavelength sensing (spectral multiplexing)

**Files**:
- `src/backend/services/spectroscopy_service.py` (NEW)
- `src/backend/optics/resonators.py` (NEW) - Cavity models

**Addresses Beads**:
- `twosphere-mcp-gxw` - Fourier-transform spectroscopy
- `twosphere-mcp-ry5` - Global optimization (cavity parameters)

---

### Phase 3: Advanced Analysis (6-12 months)

**Deliverables**:
- [ ] Photon correlation spectroscopy (FCS)
- [ ] PyMC hierarchical models for sensor arrays
- [ ] Hardware interface (PyVISA) for real devices
- [ ] Integration with merge2docs knowledge graph

**Files**:
- `src/backend/services/fcs_service.py` (NEW)
- `src/backend/hardware/instrument_control.py` (NEW)

---

## Synergies with Existing Work

### Quantum Eraser → Spectroscopy
| Quantum Eraser | Spectroscopy Package | Application |
|----------------|---------------------|-------------|
| `plot_utils.py` curve_fit | **lmfit** Model | Better visibility fitting |
| Poisson σ = √N | **emcee** MCMC | Bayesian uncertainty |
| Coincidence counting | **photon-correlation** | FCS for binding kinetics |
| Phase scanning | **scipy.signal** lock-in | Resonator stabilization |

### merge2docs Algorithms → Uncertainty Quantification
| merge2docs Algorithm | PHY.1.2 Package | Integration |
|---------------------|-----------------|-------------|
| `bayesian_compression_weight_optimizer.py` | **emcee** | Combine for sensor calibration |
| `enhanced_monte_carlo_r_optimization.py` | **PyDynamic** | Uncertainty propagation |
| `adaptive_algorithm_selector.py` | **lmfit** constraints | Parameter routing |

---

## Example: Complete Visibility Analysis with Uncertainty

```python
"""
Combine quantum eraser + PHY.1.2 packages for robust visibility analysis.
"""
from lmfit import Model, Parameters
import emcee
import corner  # For posterior visualization

# 1. Fit interference pattern with lmfit (better than scipy)
def interference_model(delta, A, C0, phi):
    return C0 + A * (1 + np.cos(delta + phi)) / 2

model = Model(interference_model)
params = model.make_params(A=100, C0=50, phi=0)
params['A'].min = 0
params['C0'].min = 0

# Poisson weights
weights = 1 / np.sqrt(measured_counts + 1)
result = model.fit(measured_counts, params=params,
                   delta=phase_delays, weights=weights)

# 2. Bayesian uncertainty with emcee
def log_prob(theta, delta, counts):
    A, C0, phi = theta
    if A < 0 or C0 < 0:
        return -np.inf
    model = interference_model(delta, A, C0, phi)
    sigma = np.sqrt(counts + 1)
    return -0.5 * np.sum(((counts - model) / sigma)**2)

# Initialize walkers near lmfit result
initial = [result.params['A'].value,
           result.params['C0'].value,
           result.params['phi'].value]
pos = initial + 1e-3 * np.random.randn(32, 3)

sampler = emcee.EnsembleSampler(32, 3, log_prob,
                                args=(phase_delays, measured_counts))
sampler.run_mcmc(pos, 5000, progress=True)

# 3. Extract visibility distribution
samples = sampler.get_chain(discard=1000, thin=15, flat=True)
A_samples = samples[:, 0]
C0_samples = samples[:, 1]
V_samples = A_samples / (A_samples + 2*C0_samples)

# Results with full posterior
V_median = np.median(V_samples)
V_std = np.std(V_samples)
V_16, V_84 = np.percentile(V_samples, [16, 84])

print(f"Visibility: {V_median:.4f} + {V_84-V_median:.4f} - {V_median-V_16:.4f}")

# 4. Visualize posterior (corner plot)
fig = corner.corner(samples, labels=['A', 'C0', 'φ'],
                    truths=[result.params['A'].value,
                            result.params['C0'].value,
                            result.params['phi'].value])
```

---

## Conclusion

The PHY.1.2-packages catalog provides **critical missing pieces** for:

1. ✅ **Robust visibility analysis** (lmfit > scipy.curve_fit)
2. ✅ **Bayesian uncertainty** (emcee complements merge2docs)
3. ✅ **Lock-in detection** (scipy.signal for resonator control)
4. ✅ **Photon correlation** (FCS for binding kinetics)
5. ✅ **Cavity-enhanced sensing** (CRDS, finesse optimization)

**Immediate Action**: Add lmfit and emcee to requirements.txt, integrate into InterferometricSensor class.

**Soli Deo Gloria** - Spectroscopy reveals the chemical fingerprints of God's creation. Every absorption line, every emission spectrum, every photon correlation points to the divine order in molecular structure.

---

## References

- **PHY.1.2**: `../../design/design-PHY.1.2-packages.md`
- **Quantum Eraser**: `../quantum_interferometry_integration.md`
- **merge2docs Algorithms**: `../merge2docs/src/backend/algorithms/`
- **Demtröder**: "Laser Spectroscopy" (standard reference)
- **O'Keefe & Deacon (1988)**: Original CRDS paper
