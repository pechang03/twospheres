# Design PHY.1.2: Python Packages for Low-Power Dye Laser Spectroscopy

## Overview
This document catalogs Python packages suitable for simulating and analyzing low-power dye laser spectroscopy experiments, where signal-to-noise ratio is critical and photon counting statistics dominate.

## Core Requirements
- Signal processing for low-SNR measurements
- Photon counting statistics and shot noise modeling
- Cavity-enhanced spectroscopy techniques
- Noise reduction and weak signal extraction
- Bayesian inference for parameter estimation from noisy data

## Recommended Packages

### Signal Processing & Noise Analysis
- **scipy.signal** - Lock-in amplifier simulation, filtering, digital signal processing
  - `savgol_filter` for smoothing noisy spectra
  - `welch` for power spectral density estimation
  - `butter`, `bessel` for filter design
  
- **PyDynamic** - Dynamic metrology with uncertainty propagation
  - Critical for propagating measurement uncertainties in low-power regimes
  - Handles time-dependent calibration data

- **scikit-learn** - Denoising and dimensionality reduction
  - PCA for extracting weak signals from multi-dimensional data
  - Robust regression techniques

### Photon Counting & Statistical Analysis
- **photon-correlation** - Photon statistics and correlation functions
  - g²(τ) correlation analysis
  - Photon bunching/antibunching characterization

- **emcee** - MCMC sampling for Bayesian parameter estimation
  - Essential when fitting models to photon-limited data
  - Proper uncertainty quantification

- **PyMC** - Probabilistic programming for hierarchical Bayesian models
  - More flexible than emcee for complex likelihood functions
  - Built-in diagnostics and visualization

### Spectroscopy-Specific Tools
- **lmfit** - Non-linear least squares fitting with proper error propagation
  - Voigt, Lorentzian, Gaussian lineshape models
  - Constraint handling for physical parameters
  - Weighting for Poisson noise

- **rampy** - Raman spectroscopy toolkit (adaptable to fluorescence)
  - Baseline correction algorithms
  - Peak deconvolution

- **specutils** - Spectroscopic data handling
  - Standard formats and units
  - Wavelength calibration utilities

### Cavity-Enhanced Techniques
For cavity ring-down spectroscopy (CRDS) and cavity-enhanced absorption spectroscopy (CEAS):

- **scipy.optimize.curve_fit** - Exponential decay fitting for ring-down signals
- **lmfit.Model** - More robust fitting with confidence intervals
- Custom implementations for:
  - Multi-exponential decays
  - Mode-matching efficiency calculations
  - Cavity finesse determination

### Optical System Modeling
- **pyoptools** - Ray tracing and optical design (already in use)
- **poppy** - Physical optics propagation (Fresnel/Fraunhofer diffraction)
- **lumerical-python** - If interfacing with commercial FDTD software

### Hardware Interface (Future)
- **PyVISA** - Generic instrument control (spectrometers, monochromators)
- **nidaqmx** - National Instruments data acquisition
- **pySerial** - PMT/photodiode direct readout
- **python-ivi** - Interchangeable Virtual Instruments

## Implementation Priorities

### Phase 1: Core Simulation (PHY.1.2)
1. Install and validate `scipy`, `lmfit`, `emcee`
2. Create example: Photon shot noise limited absorption spectroscopy
3. Implement Voigt profile fitting with proper Poisson weighting

### Phase 2: Advanced Analysis
1. Add `PyDynamic` for uncertainty propagation
2. Implement cavity ring-down simulation
3. Create lock-in amplifier digital model

### Phase 3: Integration
1. Connect to existing `twosphere-mcp` optical ray tracing
2. Add MCP tools for spectroscopy simulation
3. Document workflows in `AGENTS.md`

## Example Workflow

```python
import numpy as np
from scipy.signal import savgol_filter
from lmfit import Model
import emcee

# Simulate weak absorption signal
def voigt_absorption(wavelength, center, gamma_L, gamma_G, depth):
    """Voigt profile absorption line"""
    # ... implementation
    
# Generate photon-limited data
true_signal = voigt_absorption(wavelength, 632.8, 0.01, 0.02, 0.001)
photon_counts = 100  # Low power regime
measured = np.random.poisson(photon_counts * (1 - true_signal))

# Denoise
smoothed = savgol_filter(measured, window_length=11, polyorder=3)

# Fit with proper weighting (Poisson → weights = 1/sqrt(counts))
model = Model(voigt_absorption)
params = model.make_params(center=632.8, gamma_L=0.01, gamma_G=0.02, depth=0.001)
weights = 1 / np.sqrt(measured + 1)  # +1 to avoid divide by zero
result = model.fit(smoothed, params=params, wavelength=wavelength, weights=weights)

# Bayesian uncertainty quantification with emcee
# ... MCMC sampling for full posterior distribution
```

## Package Installation

```bash
# Core packages
pip install scipy lmfit emcee corner

# Spectroscopy tools
pip install specutils rampy

# Advanced (optional)
pip install PyDynamic pymc photon-correlation

# Optical simulation (already installed)
pip install pyoptools>=0.3.7
```

## Integration with twosphere-mcp

### New MCP Tools to Add
1. `simulate_absorption_spectroscopy` - Photon-limited absorption measurement
2. `cavity_ringdown` - CRDS simulation with finesse parameter
3. `lock_in_detection` - Digital lock-in amplifier model
4. `fit_spectral_line` - Automated lineshape fitting with uncertainty

### Data Flow
```
MRI/Brain Geometry (existing) → Optical Ray Tracing → Spectroscopy Simulation
                                      ↓
                              Low-Power Detection Models
                                      ↓
                              Signal Processing & Analysis
```

## References
- Demtröder, W. "Laser Spectroscopy" (standard reference)
- O'Keefe & Deacon (1988) - Original CRDS paper
- Hodgkinson & Tatam (2013) - Optical gas sensing review

## Related Tasks
- PHY.1.1 - Core optical physics simulation (parent)
- PHY.2.x - MRI spherical geometry analysis
- Document in `../merge2docs/` knowledge graph

## Notes
- Prioritize packages that handle Poisson noise properly
- Consider shot noise limit: σ = √N for N photons
- Lock-in bandwidth vs. scan time tradeoffs critical at low power
- Cavity finesse determines minimum detectable absorption in CEAS
