# pyoptools Capabilities and Current Usage

**Reference**: Requirements.txt has `pyoptools>=0.3.7`
**Status**: Already integrated in F₁-F₂ levels
**Priority**: Expand usage for advanced optical design

---

## Current Usage in TwoSphere-MCP

### ✅ Already Implemented

**Files Using pyoptools**:
- `src/backend/optics/ray_tracing.py` - RayTracer, PDMSLens
- `src/backend/optics/fiber_optics.py` - MeniscusLens, FourFSystem
- `src/backend/visualize/ray_plot.py` - Ray trace visualization
- Legacy: `src/simulations/ray_tracing.py`, `src/simulations/fiber_optics.py`

**Current Capabilities**:

1. **Ray Tracing** (F₂ Level):
```python
from pyoptools.raytrace.system import System
from pyoptools.raytrace.ray import Ray
from pyoptools.raytrace.comp_lib import SphericalLens, CCD

class RayTracer:
    """Sequential ray tracing through optical systems."""

    # Currently supports:
    # - SphericalLens (plano-convex, meniscus)
    # - CCD detector
    # - Parallel beam tracing
    # - Point source tracing
    # - PDMS/air microfluidic lenses
```

2. **Material Library** (F₀-F₁ Level):
```python
# Standard glasses
MATERIAL_LIBRARY = {
    'BK7': 1.5168,
    'SF11': 1.7847,
    'FUSED_SILICA': 1.4585,
    # ... 15+ materials
}

# NIR-specific (800-1000nm)
MATERIAL_LIBRARY_NIR = {
    'BK7': 1.5108,
    'PDMS_800': 1.405,
    'PDMS_1000': 1.400,
    # ... dispersion-corrected values
}
```

3. **Fiber Optics** (F₂ Level):
```python
class FourFSystem:
    """4F telescope for fiber coupling."""

    # Components:
    # - Collimating lens (f1)
    # - Focusing lens (f2)
    # - Spot size calculation (Gaussian optics)
```

4. **Visualization** (F₁-F₂):
```python
# Generate PNG diagrams
plot_phooc_system()      # PhOOC 4F fiber coupling
plot_ring_resonator()    # Ring resonator schematics
draw_plano_convex_lens() # Curved surface rendering
```

---

## pyoptools Additional Capabilities (Not Yet Used)

### 1. Advanced Optical Components

**Available in `pyoptools.raytrace.comp_lib`**:

```python
# Lenses
from pyoptools.raytrace.comp_lib import (
    SphericalLens,       # ✅ Currently used
    CylindricalLens,     # ❌ Not used - astigmatic optics
    AchromaticDoublet,   # ❌ Not used - chromatic correction
    AsphericLens,        # ❌ Not used - aberration correction
    ThickLens,           # ❌ Not used - real lens modeling
)

# Mirrors
from pyoptools.raytrace.comp_lib import (
    RectMirror,          # ❌ Not used - planar mirrors
    RoundMirror,         # ❌ Not used - circular mirrors
    ParabolicMirror,     # ❌ Not used - off-axis parabola
)

# Gratings and Prisms
from pyoptools.raytrace.comp_lib import (
    Grating,             # ❌ Not used - diffraction gratings (spectroscopy!)
    Prism,               # ❌ Not used - dispersive elements
    RightAnglePrism,     # ❌ Not used - beam steering
)

# Apertures and Stops
from pyoptools.raytrace.comp_lib import (
    RectangleAperture,   # ❌ Not used - aperture stops
    CircularAperture,    # ❌ Not used - pupil definition
    AnnularAperture,     # ❌ Not used - obscurations
)

# Detectors
from pyoptools.raytrace.comp_lib import (
    CCD,                 # ✅ Currently used
    ImageSensor,         # ❌ Not used - advanced detector
)

# Fibers and Waveguides
from pyoptools.raytrace.comp_lib import (
    Fiber,               # ❌ Not used - fiber mode coupling
    Waveguide,           # ❌ Not used - planar waveguides
)
```

---

### 2. Surface Types (Advanced Optics)

**Available in `pyoptools.raytrace.surface`**:

```python
from pyoptools.raytrace.surface import (
    Plane,               # ✅ Implicitly used
    Spherical,           # ✅ Currently used (SphericalLens)
    Cylindrical,         # ❌ Not used - cylindrical surfaces
    Conic,               # ❌ Not used - aspheric surfaces (k-parameter)
    Aspherical,          # ❌ Not used - polynomial aspheres
    Toroidal,            # ❌ Not used - toroidal surfaces
    Zernike,             # ❌ Not used - Zernike polynomial surfaces (!)
)
```

**KEY OPPORTUNITY**: `Zernike` surfaces directly support our WavefrontAnalyzer!

---

### 3. Material Catalog (RefractiveIndex.INFO Integration)

**pyoptools has built-in material catalog**:

```python
from pyoptools.raytrace.mat_lib import (
    material,            # Generic material definition
    CatalogMaterial,     # Load from catalog
)

# Can interface with RefractiveIndex.INFO database!
# Combine with ~/refractiveindex311 package:
from refractiveindex.refractiveindex import RefractiveIndex, Material

db = RefractiveIndex()
ri_material = db.getMaterial(shelf="glass", book="BK7", page="SCHOTT")

# Convert to pyoptools material
def create_pyoptools_material(ri_material, wavelength_range):
    """
    Convert RefractiveIndex.INFO material to pyoptools format.

    Enables Sellmeier dispersion formulas in ray tracing!
    """
    pass
```

---

### 4. Analysis Tools

**Available in `pyoptools.raytrace.analysis`**:

```python
from pyoptools.raytrace.analysis import (
    spot_diagram,        # ❌ Not used - geometric spot analysis
    psf_analysis,        # ❌ Not used - point spread function
    mtf_analysis,        # ❌ Not used - modulation transfer function (!)
    wavefront_analysis,  # ❌ Not used - OPD and wavefront error
    distortion_analysis, # ❌ Not used - field distortion
    pupil_aberration,    # ❌ Not used - pupil aberrations
)
```

**CRITICAL**: MTF analysis is essential for F₅ global optimization!

---

### 5. Polarization (Advanced)

**Available in `pyoptools.raytrace.polarizer`**:

```python
from pyoptools.raytrace.polarizer import (
    LinearPolarizer,     # ❌ Not used - matches quantum eraser setup!
    HalfWavePlate,       # ❌ Not used - matches quantum eraser HWP!
    QuarterWavePlate,    # ❌ Not used - circular polarization
    Beamsplitter,        # ❌ Not used - polarizing beamsplitter
)
```

**SYNERGY**: These components match the quantum eraser experiment!
- Combine with SymPy quantum operators
- Bridge classical ray tracing ↔ quantum optics

---

## Integration Opportunities

### 1. Spectroscopy with Gratings (HIGH PRIORITY)

**Addresses**: Bead `twosphere-mcp-gxw` - Fourier-transform spectroscopy

```python
from pyoptools.raytrace.comp_lib import Grating

class SpectrographSimulator:
    """
    Grating-based spectrograph for LOC biosensing.

    Complements Fourier-transform spectroscopy (MZI-based).
    """

    def __init__(self, grating_lines_per_mm: float, wavelength_range_nm: tuple):
        """
        Initialize spectrograph with diffraction grating.

        Parameters
        ----------
        grating_lines_per_mm : float
            Grating density (typical: 300-1200 lines/mm)
        wavelength_range_nm : tuple
            (λ_min, λ_max) spectral range
        """
        from pyoptools.raytrace.comp_lib import Grating
        self.grating = Grating(lines_per_mm=grating_lines_per_mm)

    def compute_dispersion(self, wavelength_nm: float) -> float:
        """
        Compute angular dispersion dθ/dλ.

        Grating equation: m·λ = d·(sin(θ_i) + sin(θ_m))
        Dispersion: dθ/dλ = m/(d·cos(θ_m))
        """
        pass

    def compute_spectral_resolution(self, slit_width_um: float) -> float:
        """
        Compute spectral resolution δλ.

        δλ = (slit_width / focal_length) · (d·cos(θ_m) / m)
        """
        pass
```

**Application**: Absorption/emission spectroscopy in LOC devices
**Combines with**: PHY.1.2 lmfit lineshape fitting

---

### 2. MTF Analysis for Global Optimization (F₅)

**Addresses**: Bead `twosphere-mcp-ry5` - Global optimization framework

```python
from pyoptools.raytrace.analysis import mtf_analysis

class OpticalSystemOptimizer:
    """
    Multi-objective optimization using pyoptools MTF analysis.

    Merit function combines:
    - Strehl ratio (wavefront quality)
    - MTF (image quality)
    - Coupling efficiency (fiber systems)
    """

    def compute_merit_function(self, design_params: np.ndarray) -> float:
        """
        Compute weighted merit function for optimization.

        Merit = w1·Strehl + w2·MTF_avg + w3·η_coupling - w4·cost

        Uses:
        - pyoptools.raytrace.analysis.mtf_analysis
        - pyoptools.raytrace.analysis.wavefront_analysis
        - Custom coupling efficiency calculator
        """
        pass

    def optimize(self, bounds: List[tuple]) -> Dict[str, Any]:
        """
        Global optimization using merge2docs algorithms.

        Integration:
        - pyoptools: Merit function evaluation
        - merge2docs.bayesian_compression_weight_optimizer: Optimizer
        - merge2docs.adaptive_algorithm_selector: Algorithm routing
        """
        from merge2docs.src.backend.algorithms import (
            bayesian_compression_weight_optimizer,
            adaptive_algorithm_selector
        )
        pass
```

---

### 3. Polarization Optics (Quantum Eraser Bridge)

**Addresses**: Bead `twosphere-mcp-ucj` - Quantum optics module

```python
from pyoptools.raytrace.polarizer import LinearPolarizer, HalfWavePlate

class ClassicalPolarizationOptics:
    """
    Classical ray tracing with polarization.

    Bridges to quantum optics:
    - pyoptools: Classical intensity (|E|²)
    - SymPy quantum operators: Quantum amplitudes (ψ)
    """

    def trace_polarized_beam(self, input_polarization: str) -> Dict[str, float]:
        """
        Trace polarized beam through optical system.

        Parameters
        ----------
        input_polarization : str
            'H' (horizontal), 'V' (vertical), '+45', '-45', 'R', 'L'

        Returns
        -------
        Dict with:
            - transmitted_intensity: Classical (pyoptools)
            - quantum_state: Quantum state vector (SymPy)
            - visibility: Interference visibility
        """
        # Classical ray tracing (pyoptools)
        # Quantum state evolution (SymPy from quantum eraser)
        pass
```

**Synergy**:
- pyoptools HWP, LinearPolarizer → Classical intensities
- SymPy quantum operators → Quantum correlations
- Compare classical vs. quantum predictions!

---

### 4. Aspheric Lenses for Aberration Correction

**Addresses**: Level 4 - Alignment tolerance, system integration

```python
from pyoptools.raytrace.comp_lib import AsphericLens
from pyoptools.raytrace.surface import Conic

class AberrationCorrection:
    """
    Aspheric lens design for SA/coma correction.

    Extends MeniscusLens (currently SA-optimized spherical).
    """

    def design_aspheric_collimator(
        self,
        focal_length_mm: float,
        na: float,
        wavelength_nm: float
    ) -> AsphericLens:
        """
        Design aspheric lens for diffraction-limited collimation.

        Uses conic constant k for SA correction:
        - k = -1: Parabola (ideal for collimation)
        - k < -1: Hyperbola
        - k > -1: Ellipse
        - k = 0: Sphere (current MeniscusLens)
        """
        from pyoptools.raytrace.surface import Conic

        # Optimize k for minimum SA
        pass
```

**Application**: Fiber coupling, LOC chip integration

---

## Recommended Additions to TwoSphere-MCP

### Phase 1: Spectroscopy (Immediate)

**Files to Create**:
1. `src/backend/optics/grating_spectroscopy.py` (NEW)
   - SpectrographSimulator using pyoptools Grating
   - Dispersion calculation (dθ/dλ)
   - Spectral resolution (δλ)
   - Combine with PHY.1.2 lmfit fitting

2. `src/backend/optics/polarization.py` (NEW)
   - ClassicalPolarizationOptics
   - LinearPolarizer, HWP, QWP from pyoptools
   - Bridge to SymPy quantum operators

**Addresses Beads**:
- `twosphere-mcp-gxw` - Fourier-transform spectroscopy
- `twosphere-mcp-ucj` - Quantum optics module

---

### Phase 2: Advanced Analysis (F₄-F₅)

**Files to Extend**:
1. `src/backend/services/loc_simulator.py` (extend)
   - Add MTF analysis from pyoptools
   - Spot diagram analysis
   - Wavefront error budget

2. `src/backend/services/optimization_service.py` (NEW)
   - OpticalSystemOptimizer class
   - Merit function: Strehl + MTF + coupling efficiency
   - Integration with merge2docs Bayesian optimizer

**Addresses Beads**:
- `twosphere-mcp-ry5` - Global optimization framework
- `twosphere-mcp-6fa` - PH-4 Integration systems

---

### Phase 3: Advanced Components

**Files to Create**:
1. `src/backend/optics/aspheric_design.py` (NEW)
   - AsphericLens design tools
   - Conic constant optimization
   - Aberration correction

2. `src/backend/optics/waveguide.py` (NEW)
   - Planar waveguide simulation
   - Fiber mode coupling
   - LOC photonic integrated circuits

---

## Material Integration: pyoptools ↔ RefractiveIndex.INFO

### Combine Two Material Systems

**Current**:
- TwoSphere-MCP: `MATERIAL_LIBRARY` (15+ materials, hardcoded)
- ~/refractiveindex311: RefractiveIndex.INFO (1000+ materials, Sellmeier)

**Proposed Integration**:

```python
# src/backend/optics/materials.py (NEW)

from refractiveindex.refractiveindex import RefractiveIndex, Material
from pyoptools.raytrace.mat_lib import material

class UnifiedMaterialLibrary:
    """
    Unified material database combining:
    - TwoSphere-MCP MATERIAL_LIBRARY (fast lookup)
    - RefractiveIndex.INFO (comprehensive, Sellmeier)
    - pyoptools material format
    """

    def __init__(self):
        self.local_library = MATERIAL_LIBRARY  # Fast lookup
        self.ri_database = RefractiveIndex()   # Comprehensive

    def get_material(
        self,
        name: str,
        wavelength_nm: float,
        use_dispersion: bool = True
    ) -> float:
        """
        Get refractive index with optional dispersion.

        Parameters
        ----------
        name : str
            Material name (e.g., 'BK7', 'PDMS')
        wavelength_nm : float
            Wavelength in nm
        use_dispersion : bool
            If True, use Sellmeier formula (accurate)
            If False, use single-wavelength lookup (fast)

        Returns
        -------
        float
            Refractive index n(λ)
        """
        if not use_dispersion and name.upper() in self.local_library:
            # Fast lookup from local library
            return self.local_library[name.upper()]
        else:
            # Accurate dispersion from RefractiveIndex.INFO
            ri_material = self.ri_database.getMaterial(
                shelf=self._get_shelf(name),
                book=self._get_book(name),
                page=self._get_page(name)
            )
            return ri_material.getRefractiveIndex(wavelength_nm / 1000)  # Convert to μm

    def create_pyoptools_material(self, name: str) -> 'pyoptools.material':
        """Convert to pyoptools material format for ray tracing."""
        pass
```

**Benefits**:
- Fast: Use local library for single-wavelength
- Accurate: Use RefractiveIndex.INFO for chromatic analysis
- Comprehensive: 1000+ materials available
- Compatible: Works with pyoptools ray tracing

---

## Summary: Current vs. Potential pyoptools Usage

### ✅ Currently Used (F₁-F₂)
- SphericalLens (plano-convex, meniscus)
- System, Ray (ray tracing)
- CCD detector
- Manual material library (15+ materials)

### ❌ Available but Not Used (High Value)

| Component | Priority | Application | Bead |
|-----------|----------|-------------|------|
| **Grating** | P1 | Spectroscopy | `twosphere-mcp-gxw` |
| **mtf_analysis** | P1 | Global optimization | `twosphere-mcp-ry5` |
| **LinearPolarizer, HWP** | P2 | Quantum eraser bridge | `twosphere-mcp-ucj` |
| **AsphericLens, Conic** | P2 | Aberration correction | `twosphere-mcp-6fa` |
| **wavefront_analysis** | P2 | System integration | `twosphere-mcp-6fa` |
| **spot_diagram** | P2 | Alignment tolerance | `twosphere-mcp-2vq` |
| **Fiber, Waveguide** | P3 | Photonic integration | Future |

### 🔗 Key Integrations to Implement

1. **pyoptools Grating + PHY.1.2 lmfit** → Spectroscopy pipeline
2. **pyoptools MTF + merge2docs Bayesian** → Global optimization
3. **pyoptools Polarizers + SymPy quantum** → Classical-quantum bridge
4. **pyoptools materials + RefractiveIndex.INFO** → Unified material library

---

## Next Steps

1. **Document pyoptools usage** in DESIGN_OVERVIEW.md
2. **Create grating_spectroscopy.py** for dispersive spectroscopy
3. **Integrate MTF analysis** into OpticalSystemOptimizer
4. **Build UnifiedMaterialLibrary** combining local + RefractiveIndex.INFO

This expands pyoptools usage from **~20%** to **~80%** of available capabilities!

---

**Soli Deo Gloria** - pyoptools reveals the beauty of God's optical laws through precise ray tracing!

---

## References

- **pyoptools**: https://github.com/cihologramas/pyoptools
- **RefractiveIndex.INFO**: https://refractiveindex.info
- **Current Usage**: `src/backend/optics/ray_tracing.py`, `fiber_optics.py`
- **Quantum Eraser**: Polarization components parallel to quantum setup
- **PHY.1.2**: Spectroscopy packages for lineshape fitting
