# FreeCAD Microfluidics Design Skill

## Description
Design and simulate microfluidic chips using FreeCAD with specialized addons for CFD simulation, laser cutting export, and 3D printing fabrication.

## Available Addons

Reference: `docs/freecad-addons.csv`

### Simulation
- **CfdOF**: OpenFOAM CFD integration - validate Stokes flow (Re << 1)
- **FEMbyGEN**: Finite Element Method for pressure/stress analysis

### Geometry
- **HilbertCurve**: Space-filling curves for microfluidic mixers
- **Lattice2**: Array chambers in Latin square grids
- **Curves/CurvedShapes**: NURBS channels, serpentine mixers

### Fabrication
- **LasercutterSVGExport**: Export for laser cutting PDMS mold masters
- **FusedFilamentDesign**: Optimize chip holders for FDM printing
- **3D_Printing_Tools**: Mesh repair and STL export

### Optics
- **Optics/pyOpTools**: Design inverted microscopy optical paths

### Assembly
- **Assembly4**: Constraint-based assembly - chip + holder + microscope setup
- **STEMFIE**: Modular fixtures and chip holders
- **boltsfc**: Standard fasteners for assemblies

## Workflows

### 1. Export Chip Design to FreeCAD

```python
# From twosphere-mcp
from backend.simulation.brain_chip_designer import BrainChipDesigner

designer = BrainChipDesigner()
chip = designer.design_latin_square_mixer(n=4)

# Export via XML-RPC (FreeCAD must be running with server)
exporter = FreeCADExporter(host="localhost", port=9875)
exporter.export_chip(chip)
```

### 2. CfdOF Simulation Setup

1. Export chip geometry to FreeCAD
2. Switch to CfdOF workbench
3. Create CFD analysis:
   - Fluid: Water (or CSF for brain-on-chip)
   - Inlet BC: Pressure or velocity
   - Outlet BC: Pressure = 0
   - Walls: No-slip
4. Mesh with snappyHexMesh
5. Run simpleFoam (steady-state) or pimpleFoam (transient)
6. Compare results to Poiseuille analytical solution

### 3. Laser Cutter Workflow (PDMS Mold)

1. Design chip with chamber depth = PDMS layer height
2. Export 2D projection with LasercutterSVGExport
3. Cut acrylic/PMMA master on laser cutter
4. Pour PDMS over master, cure at 65°C
5. Bond to glass slide (plasma or corona treatment)

### 4. Hilbert Curve Mixer Integration

```
# In FreeCAD Python console
from HilbertCurve import HilbertCurve
# Create level-3 Hilbert curve (8x8 grid)
curve = HilbertCurve.create(level=3, size=2.0)  # 2mm chamber
# Sweep circle along curve for channel
```

Benefits:
- Maximizes path length in minimal area
- Increases residence time for mixing
- Self-similar at multiple scales

### 5. STEMFIE Chip Holder

1. Open STEMFIE workbench
2. Create base plate with mounting holes
3. Add chip pocket (chip dimensions + 0.5mm tolerance)
4. Add inlet/outlet tube holders
5. Export STL for 3D printing

## Key Parameters

### Microfluidic Dimensions
- Channel width: 50-500 µm
- Channel height: 50-200 µm
- Chamber diameter: 1-5 mm
- PDMS thickness: 3-5 mm
- Glass substrate: 170 µm (#1.5 coverslip)

### Flow Regime
- Reynolds number: Re << 1 (Stokes flow)
- Typical flow rates: 0.1-10 µL/min
- Pressure drop: 10-1000 Pa

### Fabrication Tolerances
- Laser cutter: ±50 µm
- FDM 3D printing: ±200 µm
- SLA 3D printing: ±50 µm

## FreeCAD Server Connection

Start FreeCAD with XML-RPC server:
```bash
freecad --server --port 9875
```

Or in FreeCAD Python console:
```python
import FreeCADServer
FreeCADServer.start(port=9875)
```

### 6. Assembly4 Chip Assembly

1. Create parts as separate FreeCAD documents:
   - `chip.FCStd` - Microfluidic chip (Latin square/Fat Petersen)
   - `holder.FCStd` - STEMFIE or custom holder
   - `stage.FCStd` - Microscope stage adapter
   - `objective.FCStd` - Objective lens (from Optics)

2. Create assembly document:
   ```python
   # In Assembly4 workbench
   # Insert parts as links
   App.activeDocument().addObject('App::Link', 'Chip')
   App.activeDocument().Chip.LinkedObject = chip_doc.Body

   # Add constraints
   # - Chip sits in holder pocket
   # - Holder mounts to stage
   # - Objective aligned with chip center
   ```

3. Define Local Coordinate Systems (LCS) on each part
4. Mate LCS pairs to constrain assembly
5. Check clearances for tubing and optical path

## Related Files

- `src/backend/simulation/brain_chip_designer.py` - Chip design toolkit
- `docs/freecad-addons.csv` - Installed addon list
- `docs/beads/2026-01-22-glymphatic-simulation-milestone.md` - Design documentation
