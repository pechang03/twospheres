# FreeCAD Microfluidics Design Skill

## Description
Design and simulate microfluidic chips using FreeCAD with specialized addons for CFD simulation, laser cutting export, and 3D printing fabrication.

## Available Addons

Reference: `docs/freecad-addons.csv`

### Simulation
- **CfdOF**: OpenFOAM CFD integration - validate Stokes flow (Re << 1)
- **FEMbyGEN**: Finite Element Method for pressure/stress analysis
- **ElectroMagnetic**: EM fields for electroosmotic (EOF) and magnetohydrodynamic (MHD) pumping

### Geometry
- **HilbertCurve**: Space-filling curves for microfluidic mixers
- **Lattice2**: Array chambers in Latin square grids
- **Curves/CurvedShapes**: NURBS channels, serpentine mixers

### Fabrication
- **LasercutterSVGExport**: Export for laser cutting PDMS mold masters
- **LCInterlocking**: Finger joints for laser-cut enclosures/holders
- **FusedFilamentDesign**: Optimize chip holders for FDM printing
- **3D_Printing_Tools**: Mesh repair and STL export

### Optics
- **Optics/pyOpTools**: Design inverted microscopy optical paths

### Assembly
- **Assembly4**: Constraint-based assembly - chip + holder + microscope setup
- **STEMFIE**: Modular fixtures and chip holders
- **Fasteners**: Screws, nuts, washers - more comprehensive than boltsfc
- **boltsfc**: Standard parts library (bolts/fasteners)

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

Start FreeCAD with MCP/XML-RPC server on port 9875.

### Server API (discovered)

```python
import xmlrpc.client
server = xmlrpc.client.ServerProxy('http://localhost:9875')

# Test connection
server.ping()  # Returns True

# Execute Python code in FreeCAD
result = server.execute_code('''
import FreeCAD as App
import Part
doc = App.newDocument("Test")
box = Part.makeBox(10, 10, 10)
obj = doc.addObject("Part::Feature", "Box")
obj.Shape = box
doc.recompute()
"Done"
''')
# Returns: {'success': True, 'message': '...'}
```

### Verified Working Examples

Successfully created via server:
- Plano-convex lens (25.4mm diameter, R=50mm)
- Barbed hose fitting (3 barbs, 1.6mm ID tubing)
- Chip holder (pocket, viewing window, M3 mounting holes)

### 6. Electroosmotic/Magnetic Flow Control

**Addon**: ElectroMagnetic

Design electrode placement for EOF pumping or magnetic valves:

**Electroosmotic Flow (EOF)**:
- Place electrodes at channel ends
- Electric field drives flow (no moving parts)
- Typical: 100-500 V/cm
- Flow velocity: v = μ_eo × E (μ_eo ≈ 5×10⁻⁸ m²/V·s for PDMS)

**Magnetic Valves**:
- Embed ferrofluid or magnetic beads
- External magnet stops/redirects flow
- No contact with fluid

**Workflow**:
1. Define electrode positions in chip
2. Use ElectroMagnetic to simulate E-field
3. Calculate EOF velocity profile
4. Verify field uniformity in channels
5. Check for electrolysis at electrodes (keep V < 1.2V for water)

**Applications**:
- Programmable flow routing (no external pump)
- Selective channel activation for Latin square
- Magnetic sorting of cells/beads

### 7. LCInterlocking Chip Enclosure

Create a laser-cut acrylic enclosure with finger joints:

1. Design box panels in Sketcher (top, bottom, sides)
2. Switch to LCInterlocking workbench
3. Select edges to add finger joints
4. Set parameters:
   - Finger width: 3-5mm (for 3mm acrylic)
   - Material thickness: 3mm
   - Tolerance: 0.1mm for press-fit
5. Generate interlocking features
6. Export with LasercutterSVGExport
7. Cut from acrylic, snap together (no glue needed)

Use for: chip holders, pump enclosures, electronics boxes

### 7. Assembly4 Chip Assembly

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

## CAD Library

Pre-made microfluidic designs available in `resources/microfluidic-cad-library/`:

### ELEXAN Templates (F3D - Fusion 360)
- Luer lock and 1/4-28 threaded ports
- Single channel (CNC and 3D print versions)
- Flow cells with configurable ports
- **Herringbone mixer** - reference design
- **Droplet generators** - dual inlet, multi-inlet
- **Inertial sorter** - particle separation

### Research Designs
- **Component library** (Nature 2024): Droplet generator, picoinjector, sorters, anchoring
- **Blood-brain barrier chip** (Nature Protocols 2021): Mask + STL frame
- **Drop-seq device** (Cell 2015): Single-cell RNA-seq

See `resources/microfluidic-cad-library/README.md` for full inventory.

## Related Files

- `src/backend/simulation/brain_chip_designer.py` - Chip design toolkit
- `docs/freecad-addons.csv` - Installed addon list
- `docs/beads/2026-01-22-glymphatic-simulation-milestone.md` - Design documentation
- `resources/microfluidic-cad-library/` - Pre-made CAD designs
