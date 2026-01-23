# Microfluidic Chip Design Improvements

Using FreeCAD addons to enhance our topology-based designs.

## Current Limitations

| Issue | Current State | Impact |
|-------|---------------|--------|
| Simple chambers | Rectangular boxes | Poor mixing |
| Straight channels | Linear paths | Short residence time |
| 2D only | Single PDMS layer | Limited routing |
| Analytical flow | Poiseuille approximation | Ignores entrance effects |
| No validation | Simulation only | Unknown real-world accuracy |

## Proposed Improvements

### 1. Hilbert Curve Mixing Chambers

**Addon**: HilbertCurve

Replace rectangular chambers with Hilbert curve channels:

```
Current:          Improved:
┌────────┐        ┌──┬──┬──┐
│        │        │  │  │  │
│ Chamber│   →    ├──┼──┼──┤
│        │        │  │  │  │
└────────┘        └──┴──┴──┘
                  (Hilbert L2)
```

**Benefits**:
- 4x path length in same footprint (level 2)
- 16x path length with level 3
- Better drug mixing before reaching organoid
- Self-similar structure scales with chamber size

**Implementation**:
```python
def add_hilbert_mixer(chamber, level=2):
    """Replace chamber with Hilbert curve channel."""
    # Hilbert curve length = 4^level * unit_size
    # Level 2 in 2mm chamber = 16mm path
    # Level 3 in 2mm chamber = 64mm path
```

### 2. Serpentine Channel Connectors

**Addon**: Curves workbench

Replace straight inter-chamber channels with serpentines:

```
Current:          Improved:
○────────○        ○~~╲╱~~╲╱~~○
                   serpentine
```

**Benefits**:
- Increased residence time (3-5x longer path)
- Dean vortices at curves enhance mixing
- Acts as flow resistor for balancing

**Implementation**:
- Use B-spline curves for smooth bends
- Minimum bend radius = 5× channel width (reduce dead zones)
- Add to channels between Latin square chambers

### 3. Tapered Flow Distribution

**Addon**: CurvedShapes

Problem: Parallel channels have unequal flow (edge vs center).

Solution: Tapered manifolds that equalize pressure:

```
Current:              Improved:
    │ │ │ │               ╲│ │ │╱
    │ │ │ │          ══════╲│╱══════
────┴─┴─┴─┴────      tapered manifold
  unequal flow         equal flow
```

**Implementation**:
- Inlet manifold widens toward edges
- Channel lengths adjusted to equalize resistance
- Use CfdOF to validate flow uniformity

### 4. Multi-Layer 3D Routing

**Addon**: Assembly4, Lattice2

Current 2D designs limit topology options. 3D enables:

```
Layer 2:  ═══╪═══    (crossover)
              │
Layer 1:  ───┼───    (underpass)
```

**Benefits**:
- True non-planar topologies (K₅, K₃,₃)
- Separate drug delivery layers
- Vertical vias connect layers

**Implementation**:
- Design each layer as separate part
- Use Assembly4 to stack with alignment features
- Vias: 200µm diameter through-holes

### 5. CfdOF Flow Validation

**Addon**: CfdOF (OpenFOAM)

Validate our Poiseuille calculations against real CFD:

| Parameter | Analytical | CFD Needed |
|-----------|------------|------------|
| Straight channel | ✓ Accurate | Baseline |
| Junctions | ✗ Ignored | Entrance effects |
| Bends | ✗ Ignored | Secondary flows |
| Chambers | ✗ Simplified | Recirculation zones |

**Workflow**:
1. Export Latin square to FreeCAD
2. Mesh with snappyHexMesh (10µm resolution)
3. Run simpleFoam (steady laminar)
4. Compare pressure drop to Poiseuille prediction
5. Identify dead zones and recirculation

### 6. FEM Pressure Analysis

**Addon**: FEMbyGEN

Verify PDMS chip survives operating pressure:

- Typical pressure: 10-1000 Pa
- PDMS modulus: ~2 MPa
- Check: channel deformation < 10% height

**Critical areas**:
- Thin walls between adjacent channels
- Chamber roofs (may bulge)
- Inlet/outlet ports (stress concentration)

### 7. Optimized Laser-Cut Molds

**Addon**: LCInterlocking, LasercutterSVGExport

Current: Simple 2D outlines

Improved:
- Finger-joint enclosure holds chip + tubing
- Alignment features for multi-layer assembly
- Kerf compensation (laser removes ~0.1mm)

```
┌─┬─┬─┬─┬─┐
│ │ │ │ │ │  ← finger joints
├─┴─┴─┴─┴─┤
│  CHIP   │  ← pocket for chip
│  ┌───┐  │
│  │   │  │  ← viewing window
├─┬─┬─┬─┬─┤
│ │ │ │ │ │
└─┴─┴─┴─┴─┘
```

### 8. Optical Path Integration

**Addon**: Optics, pyOpTools

Design microscopy setup integrated with chip:

- Working distance constraint (objective to chip)
- Illumination path (transmitted or epi)
- Avoid tubing shadows in light path

**Parameters**:
- 20x objective: WD = 2.1mm, NA = 0.4
- 40x objective: WD = 0.6mm, NA = 0.65
- Glass substrate: 170µm (#1.5 coverslip)

## Working Examples

See `.claude/skills/freecad-examples.md` for copy-paste Python scripts:

1. **Lens Design** - Plano-convex and aspheric profiles
2. **Hose Fittings** - Barbed connectors and Luer locks
3. **CFD Setup** - CfdOF workflow and Reynolds calculator
4. **Fasteners** - M2/M3/M4 screws with clearance tables
5. **Chip Holder** - Complete assembly with pocket, window, ports

## Priority Implementation Order

1. **Hilbert mixers** - Biggest impact on mixing efficiency
2. **CfdOF validation** - Verify our flow calculations
3. **Serpentine channels** - Increase residence time
4. **LCInterlocking enclosure** - Better fabrication
5. **Multi-layer routing** - Enable non-planar topologies
6. **Optical integration** - Complete microscopy design

## Updated Design Metrics

| Metric | Current | With Improvements |
|--------|---------|-------------------|
| Chamber mixing path | 2mm | 32mm (Hilbert L3) |
| Channel residence time | 0.5s | 2.5s (serpentine) |
| Flow uniformity | ~80% | >95% (tapered manifold) |
| Topology options | Planar only | Full K₅, K₃,₃ |
| Validation | Analytical | CFD-verified |
