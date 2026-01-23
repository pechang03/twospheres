# FreeCAD Addon Examples

Practical examples for microfluidics design. Copy-paste into FreeCAD Python console.

## Setup

Start FreeCAD, then open Python console: View → Panels → Python console

```python
import FreeCAD as App
import FreeCADGui as Gui
import Part
import Draft
```

---

## 1. Lens Design (Optics + Curves)

### Simple Plano-Convex Lens

```python
# Create a plano-convex lens
# Flat on one side, spherical on other

import FreeCAD as App
import Part

doc = App.newDocument("Lens")

# Parameters
diameter = 25.4  # mm (1 inch)
radius_of_curvature = 50.0  # mm
center_thickness = 5.0  # mm

# Create spherical surface (convex side)
sphere = Part.makeSphere(radius_of_curvature)

# Position sphere so it intersects at correct thickness
# Sagitta (sag) = R - sqrt(R² - (D/2)²)
import math
sag = radius_of_curvature - math.sqrt(radius_of_curvature**2 - (diameter/2)**2)
sphere.translate(App.Vector(0, 0, radius_of_curvature - sag))

# Create cylinder to cut lens diameter
cyl = Part.makeCylinder(diameter/2, center_thickness + sag)

# Intersect to get lens
lens = sphere.common(cyl)

# Add to document
lens_obj = doc.addObject("Part::Feature", "PlanoConvexLens")
lens_obj.Shape = lens

doc.recompute()
Gui.ActiveDocument.ActiveView.fitAll()
```

### Aspheric Lens (using Curves workbench)

```python
# Aspheric lens profile using Curves workbench
# Profile: z = (r²/R) / (1 + sqrt(1 - (1+k)(r/R)²)) + A4*r⁴ + A6*r⁶

import FreeCAD as App
import Part
import numpy as np

doc = App.activeDocument() or App.newDocument("AsphericLens")

# Aspheric parameters
R = 50.0      # Radius of curvature
k = -1.0      # Conic constant (k=-1 is parabola)
diameter = 20.0
A4 = 1e-6     # 4th order coefficient
A6 = 1e-9     # 6th order coefficient

# Generate profile points
r_values = np.linspace(0, diameter/2, 50)
z_values = []

for r in r_values:
    if r == 0:
        z = 0
    else:
        term1 = (r**2 / R) / (1 + np.sqrt(1 - (1+k)*(r/R)**2))
        term2 = A4 * r**4 + A6 * r**6
        z = term1 + term2
    z_values.append(z)

# Create profile as BSpline
points = [App.Vector(r, 0, z) for r, z in zip(r_values, z_values)]
# Revolve around Z axis to create lens surface
# Use Curves workbench for smoother result

print(f"Aspheric sag at edge: {z_values[-1]:.3f} mm")
print("Use Curves workbench to create BSpline from points, then revolve")
```

---

## 2. Hose Fitting (Barb Connector)

### Barbed Hose Fitting for Microfluidics

```python
# Barbed fitting for silicone tubing
# Standard sizes: 1/16", 1/8", 3/32" ID tubing

import FreeCAD as App
import Part
import math

doc = App.newDocument("HoseFitting")

# Parameters (for 1/16" = 1.6mm ID tubing)
tubing_id = 1.6  # mm
barb_od = tubing_id * 1.3  # 30% oversize for grip
barb_id = 0.8  # mm (flow channel)
barb_length = 3.0  # mm per barb
num_barbs = 3
base_diameter = 4.0  # mm
base_height = 2.0  # mm

# Create barbed section
shapes = []

# Base (for bonding to chip)
base = Part.makeCylinder(base_diameter/2, base_height)
shapes.append(base)

# Barbs (tapered cylinders)
z = base_height
for i in range(num_barbs):
    # Each barb is a cone frustum
    barb = Part.makeCone(
        tubing_id/2 * 0.9,  # bottom radius (slightly under tubing ID)
        barb_od/2,           # top radius (oversize)
        barb_length * 0.7    # taper height
    )
    barb.translate(App.Vector(0, 0, z))
    shapes.append(barb)

    # Straight section
    straight = Part.makeCylinder(barb_od/2, barb_length * 0.3)
    straight.translate(App.Vector(0, 0, z + barb_length * 0.7))
    shapes.append(straight)

    z += barb_length

# Fuse all sections
fitting_solid = shapes[0]
for s in shapes[1:]:
    fitting_solid = fitting_solid.fuse(s)

# Cut center hole (flow channel)
hole = Part.makeCylinder(barb_id/2, z + 1)
hole.translate(App.Vector(0, 0, -0.5))
fitting_solid = fitting_solid.cut(hole)

# Add to document
fitting = doc.addObject("Part::Feature", "HoseBarb")
fitting.Shape = fitting_solid

doc.recompute()
Gui.ActiveDocument.ActiveView.fitAll()

print(f"Fitting for {tubing_id}mm ID tubing")
print(f"Total height: {z:.1f}mm")
print(f"Flow channel: {barb_id}mm diameter")
```

### Luer Lock Fitting

```python
# Female Luer Lock fitting (ISO 80369-7)
import FreeCAD as App
import Part

doc = App.activeDocument() or App.newDocument("LuerLock")

# Luer taper: 6% (1:16.17)
# Female ID at tip: 3.925 mm
# Female ID at base: 4.225 mm
# Taper length: 5 mm

taper_length = 5.0
id_tip = 3.925
id_base = 4.225
wall_thickness = 1.0
thread_od = 9.0
thread_height = 3.0

# Inner taper (cone)
inner_cone = Part.makeCone(id_tip/2, id_base/2, taper_length)

# Outer cylinder
outer_cyl = Part.makeCylinder(id_base/2 + wall_thickness, taper_length)

# Thread section (simplified as cylinder with wings)
thread_base = Part.makeCylinder(thread_od/2, thread_height)
thread_base.translate(App.Vector(0, 0, -thread_height))

# Combine
body = outer_cyl.fuse(thread_base)
body = body.cut(inner_cone)

luer = doc.addObject("Part::Feature", "LuerFemale")
luer.Shape = body

doc.recompute()
```

---

## 3. Liquid Flow Visualization (CfdOF)

### Setup CfdOF Analysis

```python
# CfdOF setup for microfluidic channel
# Requires: CfdOF addon installed, OpenFOAM installed

import FreeCAD as App

# First create or import channel geometry
doc = App.activeDocument()

# CfdOF workflow (after geometry exists):
"""
1. Switch to CfdOF workbench
2. Select solid body → CfdOF → Create Analysis
3. Set fluid properties:
   - Density: 1000 kg/m³ (water)
   - Kinematic viscosity: 1e-6 m²/s

4. Create mesh:
   - Base element size: 0.01 mm (for 100µm channels)
   - Refinement at walls

5. Set boundary conditions:
   - Inlet: Velocity = 0.001 m/s (or Pressure)
   - Outlet: Pressure = 0
   - Walls: No-slip

6. Solver settings:
   - Steady-state (simpleFoam) for most cases
   - Transient (pimpleFoam) for time-varying

7. Run and post-process
"""

# Example: Create simple channel for CFD
channel_length = 10  # mm
channel_width = 0.2  # mm (200 µm)
channel_height = 0.1  # mm (100 µm)

channel = Part.makeBox(channel_length, channel_width, channel_height)
channel_obj = doc.addObject("Part::Feature", "MicroChannel")
channel_obj.Shape = channel

print("Channel created. Now:")
print("1. Switch to CfdOF workbench")
print("2. Select channel → Analysis → Create")
print("3. Set inlet velocity: 1 mm/s")
print("4. Expected Re = ρVD/μ = 1000 * 0.001 * 0.0001 / 0.001 = 0.1")
```

### Reynolds Number Calculator

```python
# Microfluidic Reynolds number calculator
def calc_reynolds(velocity_m_s, hydraulic_diameter_m,
                  density=1000, viscosity=0.001):
    """
    Calculate Reynolds number for microfluidic channel.

    Args:
        velocity_m_s: Flow velocity (m/s)
        hydraulic_diameter_m: D_h = 4*Area/Perimeter (m)
        density: Fluid density (kg/m³), default water
        viscosity: Dynamic viscosity (Pa·s), default water

    Returns:
        Reynolds number (dimensionless)
    """
    Re = density * velocity_m_s * hydraulic_diameter_m / viscosity

    if Re < 1:
        regime = "Stokes flow (Re << 1)"
    elif Re < 2300:
        regime = "Laminar"
    else:
        regime = "Turbulent"

    print(f"Re = {Re:.4f} → {regime}")
    return Re

# Example: 100µm x 50µm channel at 1 mm/s
width = 100e-6  # m
height = 50e-6  # m
D_h = 4 * (width * height) / (2 * (width + height))
print(f"Hydraulic diameter: {D_h*1e6:.1f} µm")

calc_reynolds(velocity_m_s=0.001, hydraulic_diameter_m=D_h)
# Output: Re = 0.0667 → Stokes flow
```

---

## 4. Fasteners (Fasteners Workbench)

### Add Screw to Assembly

```python
# Using Fasteners workbench
# Must have Fasteners addon installed

import FreeCAD as App
import FreeCADGui as Gui

doc = App.activeDocument() or App.newDocument("FastenerTest")

# Method 1: Via GUI
"""
1. Switch to Fasteners workbench
2. Click screw icon → select type (e.g., ISO 4762 Socket Head)
3. Set diameter (M3) and length (10mm)
4. Position using Assembly4 constraints
"""

# Method 2: Via Python (if Fasteners module available)
try:
    import FastenersCmd

    # Create M3x10 socket head cap screw
    Gui.activateWorkbench("FastenersWorkbench")
    Gui.runCommand('Fasteners_ISO4762', 0)  # Socket head

    # Parameters set via properties panel:
    # - diameter: M3
    # - length: 10

except ImportError:
    print("Fasteners workbench not loaded. Use GUI method.")

# Create mounting hole for M3 screw
hole_diameter = 3.2  # mm (clearance for M3)
hole_depth = 15  # mm

plate = Part.makeBox(20, 20, 5)
hole = Part.makeCylinder(hole_diameter/2, hole_depth)
hole.translate(App.Vector(10, 10, -hole_depth + 5))
plate = plate.cut(hole)

plate_obj = doc.addObject("Part::Feature", "MountingPlate")
plate_obj.Shape = plate

doc.recompute()
print("Created plate with M3 clearance hole at center")
```

### Fastener Reference Table

```python
# Common fasteners for microfluidic assemblies

fastener_table = """
METRIC FASTENERS FOR MICROFLUIDICS
==================================

Screw Type          | Use Case                    | Torque
--------------------|-----------------------------|---------
M2 Socket Head      | Small chip clamps           | 0.2 Nm
M3 Socket Head      | Standard chip holders       | 0.5 Nm
M4 Socket Head      | Stage mounting              | 1.2 Nm
M3 Thumbscrew       | Quick-release fixtures      | Hand tight

CLEARANCE HOLES (mm)
--------------------
Screw   Close Fit   Normal Fit
M2      2.2         2.4
M3      3.2         3.4
M4      4.3         4.5

TAP DRILL SIZES (mm)
--------------------
Thread  Tap Drill   Depth (2xD)
M2      1.6         4
M3      2.5         6
M4      3.3         8

NYLON/PEEK SCREWS
-----------------
Use for:
- Avoiding metal contamination
- Electrical isolation
- Chemical compatibility
Available: M2, M3, M4 in various heads
"""

print(fastener_table)
```

---

## 5. Complete Example: Chip Holder Assembly

```python
# Complete chip holder with all elements
import FreeCAD as App
import Part

doc = App.newDocument("ChipHolder")

# Parameters
chip_width = 25  # mm
chip_length = 50  # mm
chip_thickness = 5  # mm (PDMS + glass)
wall_thickness = 3  # mm
holder_height = 10  # mm

# Base plate with pocket for chip
base_width = chip_width + 2 * wall_thickness
base_length = chip_length + 2 * wall_thickness

base = Part.makeBox(base_length, base_width, holder_height)

# Chip pocket (with 0.5mm tolerance)
pocket = Part.makeBox(chip_length + 1, chip_width + 1, chip_thickness + 1)
pocket.translate(App.Vector(
    wall_thickness - 0.5,
    wall_thickness - 0.5,
    holder_height - chip_thickness - 1
))
base = base.cut(pocket)

# Viewing window (for inverted microscope)
window_size = min(chip_width, chip_length) * 0.6
window = Part.makeCylinder(window_size/2, holder_height)
window.translate(App.Vector(base_length/2, base_width/2, 0))
base = base.cut(window)

# Mounting holes (M3) at corners
hole_inset = 5
for x in [hole_inset, base_length - hole_inset]:
    for y in [hole_inset, base_width - hole_inset]:
        hole = Part.makeCylinder(1.6, holder_height)  # M3 clearance
        hole.translate(App.Vector(x, y, 0))
        base = base.cut(hole)

# Tubing port holes (for hose fittings)
port_diameter = 4  # mm (for press-fit barb)
for x_offset in [10, base_length - 10]:
    port = Part.makeCylinder(port_diameter/2, wall_thickness + 1)
    port.rotate(App.Vector(0,0,0), App.Vector(0,1,0), 90)
    port.translate(App.Vector(x_offset, base_width/2, holder_height - 2))
    base = base.cut(port)

holder = doc.addObject("Part::Feature", "ChipHolder")
holder.Shape = base

doc.recompute()
Gui.ActiveDocument.ActiveView.fitAll()

print(f"Chip holder created:")
print(f"  Outer: {base_length} x {base_width} x {holder_height} mm")
print(f"  Pocket: {chip_length+1} x {chip_width+1} x {chip_thickness+1} mm")
print(f"  Window: {window_size}mm diameter")
print(f"  Mounting: 4x M3 holes")
print(f"  Ports: 2x {port_diameter}mm for tubing")
```

---

## Notes & Tips

### General FreeCAD Tips
- Always `doc.recompute()` after making changes
- Use `Gui.ActiveDocument.ActiveView.fitAll()` to see result
- Check `App.Console.PrintMessage()` for debug output

### Microfluidic-Specific
- Channel aspect ratio (width:height) affects flow profile
- Keep Re < 1 for predictable Stokes flow
- PDMS shrinks ~1% during curing - compensate in mold

### Addon-Specific
- **CfdOF**: Requires OpenFOAM installed separately
- **Fasteners**: Use "Fasteners" workbench, not "boltsfc" for more options
- **Curves**: Great for serpentine channels - use "Pipeshell" for swept profiles
- **Assembly4**: Define LCS (Local Coordinate System) on each part first

### Export Formats
- STL: 3D printing (check mesh quality)
- STEP: CAD interchange (keeps solids)
- SVG: Laser cutting (2D projection)
- FCStd: FreeCAD native (keeps parameters)
