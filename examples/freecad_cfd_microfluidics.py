#!/usr/bin/env python3
"""
Microfluidic CFD Simulation via FreeCAD CfdOF.

Applications:
1. PHLoC (Photonic Lab-on-Chip) - organoid culture fluid exchange
2. Glymphatic system - brain perivascular CSF/ISF flow simulation

Both operate in the same regime:
- Low Reynolds number (Re << 1)
- Laminar flow
- Channel diameters: 10-100 µm
- Velocities: µm/s to mm/s

Prerequisites:
- FreeCAD with CfdOF workbench installed
- OpenFOAM (macOS: brew install openfoam, mount volume first)
- FreeCAD MCP addon running (port 9875)

Usage:
    python examples/freecad_cfd_microfluidics.py

Related:
- examples/freecad_phloc_raytrace.py - Optical ray tracing
- src/backend/mri/ - Brain network analysis
"""

import xmlrpc.client
import sys


def connect_freecad(host="localhost", port=9875):
    """Connect to FreeCAD RPC server."""
    server = xmlrpc.client.ServerProxy(f'http://{host}:{port}', allow_none=True)
    try:
        if server.ping():
            print(f"Connected to FreeCAD at {host}:{port}")
            return server
    except ConnectionRefusedError:
        print("Error: FreeCAD RPC server not running")
        print("Start it in FreeCAD: MCP Addon toolbar → Start RPC Server")
        sys.exit(1)
    return None


def setup_openfoam(server):
    """Configure CfdOF to use OpenFOAM (macOS app version)."""

    code = '''
import FreeCAD
import subprocess

# Mount OpenFOAM volume (macOS app)
try:
    subprocess.run([
        "/Applications/OpenFOAM-v2512.app/Contents/Resources/volume",
        "mount"
    ], capture_output=True, timeout=10)
    print("OpenFOAM volume mounted")
except Exception as e:
    print(f"Mount note: {e}")

# Configure CfdOF preferences
prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/CfdOF")
prefs.SetString("InstallationPath", "/Volumes/OpenFOAM-v2512")

# Verify
import CfdOF.CfdTools as CfdTools
try:
    runtime = CfdTools.getFoamRuntime()
    foam_dir = CfdTools.getFoamDir()
    print(f"CfdOF configured: {runtime} at {foam_dir}")
except Exception as e:
    print(f"CfdOF config error: {e}")
    print("Make sure OpenFOAM volume is mounted")
'''

    result = server.execute_code(code)
    return result


def create_microfluidic_geometry(server, geometry_type="phloc"):
    """
    Create fluid domain geometry for CFD.

    Args:
        server: FreeCAD RPC connection
        geometry_type: "phloc" for lab-on-chip, "glymphatic" for brain vessels
    """

    if geometry_type == "phloc":
        code = '''
import Part
import FreeCAD as App
from FreeCAD import Vector

doc = App.newDocument("PHLoC_CFD")

# PHLoC fluid domain (scaled for meshing)
# Real: 10µm channels, here: 0.5mm for demo
channel_d = 0.5   # mm (scale up from 10µm for meshing)
chamber_l = 1.5   # mm (matches optical path)
chamber_w = 1.0
chamber_h = 1.0
inlet_len = 3.0
outlet_len = 3.0

# Inlet channel
inlet = Part.makeCylinder(channel_d/2, inlet_len,
                          Vector(-chamber_l/2 - inlet_len, 0, 0),
                          Vector(1, 0, 0))

# Chamber
chamber = Part.makeBox(chamber_l, chamber_w, chamber_h,
                       Vector(-chamber_l/2, -chamber_w/2, -chamber_h/2))

# Outlet channel
outlet = Part.makeCylinder(channel_d/2, outlet_len,
                           Vector(chamber_l/2, 0, 0),
                           Vector(1, 0, 0))

# Fuse
fluid_domain = inlet.fuse(chamber).fuse(outlet)

fluid_obj = doc.addObject("Part::Feature", "FluidDomain")
fluid_obj.Shape = fluid_domain
fluid_obj.ViewObject.ShapeColor = (0.3, 0.6, 1.0)
fluid_obj.ViewObject.Transparency = 30

doc.recompute()
print("Created PHLoC fluid domain")
print(f"  Channel: {channel_d}mm diameter")
print(f"  Chamber: {chamber_l}x{chamber_w}x{chamber_h}mm")
'''

    elif geometry_type == "glymphatic":
        code = '''
import Part
import FreeCAD as App
from FreeCAD import Vector

doc = App.newDocument("Glymphatic_CFD")

# Glymphatic perivascular space geometry
# Annular channel around "blood vessel"
# Outer diameter ~50µm, inner (vessel) ~30µm
# Scaled up for meshing

vessel_d = 1.5    # mm (scaled from ~30µm)
space_d = 2.5     # mm (scaled from ~50µm)
length = 10.0     # mm segment

# Outer cylinder (perivascular space boundary)
outer = Part.makeCylinder(space_d/2, length,
                          Vector(0, 0, 0),
                          Vector(1, 0, 0))

# Inner cylinder (blood vessel - to be subtracted)
inner = Part.makeCylinder(vessel_d/2, length,
                          Vector(0, 0, 0),
                          Vector(1, 0, 0))

# Annular fluid domain
fluid_domain = outer.cut(inner)

fluid_obj = doc.addObject("Part::Feature", "PerivascularSpace")
fluid_obj.Shape = fluid_domain
fluid_obj.ViewObject.ShapeColor = (0.2, 0.8, 0.4)
fluid_obj.ViewObject.Transparency = 30

doc.recompute()
print("Created glymphatic perivascular space")
print(f"  Vessel diameter: {vessel_d}mm (scaled)")
print(f"  Space outer diameter: {space_d}mm (scaled)")
print(f"  Annular gap: {(space_d-vessel_d)/2}mm")
print(f"  Length: {length}mm")
'''

    result = server.execute_code(code)
    return result


def setup_cfd_analysis(server, fluid="water", velocity_ms=0.01):
    """
    Set up CfdOF analysis with physics and materials.

    Args:
        server: FreeCAD RPC connection
        fluid: "water" or "csf" (cerebrospinal fluid)
        velocity_ms: inlet velocity in m/s
    """

    # Material properties
    if fluid == "water":
        density = "998"  # kg/m³
        viscosity = "1.0e-3"  # Pa·s
    elif fluid == "csf":
        # CSF properties similar to water but slightly different
        density = "1007"  # kg/m³
        viscosity = "0.7e-3"  # Pa·s (slightly lower than water)
    else:
        density = "998"
        viscosity = "1.0e-3"

    code = f'''
import FreeCAD as App
import FreeCADGui

FreeCADGui.activateWorkbench("CfdOFWorkbench")
doc = App.ActiveDocument

# Create CFD Analysis
from CfdOF import CfdAnalysis
from CfdOF.Solve import CfdPhysicsSelection
from CfdOF.Solve import CfdFluidMaterial

analysis = CfdAnalysis.makeCfdAnalysis("CfdAnalysis")

# Physics: Isothermal, Laminar (microfluidic regime)
physics = CfdPhysicsSelection.makeCfdPhysicsSelection("PhysicsModel")
# Default is already Isothermal, Laminar, Steady, Single phase
print("Physics: Isothermal, Laminar, Steady state")

# Fluid material
material = CfdFluidMaterial.makeCfdFluidMaterial("FluidProperties")
material.Material = {{
    "Name": "{fluid.upper()}",
    "Type": "Isothermal",
    "Density": "{density} kg/m^3",
    "DynamicViscosity": "{viscosity} kg/m/s"
}}
print(f"Material: {fluid}")
print(f"  Density: {density} kg/m³")
print(f"  Viscosity: {viscosity} Pa·s")

# Mesh settings
from CfdOF.Mesh import CfdMesh
mesh = CfdMesh.makeCfdMesh("CFDMesh")
fluid_domain = doc.getObject("FluidDomain") or doc.getObject("PerivascularSpace")
if fluid_domain:
    mesh.Part = fluid_domain
mesh.CharacteristicLengthMax = "0.2 mm"
mesh.MeshUtility = "gmsh"
print("Mesh: Gmsh, max cell 0.2mm")

# Boundary conditions
from CfdOF.Solve import CfdFluidBoundary

# Inlet
inlet_bc = CfdFluidBoundary.makeCfdFluidBoundary("Inlet")
inlet_bc.BoundaryType = "inlet"
inlet_bc.BoundarySubType = "uniformVelocityInlet"
inlet_bc.Ux = "{velocity_ms} m/s"
inlet_bc.Uy = "0 m/s"
inlet_bc.Uz = "0 m/s"
print(f"Inlet BC: {velocity_ms} m/s velocity")

# Outlet
outlet_bc = CfdFluidBoundary.makeCfdFluidBoundary("Outlet")
outlet_bc.BoundaryType = "outlet"
outlet_bc.BoundarySubType = "staticPressureOutlet"
outlet_bc.Pressure = "0 Pa"
print("Outlet BC: 0 Pa pressure")

# Walls
wall_bc = CfdFluidBoundary.makeCfdFluidBoundary("Walls")
wall_bc.BoundaryType = "wall"
wall_bc.BoundarySubType = "fixedWall"
print("Wall BC: no-slip")

doc.recompute()
print("\\nCFD analysis setup complete!")
print("Next: Assign faces to BCs in FreeCAD GUI, then mesh and solve")
'''

    result = server.execute_code(code)
    return result


def calculate_reynolds(velocity_ms, diameter_m, density=998, viscosity=1e-3):
    """
    Calculate Reynolds number for microfluidic flow.

    Re = ρvD/μ

    Microfluidic regime: Re << 1 (Stokes flow)
    """
    Re = density * velocity_ms * diameter_m / viscosity
    return Re


def main():
    print("=" * 60)
    print("Microfluidic CFD Simulation")
    print("Applications: PHLoC organoid culture, Glymphatic CSF flow")
    print("=" * 60)

    # Connect to FreeCAD
    server = connect_freecad()

    # Setup OpenFOAM
    print("\n1. Configuring OpenFOAM...")
    result = setup_openfoam(server)
    print(result.get('output', result.get('message', str(result))))

    # Create geometry (default: PHLoC)
    print("\n2. Creating fluid domain geometry...")
    result = create_microfluidic_geometry(server, geometry_type="phloc")
    print(result.get('output', result.get('message', str(result))))

    # Setup CFD analysis
    print("\n3. Setting up CFD analysis...")
    result = setup_cfd_analysis(server, fluid="water", velocity_ms=0.01)
    print(result.get('output', result.get('message', str(result))))

    # Calculate Reynolds number
    print("\n4. Flow regime analysis:")
    # PHLoC: 10µm channel, 10mm/s velocity
    Re_phloc = calculate_reynolds(0.01, 10e-6)
    print(f"   PHLoC (10µm, 10mm/s): Re = {Re_phloc:.2e} (Stokes flow)")

    # Glymphatic: 20µm gap, ~10µm/s velocity
    Re_glymph = calculate_reynolds(10e-6, 20e-6, density=1007, viscosity=0.7e-3)
    print(f"   Glymphatic (20µm, 10µm/s): Re = {Re_glymph:.2e} (Stokes flow)")

    print("\n" + "=" * 60)
    print("Setup complete! In FreeCAD:")
    print("1. Assign faces to Inlet/Outlet/Walls boundary conditions")
    print("2. Generate mesh (CfdOF toolbar)")
    print("3. Run solver (CfdOF toolbar)")
    print("4. View results in ParaView")
    print("=" * 60)


if __name__ == "__main__":
    main()
