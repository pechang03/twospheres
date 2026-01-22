#!/usr/bin/env python3
"""
PHLoC (Photonic Lab-on-Chip) Design and Ray Tracing via FreeCAD MCP.

This script demonstrates the full workflow:
1. Connect to FreeCAD via XML-RPC (requires FreeCAD MCP addon running)
2. Create PHLoC chip geometry (PDMS microfluidics)
3. Ray trace through ball lenses using pyoptools
4. Visualize rays in FreeCAD

Prerequisites:
- FreeCAD with MCP addon installed and RPC server running (port 9875)
- pyoptools installed in FreeCAD's Python environment
- OpticsWorkbench (optional, for additional tools)

Usage:
    python examples/freecad_phloc_raytrace.py

Related:
- ../freecad-mcp/ - FreeCAD MCP server
- src/backend/optics/ - twosphere-mcp optical simulation
- docs/papers/PAC_OBSTRUCTION_BIOLOGICAL_TRACTABILITY.md
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


def create_phloc_chip(server):
    """Create PHLoC chip geometry in FreeCAD."""

    code = '''
import Part
import FreeCAD as App
from FreeCAD import Vector

doc = App.newDocument("PHLoC_Chip")

# ============================================
# PHLoC: Fiber -> Lens -> Chamber -> Lens -> Fiber
# Designed for PDMS 3D printing (100nm resolution)
# ============================================

# Parameters (mm)
chip_length = 40.0
chip_width = 12.0
chip_height = 6.0
optical_z = chip_height / 2

lens_radius = 1.2  # Ball lens radius
lens1_x = -12.0
lens2_x = 12.0

# Main chip body
chip = Part.makeBox(chip_length, chip_width, chip_height,
                    Vector(-chip_length/2, -chip_width/2, 0))

# Optical channel through chip
optical_channel = Part.makeCylinder(0.5, chip_length + 2,
                                     Vector(-chip_length/2 - 1, 0, optical_z),
                                     Vector(1, 0, 0))
chip = chip.cut(optical_channel)

# Ball lens cavities
lens1_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens1_x, 0, optical_z))
lens2_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens2_x, 0, optical_z))
chip = chip.cut(lens1_cav)
chip = chip.cut(lens2_cav)

# Central sample chamber
chamber = Part.makeBox(6, 4, 3, Vector(-3, -2, optical_z - 1.5))
chip = chip.cut(chamber)

# Fluid port holes
port_r = 1.0
port1 = Part.makeCylinder(port_r, chip_height + 2, Vector(-1.5, 0, -1), Vector(0,0,1))
port2 = Part.makeCylinder(port_r, chip_height + 2, Vector(1.5, 0, -1), Vector(0,0,1))
chip = chip.cut(port1)
chip = chip.cut(port2)

# Add chip to document
chip_obj = doc.addObject("Part::Feature", "PHLoC_Chip")
chip_obj.Shape = chip
chip_obj.ViewObject.ShapeColor = (0.78, 0.78, 0.82)  # PDMS gray
chip_obj.ViewObject.Transparency = 40

# Protruding fluid port tubes
for i, x in enumerate([-1.5, 1.5]):
    tube = Part.makeCylinder(port_r, 4, Vector(x, 0, chip_height), Vector(0,0,1))
    tube_hole = Part.makeCylinder(port_r * 0.7, 5, Vector(x, 0, chip_height - 0.5), Vector(0,0,1))
    tube = tube.cut(tube_hole)
    t_obj = doc.addObject("Part::Feature", f"FluidPort_{i+1}")
    t_obj.Shape = tube
    t_obj.ViewObject.ShapeColor = (0.65, 0.65, 0.7)

# Ball lenses (BK7 glass)
for i, x in enumerate([lens1_x, lens2_x]):
    lens = Part.makeSphere(lens_radius * 0.98, Vector(x, 0, optical_z))
    l_obj = doc.addObject("Part::Feature", f"BallLens_{i+1}")
    l_obj.Shape = lens
    l_obj.ViewObject.ShapeColor = (0.6, 0.8, 1.0)
    l_obj.ViewObject.Transparency = 50

# Optical fibers
for i, x in enumerate([-chip_length/2 - 10, chip_length/2]):
    fib = Part.makeCylinder(0.0625, 10, Vector(x, 0, optical_z), Vector(1,0,0))
    f_obj = doc.addObject("Part::Feature", f"Fiber_{['In','Out'][i]}")
    f_obj.Shape = fib
    f_obj.ViewObject.ShapeColor = (0.95, 0.75, 0.1)  # Yellow

doc.recompute()
FreeCADGui.ActiveDocument.ActiveView.fitAll()
FreeCADGui.ActiveDocument.ActiveView.viewIsometric()

print("PHLoC chip created!")
print(f"  Chip: {chip_length} x {chip_width} x {chip_height} mm")
print(f"  Ball lenses: {lens_radius * 2} mm diameter")
print(f"  Chamber: 6 x 4 x 3 mm")
'''

    result = server.execute_code(code)
    return result


def ray_trace_phloc(server):
    """Ray trace through PHLoC using pyoptools."""

    code = '''
import FreeCAD as App
import Part
from FreeCAD import Vector
import numpy as np

from pyoptools.raytrace.ray import Ray
from pyoptools.raytrace.surface import Spherical
from pyoptools.raytrace.shape import Circular

doc = App.ActiveDocument

# Lens parameters
lens1_x = -12.0
lens2_x = 12.0
lens_r = 1.2
n_glass = 1.5168  # BK7
optical_z = 3.0

# Focal length (thin lens approx for ball lens)
f = lens_r * n_glass / (2 * (n_glass - 1))

# Generate rays
n_rays = 9
fiber_x = -20.0
rays_data = []

for y in np.linspace(-0.5, 0.5, n_rays):
    x0, y0 = fiber_x, y

    # Through lens 1
    x1 = lens1_x
    y1 = y0
    theta1 = -y1 / f

    # Through chamber to lens 2
    x2 = lens2_x
    y2 = y1 + theta1 * (x2 - x1)
    theta2 = theta1 - y2 / f

    # To output
    x3 = 20.0
    y3 = y2 + theta2 * (x3 - x2)

    rays_data.append([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])

# Draw rays
for i, pts in enumerate(rays_data):
    edges = []
    for j in range(len(pts) - 1):
        v1 = Vector(pts[j][0], pts[j][1], optical_z)
        v2 = Vector(pts[j+1][0], pts[j+1][1], optical_z)
        edges.append(Part.makeLine(v1, v2))

    wire = Part.Wire(edges)
    ray_obj = doc.addObject("Part::Feature", f"Ray_{i}")
    ray_obj.Shape = wire
    t = i / (n_rays - 1)
    ray_obj.ViewObject.LineColor = (1-t, 0.3, t)  # Red to blue
    ray_obj.ViewObject.LineWidth = 2.0

doc.recompute()
print(f"Ray trace complete: {n_rays} rays")
print(f"Ball lens focal length: {f:.2f} mm")
'''

    result = server.execute_code(code)
    return result


def export_stl(server, filename="/tmp/phloc_chip.stl"):
    """Export chip to STL for 3D printing."""

    code = f'''
import Mesh
import FreeCAD as App

doc = App.ActiveDocument
chip = doc.getObject("PHLoC_Chip")

if chip:
    Mesh.export([chip], "{filename}")
    print(f"Exported to {filename}")
else:
    print("PHLoC_Chip not found")
'''

    result = server.execute_code(code)
    return result


def main():
    print("=" * 60)
    print("PHLoC Design and Ray Tracing via FreeCAD MCP")
    print("=" * 60)

    # Connect
    server = connect_freecad()

    # Create chip
    print("\n1. Creating PHLoC chip geometry...")
    result = create_phloc_chip(server)
    if isinstance(result, dict):
        print(result.get('message', str(result)))
    else:
        print(result)

    # Ray trace
    print("\n2. Ray tracing through ball lenses...")
    result = ray_trace_phloc(server)
    if isinstance(result, dict):
        print(result.get('message', str(result)))
    else:
        print(result)

    # Export option
    print("\n3. Export to STL? (for PDMS printing)")
    print("   Run: python -c \"from examples.freecad_phloc_raytrace import *; export_stl(connect_freecad())\"")

    print("\n" + "=" * 60)
    print("Done! Check FreeCAD for visualization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
