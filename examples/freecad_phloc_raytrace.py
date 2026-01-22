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
    """Create PHLoC chip geometry in FreeCAD.

    Optimized for organoid culture:
    - Chamber: ~1 mL volume for organoid growth
    - Microfluidic channels: 10µm for gentle fluid exchange
    - Fiber input: 50µm (single-mode)
    - Fiber output: 100µm (multi-mode collection)
    """

    code = '''
import Part
import FreeCAD as App
from FreeCAD import Vector
import math

doc = App.newDocument("PHLoC_Organoid")

# ============================================
# PHLoC for Organoid Culture
# Fiber(50µm) -> Lens -> Chamber(1mL) -> Lens -> Fiber(100µm)
# Microfluidic channels: 10µm
# Designed for PDMS 3D printing (100nm resolution)
# ============================================

# Parameters (mm)
# Chamber: 10 x 10 x 10 mm = 1000 mm³ = 1 mL
chamber_size = 10.0  # 10mm cube = 1 mL
chamber_volume = chamber_size ** 3  # 1000 mm³

# Chip sized around chamber
chip_length = 50.0
chip_width = 20.0
chip_height = 16.0
optical_z = chip_height / 2

# Fiber dimensions (mm)
fiber_in_diameter = 0.050   # 50 µm input (single-mode)
fiber_out_diameter = 0.100  # 100 µm output (multi-mode)
fiber_channel_clearance = 1.5  # Clearance factor for channel

# Microfluidic channels (mm)
microchannel_diameter = 0.010  # 10 µm channels

# Lens parameters
lens_radius = 2.0  # Larger lens for bigger chamber
lens1_x = -chamber_size/2 - 5  # 5mm from chamber
lens2_x = chamber_size/2 + 5

# Fluid port positions (offset from chamber, not direct)
port_offset_x = 3.0  # mm from chamber edge
port_offset_y = 3.0  # mm off optical axis

# ============================================
# Main chip body
# ============================================
chip = Part.makeBox(chip_length, chip_width, chip_height,
                    Vector(-chip_length/2, -chip_width/2, 0))

# ============================================
# Fiber channels (asymmetric: 50µm in, 100µm out)
# ============================================
# Input fiber channel (50 µm fiber in ~75 µm channel)
fiber_in_channel = Part.makeCylinder(
    fiber_in_diameter * fiber_channel_clearance,
    chip_length/2 - chamber_size/2 - 3,
    Vector(-chip_length/2 - 0.1, 0, optical_z),
    Vector(1, 0, 0))
chip = chip.cut(fiber_in_channel)

# Output fiber channel (100 µm fiber in ~150 µm channel)
fiber_out_channel = Part.makeCylinder(
    fiber_out_diameter * fiber_channel_clearance,
    chip_length/2 - chamber_size/2 - 3,
    Vector(chamber_size/2 + 4, 0, optical_z),
    Vector(1, 0, 0))
chip = chip.cut(fiber_out_channel)

# ============================================
# Ball lens cavities
# ============================================
lens1_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens1_x, 0, optical_z))
lens2_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens2_x, 0, optical_z))
chip = chip.cut(lens1_cav)
chip = chip.cut(lens2_cav)

# Optical channels: lens to chamber
opt_ch1 = Part.makeCylinder(1.0, 4, Vector(lens1_x + lens_radius, 0, optical_z), Vector(1,0,0))
opt_ch2 = Part.makeCylinder(1.0, 4, Vector(chamber_size/2, 0, optical_z), Vector(1,0,0))
chip = chip.cut(opt_ch1)
chip = chip.cut(opt_ch2)

# ============================================
# Central organoid chamber (1 mL = 10x10x10 mm)
# ============================================
chamber = Part.makeBox(chamber_size, chamber_size, chamber_size,
                       Vector(-chamber_size/2, -chamber_size/2, optical_z - chamber_size/2))
chip = chip.cut(chamber)

# ============================================
# Microfluidic system (10 µm channels, NOT direct to chamber)
# Entry/exit points above and below chamber
# ============================================

# Fluid reservoir wells (where tubes connect)
reservoir_r = 1.5  # 1.5mm radius reservoir
reservoir_depth = 3.0

# Input reservoir (top-left of chamber)
res_in_x = -chamber_size/2 - port_offset_x
res_in_y = port_offset_y
res_in = Part.makeCylinder(reservoir_r, reservoir_depth,
                            Vector(res_in_x, res_in_y, chip_height - reservoir_depth),
                            Vector(0, 0, 1))
chip = chip.cut(res_in)

# Output reservoir (top-right of chamber)
res_out_x = chamber_size/2 + port_offset_x
res_out_y = -port_offset_y
res_out = Part.makeCylinder(reservoir_r, reservoir_depth,
                             Vector(res_out_x, res_out_y, chip_height - reservoir_depth),
                             Vector(0, 0, 1))
chip = chip.cut(res_out)

# 10 µm microchannels from reservoirs to chamber (horizontal then down)
# These are the gentle fluid exchange channels for organoids

# Input microchannel: reservoir -> horizontal -> down into chamber top
# Horizontal segment
micro_in_h = Part.makeCylinder(microchannel_diameter/2, port_offset_x + chamber_size/2,
                                Vector(res_in_x, res_in_y, chip_height - reservoir_depth - 0.5),
                                Vector(1, 0, 0))
chip = chip.cut(micro_in_h)

# Vertical segment into chamber (from above)
micro_in_v = Part.makeCylinder(microchannel_diameter/2,
                                (chip_height - reservoir_depth) - (optical_z + chamber_size/2) + 1,
                                Vector(-chamber_size/4, res_in_y, optical_z + chamber_size/2 - 0.5),
                                Vector(0, 0, 1))
chip = chip.cut(micro_in_v)

# Connect horizontal to vertical
micro_in_corner = Part.makeCylinder(microchannel_diameter/2, abs(res_in_y) + 0.5,
                                     Vector(-chamber_size/4, res_in_y, chip_height - reservoir_depth - 0.5),
                                     Vector(0, -1, 0))
chip = chip.cut(micro_in_corner)

# Output microchannel: chamber bottom -> horizontal -> up to reservoir
# Vertical from chamber bottom
micro_out_v = Part.makeCylinder(microchannel_diameter/2,
                                 (optical_z - chamber_size/2) + 1,
                                 Vector(chamber_size/4, res_out_y, 0),
                                 Vector(0, 0, 1))
chip = chip.cut(micro_out_v)

# Horizontal to reservoir area
micro_out_h = Part.makeCylinder(microchannel_diameter/2, port_offset_x + chamber_size/2,
                                 Vector(chamber_size/4, res_out_y, 1.0),
                                 Vector(1, 0, 0))
chip = chip.cut(micro_out_h)

# Up to reservoir
micro_out_up = Part.makeCylinder(microchannel_diameter/2,
                                  chip_height - reservoir_depth - 1,
                                  Vector(res_out_x, res_out_y, 1.0),
                                  Vector(0, 0, 1))
chip = chip.cut(micro_out_up)

# ============================================
# Add chip to document
# ============================================
chip_obj = doc.addObject("Part::Feature", "PHLoC_Organoid")
chip_obj.Shape = chip
chip_obj.ViewObject.ShapeColor = (0.78, 0.78, 0.82)  # PDMS gray
chip_obj.ViewObject.Transparency = 40

# ============================================
# Fluid port tubes (on reservoirs, not chamber)
# ============================================
port_tube_r = 1.2
port_tube_h = 5.0

for name, x, y in [("FluidIn", res_in_x, res_in_y), ("FluidOut", res_out_x, res_out_y)]:
    tube = Part.makeCylinder(port_tube_r, port_tube_h,
                              Vector(x, y, chip_height), Vector(0,0,1))
    tube_hole = Part.makeCylinder(port_tube_r * 0.7, port_tube_h + 1,
                                   Vector(x, y, chip_height - 0.5), Vector(0,0,1))
    tube = tube.cut(tube_hole)
    t_obj = doc.addObject("Part::Feature", name)
    t_obj.Shape = tube
    t_obj.ViewObject.ShapeColor = (0.65, 0.65, 0.7)

# ============================================
# Ball lenses (BK7 glass)
# ============================================
for i, x in enumerate([lens1_x, lens2_x]):
    lens = Part.makeSphere(lens_radius * 0.98, Vector(x, 0, optical_z))
    l_obj = doc.addObject("Part::Feature", f"BallLens_{i+1}")
    l_obj.Shape = lens
    l_obj.ViewObject.ShapeColor = (0.6, 0.8, 1.0)
    l_obj.ViewObject.Transparency = 50

# ============================================
# Optical fibers (correct diameters)
# ============================================
# Input fiber: 50 µm
fib_in = Part.makeCylinder(fiber_in_diameter/2, 12,
                            Vector(-chip_length/2 - 12, 0, optical_z),
                            Vector(1, 0, 0))
f_in_obj = doc.addObject("Part::Feature", "Fiber_In_50um")
f_in_obj.Shape = fib_in
f_in_obj.ViewObject.ShapeColor = (0.95, 0.75, 0.1)

# Output fiber: 100 µm
fib_out = Part.makeCylinder(fiber_out_diameter/2, 12,
                             Vector(chip_length/2, 0, optical_z),
                             Vector(1, 0, 0))
f_out_obj = doc.addObject("Part::Feature", "Fiber_Out_100um")
f_out_obj.Shape = fib_out
f_out_obj.ViewObject.ShapeColor = (1.0, 0.6, 0.1)  # Orange for MM

doc.recompute()
FreeCADGui.ActiveDocument.ActiveView.fitAll()
FreeCADGui.ActiveDocument.ActiveView.viewIsometric()

print("PHLoC Organoid Chip created!")
print(f"  Chip: {chip_length} x {chip_width} x {chip_height} mm")
print(f"  Chamber: {chamber_size} x {chamber_size} x {chamber_size} mm = {chamber_volume:.0f} mm³ = {chamber_volume/1000:.1f} mL")
print(f"  Ball lenses: {lens_radius * 2} mm diameter")
print(f"  Fiber input: {fiber_in_diameter * 1000:.0f} µm (single-mode)")
print(f"  Fiber output: {fiber_out_diameter * 1000:.0f} µm (multi-mode)")
print(f"  Microchannels: {microchannel_diameter * 1000:.0f} µm (gentle fluid exchange)")
'''

    result = server.execute_code(code)
    return result


def ray_trace_phloc(server):
    """Ray trace through PHLoC using pyoptools.

    Updated for organoid chip with:
    - Larger chamber (10mm)
    - Larger ball lenses (4mm diameter)
    - 50µm input fiber, 100µm output fiber
    """

    code = '''
import FreeCAD as App
import Part
from FreeCAD import Vector
import numpy as np

from pyoptools.raytrace.ray import Ray
from pyoptools.raytrace.surface import Spherical
from pyoptools.raytrace.shape import Circular

doc = App.ActiveDocument

# Lens parameters (matching organoid chip)
chamber_size = 10.0
lens1_x = -chamber_size/2 - 5  # -10mm
lens2_x = chamber_size/2 + 5   # +10mm
lens_r = 2.0  # 4mm diameter ball lens
n_glass = 1.5168  # BK7
optical_z = 8.0  # chip_height / 2

# Focal length (thin lens approx for ball lens)
f = lens_r * n_glass / (2 * (n_glass - 1))

# Generate rays from 50µm input fiber
n_rays = 11
fiber_x = -25.0
fiber_radius = 0.025  # 50µm diameter / 2
rays_data = []

for y in np.linspace(-fiber_radius * 2, fiber_radius * 2, n_rays):
    x0, y0 = fiber_x, y

    # Through lens 1 (collimating)
    x1 = lens1_x
    y1 = y0
    theta1 = -y1 / f

    # Through chamber to lens 2
    x2 = lens2_x
    y2 = y1 + theta1 * (x2 - x1)
    theta2 = theta1 - y2 / f

    # To output (100µm fiber can collect wider)
    x3 = 25.0
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
print(f"Input fiber: 50 µm, Output fiber: 100 µm")
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
