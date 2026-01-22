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
    - Chamber: 1.5 x 1 x 1 mm = 1.5 µL (1.5mm optical path for absorption)
    - Microfluidic channels: 10µm for gentle fluid exchange (TOP entry, routes to chamber above/below)
    - PDMS injection channels: 45% wider than fiber, at fiber termination points
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
# Fiber(50µm) -> Lens -> Chamber(1.5µL) -> Lens -> Fiber(100µm)
#
# Channel systems:
# 1. Microfluidic (10µm): TOP entry -> routes to chamber above/below
# 2. PDMS injection (45% wider): at fiber termination points
#
# Designed for PDMS 3D printing (100nm resolution)
# ============================================

# Parameters (mm)
# Chamber: 1.5 x 1 x 1 mm = 1.5 µL (1.5mm for Beer-Lambert absorption)
chamber_length = 1.5   # Optical path length
chamber_width = 1.0
chamber_height = 1.0
chamber_volume_uL = chamber_length * chamber_width * chamber_height  # 1.5 µL

# Chip sized around chamber
chip_length = 20.0
chip_width = 8.0
chip_height = 6.0
optical_z = chip_height / 2

# Fiber dimensions (mm)
fiber_in_diameter = 0.050   # 50 µm input (single-mode)
fiber_out_diameter = 0.100  # 100 µm output (multi-mode)

# PDMS injection channels: 45% wider than fiber (at fiber termination)
pdms_inject_in = fiber_in_diameter * 1.45   # ~72.5 µm
pdms_inject_out = fiber_out_diameter * 1.45  # ~145 µm

# Microfluidic channels (mm)
microchannel_diameter = 0.010  # 10 µm channels

# Lens parameters (scaled for small chamber)
lens_radius = 0.5  # 1mm diameter ball lens
lens1_x = -chamber_length/2 - 1.5  # 1.5mm from chamber
lens2_x = chamber_length/2 + 1.5

# ============================================
# Main chip body
# ============================================
chip = Part.makeBox(chip_length, chip_width, chip_height,
                    Vector(-chip_length/2, -chip_width/2, 0))

# ============================================
# Fiber channels (asymmetric: 50µm in, 100µm out)
# Slightly larger than fiber for insertion clearance
# ============================================
fiber_clearance = 1.1  # 10% clearance

# Input fiber channel (50 µm fiber)
fiber_in_len = chip_length/2 - abs(lens1_x) - lens_radius
fiber_in_channel = Part.makeCylinder(
    fiber_in_diameter * fiber_clearance / 2,
    fiber_in_len,
    Vector(-chip_length/2 - 0.1, 0, optical_z),
    Vector(1, 0, 0))
chip = chip.cut(fiber_in_channel)

# Output fiber channel (100 µm fiber)
fiber_out_len = chip_length/2 - abs(lens2_x) - lens_radius
fiber_out_channel = Part.makeCylinder(
    fiber_out_diameter * fiber_clearance / 2,
    fiber_out_len,
    Vector(lens2_x + lens_radius, 0, optical_z),
    Vector(1, 0, 0))
chip = chip.cut(fiber_out_channel)

# ============================================
# PDMS injection channels (45% wider, at fiber termination)
# These allow liquid PDMS to fill around fiber tips
# ============================================
# Input side PDMS injection (from top, to fiber termination point)
pdms_in_x = -chip_length/2 + fiber_in_len - 0.5  # Near fiber end
pdms_in_channel = Part.makeCylinder(
    pdms_inject_in / 2,
    chip_height - optical_z + 0.5,
    Vector(pdms_in_x, 0, optical_z - 0.25),
    Vector(0, 0, 1))
chip = chip.cut(pdms_in_channel)

# Output side PDMS injection (from top, to fiber termination point)
pdms_out_x = lens2_x + lens_radius + 0.5  # Near fiber end
pdms_out_channel = Part.makeCylinder(
    pdms_inject_out / 2,
    chip_height - optical_z + 0.5,
    Vector(pdms_out_x, 0, optical_z - 0.25),
    Vector(0, 0, 1))
chip = chip.cut(pdms_out_channel)

# ============================================
# Ball lens cavities
# ============================================
lens1_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens1_x, 0, optical_z))
lens2_cav = Part.makeSphere(lens_radius * 1.02, Vector(lens2_x, 0, optical_z))
chip = chip.cut(lens1_cav)
chip = chip.cut(lens2_cav)

# Optical channels: lens to chamber (clear optical path)
opt_ch1 = Part.makeCylinder(0.3, chamber_length/2 + 1.5 - lens_radius,
                             Vector(lens1_x + lens_radius, 0, optical_z), Vector(1,0,0))
opt_ch2 = Part.makeCylinder(0.3, chamber_length/2 + 1.5 - lens_radius,
                             Vector(chamber_length/2, 0, optical_z), Vector(1,0,0))
chip = chip.cut(opt_ch1)
chip = chip.cut(opt_ch2)

# ============================================
# Central organoid chamber (1.5 x 1 x 1 mm = 1.5 µL)
# ============================================
chamber = Part.makeBox(chamber_length, chamber_width, chamber_height,
                       Vector(-chamber_length/2, -chamber_width/2, optical_z - chamber_height/2))
chip = chip.cut(chamber)

# ============================================
# Microfluidic system (10 µm channels)
# BOTH fluid ports on TOP of chip
# Internal channels route to chamber from ABOVE and BELOW
# ============================================

# Fluid reservoir wells (both on TOP surface)
reservoir_r = 0.8  # 0.8mm radius reservoir
reservoir_depth = 1.5

# Input reservoir (top surface, offset from optical axis)
res_in_x = -2.0  # Left of chamber
res_in_y = 2.0   # Off optical axis (Y direction)
res_in = Part.makeCylinder(reservoir_r, reservoir_depth,
                            Vector(res_in_x, res_in_y, chip_height - reservoir_depth),
                            Vector(0, 0, 1))
chip = chip.cut(res_in)

# Output reservoir (top surface, offset from optical axis)
res_out_x = 2.0   # Right of chamber
res_out_y = 2.0   # Same Y side (both tubes on top)
res_out = Part.makeCylinder(reservoir_r, reservoir_depth,
                             Vector(res_out_x, res_out_y, chip_height - reservoir_depth),
                             Vector(0, 0, 1))
chip = chip.cut(res_out)

# ============================================
# 10 µm microchannels routing to CHAMBER (not lenses!)
# Input: TOP reservoir -> down -> horizontal -> enters chamber from ABOVE
# Output: chamber BELOW -> horizontal -> up -> TOP reservoir
# ============================================

# --- INPUT CHANNEL: enters chamber from ABOVE ---
# Vertical down from reservoir
micro_in_v1 = Part.makeCylinder(microchannel_diameter/2,
                                 chip_height - reservoir_depth - (optical_z + chamber_height/2) - 0.2,
                                 Vector(res_in_x, res_in_y, optical_z + chamber_height/2 + 0.3),
                                 Vector(0, 0, 1))
chip = chip.cut(micro_in_v1)

# Horizontal toward chamber (in Y direction)
micro_in_h = Part.makeCylinder(microchannel_diameter/2,
                                abs(res_in_y) - chamber_width/2 + 0.1,
                                Vector(res_in_x, chamber_width/2 - 0.1, optical_z + chamber_height/2 + 0.3),
                                Vector(0, 1, 0))
chip = chip.cut(micro_in_h)

# Short vertical into chamber top
micro_in_v2 = Part.makeCylinder(microchannel_diameter/2, 0.4,
                                 Vector(res_in_x, 0, optical_z + chamber_height/2 - 0.1),
                                 Vector(0, 0, 1))
chip = chip.cut(micro_in_v2)

# --- OUTPUT CHANNEL: exits chamber from BELOW ---
# Vertical down from chamber bottom
micro_out_v1 = Part.makeCylinder(microchannel_diameter/2,
                                  optical_z - chamber_height/2,
                                  Vector(res_out_x, 0, 0),
                                  Vector(0, 0, 1))
chip = chip.cut(micro_out_v1)

# Horizontal toward reservoir Y position
micro_out_h = Part.makeCylinder(microchannel_diameter/2,
                                 abs(res_out_y) - 0.1,
                                 Vector(res_out_x, 0, 0.5),
                                 Vector(0, 1, 0))
chip = chip.cut(micro_out_h)

# Vertical up to reservoir
micro_out_v2 = Part.makeCylinder(microchannel_diameter/2,
                                  chip_height - reservoir_depth - 0.5,
                                  Vector(res_out_x, res_out_y, 0.5),
                                  Vector(0, 0, 1))
chip = chip.cut(micro_out_v2)

# ============================================
# Add chip to document
# ============================================
chip_obj = doc.addObject("Part::Feature", "PHLoC_Organoid")
chip_obj.Shape = chip
chip_obj.ViewObject.ShapeColor = (0.78, 0.78, 0.82)  # PDMS gray
chip_obj.ViewObject.Transparency = 40

# ============================================
# Fluid port tubes (both on TOP)
# ============================================
port_tube_r = 0.6
port_tube_h = 3.0

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
fib_in = Part.makeCylinder(fiber_in_diameter/2, 8,
                            Vector(-chip_length/2 - 8, 0, optical_z),
                            Vector(1, 0, 0))
f_in_obj = doc.addObject("Part::Feature", "Fiber_In_50um")
f_in_obj.Shape = fib_in
f_in_obj.ViewObject.ShapeColor = (0.95, 0.75, 0.1)

# Output fiber: 100 µm
fib_out = Part.makeCylinder(fiber_out_diameter/2, 8,
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
print(f"  Chamber: {chamber_length} x {chamber_width} x {chamber_height} mm = {chamber_volume_uL:.1f} µL")
print(f"  Ball lenses: {lens_radius * 2} mm diameter")
print(f"  Fiber input: {fiber_in_diameter * 1000:.0f} µm (single-mode)")
print(f"  Fiber output: {fiber_out_diameter * 1000:.0f} µm (multi-mode)")
print(f"  Microchannels: {microchannel_diameter * 1000:.0f} µm (to chamber above/below)")
print(f"  PDMS injection: {pdms_inject_in * 1000:.0f} µm in, {pdms_inject_out * 1000:.0f} µm out (at fiber ends)")
'''

    result = server.execute_code(code)
    return result


def ray_trace_phloc(server):
    """Ray trace through PHLoC using pyoptools.

    Updated for organoid chip with:
    - Chamber: 1.5 x 1 x 1 mm (1.5mm optical path)
    - Ball lenses: 1mm diameter
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
chamber_length = 1.5
lens_r = 0.5  # 1mm diameter ball lens
lens1_x = -chamber_length/2 - 1.5  # -2.25mm
lens2_x = chamber_length/2 + 1.5   # +2.25mm
n_glass = 1.5168  # BK7
optical_z = 3.0  # chip_height / 2

# Focal length (thin lens approx for ball lens)
f = lens_r * n_glass / (2 * (n_glass - 1))

# Generate rays from 50µm input fiber
n_rays = 11
fiber_x = -10.0
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
    x3 = 10.0
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
print(f"Chamber optical path: {chamber_length} mm")
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
