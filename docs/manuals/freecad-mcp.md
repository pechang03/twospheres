# FreeCAD MCP Integration Manual

## Overview

FreeCAD integration via XML-RPC server for programmatic CAD creation. Used for microfluidic chip design, 3D modeling, and fabrication file export.

## Server Setup

### Start FreeCAD with MCP Server

```bash
# In FreeCAD Python console or startup script
import FreeCAD
import xmlrpc.server

class FreeCADServer:
    def ping(self):
        return True

    def execute_code(self, code):
        try:
            exec(code, globals())
            return {"success": True, "message": "OK"}
        except Exception as e:
            return {"success": False, "error": str(e)}

server = xmlrpc.server.SimpleXMLRPCServer(('localhost', 9875))
server.register_instance(FreeCADServer())
server.serve_forever()
```

Default port: **9875**

## Python Client Usage

### Basic Connection

```python
import xmlrpc.client

server = xmlrpc.client.ServerProxy('http://localhost:9875')

# Test connection
if server.ping():
    print("Connected to FreeCAD")
```

### Execute FreeCAD Code

```python
result = server.execute_code('''
import FreeCAD as App
import Part

doc = App.newDocument("MyDoc")
box = Part.makeBox(10, 10, 10)
obj = doc.addObject("Part::Feature", "Box")
obj.Shape = box
doc.recompute()
"Created box"
''')

print(result)  # {'success': True, 'message': 'Created box'}
```

## Microfluidic Components

### Chip Base

```python
code = '''
import FreeCAD as App
import Part
from FreeCAD import Vector

doc = App.newDocument("MicrofluidicChip")

# Standard microscope slide size: 75mm x 25mm x 3mm
chip = Part.makeBox(75, 25, 3, Vector(0, 0, 0))

obj = doc.addObject("Part::Feature", "ChipBase")
obj.Shape = chip
obj.ViewObject.ShapeColor = (0.8, 0.85, 0.9)  # Light blue (PDMS)
obj.ViewObject.Transparency = 50

doc.recompute()
'''
server.execute_code(code)
```

### Channels

```python
code = '''
import Part
from FreeCAD import Vector

# Channel parameters
start = Vector(5, 12.5, 1.5)
end = Vector(70, 12.5, 1.5)
diameter = 0.1  # 100 µm

direction = end - start
length = direction.Length
direction.normalize()

channel = Part.makeCylinder(diameter/2, length, start, direction)

# Cut from chip
chip = doc.getObject("ChipBase")
chip.Shape = chip.Shape.cut(channel)
doc.recompute()
'''
server.execute_code(code)
```

### Chambers

```python
code = '''
import Part
from FreeCAD import Vector

# Circular chamber: 2mm diameter, 0.5mm deep
center = Vector(37.5, 12.5, 2.5)
chamber = Part.makeCylinder(1, 0.5, center, Vector(0, 0, 1))

chip = doc.getObject("ChipBase")
chip.Shape = chip.Shape.cut(chamber)
doc.recompute()
'''
server.execute_code(code)
```

### Ports (Inlet/Outlet)

```python
code = '''
import Part
from FreeCAD import Vector

# Port: through-hole for tubing, 1.6mm diameter (for 1/16" tubing)
port_pos = Vector(5, 12.5, 0)
port = Part.makeCylinder(0.8, 3, port_pos, Vector(0, 0, 1))

chip = doc.getObject("ChipBase")
chip.Shape = chip.Shape.cut(port)
doc.recompute()
'''
server.execute_code(code)
```

## BrainChipDesigner Integration

```python
from backend.simulation.brain_chip_designer import BrainChipDesigner, FreeCADExporter

designer = BrainChipDesigner()
chip = designer.design_latin_square_mixer(n=4)

exporter = FreeCADExporter(host="localhost", port=9875)
if exporter.connect():
    result = exporter.export_design(chip, doc_name="LatinSquareChip")
    print(result)
```

## Export Formats

### STL (3D Printing)

```python
code = '''
import Mesh

mesh = doc.getObject("ChipBase").Shape.tessellate(0.1)
Mesh.Mesh(mesh).write("/tmp/chip.stl")
"Exported STL"
'''
server.execute_code(code)
```

### STEP (CAD Exchange)

```python
code = '''
import Part

Part.export([doc.getObject("ChipBase")], "/tmp/chip.step")
"Exported STEP"
'''
server.execute_code(code)
```

### DXF (Laser Cutting)

```python
code = '''
import importDXF

# Get top face for 2D export
shape = doc.getObject("ChipBase").Shape
top_face = shape.Faces[0]  # Adjust index for correct face

importDXF.export([top_face], "/tmp/chip.dxf")
"Exported DXF"
'''
server.execute_code(code)
```

## Useful Addons

Reference: `docs/freecad-addons.csv`

| Addon | Use |
|-------|-----|
| CfdOF | CFD simulation with OpenFOAM |
| LasercutterSVGExport | 2D export for laser cutting |
| Assembly4 | Constraint-based assembly |
| Curves | NURBS curves for serpentine channels |
| HilbertCurve | Space-filling mixer patterns |
| 3D_Printing_Tools | Mesh repair for STL export |

## Common Patterns

### Serpentine Channel

```python
code = '''
import Part
from FreeCAD import Vector
import math

def create_serpentine(n_turns, turn_radius, straight_length, channel_width, z_height):
    """Create serpentine mixer path."""
    points = []
    x, y = 0, 0
    direction = 1

    for i in range(n_turns):
        points.append(Vector(x, y, z_height))
        x += direction * straight_length
        points.append(Vector(x, y, z_height))
        y += turn_radius * 2
        direction *= -1

    points.append(Vector(x, y, z_height))

    # Create wire and pipe
    wire = Part.makePolygon(points)
    circle = Part.makeCircle(channel_width/2)
    pipe = wire.makePipe(circle)

    return pipe

serpentine = create_serpentine(
    n_turns=10,
    turn_radius=1,
    straight_length=5,
    channel_width=0.1,
    z_height=1.5
)

obj = doc.addObject("Part::Feature", "Serpentine")
obj.Shape = serpentine
doc.recompute()
'''
server.execute_code(code)
```

### Latin Square Chamber Array

```python
code = '''
import Part
from FreeCAD import Vector

def create_latin_square_chambers(n, chamber_diameter, spacing, z_depth):
    """Create n×n chamber array."""
    chambers = []

    for i in range(n):
        for j in range(n):
            x = spacing * (i + 0.5)
            y = spacing * (j + 0.5)
            center = Vector(x, y, 3 - z_depth)
            chamber = Part.makeCylinder(chamber_diameter/2, z_depth, center, Vector(0,0,1))
            chambers.append(chamber)

    return chambers

chambers = create_latin_square_chambers(n=4, chamber_diameter=2, spacing=5, z_depth=0.5)

# Combine and cut from chip
chip = doc.getObject("ChipBase")
for i, chamber in enumerate(chambers):
    chip.Shape = chip.Shape.cut(chamber)

doc.recompute()
'''
server.execute_code(code)
```

## Troubleshooting

### Connection Refused

FreeCAD server not running. Start FreeCAD and execute server script.

### execute_code Returns Error

Check FreeCAD console for Python errors. Common issues:
- Missing imports (add `import Part`, etc.)
- Object name conflicts
- Invalid geometry operations

### Slow Operations

Complex boolean operations (cut, fuse) can be slow. Use `doc.recompute()` only at the end.

### View Not Updating

```python
code = '''
try:
    FreeCADGui.ActiveDocument.ActiveView.fitAll()
except:
    pass  # GUI not available in headless mode
'''
```

## Related Files

- `src/backend/simulation/brain_chip_designer.py` - ChipDesign and FreeCADExporter classes
- `.claude/skills/freecad-microfluidics.md` - FreeCAD skill documentation
- `.claude/skills/freecad-examples.md` - Code examples
- `docs/freecad-addons.csv` - Addon inventory
