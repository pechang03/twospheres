# Fluigi Quickstart Manual

## Overview

Fluigi is a microfluidic design automation tool from CIDAR Lab. It takes MINT (Microfluidic Netlist) descriptions and performs place-and-route to generate physical layouts.

## Installation

Located at: `~/pyfluigi/`

```bash
cd ~/pyfluigi
poetry install
poetry shell
fluigi --help
```

Or use conda environment:
```bash
conda activate twosphere
fluigi --help
```

## Commands

### 1. convert-to-parchmint

Convert MINT to Parchmint JSON format.

```bash
fluigi convert-to-parchmint input.mint -o output/
```

Generates:
- `output/input.json` - Parchmint device description
- `output/input.dot` - GraphViz graph
- `output/input.pdf` - Visualization

### 2. mint-compile

Full place-and-route pipeline (requires primitives server).

```bash
fluigi mint-compile input.mint -o output/ --render-svg true
```

Options:
- `--route BOOLEAN` - Only perform routing
- `--render-svg BOOLEAN` - Generate SVG previews
- `--ignore-layout-constraints BOOLEAN` - Skip constraints

### 3. lfr-compile

Compile LFR (Living Fabric Runtime) files.

```bash
fluigi lfr-compile input.lfr -o output/ --technology standard
```

### 4. synthesize

Full synthesis pipeline from high-level description.

```bash
fluigi synthesize input.lfr -o output/ --technology standard
```

### 5. utils-render-svg

Render SVG from Parchmint JSON.

```bash
fluigi utils-render-svg output/device.json
```

## MINT Format

Basic structure:
```mint
DEVICE my_device

LAYER FLOW

PORT inlet portRadius=500 ;
PORT outlet portRadius=500 ;
MIXER mixer1 width=200 ;
NODE junction1, junction2 ;

CHANNEL c1 from inlet to mixer1 1 width=100 ;
CHANNEL c2 from mixer1 1 to outlet width=100 ;

END LAYER
```

### Component Types

| Type | Description | Parameters |
|------|-------------|------------|
| PORT | I/O port | portRadius |
| NODE | Junction node | - |
| CHANNEL | Flow channel | width, length |
| MIXER | Generic mixer | width |
| CHAMBER | Reaction chamber | width, height |
| VALVE | Control valve | width |
| NOZZLE DROPLET GENERATOR | Droplet generator | oilChannelWidth, waterChannelWidth |

## Primitives Server

For full functionality, run the 3DuF primitives server:

```bash
cd ~/3duf
git checkout primitives-server
docker build -f primitives-server.Dockerfile -t primitives-server:latest .
docker run -p 6060:6060 primitives-server
```

The server provides:
- Default component parameters
- Component dimensions
- Terminal positions

## NetworkX Integration

Use `nx_bridge.py` to convert NetworkX graphs to MINT:

```python
from fluigi.nx_bridge import nx_to_mint, MintComponentType
import networkx as nx

G = nx.DiGraph()
G.add_node('inlet', mint_type=MintComponentType.PORT)
G.add_node('mixer', mint_type=MintComponentType.MIXER)
G.add_node('outlet', mint_type=MintComponentType.PORT)
G.add_edge('inlet', 'mixer', width=100)
G.add_edge('mixer', 'outlet', width=100)

mint = nx_to_mint(G, device_name='my_chip')

# Save to file
with open('my_chip.mint', 'w') as f:
    f.write(mint)
```

Then compile:
```bash
fluigi convert-to-parchmint my_chip.mint -o output/
```

## Workflow

```
NetworkX Graph
     │
     ▼
nx_to_mint()
     │
     ▼
MINT file (.mint)
     │
     ▼
fluigi convert-to-parchmint
     │
     ▼
Parchmint JSON + DOT + PDF
     │
     ▼
fluigi mint-compile (with primitives server)
     │
     ▼
Physical Layout
     │
     ▼
3DuF / FreeCAD / Fabrication
```

## Troubleshooting

### ImportError with _place_and_route.so

The P&R module has a compiled binary. Rebuild for your architecture:
```bash
cd ~/pyfluigi
poetry install
# May need to rebuild Cython extensions
```

### Connection refused to localhost:6060

Start the primitives server or use `convert-to-parchmint` which works without it.

### ANTLR version mismatch

Warning only - usually works despite mismatch.

## Related

- `~/pyfluigi/` - Fluigi source
- `~/3duf/` - 3DuF web CAD
- `fluigi/nx_bridge.py` - NetworkX bridge
- MINT language spec: https://github.com/CIDARLAB/mint-lang
