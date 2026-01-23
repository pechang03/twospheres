# NetworkX to Fluigi Bridge

**Date**: 2026-01-23
**Type**: Tool / Infrastructure
**Status**: Complete

## Summary

Created `nx_bridge.py` module in pyfluigi that converts NetworkX graphs to MINT (Microfluidic Netlist) format. This enables any graph-based workflow to generate physical microfluidic layouts via Fluigi place-and-route.

## Location

`~/pyfluigi/fluigi/nx_bridge.py`

## Motivation

Graph structures are central to many analyses:
- Brain connectivity matrices → microfluidic chips
- Gene interaction networks → assay devices
- Pathway analysis → lab-on-chip designs
- Functor/category structures → physical topologies

NetworkX is the universal hub - all these convert to/from NetworkX easily. The missing piece was the bridge from NetworkX to physical fabrication.

## Architecture

```
Sources                    Hub              Outputs
────────────────────────────────────────────────────
Adjacency matrix   ──┐                  ┌──► MINT format
Edge list          ──┤                  │
Brain connectivity ──┼──► NetworkX ─────┼──► Fluigi P&R ──► Layout
Gene networks      ──┤      Graph       │
Functor edges      ──┘                  └──► 3DuF / FreeCAD
```

## API

### Core Functions

```python
from fluigi.nx_bridge import nx_to_mint, MintComponentType

# Convert graph to MINT
mint_code = nx_to_mint(G, device_name="my_chip")

# Parse MINT back to graph
G = mint_to_nx(mint_code)
```

### Convenience Constructors

```python
from fluigi.nx_bridge import from_adjacency_matrix, from_edge_list

# From connectivity matrix
G = from_adjacency_matrix(matrix, node_names, node_types)

# From edge list
G = from_edge_list(edges, node_types, edge_widths)
```

### Component Types

```python
class MintComponentType(Enum):
    PORT = "PORT"
    NODE = "NODE"
    CHANNEL = "CHANNEL"
    CHAMBER = "CHAMBER"
    MIXER = "MIXER"
    DROPLET_GENERATOR = "NOZZLE DROPLET GENERATOR"
    VALVE = "VALVE"
    FILTER = "FILTER"
    # ... more
```

## Usage Examples

### Simple Mixer

```python
import networkx as nx
from fluigi.nx_bridge import nx_to_mint, MintComponentType

G = nx.DiGraph()
G.add_node('inlet1', mint_type=MintComponentType.PORT)
G.add_node('inlet2', mint_type=MintComponentType.PORT)
G.add_node('mixer', mint_type=MintComponentType.MIXER, width=200)
G.add_node('outlet', mint_type=MintComponentType.PORT)

G.add_edge('inlet1', 'mixer', width=100)
G.add_edge('inlet2', 'mixer', width=100)
G.add_edge('mixer', 'outlet', width=150)

mint = nx_to_mint(G, device_name='test_chip')
```

Output:
```mint
DEVICE test_chip

LAYER FLOW

PORT inlet1 portRadius=500 ;
PORT inlet2 portRadius=500 ;
PORT outlet portRadius=500 ;

MIXER mixer width=200 ;

CHANNEL c0 from inlet1 to mixer 1 width=100 ;
CHANNEL c1 from inlet2 to mixer 1 width=100 ;
CHANNEL c2 from mixer 1 to outlet width=150 ;

END LAYER
```

### From Adjacency Matrix

```python
import numpy as np
from fluigi.nx_bridge import from_adjacency_matrix, nx_to_mint, MintComponentType

# Brain region connectivity (weights = channel widths)
matrix = np.array([
    [0, 100, 0, 0],
    [0, 0, 150, 80],
    [0, 0, 0, 120],
    [0, 0, 0, 0],
])

node_names = ['inlet', 'region_A', 'region_B', 'outlet']
node_types = [
    MintComponentType.PORT,
    MintComponentType.CHAMBER,
    MintComponentType.CHAMBER,
    MintComponentType.PORT,
]

G = from_adjacency_matrix(matrix, node_names, node_types)
mint = nx_to_mint(G, device_name='brain_network_chip')
```

## Integration Points

### With twosphere-mcp

```python
# In BrainChipDesigner
def export_to_mint(self, design: ChipDesign) -> str:
    G = design.to_networkx()  # Need to implement
    return nx_to_mint(G, device_name=design.network_type)
```

### With biosearch

```python
# Convert gene interaction network to microfluidic assay chip
from fluigi.nx_bridge import nx_to_mint, MintComponentType

def gene_network_to_chip(gene_graph):
    # Map genes to chambers, interactions to channels
    for node in gene_graph.nodes():
        gene_graph.nodes[node]['mint_type'] = MintComponentType.CHAMBER

    # Leaf nodes become ports
    for node in gene_graph.nodes():
        if gene_graph.degree(node) == 1:
            gene_graph.nodes[node]['mint_type'] = MintComponentType.PORT

    return nx_to_mint(gene_graph, device_name='gene_assay')
```

## Running Fluigi

After generating MINT:

```bash
# Compile MINT to layout
fluigi mint-compile chip.mint -o output/

# Render SVG preview
fluigi utils-render-svg output/chip.json
```

## Node Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `mint_type` | MintComponentType or str | Component type |
| `width` | float | Component width (µm) |
| `height` | float | Component height (µm) |
| `radius` | float | Chamber/port radius (µm) |
| `port_radius` | float | Port radius (µm) |

## Edge Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `width` | float | Channel width (µm), default 100 |
| `length` | float | Channel length (optional) |

## Related

- `~/pyfluigi/` - Fluigi place-and-route tool
- `~/3duf/` - 3DuF web CAD tool
- `src/backend/simulation/brain_chip_designer.py` - Brain chip designs
- `~/code/aider/biosearch/` - Biological network analysis

## Next Steps

1. Add `to_networkx()` method to ChipDesign class
2. Create MINT → 3DuF JSON converter
3. Add position constraint support for pre-placed nodes
4. Test with Fluigi full pipeline (P&R → render)
