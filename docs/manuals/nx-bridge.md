# NetworkX to Microfluidics Bridge Manual

## Overview

The `nx_bridge` module converts NetworkX graphs to MINT format for Fluigi place-and-route. This enables graph-based design workflows to generate physical microfluidic devices.

## Location

`~/pyfluigi/fluigi/nx_bridge.py`

## Quick Start

```python
import networkx as nx
from fluigi.nx_bridge import nx_to_mint, MintComponentType

# Create graph
G = nx.DiGraph()
G.add_node('inlet', mint_type=MintComponentType.PORT)
G.add_node('chamber', mint_type=MintComponentType.CHAMBER)
G.add_node('outlet', mint_type=MintComponentType.PORT)
G.add_edge('inlet', 'chamber', width=100)
G.add_edge('chamber', 'outlet', width=100)

# Convert to MINT
mint_code = nx_to_mint(G, device_name='my_device')
print(mint_code)
```

## API Reference

### nx_to_mint(G, ...)

Convert NetworkX graph to MINT format string.

```python
def nx_to_mint(
    G: nx.Graph,
    device_name: str = "device",
    layer_name: str = "FLOW",
    default_channel_width: float = 100.0,
    default_port_radius: float = 500.0,
    include_positions: bool = False,
) -> str
```

**Parameters:**
- `G` - NetworkX DiGraph or Graph
- `device_name` - Name for MINT device
- `layer_name` - Layer name ("FLOW" or "CONTROL")
- `default_channel_width` - Default channel width in µm
- `default_port_radius` - Default port radius in µm

**Returns:** MINT format string

### mint_to_nx(mint_code)

Parse MINT back to NetworkX graph.

```python
def mint_to_nx(mint_code: str) -> nx.DiGraph
```

### from_adjacency_matrix(matrix, ...)

Create graph from connectivity matrix.

```python
def from_adjacency_matrix(
    matrix,                                    # 2D array
    node_names: Optional[List[str]] = None,    # Node labels
    node_types: Optional[List[MintComponentType]] = None,
    threshold: float = 0.0,                    # Edge threshold
) -> nx.DiGraph
```

**Example:**
```python
import numpy as np
from fluigi.nx_bridge import from_adjacency_matrix, MintComponentType

matrix = np.array([
    [0, 100, 0],
    [0, 0, 150],
    [0, 0, 0],
])

G = from_adjacency_matrix(
    matrix,
    node_names=['inlet', 'chamber', 'outlet'],
    node_types=[MintComponentType.PORT, MintComponentType.CHAMBER, MintComponentType.PORT]
)
```

### from_edge_list(edges, ...)

Create graph from edge list.

```python
def from_edge_list(
    edges: List[Tuple[str, str]],
    node_types: Optional[Dict[str, MintComponentType]] = None,
    edge_widths: Optional[Dict[Tuple[str, str], float]] = None,
) -> nx.DiGraph
```

## Component Types

```python
class MintComponentType(Enum):
    # Basic
    PORT = "PORT"
    NODE = "NODE"
    CHANNEL = "CHANNEL"

    # Chambers
    CHAMBER = "CHAMBER"
    CIRCULAR_CHAMBER = "CIRCULAR CHAMBER"
    RECTANGULAR_CHAMBER = "RECTANGULAR CHAMBER"

    # Mixers
    MIXER = "MIXER"
    GRADIENT_GENERATOR = "GRADIENT GENERATOR"
    SERPENTINE_MIXER = "SERPENTINE MIXER"
    HERRINGBONE_MIXER = "HERRINGBONE MIXER"

    # Droplet generators
    DROPLET_GENERATOR = "NOZZLE DROPLET GENERATOR"
    T_JUNCTION = "T JUNCTION"
    FLOW_FOCUS = "FLOW FOCUS"

    # Valves
    VALVE = "VALVE"
    MEMBRANE_VALVE = "MEMBRANE VALVE"

    # Filters
    FILTER = "FILTER"
    PILLAR_ARRAY = "PILLAR ARRAY"
```

## Node Attributes

Set on NetworkX nodes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `mint_type` | MintComponentType or str | Component type |
| `width` | float | Width in µm |
| `height` | float | Height in µm |
| `radius` | float | Radius in µm |
| `port_radius` | float | Port radius in µm |
| `num_turns` | int | For serpentine mixers |
| `oil_channel_width` | float | For droplet generators |
| `water_channel_width` | float | For droplet generators |

**Example:**
```python
G.add_node('mixer1',
    mint_type=MintComponentType.SERPENTINE_MIXER,
    width=200,
    num_turns=10
)
```

## Edge Attributes

Set on NetworkX edges:

| Attribute | Type | Description |
|-----------|------|-------------|
| `width` | float | Channel width in µm (default 100) |
| `length` | float | Channel length in µm (optional) |

**Example:**
```python
G.add_edge('inlet', 'mixer', width=150)
```

## Auto-Detection

If `mint_type` is not set, the module infers:

1. Nodes with degree 1 → PORT
2. Nodes with degree > 2 → NODE (junction)
3. Node names containing "inlet"/"outlet" → PORT
4. Node names containing "chamber" → CHAMBER
5. Node names containing "mixer" → MIXER

## Complete Example

```python
import networkx as nx
import numpy as np
from fluigi.nx_bridge import (
    nx_to_mint,
    from_adjacency_matrix,
    MintComponentType
)

# From brain connectivity matrix
conn_matrix = np.array([
    [0, 0.8, 0.3, 0],
    [0, 0, 0.6, 0.4],
    [0, 0, 0, 0.7],
    [0, 0, 0, 0],
])

regions = ['CSF_inlet', 'cortex', 'hippocampus', 'drain']
types = [
    MintComponentType.PORT,
    MintComponentType.CHAMBER,
    MintComponentType.CHAMBER,
    MintComponentType.PORT,
]

# Scale weights to channel widths (µm)
conn_matrix = conn_matrix * 200  # 0-200 µm range

G = from_adjacency_matrix(conn_matrix, regions, types, threshold=50)

# Add component parameters
G.nodes['cortex']['width'] = 2000  # 2mm chamber
G.nodes['hippocampus']['width'] = 1500

# Generate MINT
mint = nx_to_mint(G, device_name='brain_region_chip')

# Save
with open('brain_chip.mint', 'w') as f:
    f.write(mint)
```

## Integration with biosearch

Convert gene networks to microfluidic assays:

```python
# Assume gene_network is a NetworkX graph from biosearch
from fluigi.nx_bridge import nx_to_mint, MintComponentType

# Map genes to chambers
for gene in gene_network.nodes():
    gene_network.nodes[gene]['mint_type'] = MintComponentType.CHAMBER
    gene_network.nodes[gene]['width'] = 1000  # 1mm chambers

# Add inlet/outlet
gene_network.add_node('sample_inlet', mint_type=MintComponentType.PORT)
gene_network.add_node('waste_outlet', mint_type=MintComponentType.PORT)

# Connect to network
first_gene = list(gene_network.nodes())[0]
gene_network.add_edge('sample_inlet', first_gene, width=100)

# Generate device
mint = nx_to_mint(gene_network, device_name='gene_assay_chip')
```

## Troubleshooting

### Invalid identifier names

Node names are sanitized automatically:
- Special characters → underscore
- Names starting with number → prefixed with "n_"

### Missing connections

Ensure edges exist in the graph:
```python
print(list(G.edges()))  # Check edges
```

### Wrong component types

Explicitly set `mint_type` attribute:
```python
G.nodes['my_node']['mint_type'] = MintComponentType.MIXER
```

## Related Files

- `~/pyfluigi/fluigi/nx_bridge.py` - Source code
- `docs/manuals/fluigi-quickstart.md` - Fluigi manual
- `docs/beads/2026-01-23-nx-bridge-fluigi.md` - Development notes
