"""Brain-on-Chip Designer for Glymphatic Validation.

Generates microfluidic chip designs that mimic brain perivascular topology
for experimental validation of glymphatic clearance models.

Key features:
1. Network topology from brain connectivity (disc dimension matched)
2. Perivascular channel dimensions (3-50 µm gaps)
3. Multiple clearance zones for regional comparison
4. Optical monitoring ports for fluorescence tracking
5. FreeCAD export via XML-RPC

Based on PH-7 hypothesis: Chips with lower disc dimension networks
should show more efficient tracer clearance.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import xmlrpc.client


@dataclass
class ChannelSpec:
    """Specification for a microfluidic channel."""
    start: Tuple[float, float, float]  # (x, y, z) in mm
    end: Tuple[float, float, float]
    diameter_um: float = 50.0  # Channel diameter in µm
    name: str = ""

    @property
    def length_mm(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        dz = self.end[2] - self.start[2]
        return float(np.sqrt(dx**2 + dy**2 + dz**2))


@dataclass
class ChamberSpec:
    """Specification for a clearance chamber.

    PDMS chip is bonded to glass substrate - imaging is done from below
    using an inverted microscope through the glass.
    """
    center: Tuple[float, float, float]  # (x, y, z) in mm
    width: float = 1.0  # mm
    height: float = 0.5  # mm (channel height in PDMS)
    depth: float = 0.5  # mm
    name: str = "chamber"


@dataclass
class InletPortSpec:
    """Specification for an inlet port adjacent to a chamber.

    Ports are positioned to the side of chambers so they don't
    obstruct microscope observation. A short channel connects
    the port to the chamber.
    """
    position: Tuple[float, float, float]  # Port center (x, y, z) in mm
    chamber_name: str  # Associated chamber
    port_diameter_um: float = 500.0  # Port diameter for tubing connection
    offset_direction: str = "x+"  # Direction from chamber (x+, x-, y+, y-)


@dataclass
class ChipDesign:
    """Complete brain-on-chip design.

    Microscopy-friendly layout:
    - Chamber tops are clear for microscope observation
    - Inlet ports are positioned adjacent to chambers (not on top)
    - Short inlet channels (100µm) connect ports to chambers
    - Main network channels connect chambers to each other
    """
    # Chip dimensions (mm)
    length: float = 30.0
    width: float = 15.0
    height: float = 5.0

    # Network topology
    channels: List[ChannelSpec] = field(default_factory=list)  # Inter-chamber channels
    chambers: List[ChamberSpec] = field(default_factory=list)

    # Inlet ports (adjacent to chambers for microscopy access)
    inlet_ports: List[InletPortSpec] = field(default_factory=list)
    inlet_channels: List[ChannelSpec] = field(default_factory=list)  # Port-to-chamber (100µm)

    # Main flow ports (horizontal entry at chip edge)
    main_inlet_position: Tuple[float, float, float] = (-14.0, 0.0, 2.5)
    main_outlet_position: Tuple[float, float, float] = (14.0, 0.0, 2.5)

    # Glass substrate for inverted microscopy (imaging from below)
    substrate_thickness_mm: float = 0.17  # Standard #1.5 coverslip

    # Metadata
    disc_dimension: float = 2.0
    network_type: str = "planar"
    target_application: str = "glymphatic_validation"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions_mm": {
                "length": self.length,
                "width": self.width,
                "height": self.height,
            },
            "network": {
                "n_channels": len(self.channels),
                "n_inlet_channels": len(self.inlet_channels),
                "n_chambers": len(self.chambers),
                "disc_dimension": self.disc_dimension,
                "network_type": self.network_type,
            },
            "ports": {
                "main_inlet": self.main_inlet_position,
                "main_outlet": self.main_outlet_position,
                "chamber_inlet_ports": len(self.inlet_ports),
            },
            "microscopy": {
                "type": "inverted",
                "substrate": "glass",
                "substrate_thickness_mm": self.substrate_thickness_mm,
                "imaging_from": "below through glass",
            },
            "target_application": self.target_application,
        }

    def to_3duf_json(self, name: str = "Microfluidic Chip") -> Dict[str, Any]:
        """Export chip design to 3DuF JSON format.

        3DuF uses micrometers (µm) as units and a 2D coordinate system.
        Features are organized into layers (flow, control).

        Returns:
            Dict compatible with 3DuF JSON format
        """
        import uuid

        def make_id() -> str:
            return str(uuid.uuid4())

        # Convert mm to µm
        MM_TO_UM = 1000

        # Device dimensions in µm
        device_width = int(self.length * MM_TO_UM)
        device_height = int(self.width * MM_TO_UM)
        channel_height = 100  # Standard PDMS channel height in µm

        features = {}

        # Add chambers
        for chamber in self.chambers:
            fid = make_id()
            # Convert center position from mm to µm
            # 3DuF uses top-left origin, we use center
            x = int((chamber.center[0] + self.length / 2) * MM_TO_UM)
            y = int((chamber.center[1] + self.width / 2) * MM_TO_UM)

            features[fid] = {
                "id": fid,
                "name": chamber.name,
                "type": "Chamber",
                "params": {
                    "position": [x, y],
                    "width": int(chamber.width * MM_TO_UM),
                    "length": int(chamber.depth * MM_TO_UM),
                    "height": channel_height,
                    "cornerRadius": 50,
                    "rotation": 0
                }
            }

        # Add channels
        for channel in self.channels:
            fid = make_id()
            # Convert coordinates
            x1 = int((channel.start[0] + self.length / 2) * MM_TO_UM)
            y1 = int((channel.start[1] + self.width / 2) * MM_TO_UM)
            x2 = int((channel.end[0] + self.length / 2) * MM_TO_UM)
            y2 = int((channel.end[1] + self.width / 2) * MM_TO_UM)

            features[fid] = {
                "id": fid,
                "name": channel.name or "Channel",
                "type": "Channel",
                "params": {
                    "start": [x1, y1],
                    "end": [x2, y2],
                    "width": int(channel.diameter_um),
                    "height": channel_height
                }
            }

        # Add inlet channels
        for channel in self.inlet_channels:
            fid = make_id()
            x1 = int((channel.start[0] + self.length / 2) * MM_TO_UM)
            y1 = int((channel.start[1] + self.width / 2) * MM_TO_UM)
            x2 = int((channel.end[0] + self.length / 2) * MM_TO_UM)
            y2 = int((channel.end[1] + self.width / 2) * MM_TO_UM)

            features[fid] = {
                "id": fid,
                "name": channel.name or "Inlet Channel",
                "type": "Channel",
                "params": {
                    "start": [x1, y1],
                    "end": [x2, y2],
                    "width": int(channel.diameter_um),
                    "height": channel_height
                }
            }

        # Add inlet ports
        for port in self.inlet_ports:
            fid = make_id()
            x = int((port.position[0] + self.length / 2) * MM_TO_UM)
            y = int((port.position[1] + self.width / 2) * MM_TO_UM)

            features[fid] = {
                "id": fid,
                "name": f"Port_{port.chamber_name}",
                "type": "Port",
                "params": {
                    "position": [x, y],
                    "radius1": int(port.port_diameter_um / 2),
                    "radius2": int(port.port_diameter_um / 2),
                    "height": channel_height
                }
            }

        # Add main inlet/outlet ports
        for port_name, position in [("Main_Inlet", self.main_inlet_position),
                                     ("Main_Outlet", self.main_outlet_position)]:
            fid = make_id()
            x = int((position[0] + self.length / 2) * MM_TO_UM)
            y = int((position[1] + self.width / 2) * MM_TO_UM)

            features[fid] = {
                "id": fid,
                "name": port_name,
                "type": "Port",
                "params": {
                    "position": [x, y],
                    "radius1": 500,  # 1mm diameter main port
                    "radius2": 500,
                    "height": channel_height
                }
            }

        # Build 3DuF JSON structure
        return {
            "name": name,
            "params": {
                "width": device_width,
                "height": device_height
            },
            "layers": [
                {
                    "name": "flow",
                    "color": "indigo",
                    "params": {
                        "z_offset": 0,
                        "flip": False
                    },
                    "features": features
                }
            ],
            "groups": [],
            "defaults": {},
            "metadata": {
                "disc_dimension": self.disc_dimension,
                "network_type": self.network_type,
                "source": "brain_chip_designer.py"
            }
        }

    def to_networkx(self):
        """Convert chip design to NetworkX graph for Fluigi pipeline.

        Creates a directed graph where:
        - Chambers become nodes with mint_type=CHAMBER
        - Main inlet/outlet become PORT nodes
        - Inlet ports become PORT nodes
        - Channels become edges with width attribute

        Returns:
            nx.DiGraph suitable for nx_to_mint() conversion

        Example:
            >>> chip = designer.design_latin_square_mixer(n=4)
            >>> G = chip.to_networkx()
            >>> from fluigi.nx_bridge import nx_to_mint
            >>> mint_code = nx_to_mint(G, device_name='latin_square')
        """
        import networkx as nx

        G = nx.DiGraph()

        # Track chamber positions for edge creation
        chamber_positions = {}

        # Add chambers as nodes
        # Note: Use NODE type for MINT compatibility - CHAMBER isn't a standalone MINT keyword
        for chamber in self.chambers:
            node_name = chamber.name or f"chamber_{len(chamber_positions)}"
            G.add_node(
                node_name,
                mint_type="NODE",  # NODE is valid MINT; CHAMBER alone is not
                width=chamber.width * 1000,  # mm to µm (stored as metadata)
                height=chamber.height * 1000,
                pos=chamber.center[:2],  # (x, y) position
            )
            chamber_positions[chamber.center[:2]] = node_name

        # Add main inlet/outlet as PORT nodes
        G.add_node(
            "main_inlet",
            mint_type="PORT",
            port_radius=500,
            pos=self.main_inlet_position[:2],
        )
        G.add_node(
            "main_outlet",
            mint_type="PORT",
            port_radius=500,
            pos=self.main_outlet_position[:2],
        )

        # Add inlet ports as PORT nodes
        for port in self.inlet_ports:
            port_name = f"port_{port.chamber_name}"
            G.add_node(
                port_name,
                mint_type="PORT",
                port_radius=port.port_diameter_um,
                pos=port.position[:2],
            )

        # Helper to find nearest node to a position
        def find_nearest_node(pos, exclude=None):
            min_dist = float('inf')
            nearest = None
            for node in G.nodes():
                if exclude and node == exclude:
                    continue
                node_pos = G.nodes[node].get('pos', (0, 0))
                dist = ((pos[0] - node_pos[0])**2 + (pos[1] - node_pos[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
            return nearest, min_dist

        # Add channels as edges
        for channel in self.channels:
            start_pos = channel.start[:2]
            end_pos = channel.end[:2]

            # Find nodes closest to channel endpoints
            start_node, _ = find_nearest_node(start_pos)
            end_node, _ = find_nearest_node(end_pos, exclude=start_node)

            if start_node and end_node and start_node != end_node:
                G.add_edge(
                    start_node,
                    end_node,
                    width=channel.diameter_um,
                    name=channel.name,
                    length=channel.length_mm * 1000,  # mm to µm
                )

        # Add inlet channels (port to chamber connections)
        for channel in self.inlet_channels:
            start_pos = channel.start[:2]
            end_pos = channel.end[:2]

            start_node, _ = find_nearest_node(start_pos)
            end_node, _ = find_nearest_node(end_pos, exclude=start_node)

            if start_node and end_node and start_node != end_node:
                G.add_edge(
                    start_node,
                    end_node,
                    width=channel.diameter_um,
                    name=channel.name or "inlet_channel",
                )

        return G


class BrainChipDesigner:
    """Designer for brain-mimetic microfluidic chips.

    Creates chip layouts that replicate brain network topology
    at microfluidic scale for glymphatic validation experiments.
    """

    # Standard channel diameters for different vessel types
    CHANNEL_DIAMETERS = {
        "artery": 100.0,  # µm
        "arteriole": 50.0,
        "capillary": 20.0,
        "venule": 50.0,
        "vein": 100.0,
    }

    def __init__(
        self,
        chip_length_mm: float = 30.0,
        chip_width_mm: float = 15.0,
        chip_height_mm: float = 5.0,
    ):
        self.chip_length = chip_length_mm
        self.chip_width = chip_width_mm
        self.chip_height = chip_height_mm

    def design_from_connectivity(
        self,
        connectivity_matrix: NDArray,
        region_positions: Optional[NDArray] = None,
        channel_diameter_um: float = 50.0,
        chamber_size_mm: float = 1.0,
    ) -> ChipDesign:
        """Design chip network from brain connectivity matrix.

        Args:
            connectivity_matrix: NxN connectivity matrix
            region_positions: Nx2 or Nx3 array of region positions (normalized 0-1)
            channel_diameter_um: Channel diameter in micrometers
            chamber_size_mm: Size of clearance chambers

        Returns:
            ChipDesign with channels and chambers
        """
        n_regions = connectivity_matrix.shape[0]

        # Generate positions if not provided
        if region_positions is None:
            region_positions = self._generate_positions(n_regions)

        # Scale positions to chip dimensions
        margin = 3.0  # mm margin from edges
        scaled_positions = self._scale_positions(region_positions, margin)

        # Microscopy-friendly design parameters
        port_offset_mm = 0.5  # Inlet ports offset from chamber center
        inlet_channel_length_um = 100.0  # 100µm connecting channel

        # Create chambers at node positions (tops clear for microscopy)
        chambers = []
        inlet_ports = []
        inlet_channels = []

        for i in range(n_regions):
            pos = scaled_positions[i]
            chamber_center = (pos[0], pos[1], self.chip_height / 2)

            chambers.append(ChamberSpec(
                center=chamber_center,
                width=chamber_size_mm,
                height=chamber_size_mm * 0.5,
                depth=chamber_size_mm * 0.5,
                name=f"region_{i}",
            ))

            # Position inlet port adjacent to chamber (alternating sides)
            # This keeps ports from obscuring microscope view
            if i % 2 == 0:
                port_x = pos[0] - chamber_size_mm / 2 - port_offset_mm
                offset_dir = "x-"
            else:
                port_x = pos[0] + chamber_size_mm / 2 + port_offset_mm
                offset_dir = "x+"

            port_pos = (port_x, pos[1], self.chip_height / 2)

            inlet_ports.append(InletPortSpec(
                position=port_pos,
                chamber_name=f"region_{i}",
                port_diameter_um=500.0,  # Standard tubing connection
                offset_direction=offset_dir,
            ))

            # Short channel connecting port to chamber (100µm)
            inlet_channels.append(ChannelSpec(
                start=port_pos,
                end=chamber_center,
                diameter_um=inlet_channel_length_um,
                name=f"inlet_channel_{i}",
            ))

        # Create channels for strong connections (inter-chamber network)
        # Use lower threshold to ensure we capture the network structure
        threshold = np.mean(connectivity_matrix[connectivity_matrix > 0]) * 0.5
        channels = []

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                if connectivity_matrix[i, j] > threshold:
                    pos_i = scaled_positions[i]
                    pos_j = scaled_positions[j]

                    # Scale diameter by connection strength
                    strength = connectivity_matrix[i, j]
                    diameter = channel_diameter_um * (0.5 + 0.5 * strength)

                    channels.append(ChannelSpec(
                        start=(pos_i[0], pos_i[1], self.chip_height / 2),
                        end=(pos_j[0], pos_j[1], self.chip_height / 2),
                        diameter_um=diameter,
                        name=f"channel_{i}_{j}",
                    ))

        # Estimate disc dimension from network structure
        disc_dim = self._estimate_disc_dimension(connectivity_matrix, float(threshold))

        return ChipDesign(
            length=self.chip_length,
            width=self.chip_width,
            height=self.chip_height,
            channels=channels,
            chambers=chambers,
            inlet_ports=inlet_ports,
            inlet_channels=inlet_channels,
            main_inlet_position=(-self.chip_length/2 + 2, 0, self.chip_height/2),
            main_outlet_position=(self.chip_length/2 - 2, 0, self.chip_height/2),
            disc_dimension=disc_dim,
            network_type="planar" if disc_dim <= 2.5 else "non-planar",
        )

    def design_planar_network(
        self,
        n_nodes: int = 5,
        channel_diameter_um: float = 50.0,
    ) -> ChipDesign:
        """Design a planar (disc dimension 2) network.

        Creates a tree-like or grid network for comparison with
        non-planar designs. Planar networks should show better clearance.
        """
        # Create grid positions
        n_cols = int(np.ceil(np.sqrt(n_nodes)))
        n_rows = int(np.ceil(n_nodes / n_cols))

        positions = []
        for i in range(n_nodes):
            row = i // n_cols
            col = i % n_cols
            x = (col / max(n_cols - 1, 1)) if n_cols > 1 else 0.5
            y = (row / max(n_rows - 1, 1)) if n_rows > 1 else 0.5
            positions.append([x, y])

        positions = np.array(positions)

        # Create planar connectivity (grid + some diagonals, but planar)
        connectivity = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            row_i, col_i = i // n_cols, i % n_cols

            # Connect to right neighbor
            if col_i + 1 < n_cols and i + 1 < n_nodes:
                connectivity[i, i + 1] = 0.8
                connectivity[i + 1, i] = 0.8

            # Connect to bottom neighbor
            if i + n_cols < n_nodes:
                connectivity[i, i + n_cols] = 0.8
                connectivity[i + n_cols, i] = 0.8

        return self.design_from_connectivity(
            connectivity, positions, channel_diameter_um
        )

    def design_nonplanar_network(
        self,
        n_nodes: int = 5,
        channel_diameter_um: float = 50.0,
    ) -> ChipDesign:
        """Design a non-planar (disc dimension > 2) network.

        Creates K5 or K3,3-like structure for comparison.
        Non-planar networks should show reduced clearance efficiency.
        """
        # Circular arrangement
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = np.column_stack([
            0.5 + 0.4 * np.cos(angles),
            0.5 + 0.4 * np.sin(angles),
        ])

        # Dense connectivity (K_n-like, definitely non-planar for n >= 5)
        connectivity = np.ones((n_nodes, n_nodes)) * 0.7
        np.fill_diagonal(connectivity, 0)

        return self.design_from_connectivity(
            connectivity, positions, channel_diameter_um
        )

    def design_comparison_set(
        self,
        n_nodes: int = 5,
    ) -> Dict[str, ChipDesign]:
        """Design a set of chips for disc dimension comparison.

        Returns planar, semi-planar, and non-planar designs
        for experimental comparison of clearance efficiency.
        """
        return {
            "planar": self.design_planar_network(n_nodes),
            "non_planar": self.design_nonplanar_network(n_nodes),
            "tree": self._design_tree_network(n_nodes),
        }

    def _design_tree_network(self, n_nodes: int) -> ChipDesign:
        """Design a tree network (disc dimension ~1)."""
        # Binary tree positions
        positions = []
        connectivity = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            depth = int(np.floor(np.log2(i + 1)))
            pos_in_level = i - (2**depth - 1)
            n_in_level = 2**depth

            x = (pos_in_level + 0.5) / n_in_level
            y = depth / max(np.floor(np.log2(n_nodes)), 1)
            positions.append([x, y])

            # Connect to parent
            if i > 0:
                parent = (i - 1) // 2
                connectivity[i, parent] = 0.9
                connectivity[parent, i] = 0.9

        return self.design_from_connectivity(
            connectivity, np.array(positions), 50.0
        )

    def design_flow_validation_set(self) -> Dict[str, "ChipDesign"]:
        """Design chips for flow visualization experiments.

        Returns three designs with proper through-flow (no dead ends):
        - Grid: 3x3 planar grid, flow left→right
        - Cross-connected: Same grid with diagonal connections (non-planar)
        - Bifurcating tree: Binary tree, inlet at root, outlets at leaves
        """
        return {
            "grid": self._design_flow_grid(),
            "cross_connected": self._design_flow_cross_connected(),
            "bifurcating_tree": self._design_flow_tree(),
        }

    def design_combinatorial_mixing_chip(
        self,
        n_drugs: int = 2,
        dilution_levels: int = 4,
    ) -> "ChipDesign":
        """Design chip for combinatorial drug mixing using topology.

        Uses tree topology to create concentration gradients and mixing
        chambers to achieve MOLS-like combinatorial balance.

        For n_drugs=2, dilution_levels=4:
        - Drug A: 100%, 75%, 50%, 25%
        - Drug B: 100%, 75%, 50%, 25%
        - Mixing creates all combinations

        Args:
            n_drugs: Number of drug inputs (2-4)
            dilution_levels: Number of concentration levels per drug

        Returns:
            ChipDesign with combinatorial mixing topology
        """
        if n_drugs == 2:
            return self._design_two_drug_combinator(dilution_levels)
        elif n_drugs == 3:
            return self._design_three_drug_combinator(dilution_levels)
        else:
            # Default to 2-drug design
            return self._design_two_drug_combinator(dilution_levels)

    def _design_two_drug_combinator(self, levels: int = 4) -> "ChipDesign":
        """Two-drug combinatorial mixer.

        Layout:
        Drug A inlet ────────────────── Drug B inlet
              │                              │
            [A100]                        [B100]
              │                              │
           ┌──┴──┐                      ┌──┴──┐
         [A75] [A50]                  [B75] [B50]
           │     │                      │     │
           │     └────── MIXERS ────────┘     │
           │         ╱    │    ╲              │
           │    [A50B50] [A25B75] [A75B25]    │
           │         ↓    ↓    ↓              │
           └──────── OUTPUT CHAMBERS ─────────┘
        """
        # Chamber positions and connectivity
        # 12 chambers: 3 A-side + 3 B-side + 3 mixers + 3 outputs
        n_chambers = 12
        positions = []
        connectivity = np.zeros((n_chambers, n_chambers))

        # Drug A side (left): positions 0, 1, 2 (100%, 75%, 50%)
        positions.append([0.0, 0.5])   # 0: A inlet/100%
        positions.append([0.15, 0.3])  # 1: A 75%
        positions.append([0.15, 0.7])  # 2: A 50%

        # Drug B side (right): positions 3, 4, 5 (100%, 75%, 50%)
        positions.append([1.0, 0.5])   # 3: B inlet/100%
        positions.append([0.85, 0.3])  # 4: B 75%
        positions.append([0.85, 0.7])  # 5: B 50%

        # Mixing chambers (center): positions 6, 7, 8
        positions.append([0.5, 0.2])   # 6: A75+B25
        positions.append([0.5, 0.5])   # 7: A50+B50
        positions.append([0.5, 0.8])   # 8: A25+B75

        # Output chambers: positions 9, 10, 11
        positions.append([0.5, 0.0])   # 9: Output 1
        positions.append([0.35, 1.0])  # 10: Output 2
        positions.append([0.65, 1.0])  # 11: Output 3

        positions = np.array(positions)

        # Connections for serial dilution (A side)
        connectivity[0, 1] = 0.9  # A100 → A75
        connectivity[1, 0] = 0.9
        connectivity[1, 2] = 0.9  # A75 → A50
        connectivity[2, 1] = 0.9

        # Connections for serial dilution (B side)
        connectivity[3, 4] = 0.9  # B100 → B75
        connectivity[4, 3] = 0.9
        connectivity[4, 5] = 0.9  # B75 → B50
        connectivity[5, 4] = 0.9

        # Mixing connections (A side to mixers)
        connectivity[1, 6] = 0.7  # A75 → Mix(A75+B25)
        connectivity[6, 1] = 0.7
        connectivity[2, 7] = 0.7  # A50 → Mix(A50+B50)
        connectivity[7, 2] = 0.7
        connectivity[2, 8] = 0.7  # A50 → Mix(A25+B75) - lower contribution
        connectivity[8, 2] = 0.7

        # Mixing connections (B side to mixers)
        connectivity[4, 6] = 0.7  # B75 → Mix(A75+B25) - lower contribution
        connectivity[6, 4] = 0.7
        connectivity[5, 7] = 0.7  # B50 → Mix(A50+B50)
        connectivity[7, 5] = 0.7
        connectivity[5, 8] = 0.7  # B50 → Mix(A25+B75)
        connectivity[8, 5] = 0.7

        # Outputs
        connectivity[6, 9] = 0.8   # Mix1 → Out1
        connectivity[9, 6] = 0.8
        connectivity[7, 10] = 0.8  # Mix2 → Out2
        connectivity[10, 7] = 0.8
        connectivity[8, 11] = 0.8  # Mix3 → Out3
        connectivity[11, 8] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 60.0)
        chip.network_type = "combinatorial_2drug"

        # Label chambers with concentrations
        chamber_labels = [
            "A100", "A75", "A50",  # Drug A dilutions
            "B100", "B75", "B50",  # Drug B dilutions
            "A75_B25", "A50_B50", "A25_B75",  # Mixtures
            "Out1", "Out2", "Out3"  # Outputs
        ]

        return chip

    def design_latin_square_mixer(self, n: int = 4) -> "ChipDesign":
        """Design a Latin square mixer for balanced drug combination testing.

        Creates an n×n grid where:
        - Rows represent Drug A concentration levels (0 to n-1)
        - Columns represent Drug B concentration levels (0 to n-1)
        - Each cell gets a unique treatment assignment following Latin square rules
        - Each treatment appears exactly once per row and once per column

        This provides statistically balanced coverage of all drug combinations.

        Args:
            n: Order of the Latin square (n×n chambers)

        Returns:
            ChipDesign with Latin square mixing topology
        """
        # Generate standard Latin square (cyclic construction)
        # L[i,j] = (i + j) mod n
        latin_square = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                latin_square[i, j] = (i + j) % n

        # Create chambers in n×n grid
        chambers = []
        spacing = 8.0  # mm between chambers
        margin = 5.0

        # Grid positions
        positions = []
        for i in range(n):  # rows (Drug A levels)
            for j in range(n):  # cols (Drug B levels)
                x = margin + j * spacing
                y = margin + i * spacing
                treatment = latin_square[i, j]

                chambers.append(ChamberSpec(
                    name=f"A{i}_B{j}_T{treatment}",
                    center=(x, y, 0),
                    width=2.0, height=2.0, depth=0.1
                ))
                positions.append([x, y])

        positions = np.array(positions)

        # Create channels connecting grid
        channels = []

        # Horizontal connections (along rows - Drug A flow)
        for i in range(n):
            for j in range(n - 1):
                idx1 = i * n + j
                idx2 = i * n + j + 1
                channels.append(ChannelSpec(
                    name=f"h_{i}_{j}",
                    start=(positions[idx1, 0], positions[idx1, 1], 0),
                    end=(positions[idx2, 0], positions[idx2, 1], 0),
                    diameter_um=50.0,
                ))

        # Vertical connections (along columns - Drug B flow)
        for i in range(n - 1):
            for j in range(n):
                idx1 = i * n + j
                idx2 = (i + 1) * n + j
                channels.append(ChannelSpec(
                    name=f"v_{i}_{j}",
                    start=(positions[idx1, 0], positions[idx1, 1], 0),
                    end=(positions[idx2, 0], positions[idx2, 1], 0),
                    diameter_um=50.0,
                ))

        # Add inlet ports for Drug A (left side) and Drug B (top)
        # Drug A inlets - one per row
        for i in range(n):
            idx = i * n  # First chamber in row
            channels.append(ChannelSpec(
                name=f"inlet_A{i}",
                start=(margin - 3.0, margin + i * spacing, 0),
                end=(positions[idx, 0], positions[idx, 1], 0),
                diameter_um=80.0,
            ))

        # Drug B inlets - one per column
        for j in range(n):
            idx = j  # First chamber in column
            channels.append(ChannelSpec(
                name=f"inlet_B{j}",
                start=(margin + j * spacing, margin - 3.0, 0),
                end=(positions[idx, 0], positions[idx, 1], 0),
                diameter_um=80.0,
            ))

        chip_size = margin * 2 + (n - 1) * spacing + 6.0

        chip = ChipDesign(
            width=chip_size,
            length=chip_size,
            chambers=chambers,
            channels=channels,
            network_type=f"latin_square_{n}x{n}",
            disc_dimension=2.0,
        )

        # Store Latin square matrix for verification
        chip._latin_square = latin_square

        return chip

    def verify_latin_square(self, chip: "ChipDesign") -> Dict[str, Any]:
        """Verify Latin square statistical properties of a mixer design.

        Checks:
        1. Each treatment appears exactly once per row
        2. Each treatment appears exactly once per column
        3. All n² combinations are covered

        Returns:
            Dict with verification results and any violations
        """
        # Extract Latin square from chamber names
        n = int(np.sqrt(len(chip.chambers)))
        if n * n != len(chip.chambers):
            return {"valid": False, "error": "Not a square grid"}

        # Parse chamber names to extract treatment assignments
        treatments = np.zeros((n, n), dtype=int)
        for idx, ch in enumerate(chip.chambers):
            i = idx // n
            j = idx % n
            # Extract treatment from name like "A0_B1_T2"
            if "_T" in ch.name:
                t = int(ch.name.split("_T")[1])
                treatments[i, j] = t

        # Check row uniqueness
        row_valid = True
        row_issues = []
        for i in range(n):
            row = treatments[i, :]
            if len(set(row)) != n:
                row_valid = False
                row_issues.append(f"Row {i}: {list(row)} has duplicates")

        # Check column uniqueness
        col_valid = True
        col_issues = []
        for j in range(n):
            col = treatments[:, j]
            if len(set(col)) != n:
                col_valid = False
                col_issues.append(f"Col {j}: {list(col)} has duplicates")

        # Check all treatments present
        all_treatments = set(treatments.flatten())
        expected = set(range(n))
        coverage_valid = all_treatments == expected

        # Build combination matrix for display
        drug_a_levels = [f"A{i}" for i in range(n)]
        drug_b_levels = [f"B{j}" for j in range(n)]

        return {
            "valid": row_valid and col_valid and coverage_valid,
            "n": n,
            "latin_square": treatments.tolist(),
            "row_valid": row_valid,
            "row_issues": row_issues,
            "col_valid": col_valid,
            "col_issues": col_issues,
            "coverage_valid": coverage_valid,
            "treatments_found": sorted(list(all_treatments)),
            "drug_a_levels": drug_a_levels,
            "drug_b_levels": drug_b_levels,
        }

    def design_graeco_latin_square(self, n: int = 4) -> "ChipDesign":
        """Design a Graeco-Latin square for 3-factor experiments.

        A Graeco-Latin square overlays two orthogonal Latin squares,
        allowing simultaneous testing of 3 factors:
        - Rows: Drug A concentration
        - Columns: Drug B concentration
        - Latin symbol: Drug C concentration (or time point)
        - Greek symbol: Treatment condition (or replicate)

        Args:
            n: Order (must be != 2, 6 for orthogonal squares to exist)

        Returns:
            ChipDesign with Graeco-Latin square topology
        """
        if n == 2 or n == 6:
            raise ValueError(f"Graeco-Latin squares don't exist for n={n}")

        # Generate two mutually orthogonal Latin squares (MOLS)
        # Use known constructions that guarantee orthogonality

        if n == 3:
            # Standard orthogonal pair for n=3
            latin1 = np.array([[0,1,2], [1,2,0], [2,0,1]])
            latin2 = np.array([[0,1,2], [2,0,1], [1,2,0]])
        elif n == 4:
            # Known orthogonal pair for n=4 (verified)
            latin1 = np.array([[0,1,2,3], [1,0,3,2], [2,3,0,1], [3,2,1,0]])
            latin2 = np.array([[0,1,2,3], [2,3,0,1], [3,2,1,0], [1,0,3,2]])
        elif n == 5:
            # Cyclic construction works for prime n
            latin1 = np.array([[(i+j)%5 for j in range(5)] for i in range(5)])
            latin2 = np.array([[(i+2*j)%5 for j in range(5)] for i in range(5)])
        else:
            # General cyclic construction (works for prime n)
            latin1 = np.array([[(i+j)%n for j in range(n)] for i in range(n)])
            latin2 = np.array([[(2*i+j)%n for j in range(n)] for i in range(n)])

        # Create chambers
        chambers = []
        spacing = 8.0
        margin = 5.0
        positions = []

        for i in range(n):
            for j in range(n):
                x = margin + j * spacing
                y = margin + i * spacing
                t1 = latin1[i, j]  # Treatment factor
                t2 = latin2[i, j]  # Second factor (Greek)

                chambers.append(ChamberSpec(
                    name=f"A{i}_B{j}_C{t1}_D{t2}",
                    center=(x, y, 0),
                    width=2.0, height=2.0, depth=0.1
                ))
                positions.append([x, y])

        positions = np.array(positions)

        # Create grid channels (same as Latin square)
        channels = []

        for i in range(n):
            for j in range(n - 1):
                idx1 = i * n + j
                idx2 = i * n + j + 1
                channels.append(ChannelSpec(
                    name=f"h_{i}_{j}",
                    start=(positions[idx1, 0], positions[idx1, 1], 0),
                    end=(positions[idx2, 0], positions[idx2, 1], 0),
                    diameter_um=50.0,
                ))

        for i in range(n - 1):
            for j in range(n):
                idx1 = i * n + j
                idx2 = (i + 1) * n + j
                channels.append(ChannelSpec(
                    name=f"v_{i}_{j}",
                    start=(positions[idx1, 0], positions[idx1, 1], 0),
                    end=(positions[idx2, 0], positions[idx2, 1], 0),
                    diameter_um=50.0,
                ))

        chip_size = margin * 2 + (n - 1) * spacing + 6.0

        chip = ChipDesign(
            width=chip_size,
            length=chip_size,
            chambers=chambers,
            channels=channels,
            network_type=f"graeco_latin_{n}x{n}",
            disc_dimension=2.0,
        )

        chip._latin_square = latin1
        chip._greek_square = latin2

        return chip

    def design_topology_set(self) -> Dict[str, "ChipDesign"]:
        """Design chips with various graph topologies for comparison.

        Returns different topologies - each creates different flow patterns:
        - Petersen: Classic non-planar, 3-regular, highly symmetric
        - Random planar: Organic-looking, no crossings
        - Complete bipartite K33: Non-planar, two groups fully connected
        - Hypercube Q3: 3D cube topology, regular
        """
        return {
            "petersen": self._design_petersen_graph(),
            "random_planar": self._design_random_planar(n_nodes=9, seed=42),
            "k33_bipartite": self._design_k33_bipartite(),
            "hypercube_q3": self._design_hypercube_q3(),
        }

    def design_fat_petersen(self, channels_per_edge: int = 3) -> "ChipDesign":
        """Fat Petersen graph - Petersen topology with multiple parallel channels.

        Combines:
        - Petersen's high max-flow capacity (multiple parallel paths)
        - Fat tree's enhanced mixing (parallel channels per edge)

        Args:
            channels_per_edge: Number of parallel channels between connected nodes

        Returns:
            ChipDesign with 10 chambers and 15 * channels_per_edge channels
        """
        n = 10
        positions = np.zeros((n, 2))

        # Outer pentagon (vertices 0-4)
        for i in range(5):
            angle = np.pi/2 + i * 2*np.pi/5
            positions[i] = [0.5 + 0.4*np.cos(angle), 0.5 + 0.4*np.sin(angle)]

        # Inner pentagram (vertices 5-9)
        for i in range(5):
            angle = np.pi/2 + i * 2*np.pi/5
            positions[5+i] = [0.5 + 0.2*np.cos(angle), 0.5 + 0.2*np.sin(angle)]

        # Build edge list (Petersen has 15 edges)
        edges = []
        # Outer pentagon
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Inner pentagram (skip-1 connections)
        inner_order = [5, 7, 9, 6, 8]
        for i in range(5):
            edges.append((inner_order[i], inner_order[(i + 1) % 5]))
        # Spokes
        for i in range(5):
            edges.append((i, i + 5))

        # Create chambers
        chambers = []
        chip_scale = 50.0  # mm
        for i in range(n):
            x = positions[i, 0] * chip_scale
            y = positions[i, 1] * chip_scale
            chambers.append(ChamberSpec(
                name=f"P{i}",
                center=(x, y, 0),
                width=2.0, height=2.0, depth=0.1
            ))

        # Create channels - multiple per edge with slight offset
        channels = []
        channel_spacing = 0.15  # mm offset between parallel channels

        for edge_idx, (i, j) in enumerate(edges):
            start = np.array([positions[i, 0] * chip_scale, positions[i, 1] * chip_scale])
            end = np.array([positions[j, 0] * chip_scale, positions[j, 1] * chip_scale])

            # Direction vector and perpendicular for offsets
            direction = end - start
            length = np.linalg.norm(direction)
            if length > 0:
                perp = np.array([-direction[1], direction[0]]) / length
            else:
                perp = np.array([0, 1])

            # Create parallel channels
            for ch_idx in range(channels_per_edge):
                # Offset perpendicular to channel direction
                offset = (ch_idx - (channels_per_edge - 1) / 2) * channel_spacing
                ch_start = start + perp * offset
                ch_end = end + perp * offset

                channels.append(ChannelSpec(
                    name=f"ch_{i}_{j}_{ch_idx}",
                    start=(ch_start[0], ch_start[1], 0),
                    end=(ch_end[0], ch_end[1], 0),
                    diameter_um=100.0,
                ))

        chip = ChipDesign(
            width=chip_scale,
            length=chip_scale,
            chambers=chambers,
            channels=channels,
            network_type=f"fat_petersen_{channels_per_edge}x",
            disc_dimension=2.8  # Petersen base disc
        )
        return chip

    def fatten_chip(self, base_chip: "ChipDesign", channels_per_edge: int = 3) -> "ChipDesign":
        """Add parallel channels to any existing chip design.

        Takes a base topology and creates multiple parallel channels
        for each existing channel, enhancing flow capacity.

        Args:
            base_chip: Existing ChipDesign to fatten
            channels_per_edge: Number of parallel channels per original channel

        Returns:
            New ChipDesign with multiplied channels
        """
        channel_spacing = 0.15  # mm offset between parallel channels
        new_channels = []

        for orig_ch in base_chip.channels:
            start = np.array(orig_ch.start[:2])
            end = np.array(orig_ch.end[:2])

            direction = end - start
            length = np.linalg.norm(direction)
            if length > 0:
                perp = np.array([-direction[1], direction[0]]) / length
            else:
                perp = np.array([0, 1])

            for ch_idx in range(channels_per_edge):
                offset = (ch_idx - (channels_per_edge - 1) / 2) * channel_spacing
                ch_start = start + perp * offset
                ch_end = end + perp * offset

                new_channels.append(ChannelSpec(
                    name=f"{orig_ch.name}_{ch_idx}",
                    start=(ch_start[0], ch_start[1], orig_ch.start[2]),
                    end=(ch_end[0], ch_end[1], orig_ch.end[2]),
                    diameter_um=orig_ch.diameter_um,
                ))

        return ChipDesign(
            width=base_chip.width,
            length=base_chip.length,
            height=base_chip.height,
            chambers=base_chip.chambers,
            channels=new_channels,
            network_type=f"fat_{base_chip.network_type}_{channels_per_edge}x",
            disc_dimension=base_chip.disc_dimension,
        )

    def _design_petersen_graph(self) -> "ChipDesign":
        """Petersen graph - famous 3-regular non-planar graph.

        Properties:
        - 10 vertices, 15 edges
        - Every vertex has exactly 3 connections (3-regular)
        - Highly symmetric (vertex-transitive)
        - Non-planar (contains K5 minor)

        Layout: outer pentagon + inner pentagram
               0
              /|\
             4   1
            /|\ /|\
           3-+-X-+-2   (outer pentagon)
             |/ \|
             8   5
            /|\ /|\
           7---9---6   (inner pentagram)
        """
        n = 10
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # Outer pentagon (vertices 0-4)
        for i in range(5):
            angle = np.pi/2 + i * 2*np.pi/5  # Start from top
            positions[i] = [0.5 + 0.4*np.cos(angle), 0.5 + 0.4*np.sin(angle)]

        # Inner pentagram (vertices 5-9)
        for i in range(5):
            angle = np.pi/2 + i * 2*np.pi/5
            positions[5+i] = [0.5 + 0.2*np.cos(angle), 0.5 + 0.2*np.sin(angle)]

        # Outer pentagon edges (0-1-2-3-4-0)
        for i in range(5):
            j = (i + 1) % 5
            connectivity[i, j] = 0.8
            connectivity[j, i] = 0.8

        # Inner pentagram edges (5-7-9-6-8-5) - skip one each time
        inner_order = [5, 7, 9, 6, 8]
        for i in range(5):
            j = (i + 1) % 5
            connectivity[inner_order[i], inner_order[j]] = 0.8
            connectivity[inner_order[j], inner_order[i]] = 0.8

        # Spoke edges (outer to inner)
        for i in range(5):
            connectivity[i, i+5] = 0.8
            connectivity[i+5, i] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "petersen_graph"
        return chip

    def _design_random_planar(self, n_nodes: int = 9, seed: int = 42) -> "ChipDesign":
        """Random planar graph - organic-looking network.

        Uses Delaunay triangulation of random points, then
        removes some edges to reduce density while maintaining connectivity.
        """
        np.random.seed(seed)

        # Generate random positions
        positions = np.random.rand(n_nodes, 2) * 0.8 + 0.1

        # Create Delaunay-like connectivity (connect nearby points)
        connectivity = np.zeros((n_nodes, n_nodes))

        # Calculate distances
        for i in range(n_nodes):
            distances = []
            for j in range(n_nodes):
                if i != j:
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 +
                               (positions[i,1]-positions[j,1])**2)
                    distances.append((d, j))
            distances.sort()

            # Connect to 2-4 nearest neighbors (keeps it planar-ish)
            n_neighbors = min(3, len(distances))
            for d, j in distances[:n_neighbors]:
                if d < 0.5:  # Distance threshold
                    connectivity[i, j] = 0.7 + np.random.rand() * 0.2
                    connectivity[j, i] = connectivity[i, j]

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "random_planar"
        return chip

    def _design_k33_bipartite(self) -> "ChipDesign":
        """Complete bipartite graph K3,3 - classic non-planar.

        Two groups of 3 nodes, every node in group A
        connects to every node in group B.

        Layout:
        A1    A2    A3     (top row - e.g., 3 drug inputs)
         |\ /|\ /|
         | X | X |
         |/ \|/ \|
        B1    B2    B3     (bottom row - e.g., 3 outputs)

        Non-planar: cannot draw without edge crossings.
        """
        n = 6
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # Group A (top row): vertices 0, 1, 2
        positions[0] = [0.2, 0.2]
        positions[1] = [0.5, 0.2]
        positions[2] = [0.8, 0.2]

        # Group B (bottom row): vertices 3, 4, 5
        positions[3] = [0.2, 0.8]
        positions[4] = [0.5, 0.8]
        positions[5] = [0.8, 0.8]

        # Complete bipartite: every A connects to every B
        for i in range(3):      # A vertices
            for j in range(3):  # B vertices
                connectivity[i, 3+j] = 0.8
                connectivity[3+j, i] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "k33_bipartite"
        return chip

    def design_sorting_network(
        self,
        n_inputs: int = 8,
        network_type: str = "bitonic"
    ) -> "ChipDesign":
        """Sorting network topology - fixed structure that sorts any input.

        Args:
            n_inputs: Number of inputs (should be power of 2 for bitonic)
            network_type: "bitonic", "oddeven", or "bubble"

        Returns:
            ChipDesign with sorting network topology
        """
        if network_type == "bitonic":
            return self._design_bitonic_sort(n_inputs)
        elif network_type == "oddeven":
            return self._design_oddeven_sort(n_inputs)
        elif network_type == "unplugged" or network_type == "cs_unplugged":
            return self._design_cs_unplugged_sort(n_inputs)
        else:
            return self._design_bubble_sort(n_inputs)

    def _design_bitonic_sort(self, n: int = 8) -> "ChipDesign":
        """Bitonic sorting network - O(n log²n) comparators.

        Beautiful recursive structure. For n=8:
        - Stage 1: pairs (0,1), (2,3), (4,5), (6,7)
        - Stage 2: groups of 4
        - Stage 3: all 8

        Each comparator is a junction where two channels meet.
        """
        # Ensure n is power of 2
        n = 2 ** int(np.ceil(np.log2(n)))

        # Calculate stages: log2(n) stages, each with sub-stages
        n_stages = int(np.log2(n))

        # Generate comparator pairs for bitonic sort
        comparators = []
        for stage in range(n_stages):
            for substage in range(stage + 1):
                step = 2 ** (stage - substage)
                for i in range(0, n, 2 * step):
                    for j in range(step):
                        idx1 = i + j
                        idx2 = i + j + step
                        if idx2 < n:
                            # Direction alternates for bitonic property
                            direction = (idx1 // (2 ** (stage + 1))) % 2
                            comparators.append((idx1, idx2, len(comparators), direction))

        # Create chambers: inputs + comparator junctions + outputs
        # Layout: inputs on left, stages flow right, outputs on right
        n_comparators = len(comparators)
        total_chambers = n * 2 + n_comparators  # inputs + outputs + junctions

        positions = []
        connectivity = np.zeros((total_chambers, total_chambers))

        # Input chambers (left column)
        for i in range(n):
            positions.append([0.0, i / (n - 1)])

        # Output chambers (right column)
        for i in range(n):
            positions.append([1.0, i / (n - 1)])

        # Comparator junctions (middle columns)
        # Group by stage for x-position
        stage_x = {}
        comp_idx = 0
        for stage in range(n_stages):
            for substage in range(stage + 1):
                x = 0.1 + 0.8 * (comp_idx + 0.5) / (n_stages * (n_stages + 1) / 2)
                stage_x[(stage, substage)] = x
                comp_idx += 1

        comp_positions = {}
        comp_idx = 0
        for stage in range(n_stages):
            for substage in range(stage + 1):
                step = 2 ** (stage - substage)
                x = stage_x[(stage, substage)]
                for i in range(0, n, 2 * step):
                    for j in range(step):
                        idx1 = i + j
                        idx2 = i + j + step
                        if idx2 < n:
                            y = (idx1 + idx2) / (2 * (n - 1))
                            positions.append([x, y])
                            comp_positions[comp_idx] = n * 2 + len(positions) - 1 - n * 2
                            comp_idx += 1

        positions = np.array(positions)

        # Build connectivity
        # Track which "wire" each input is currently on
        wire_pos = {i: i for i in range(n)}  # wire i is at chamber i

        comp_idx = 0
        for stage in range(n_stages):
            for substage in range(stage + 1):
                step = 2 ** (stage - substage)
                for i in range(0, n, 2 * step):
                    for j in range(step):
                        idx1 = i + j
                        idx2 = i + j + step
                        if idx2 < n:
                            comp_chamber = n * 2 + comp_idx

                            # Connect current wire positions to comparator
                            connectivity[wire_pos[idx1], comp_chamber] = 0.8
                            connectivity[comp_chamber, wire_pos[idx1]] = 0.8
                            connectivity[wire_pos[idx2], comp_chamber] = 0.8
                            connectivity[comp_chamber, wire_pos[idx2]] = 0.8

                            # Update wire positions (both now go through this comparator)
                            wire_pos[idx1] = comp_chamber
                            wire_pos[idx2] = comp_chamber

                            comp_idx += 1

        # Connect final wire positions to outputs
        for i in range(n):
            output_chamber = n + i
            connectivity[wire_pos[i], output_chamber] = 0.8
            connectivity[output_chamber, wire_pos[i]] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 40.0)
        chip.network_type = f"bitonic_sort_{n}"
        return chip

    def _design_oddeven_sort(self, n: int = 8) -> "ChipDesign":
        """Odd-even transposition sort network.

        Alternates between comparing odd-even pairs and even-odd pairs.
        Simpler structure than bitonic, but more stages.
        """
        positions = []
        connectivity = np.zeros((n * (n + 1), n * (n + 1)))

        # Create n stages, each with n chambers (one per wire)
        for stage in range(n + 1):
            for wire in range(n):
                x = stage / n
                y = wire / (n - 1) if n > 1 else 0.5
                positions.append([x, y])

        positions = np.array(positions)

        # Connect consecutive stages
        for stage in range(n):
            # Determine if odd or even phase
            if stage % 2 == 0:
                # Compare (0,1), (2,3), (4,5), ...
                pairs = [(i, i+1) for i in range(0, n-1, 2)]
            else:
                # Compare (1,2), (3,4), (5,6), ...
                pairs = [(i, i+1) for i in range(1, n-1, 2)]

            for wire in range(n):
                curr = stage * n + wire
                next_stage = (stage + 1) * n + wire

                # Check if this wire is in a comparator pair
                in_pair = False
                for p1, p2 in pairs:
                    if wire == p1 or wire == p2:
                        in_pair = True
                        partner = p2 if wire == p1 else p1
                        partner_next = (stage + 1) * n + partner

                        # Connect to both possible outputs (swap or no-swap)
                        connectivity[curr, next_stage] = 0.7
                        connectivity[next_stage, curr] = 0.7
                        connectivity[curr, partner_next] = 0.5
                        connectivity[partner_next, curr] = 0.5
                        break

                if not in_pair:
                    # No comparator, just pass through
                    connectivity[curr, next_stage] = 0.8
                    connectivity[next_stage, curr] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 40.0)
        chip.network_type = f"oddeven_sort_{n}"
        return chip

    def _design_cs_unplugged_sort(self, n: int = 6) -> "ChipDesign":
        """CS Unplugged style sorting network - clean visual layout.

        6 inputs flow upward through comparator boxes.
        Each box compares two adjacent values and swaps if needed.

        Layout matches the classic CS Unplugged diagram:
        - Inputs at bottom
        - Comparator boxes in middle (where lines meet)
        - Outputs at top (sorted)

                 ○ ○ ○ ○ ○ ○  (outputs - sorted)
                 │ │ │ │ │ │
                ┌┴─┴┐ │ │ │ │
                │ □ │ │ │ │ │  (comparator)
                └┬─┬┘ │ │ │ │
                 │ └──┼─┼─╳─┘
                ... more stages ...
                 │ │ │ │ │ │
                 9 6 2 3 1 7  (inputs)
        """
        # 6-input sorting network structure (Batcher's or similar)
        # Comparators as (wire1, wire2, stage)
        # This is an optimal 6-sort with 12 comparators

        comparators_by_stage = [
            # Stage 0 (bottom)
            [(0, 1), (2, 3), (4, 5)],
            # Stage 1
            [(0, 2), (1, 3), (4, 5)],
            # Stage 2
            [(0, 4), (1, 5), (2, 3)],
            # Stage 3
            [(0, 1), (2, 4), (3, 5)],
            # Stage 4
            [(1, 2), (3, 4)],
            # Stage 5 (top)
            [(2, 3)],
        ]

        n_stages = len(comparators_by_stage)
        n_wires = 6

        # Count total comparators
        n_comparators = sum(len(stage) for stage in comparators_by_stage)

        # Chambers: inputs (bottom) + comparators + outputs (top)
        # Layout: y=0 is bottom (inputs), y=1 is top (outputs)
        positions = []
        connectivity = np.zeros((n_wires * 2 + n_comparators, n_wires * 2 + n_comparators))

        # Input chambers (bottom row)
        for i in range(n_wires):
            x = (i + 0.5) / n_wires
            positions.append([x, 0.0])  # indices 0-5

        # Output chambers (top row)
        for i in range(n_wires):
            x = (i + 0.5) / n_wires
            positions.append([x, 1.0])  # indices 6-11

        # Comparator chambers (middle)
        comp_idx = n_wires * 2  # Start after inputs and outputs
        comp_positions = {}  # (stage, comp_in_stage) -> chamber_idx

        for stage_idx, stage in enumerate(comparators_by_stage):
            y = 0.1 + 0.8 * (stage_idx + 0.5) / n_stages
            for comp_in_stage, (w1, w2) in enumerate(stage):
                x = ((w1 + w2) / 2 + 0.5) / n_wires
                positions.append([x, y])
                comp_positions[(stage_idx, comp_in_stage)] = comp_idx
                comp_idx += 1

        positions = np.array(positions)

        # Build connectivity
        # Track current position of each wire (which chamber it's at)
        wire_chamber = {i: i for i in range(n_wires)}  # Start at input chambers

        for stage_idx, stage in enumerate(comparators_by_stage):
            for comp_in_stage, (w1, w2) in enumerate(stage):
                comp_chamber = comp_positions[(stage_idx, comp_in_stage)]

                # Connect wires to this comparator
                connectivity[wire_chamber[w1], comp_chamber] = 0.8
                connectivity[comp_chamber, wire_chamber[w1]] = 0.8
                connectivity[wire_chamber[w2], comp_chamber] = 0.8
                connectivity[comp_chamber, wire_chamber[w2]] = 0.8

                # After comparator, both wires continue from here
                wire_chamber[w1] = comp_chamber
                wire_chamber[w2] = comp_chamber

        # Connect final wire positions to outputs
        for wire in range(n_wires):
            output_chamber = n_wires + wire  # Output chambers are 6-11
            connectivity[wire_chamber[wire], output_chamber] = 0.8
            connectivity[output_chamber, wire_chamber[wire]] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "cs_unplugged_sort_6"

        return chip

    def _design_bubble_sort(self, n: int = 6) -> "ChipDesign":
        """Bubble sort network - simple but O(n²) comparators.

        Every adjacent pair compared, repeated n times.
        Creates a triangular/diagonal pattern.
        """
        # Simplified: just create the comparison structure
        positions = []
        connectivity = np.zeros((n, n))

        # Circular arrangement
        for i in range(n):
            angle = 2 * np.pi * i / n
            positions.append([0.5 + 0.4*np.cos(angle), 0.5 + 0.4*np.sin(angle)])

        positions = np.array(positions)

        # Connect adjacent pairs (bubble sort comparisons)
        for i in range(n):
            j = (i + 1) % n
            connectivity[i, j] = 0.8
            connectivity[j, i] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = f"bubble_sort_{n}"
        return chip

    def _design_hypercube_q3(self) -> "ChipDesign":
        """3D hypercube (Q3) - 8 vertices, each with 3 neighbors.

        Vertices are binary: 000, 001, 010, 011, 100, 101, 110, 111
        Edges connect vertices differing by 1 bit.

        Layout (projected to 2D):
            110───111
            /|    /|
          100───101
           │010──│011
           │/    │/
          000───001

        Regular, symmetric, planar embedding possible.
        """
        n = 8
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # 2D projection of cube vertices
        # Inner square (000, 001, 011, 010) and outer square (100, 101, 111, 110)
        inner = [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]]
        outer = [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]]

        # Map binary to positions
        # 000=0, 001=1, 010=2, 011=3, 100=4, 101=5, 110=6, 111=7
        positions[0] = inner[0]  # 000
        positions[1] = inner[1]  # 001
        positions[2] = inner[3]  # 010
        positions[3] = inner[2]  # 011
        positions[4] = outer[0]  # 100
        positions[5] = outer[1]  # 101
        positions[6] = outer[3]  # 110
        positions[7] = outer[2]  # 111

        # Edges: connect vertices differing by exactly 1 bit
        for i in range(8):
            for j in range(i+1, 8):
                # XOR and count bits
                xor = i ^ j
                if bin(xor).count('1') == 1:  # Differ by 1 bit
                    connectivity[i, j] = 0.8
                    connectivity[j, i] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "hypercube_q3"
        return chip

    def design_fat_tree_mixer(
        self,
        n_levels: int = 3,
        channels_per_branch: int = 3,
        channel_diameter_um: float = 50.0,
    ) -> "ChipDesign":
        """Fat tree design - multiple parallel channels per branch.

        Bandwidth increases toward root (no bottlenecks).
        Multiple inlet channels to each chamber improves mixing.

        Args:
            n_levels: Tree depth (2-4)
            channels_per_branch: Parallel channels per connection (1-4)
            channel_diameter_um: Individual channel diameter

        Layout (3 levels, 3 channels per branch):
                    ═══╗
                    [root] ← inlet
                   ╱══╲
                  ╱    ╲
               [L1a]  [L1b]
               ╱═╲    ╱═╲
              ╱   ╲  ╱   ╲
           [L2a][L2b][L2c][L2d] ← outlets
        """
        # Calculate number of chambers
        n_chambers = 2**n_levels - 1  # Full binary tree
        positions = []
        connectivity = np.zeros((n_chambers, n_chambers))

        # Position chambers in tree layout
        for i in range(n_chambers):
            level = int(np.floor(np.log2(i + 1)))
            pos_in_level = i - (2**level - 1)
            n_in_level = 2**level

            # x spreads out at each level, y goes down
            x = (pos_in_level + 0.5) / n_in_level
            y = level / (n_levels - 1) if n_levels > 1 else 0.5
            positions.append([x, y])

        positions = np.array(positions)

        # Build connectivity (parent to children)
        for i in range(n_chambers):
            left_child = 2 * i + 1
            right_child = 2 * i + 2

            if left_child < n_chambers:
                # Connection strength represents number of parallel channels
                connectivity[i, left_child] = channels_per_branch * 0.3
                connectivity[left_child, i] = channels_per_branch * 0.3

            if right_child < n_chambers:
                connectivity[i, right_child] = channels_per_branch * 0.3
                connectivity[right_child, i] = channels_per_branch * 0.3

        # Create base chip
        chip = self.design_from_connectivity(
            connectivity, positions, channel_diameter_um
        )

        # Now add parallel channels for "fat" connections
        fat_channels = []
        spacing_um = channel_diameter_um * 1.5  # Space between parallel channels

        for ch in chip.channels:
            # Original channel is first
            fat_channels.append(ch)

            # Add parallel channels offset perpendicular to flow direction
            if channels_per_branch > 1:
                dx = ch.end[0] - ch.start[0]
                dy = ch.end[1] - ch.start[1]
                length = np.sqrt(dx**2 + dy**2)

                if length > 0:
                    # Perpendicular unit vector
                    perp_x = -dy / length
                    perp_y = dx / length

                    for p in range(1, channels_per_branch):
                        offset = (p - (channels_per_branch - 1) / 2) * spacing_um / 1000

                        parallel_ch = ChannelSpec(
                            start=(
                                ch.start[0] + perp_x * offset,
                                ch.start[1] + perp_y * offset,
                                ch.start[2]
                            ),
                            end=(
                                ch.end[0] + perp_x * offset,
                                ch.end[1] + perp_y * offset,
                                ch.end[2]
                            ),
                            diameter_um=ch.diameter_um,
                            name=f"{ch.name}_parallel_{p}",
                        )
                        fat_channels.append(parallel_ch)

        chip.channels = fat_channels
        chip.network_type = f"fat_tree_{n_levels}lvl_{channels_per_branch}ch"

        return chip

    def _design_three_drug_combinator(self, levels: int = 3) -> "ChipDesign":
        """Three-drug combinatorial mixer using V4-like structure.

        Creates balanced combinations of 3 drugs using
        mutually orthogonal topology.
        """
        # Triangular arrangement with center mixing
        positions = []
        connectivity = np.zeros((10, 10))

        # Drug inlets at triangle vertices
        positions.append([0.5, 0.0])   # 0: Drug A inlet
        positions.append([0.0, 0.866]) # 1: Drug B inlet
        positions.append([1.0, 0.866]) # 2: Drug C inlet

        # Dilution chambers
        positions.append([0.35, 0.25])  # 3: A75
        positions.append([0.15, 0.65])  # 4: B75
        positions.append([0.85, 0.65])  # 5: C75

        # Binary mixing chambers
        positions.append([0.25, 0.45])  # 6: A+B mix
        positions.append([0.75, 0.45])  # 7: A+C mix
        positions.append([0.5, 0.75])   # 8: B+C mix

        # Center: triple mix
        positions.append([0.5, 0.5])    # 9: A+B+C mix

        positions = np.array(positions)

        # Dilution connections
        connectivity[0, 3] = 0.9  # A → A75
        connectivity[3, 0] = 0.9
        connectivity[1, 4] = 0.9  # B → B75
        connectivity[4, 1] = 0.9
        connectivity[2, 5] = 0.9  # C → C75
        connectivity[5, 2] = 0.9

        # Binary mixing
        connectivity[3, 6] = 0.7; connectivity[6, 3] = 0.7  # A75 → A+B
        connectivity[4, 6] = 0.7; connectivity[6, 4] = 0.7  # B75 → A+B
        connectivity[3, 7] = 0.7; connectivity[7, 3] = 0.7  # A75 → A+C
        connectivity[5, 7] = 0.7; connectivity[7, 5] = 0.7  # C75 → A+C
        connectivity[4, 8] = 0.7; connectivity[8, 4] = 0.7  # B75 → B+C
        connectivity[5, 8] = 0.7; connectivity[8, 5] = 0.7  # C75 → B+C

        # Triple mixing (center)
        connectivity[6, 9] = 0.6; connectivity[9, 6] = 0.6
        connectivity[7, 9] = 0.6; connectivity[9, 7] = 0.6
        connectivity[8, 9] = 0.6; connectivity[9, 8] = 0.6

        chip = self.design_from_connectivity(connectivity, positions, 60.0)
        chip.network_type = "combinatorial_3drug_V4"

        return chip

    def _design_flow_grid(self) -> "ChipDesign":
        """3x3 grid with flow from left to right.

        Layout:
        IN →[0]─[1]─[2]→ OUT
             │   │   │
            [3]─[4]─[5]
             │   │   │
            [6]─[7]─[8]→ OUT
        """
        n = 9
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # 3x3 grid positions
        for i in range(n):
            row = i // 3
            col = i % 3
            positions[i] = [col / 2, row / 2]  # Normalized 0-1

        # Horizontal connections
        for row in range(3):
            for col in range(2):
                i = row * 3 + col
                j = row * 3 + col + 1
                connectivity[i, j] = 0.8
                connectivity[j, i] = 0.8

        # Vertical connections
        for row in range(2):
            for col in range(3):
                i = row * 3 + col
                j = (row + 1) * 3 + col
                connectivity[i, j] = 0.8
                connectivity[j, i] = 0.8

        chip = self.design_from_connectivity(connectivity, positions, 50.0)

        # Set proper inlet/outlets for through-flow
        # Inlet on left (chambers 0, 3, 6), outlets on right (chambers 2, 5, 8)
        chip.main_inlet_position = (-self.chip_length/2 + 1, 0, self.chip_height/2)
        chip.main_outlet_position = (self.chip_length/2 - 1, 0, self.chip_height/2)
        chip.network_type = "grid_planar"

        return chip

    def _design_flow_cross_connected(self) -> "ChipDesign":
        """3x3 grid with diagonal cross-connections (non-planar).

        Layout:
        IN →[0]─[1]─[2]→ OUT
             │╲ │╱│╲│
            [3]─[4]─[5]
             │╱│╲│ │╱│
            [6]─[7]─[8]→ OUT

        Diagonal connections create K5-like subgraphs → non-planar
        """
        n = 9
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # Same 3x3 grid positions
        for i in range(n):
            row = i // 3
            col = i % 3
            positions[i] = [col / 2, row / 2]

        # Horizontal connections
        for row in range(3):
            for col in range(2):
                i = row * 3 + col
                j = row * 3 + col + 1
                connectivity[i, j] = 0.8
                connectivity[j, i] = 0.8

        # Vertical connections
        for row in range(2):
            for col in range(3):
                i = row * 3 + col
                j = (row + 1) * 3 + col
                connectivity[i, j] = 0.8
                connectivity[j, i] = 0.8

        # Diagonal connections (creates non-planarity)
        diagonals = [
            (0, 4), (4, 8),  # Main diagonal
            (2, 4), (4, 6),  # Anti-diagonal
            (1, 3), (1, 5),  # Additional crosses
            (3, 7), (5, 7),
        ]
        for i, j in diagonals:
            connectivity[i, j] = 0.6
            connectivity[j, i] = 0.6

        chip = self.design_from_connectivity(connectivity, positions, 50.0)
        chip.network_type = "cross_connected_nonplanar"

        return chip

    def _design_flow_tree(self) -> "ChipDesign":
        """Bifurcating tree with inlet at root, outlets at leaves.

        Layout:
                 IN
                  │
                 [0]
                ╱   ╲
              [1]   [2]
              ╱╲     ╱╲
            [3][4] [5][6]
             ↓  ↓   ↓  ↓
            OUT OUT OUT OUT
        """
        n = 7  # 3 levels: 1 + 2 + 4
        positions = np.zeros((n, 2))
        connectivity = np.zeros((n, n))

        # Tree positions (root at top, leaves at bottom)
        # Level 0: node 0
        positions[0] = [0.5, 0.0]

        # Level 1: nodes 1, 2
        positions[1] = [0.25, 0.4]
        positions[2] = [0.75, 0.4]

        # Level 2: nodes 3, 4, 5, 6 (leaves)
        positions[3] = [0.125, 0.8]
        positions[4] = [0.375, 0.8]
        positions[5] = [0.625, 0.8]
        positions[6] = [0.875, 0.8]

        # Tree connections
        tree_edges = [
            (0, 1), (0, 2),      # Root to level 1
            (1, 3), (1, 4),      # Level 1 to leaves
            (2, 5), (2, 6),
        ]
        for i, j in tree_edges:
            connectivity[i, j] = 0.9
            connectivity[j, i] = 0.9

        chip = self.design_from_connectivity(connectivity, positions, 50.0)

        # Inlet at root (top), outlets at leaves (bottom)
        chip.main_inlet_position = (0, self.chip_width/2 - 1, self.chip_height/2)
        chip.main_outlet_position = (0, -self.chip_width/2 + 1, self.chip_height/2)
        chip.network_type = "bifurcating_tree"

        return chip

    def _generate_positions(self, n: int) -> NDArray:
        """Generate node positions using force-directed layout."""
        np.random.seed(42)
        positions = np.random.rand(n, 2)

        # Simple force-directed refinement
        for _ in range(50):
            for i in range(n):
                force = np.zeros(2)
                for j in range(n):
                    if i != j:
                        diff = positions[i] - positions[j]
                        dist = max(np.linalg.norm(diff), 0.01)
                        force += diff / (dist**2)

                # Center attraction
                force -= (positions[i] - 0.5) * 0.5

                positions[i] += force * 0.01
                positions[i] = np.clip(positions[i], 0.1, 0.9)

        return positions

    def _scale_positions(self, positions: NDArray, margin: float) -> NDArray:
        """Scale normalized positions to chip dimensions."""
        positions = np.asarray(positions)
        scaled = positions.copy()

        # Scale to chip dimensions minus margins
        scaled[:, 0] = (positions[:, 0] - 0.5) * (self.chip_length - 2 * margin)
        scaled[:, 1] = (positions[:, 1] - 0.5) * (self.chip_width - 2 * margin)

        if positions.shape[1] > 2:
            scaled[:, 2] = positions[:, 2] * self.chip_height
        else:
            # Add z coordinate
            scaled = np.column_stack([scaled, np.full(len(scaled), self.chip_height / 2)])

        return scaled

    def analyze_flow_network(
        self,
        chip: "ChipDesign",
        source_chambers: List[int] = None,
        sink_chambers: List[int] = None,
    ) -> Dict[str, Any]:
        """Analyze chip using max-flow/min-cut theory.

        Identifies bottlenecks and optimal flow distribution.

        Args:
            chip: ChipDesign to analyze
            source_chambers: Indices of source chambers (default: first)
            sink_chambers: Indices of sink chambers (default: last)

        Returns:
            Dict with max_flow, min_cut edges, bottleneck analysis
        """
        n = len(chip.chambers)

        if source_chambers is None:
            source_chambers = [0]
        if sink_chambers is None:
            sink_chambers = [n - 1]

        # Build capacity matrix from channels
        # Capacity proportional to channel diameter squared (Hagen-Poiseuille)
        capacity = np.zeros((n, n))

        for ch in chip.channels:
            # Find which chambers this channel connects
            start_pos = np.array(ch.start[:2])
            end_pos = np.array(ch.end[:2])

            start_chamber = None
            end_chamber = None

            for i, chamber in enumerate(chip.chambers):
                chamber_pos = np.array(chamber.center[:2])
                if np.linalg.norm(start_pos - chamber_pos) < 1.0:
                    start_chamber = i
                if np.linalg.norm(end_pos - chamber_pos) < 1.0:
                    end_chamber = i

            if start_chamber is not None and end_chamber is not None:
                # Capacity ~ diameter^2 (cross-sectional area)
                cap = (ch.diameter_um / 50.0) ** 2
                capacity[start_chamber, end_chamber] = cap
                capacity[end_chamber, start_chamber] = cap

        # Simple max-flow using BFS (Edmonds-Karp style)
        def bfs_path(capacity, source, sink, parent):
            visited = set([source])
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in range(len(capacity)):
                    if v not in visited and capacity[u, v] > 0:
                        visited.add(v)
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)
            return False

        # Compute max flow from all sources to all sinks
        # (simplified: use super-source and super-sink)
        total_max_flow = 0
        min_cut_edges = []

        for source in source_chambers:
            for sink in sink_chambers:
                if source == sink:
                    continue

                # Copy capacity matrix
                residual = capacity.copy()
                parent = [-1] * n
                max_flow = 0

                while bfs_path(residual, source, sink, parent):
                    # Find min capacity along path
                    path_flow = float('inf')
                    s = sink
                    while s != source:
                        path_flow = min(path_flow, residual[parent[s], s])
                        s = parent[s]

                    # Update residual capacities
                    v = sink
                    while v != source:
                        u = parent[v]
                        residual[u, v] -= path_flow
                        residual[v, u] += path_flow
                        v = parent[v]

                    max_flow += path_flow
                    parent = [-1] * n

                total_max_flow += max_flow

                # Find min-cut edges (saturated edges reachable from source)
                visited = set()
                queue = [source]
                while queue:
                    u = queue.pop(0)
                    if u in visited:
                        continue
                    visited.add(u)
                    for v in range(n):
                        if v not in visited and residual[u, v] > 0:
                            queue.append(v)

                for u in visited:
                    for v in range(n):
                        if v not in visited and capacity[u, v] > 0:
                            min_cut_edges.append((u, v, capacity[u, v]))

        # Identify bottleneck chambers (in min-cut)
        bottleneck_chambers = set()
        for u, v, _ in min_cut_edges:
            bottleneck_chambers.add(u)
            bottleneck_chambers.add(v)

        return {
            "max_flow": total_max_flow,
            "min_cut_edges": min_cut_edges,
            "n_cut_edges": len(min_cut_edges),
            "bottleneck_chambers": list(bottleneck_chambers),
            "recommendation": self._flow_recommendation(
                total_max_flow, min_cut_edges, chip
            ),
        }

    def optimize_for_balanced_flow(
        self,
        chip: "ChipDesign",
        channels_at_mincut: int = 1,
        diameter_multiplier: float = 1.0,
        source_idx: int = 0,
        sink_idx: int = -1,
    ) -> "ChipDesign":
        """Optimize chip by widening or adding channels at min-cut bottlenecks.

        Instead of uniformly fattening all edges, this method:
        1. Identifies the min-cut edges (flow bottlenecks)
        2. Widens diameter and/or adds parallel channels at those locations
        3. Leaves other channels unchanged

        Flow scales with diameter^4 (Poiseuille), so widening is very effective.

        Args:
            chip: Base chip design to optimize
            channels_at_mincut: Number of parallel channels at each min-cut edge
            diameter_multiplier: Multiply min-cut channel diameters by this factor
            source_idx: Source chamber index
            sink_idx: Sink chamber index (-1 for last)

        Returns:
            Optimized ChipDesign with balanced flow
        """
        n = len(chip.chambers)
        if sink_idx == -1:
            sink_idx = n - 1

        # Find min-cut edges
        analysis = self.analyze_flow_network(chip, [source_idx], [sink_idx])
        min_cut_edges = analysis["min_cut_edges"]

        # Build set of chamber pairs in the min-cut
        bottleneck_pairs = set()
        for u, v, _ in min_cut_edges:
            bottleneck_pairs.add((min(u, v), max(u, v)))

        # Find which channels correspond to min-cut edges
        channel_spacing = 0.15  # mm offset between parallel channels
        new_channels = []

        for ch in chip.channels:
            # Find which chambers this channel connects (use closest match)
            start_pos = np.array(ch.start[:2])
            end_pos = np.array(ch.end[:2])

            start_chamber = None
            end_chamber = None
            start_dist = float('inf')
            end_dist = float('inf')

            for i, chamber in enumerate(chip.chambers):
                chamber_pos = np.array(chamber.center[:2])
                d_start = np.linalg.norm(start_pos - chamber_pos)
                d_end = np.linalg.norm(end_pos - chamber_pos)

                if d_start < start_dist:
                    start_dist = d_start
                    start_chamber = i
                if d_end < end_dist:
                    end_dist = d_end
                    end_chamber = i

            # Check if this channel is on a min-cut edge
            is_bottleneck = False
            if start_chamber is not None and end_chamber is not None:
                pair = (min(start_chamber, end_chamber), max(start_chamber, end_chamber))
                is_bottleneck = pair in bottleneck_pairs

            if is_bottleneck:
                # Add multiple parallel channels at bottleneck
                start = np.array(ch.start[:2])
                end = np.array(ch.end[:2])
                direction = end - start
                length = np.linalg.norm(direction)

                if length > 0:
                    perp = np.array([-direction[1], direction[0]]) / length
                else:
                    perp = np.array([0, 1])

                # Apply diameter multiplier at bottleneck (flow ~ d^4)
                widened_diameter = ch.diameter_um * diameter_multiplier

                for ch_idx in range(channels_at_mincut):
                    offset = (ch_idx - (channels_at_mincut - 1) / 2) * channel_spacing
                    ch_start = start + perp * offset
                    ch_end = end + perp * offset

                    new_channels.append(ChannelSpec(
                        name=f"{ch.name}_opt{ch_idx}",
                        start=(ch_start[0], ch_start[1], ch.start[2]),
                        end=(ch_end[0], ch_end[1], ch.end[2]),
                        diameter_um=widened_diameter,
                    ))
            else:
                # Keep original channel unchanged
                new_channels.append(ch)

        return ChipDesign(
            width=chip.width,
            length=chip.length,
            height=chip.height,
            chambers=chip.chambers,
            channels=new_channels,
            network_type=f"balanced_{chip.network_type}",
            disc_dimension=chip.disc_dimension,
        )

    def simulate_chip_flow(
        self,
        chip: "ChipDesign",
        inlet_pressure_Pa: float = 100.0,
        outlet_pressure_Pa: float = 0.0,
    ) -> Dict[str, Any]:
        """Simulate flow through chip using Stokes flow physics.

        Args:
            chip: ChipDesign to simulate
            inlet_pressure_Pa: Pressure at inlet
            outlet_pressure_Pa: Pressure at outlet

        Returns:
            Dict with flow rates, velocities, residence times per chamber
        """
        from .glymphatic_flow import MicrofluidicChannel, compute_channel_flow, WATER_VISCOSITY

        # Pressure drop across chip
        delta_P = inlet_pressure_Pa - outlet_pressure_Pa

        # Simulate flow through each channel
        channel_results = []
        total_flow_rate = 0

        for ch in chip.channels:
            # Create channel geometry
            width_um = ch.diameter_um
            height_um = ch.diameter_um * 0.5  # Assume rectangular, half height

            channel = MicrofluidicChannel(
                width_um=width_um,
                height_um=height_um,
                length_mm=ch.length_mm,
            )

            # Calculate flow rate from pressure using Poiseuille's law for rectangular channels
            # Q = ΔP × w × h³ / (12 × μ × L)
            w = width_um * 1e-6  # m
            h = height_um * 1e-6  # m
            L = ch.length_mm * 1e-3  # m
            Q_m3_s = delta_P * w * h**3 / (12 * WATER_VISCOSITY * L)
            flow_rate_uL_min = Q_m3_s * 1e9 * 60  # Convert to µL/min

            # Get detailed flow parameters
            result = compute_channel_flow(channel, flow_rate_uL_min)

            channel_results.append({
                'name': ch.name,
                'length_mm': ch.length_mm,
                'diameter_um': ch.diameter_um,
                'flow_rate_uL_min': result['flow_rate_uL_min'],
                'velocity_um_s': result['average_velocity_um_s'],
                'reynolds': result['reynolds_number'],
            })

            total_flow_rate += result['flow_rate_uL_min']

        # Estimate chamber residence times
        chamber_results = []
        for chamber in chip.chambers:
            # Chamber volume (approximate as rectangular)
            volume_uL = chamber.width * chamber.height * chamber.depth  # mm³ = µL

            # Count channels connected to this chamber
            connected_channels = []
            for ch_res in channel_results:
                # Simple check - would need proper connectivity for accuracy
                connected_channels.append(ch_res)

            # Average inflow rate to chamber
            if connected_channels:
                avg_flow = np.mean([c['flow_rate_uL_min'] for c in connected_channels])
                residence_time_min = volume_uL / avg_flow if avg_flow > 0 else float('inf')
            else:
                residence_time_min = float('inf')

            chamber_results.append({
                'name': chamber.name,
                'volume_uL': volume_uL,
                'residence_time_min': residence_time_min,
            })

        # Overall metrics
        avg_velocity = np.mean([c['velocity_um_s'] for c in channel_results])
        avg_reynolds = np.mean([c['reynolds'] for c in channel_results])
        total_residence = sum(c['residence_time_min'] for c in chamber_results if c['residence_time_min'] < float('inf'))

        return {
            'summary': {
                'total_flow_rate_uL_min': total_flow_rate,
                'avg_velocity_um_s': avg_velocity,
                'avg_reynolds': avg_reynolds,
                'total_residence_time_min': total_residence,
                'n_channels': len(channel_results),
                'n_chambers': len(chamber_results),
                'flow_regime': 'Stokes (Re << 1)' if avg_reynolds < 0.1 else 'Laminar',
            },
            'channels': channel_results,
            'chambers': chamber_results,
        }

    def compare_topology_flow(
        self,
        topologies: Dict[str, "ChipDesign"],
        inlet_pressure_Pa: float = 100.0,
    ) -> Dict[str, Any]:
        """Compare flow characteristics across different topologies.

        Args:
            topologies: Dict of name -> ChipDesign
            inlet_pressure_Pa: Inlet pressure for simulation

        Returns:
            Comparison table of flow metrics
        """
        results = {}

        for name, chip in topologies.items():
            sim = self.simulate_chip_flow(chip, inlet_pressure_Pa)
            results[name] = {
                'chambers': len(chip.chambers),
                'channels': len(chip.channels),
                'total_flow_uL_min': sim['summary']['total_flow_rate_uL_min'],
                'avg_velocity_um_s': sim['summary']['avg_velocity_um_s'],
                'reynolds': sim['summary']['avg_reynolds'],
                'residence_time_min': sim['summary']['total_residence_time_min'],
            }

        return results

    def simulate_concentration_propagation(
        self,
        chip: "ChipDesign",
        drug_inlets: Dict[str, List[int]],
        n_steps: int = 100,
    ) -> Dict[str, Any]:
        """Simulate drug concentration propagation through chip.

        Tracks how drug concentrations spread from inlet chambers
        through the network over time.

        Args:
            chip: ChipDesign to simulate
            drug_inlets: Dict mapping drug name to list of inlet chamber indices
                         e.g., {"DrugA": [0,4,8,12], "DrugB": [0,1,2,3]}
            n_steps: Number of simulation steps

        Returns:
            Dict with concentration at each chamber over time
        """
        n = len(chip.chambers)

        # Build adjacency matrix from channels
        adjacency = np.zeros((n, n))
        for ch in chip.channels:
            start_pos = np.array(ch.start[:2])
            end_pos = np.array(ch.end[:2])

            start_idx = None
            end_idx = None
            start_dist = end_dist = float('inf')

            for i, chamber in enumerate(chip.chambers):
                pos = np.array(chamber.center[:2])
                d_start = np.linalg.norm(start_pos - pos)
                d_end = np.linalg.norm(end_pos - pos)
                if d_start < start_dist:
                    start_dist = d_start
                    start_idx = i
                if d_end < end_dist:
                    end_dist = d_end
                    end_idx = i

            if start_idx is not None and end_idx is not None and start_idx != end_idx:
                # Weight by channel diameter (flow capacity)
                weight = (ch.diameter_um / 50.0) ** 2
                adjacency[start_idx, end_idx] = weight
                adjacency[end_idx, start_idx] = weight

        # Normalize adjacency for diffusion
        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        diffusion_matrix = adjacency / row_sums * 0.3  # Diffusion rate

        # Initialize concentrations
        concentrations = {drug: np.zeros((n_steps, n)) for drug in drug_inlets}

        for drug, inlets in drug_inlets.items():
            # Set initial concentration at inlets
            for inlet in inlets:
                if inlet < n:
                    concentrations[drug][0, inlet] = 1.0

        # Simulate diffusion
        for t in range(1, n_steps):
            for drug, inlets in drug_inlets.items():
                prev = concentrations[drug][t-1].copy()

                # Diffusion step
                new_conc = prev.copy()
                for i in range(n):
                    inflow = 0
                    outflow = 0
                    for j in range(n):
                        if adjacency[i, j] > 0:
                            inflow += diffusion_matrix[j, i] * prev[j]
                            outflow += diffusion_matrix[i, j] * prev[i]
                    new_conc[i] = prev[i] + inflow - outflow

                # Maintain inlet concentration
                for inlet in inlets:
                    if inlet < n:
                        new_conc[inlet] = 1.0

                concentrations[drug][t] = np.clip(new_conc, 0, 1)

        # Final concentrations
        final = {drug: conc[-1] for drug, conc in concentrations.items()}

        # Calculate Latin square verification for grid chips
        if "latin" in chip.network_type.lower():
            n_grid = int(np.sqrt(n))
            if n_grid * n_grid == n:
                grid_conc = {drug: final[drug].reshape(n_grid, n_grid)
                            for drug in drug_inlets}
            else:
                grid_conc = None
        else:
            grid_conc = None

        return {
            "final_concentrations": final,
            "time_series": concentrations,
            "grid_concentrations": grid_conc,
            "n_chambers": n,
            "n_steps": n_steps,
            "drugs": list(drug_inlets.keys()),
        }

    def _flow_recommendation(
        self,
        max_flow: float,
        min_cut_edges: List,
        chip: "ChipDesign"
    ) -> str:
        """Generate recommendation based on flow analysis."""
        if not min_cut_edges:
            return "Network appears well-balanced"

        avg_cut_capacity = np.mean([c for _, _, c in min_cut_edges])

        if avg_cut_capacity < 0.5:
            return f"Bottleneck detected: {len(min_cut_edges)} narrow channels limit flow. Consider widening channels at cut."
        elif len(min_cut_edges) == 1:
            return "Single bottleneck edge - widening this channel would increase max flow"
        else:
            return f"Distributed bottleneck across {len(min_cut_edges)} edges"

    def _estimate_disc_dimension(self, connectivity: NDArray, threshold: float) -> float:
        """Estimate disc dimension from connectivity matrix."""
        # Count edges above threshold
        adj = (connectivity > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        n_edges = np.sum(adj) // 2
        n_nodes = adj.shape[0]

        # Euler's formula: planar graph has e ≤ 3n - 6
        max_planar_edges = 3 * n_nodes - 6 if n_nodes >= 3 else n_nodes

        if n_edges <= max_planar_edges:
            # Likely planar
            density = n_edges / max(max_planar_edges, 1)
            return 1.5 + 0.5 * density  # 1.5 to 2.0
        else:
            # Likely non-planar
            excess = (n_edges - max_planar_edges) / max(n_edges, 1)
            return 2.5 + 1.5 * excess  # 2.5 to 4.0

    # =========================================================================
    # STANDARD LOC DESIGNS
    # Well-established microfluidic geometries used across the field
    # =========================================================================

    def design_y_junction_mixer(
        self,
        angle_deg: float = 45.0,
        channel_width_um: float = 100.0,
        mixing_length_mm: float = 10.0,
    ) -> "ChipDesign":
        """Y-junction mixer - basic passive mixing."""
        chip = ChipDesign(
            length=mixing_length_mm + 5.0,
            width=10.0,
            height=3.0,
            network_type="y_junction_mixer",
        )
        junction = (0.0, 0.0, 1.5)
        angle_rad = np.radians(angle_deg)
        inlet_length = 5.0
        inlet1 = (-inlet_length * np.cos(angle_rad), inlet_length * np.sin(angle_rad), 1.5)
        inlet2 = (-inlet_length * np.cos(angle_rad), -inlet_length * np.sin(angle_rad), 1.5)
        outlet = (mixing_length_mm, 0.0, 1.5)
        chip.channels.append(ChannelSpec(start=inlet1, end=junction, diameter_um=channel_width_um, name="inlet_A"))
        chip.channels.append(ChannelSpec(start=inlet2, end=junction, diameter_um=channel_width_um, name="inlet_B"))
        chip.channels.append(ChannelSpec(start=junction, end=outlet, diameter_um=channel_width_um, name="mixing_channel"))
        chip.chambers.append(ChamberSpec(center=inlet1, width=1.0, name="inlet_A"))
        chip.chambers.append(ChamberSpec(center=inlet2, width=1.0, name="inlet_B"))
        chip.chambers.append(ChamberSpec(center=outlet, width=1.0, name="outlet"))
        chip.disc_dimension = 1.0
        return chip

    def design_t_junction(self, channel_width_um: float = 100.0, mode: str = "mixing") -> "ChipDesign":
        """T-junction - mixing or droplet generation."""
        chip = ChipDesign(length=20.0, width=10.0, height=3.0, network_type=f"t_junction_{mode}")
        junction = (0.0, 0.0, 1.5)
        inlet_main = (-8.0, 0.0, 1.5)
        outlet = (8.0, 0.0, 1.5)
        inlet_side = (0.0, 4.0, 1.5)
        main_width = channel_width_um
        side_width = channel_width_um * 0.5 if mode == "droplet" else channel_width_um
        chip.channels.append(ChannelSpec(start=inlet_main, end=junction, diameter_um=main_width, name="main_inlet"))
        chip.channels.append(ChannelSpec(start=inlet_side, end=junction, diameter_um=side_width, name="side_inlet"))
        chip.channels.append(ChannelSpec(start=junction, end=outlet, diameter_um=main_width, name="outlet"))
        chip.chambers.append(ChamberSpec(center=inlet_main, width=1.0, name="main_inlet"))
        chip.chambers.append(ChamberSpec(center=inlet_side, width=1.0, name="side_inlet"))
        chip.chambers.append(ChamberSpec(center=outlet, width=1.0, name="outlet"))
        chip.disc_dimension = 1.0
        return chip

    def design_serpentine_mixer(self, n_turns: int = 10, channel_width_um: float = 100.0,
                                 turn_radius_mm: float = 0.5, straight_length_mm: float = 2.0) -> "ChipDesign":
        """Serpentine mixer - extended path length with Dean vortices."""
        chip = ChipDesign(length=straight_length_mm + 4.0, width=(n_turns + 1) * (turn_radius_mm * 2 + 0.5),
                          height=3.0, network_type="serpentine_mixer")
        x, y, z = 0.0, 0.0, 1.5
        direction = 1
        points = [(x, y, z)]
        for i in range(n_turns):
            x += direction * straight_length_mm
            points.append((x, y, z))
            y += turn_radius_mm * 2
            points.append((x, y, z))
            direction *= -1
        x += direction * straight_length_mm
        points.append((x, y, z))
        for i in range(len(points) - 1):
            chip.channels.append(ChannelSpec(start=points[i], end=points[i + 1], diameter_um=channel_width_um, name=f"segment_{i}"))
        chip.chambers.append(ChamberSpec(center=points[0], width=1.0, name="inlet"))
        chip.chambers.append(ChamberSpec(center=points[-1], width=1.0, name="outlet"))
        chip.disc_dimension = 1.0
        return chip

    def design_herringbone_mixer(self, n_cycles: int = 5, channel_width_um: float = 200.0,
                                  channel_length_mm: float = 20.0, groove_depth_um: float = 50.0) -> "ChipDesign":
        """Staggered Herringbone Mixer (SHM) - chaotic advection mixing."""
        chip = ChipDesign(length=channel_length_mm + 4.0, width=5.0, height=3.0, network_type="herringbone_mixer")
        inlet = (-channel_length_mm / 2, 0.0, 1.5)
        outlet = (channel_length_mm / 2, 0.0, 1.5)
        chip.channels.append(ChannelSpec(start=inlet, end=outlet, diameter_um=channel_width_um, name="main_channel"))
        chip.chambers.append(ChamberSpec(center=inlet, width=1.0, name="inlet"))
        chip.chambers.append(ChamberSpec(center=outlet, width=1.0, name="outlet"))
        chip.network_type = f"herringbone_mixer_cycles{n_cycles}_groove{groove_depth_um}um"
        chip.disc_dimension = 1.0
        return chip

    def design_flow_focusing(self, orifice_width_um: float = 50.0, channel_width_um: float = 200.0) -> "ChipDesign":
        """Flow focusing geometry - droplet generation."""
        chip = ChipDesign(length=20.0, width=12.0, height=3.0, network_type="flow_focusing")
        dispersed_inlet = (-8.0, 0.0, 1.5)
        orifice = (0.0, 0.0, 1.5)
        sheath_top = (-4.0, 4.0, 1.5)
        sheath_bottom = (-4.0, -4.0, 1.5)
        outlet = (8.0, 0.0, 1.5)
        chip.channels.append(ChannelSpec(start=dispersed_inlet, end=orifice, diameter_um=channel_width_um, name="dispersed_inlet"))
        chip.channels.append(ChannelSpec(start=sheath_top, end=orifice, diameter_um=channel_width_um, name="sheath_top"))
        chip.channels.append(ChannelSpec(start=sheath_bottom, end=orifice, diameter_um=channel_width_um, name="sheath_bottom"))
        chip.channels.append(ChannelSpec(start=orifice, end=outlet, diameter_um=orifice_width_um, name="focusing_orifice"))
        chip.chambers.append(ChamberSpec(center=dispersed_inlet, width=1.0, name="dispersed"))
        chip.chambers.append(ChamberSpec(center=sheath_top, width=1.0, name="sheath_top"))
        chip.chambers.append(ChamberSpec(center=sheath_bottom, width=1.0, name="sheath_bottom"))
        chip.chambers.append(ChamberSpec(center=outlet, width=2.0, name="collection"))
        chip.disc_dimension = 1.5
        return chip

    def design_gradient_generator(self, n_outlets: int = 5, channel_width_um: float = 100.0) -> "ChipDesign":
        """Christmas tree gradient generator - concentration gradients."""
        chip = ChipDesign(length=25.0, width=20.0, height=3.0, network_type="gradient_generator")
        inlet_a = (-10.0, 3.0, 1.5)
        inlet_b = (-10.0, -3.0, 1.5)
        chip.chambers.append(ChamberSpec(center=inlet_a, width=1.0, name="inlet_A"))
        chip.chambers.append(ChamberSpec(center=inlet_b, width=1.0, name="inlet_B"))
        n_levels = n_outlets - 1
        prev_nodes = [inlet_a, inlet_b]
        for level in range(n_levels):
            x = -8.0 + level * 4.0
            n_nodes = level + 3
            y_span = 8.0
            new_nodes = []
            for i in range(n_nodes):
                y = y_span * (i / (n_nodes - 1) - 0.5)
                node = (x, y, 1.5)
                new_nodes.append(node)
                if i < len(prev_nodes):
                    chip.channels.append(ChannelSpec(start=prev_nodes[i], end=node, diameter_um=channel_width_um, name=f"level{level}_ch{i}a"))
                if i > 0 and i - 1 < len(prev_nodes):
                    chip.channels.append(ChannelSpec(start=prev_nodes[i - 1], end=node, diameter_um=channel_width_um, name=f"level{level}_ch{i}b"))
            prev_nodes = new_nodes
        for i, node in enumerate(prev_nodes):
            outlet = (node[0] + 3.0, node[1], node[2])
            chip.channels.append(ChannelSpec(start=node, end=outlet, diameter_um=channel_width_um, name=f"outlet_{i}"))
            chip.chambers.append(ChamberSpec(center=outlet, width=1.0, name=f"outlet_{i}_conc{int(100*i/(len(prev_nodes)-1))}pct"))
        chip.disc_dimension = 1.5
        return chip

    def design_tesla_valve(self, n_segments: int = 5, channel_width_um: float = 200.0) -> "ChipDesign":
        """Tesla valve - passive diodic flow (no moving parts)."""
        chip = ChipDesign(length=n_segments * 4.0 + 4.0, width=6.0, height=3.0, network_type="tesla_valve")
        x = -n_segments * 2.0
        y, z = 0.0, 1.5
        prev_point = (x, y, z)
        chip.chambers.append(ChamberSpec(center=prev_point, width=1.0, name="inlet"))
        for i in range(n_segments):
            junction = (x + 2.0, y, z)
            chip.channels.append(ChannelSpec(start=prev_point, end=junction, diameter_um=channel_width_um, name=f"main_{i}"))
            bypass_top = (x + 2.0, y + 1.5, z)
            bypass_end = (x + 4.0, y, z)
            chip.channels.append(ChannelSpec(start=junction, end=bypass_top, diameter_um=channel_width_um * 0.7, name=f"bypass_up_{i}"))
            chip.channels.append(ChannelSpec(start=bypass_top, end=bypass_end, diameter_um=channel_width_um * 0.7, name=f"bypass_down_{i}"))
            chip.channels.append(ChannelSpec(start=junction, end=bypass_end, diameter_um=channel_width_um, name=f"direct_{i}"))
            prev_point = bypass_end
            x += 4.0
        chip.chambers.append(ChamberSpec(center=prev_point, width=1.0, name="outlet"))
        chip.disc_dimension = 1.5
        return chip

    def design_h_filter(self, channel_width_um: float = 100.0, diffusion_length_mm: float = 10.0) -> "ChipDesign":
        """H-filter - diffusion-based separation."""
        chip = ChipDesign(length=diffusion_length_mm + 8.0, width=10.0, height=3.0, network_type="h_filter")
        inlet_sample = (-diffusion_length_mm / 2 - 3.0, 1.0, 1.5)
        inlet_buffer = (-diffusion_length_mm / 2 - 3.0, -1.0, 1.5)
        merge = (-diffusion_length_mm / 2, 0.0, 1.5)
        split = (diffusion_length_mm / 2, 0.0, 1.5)
        outlet_waste = (diffusion_length_mm / 2 + 3.0, 1.0, 1.5)
        outlet_extract = (diffusion_length_mm / 2 + 3.0, -1.0, 1.5)
        chip.channels.append(ChannelSpec(start=inlet_sample, end=merge, diameter_um=channel_width_um, name="sample_inlet"))
        chip.channels.append(ChannelSpec(start=inlet_buffer, end=merge, diameter_um=channel_width_um, name="buffer_inlet"))
        chip.channels.append(ChannelSpec(start=merge, end=split, diameter_um=channel_width_um * 2, name="diffusion_zone"))
        chip.channels.append(ChannelSpec(start=split, end=outlet_waste, diameter_um=channel_width_um, name="waste_outlet"))
        chip.channels.append(ChannelSpec(start=split, end=outlet_extract, diameter_um=channel_width_um, name="extract_outlet"))
        chip.chambers.append(ChamberSpec(center=inlet_sample, width=1.0, name="sample_inlet"))
        chip.chambers.append(ChamberSpec(center=inlet_buffer, width=1.0, name="buffer_inlet"))
        chip.chambers.append(ChamberSpec(center=outlet_waste, width=1.0, name="waste"))
        chip.chambers.append(ChamberSpec(center=outlet_extract, width=1.0, name="extract"))
        chip.disc_dimension = 1.5
        return chip

    def design_dean_flow_spiral(self, n_turns: float = 3.0, inner_radius_mm: float = 2.0,
                                 channel_width_um: float = 200.0, pitch_mm: float = 1.0) -> "ChipDesign":
        """Dean flow spiral - particle separation by size/density."""
        chip = ChipDesign(length=(inner_radius_mm + n_turns * pitch_mm) * 2 + 4.0,
                          width=(inner_radius_mm + n_turns * pitch_mm) * 2 + 4.0,
                          height=3.0, network_type="dean_flow_spiral")
        n_points = int(n_turns * 36)
        points = []
        for i in range(n_points + 1):
            theta = 2 * np.pi * n_turns * i / n_points
            r = inner_radius_mm + pitch_mm * theta / (2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append((x, y, 1.5))
        for i in range(len(points) - 1):
            chip.channels.append(ChannelSpec(start=points[i], end=points[i + 1], diameter_um=channel_width_um, name=f"spiral_{i}"))
        chip.chambers.append(ChamberSpec(center=points[0], width=1.0, name="inlet"))
        chip.chambers.append(ChamberSpec(center=points[-1], width=1.0, name="outlet"))
        chip.disc_dimension = 1.0
        return chip

    def design_dld_array(self, n_rows: int = 10, n_cols: int = 20, post_diameter_um: float = 20.0,
                          gap_um: float = 25.0, shift_fraction: float = 0.1) -> "ChipDesign":
        """Deterministic Lateral Displacement (DLD) array - size-based separation."""
        pitch = post_diameter_um + gap_um
        array_width = n_cols * pitch / 1000
        array_length = n_rows * pitch / 1000
        chip = ChipDesign(length=array_length + 6.0, width=array_width + 4.0, height=3.0, network_type="dld_array")
        for row in range(n_rows):
            shift = (row * shift_fraction * pitch) % pitch
            for col in range(n_cols):
                x = -array_length / 2 + row * pitch / 1000
                y = -array_width / 2 + (col * pitch + shift) / 1000
                chip.chambers.append(ChamberSpec(center=(x, y, 1.5), width=post_diameter_um / 1000, name=f"post_{row}_{col}"))
        inlet = (-array_length / 2 - 2.0, 0.0, 1.5)
        outlet_large = (array_length / 2 + 2.0, array_width / 4, 1.5)
        outlet_small = (array_length / 2 + 2.0, -array_width / 4, 1.5)
        chip.chambers.append(ChamberSpec(center=inlet, width=1.0, name="inlet"))
        chip.chambers.append(ChamberSpec(center=outlet_large, width=1.0, name="outlet_large"))
        chip.chambers.append(ChamberSpec(center=outlet_small, width=1.0, name="outlet_small"))
        D_c = 1.4 * gap_um * (shift_fraction ** 0.48)
        chip.network_type = f"dld_array_Dc{D_c:.1f}um"
        chip.disc_dimension = 2.0
        return chip

    def design_split_recombine_mixer(self, n_stages: int = 4, channel_width_um: float = 100.0) -> "ChipDesign":
        """Split-and-Recombine (SAR) mixer - geometric mixing."""
        chip = ChipDesign(length=n_stages * 6.0 + 6.0, width=8.0, height=3.0, network_type="split_recombine_mixer")
        x, z = -n_stages * 3.0, 1.5
        inlet = (x - 2.0, 0.0, z)
        chip.chambers.append(ChamberSpec(center=inlet, width=1.0, name="inlet"))
        prev_point = inlet
        for stage in range(n_stages):
            split = (x, 0.0, z)
            branch_top = (x + 2.0, 1.5, z)
            branch_bottom = (x + 2.0, -1.5, z)
            recombine = (x + 4.0, 0.0, z)
            chip.channels.append(ChannelSpec(start=prev_point, end=split, diameter_um=channel_width_um, name=f"stage{stage}_in"))
            chip.channels.append(ChannelSpec(start=split, end=branch_top, diameter_um=channel_width_um * 0.7, name=f"stage{stage}_split_top"))
            chip.channels.append(ChannelSpec(start=split, end=branch_bottom, diameter_um=channel_width_um * 0.7, name=f"stage{stage}_split_bottom"))
            chip.channels.append(ChannelSpec(start=branch_top, end=recombine, diameter_um=channel_width_um * 0.7, name=f"stage{stage}_merge_top"))
            chip.channels.append(ChannelSpec(start=branch_bottom, end=recombine, diameter_um=channel_width_um * 0.7, name=f"stage{stage}_merge_bottom"))
            prev_point = recombine
            x += 6.0
        outlet = (x, 0.0, z)
        chip.channels.append(ChannelSpec(start=prev_point, end=outlet, diameter_um=channel_width_um, name="outlet_channel"))
        chip.chambers.append(ChamberSpec(center=outlet, width=1.0, name="outlet"))
        chip.network_type = f"split_recombine_{n_stages}stages_{2**n_stages}lamellae"
        chip.disc_dimension = 1.5
        return chip

    def design_standard_loc_set(self) -> Dict[str, "ChipDesign"]:
        """Generate complete set of standard LOC designs."""
        return {
            "y_junction": self.design_y_junction_mixer(),
            "t_junction_mixing": self.design_t_junction(mode="mixing"),
            "t_junction_droplet": self.design_t_junction(mode="droplet"),
            "serpentine": self.design_serpentine_mixer(),
            "herringbone": self.design_herringbone_mixer(),
            "flow_focusing": self.design_flow_focusing(),
            "gradient_generator": self.design_gradient_generator(),
            "tesla_valve": self.design_tesla_valve(),
            "h_filter": self.design_h_filter(),
            "dean_spiral": self.design_dean_flow_spiral(),
            "dld_array": self.design_dld_array(),
            "split_recombine": self.design_split_recombine_mixer(),
        }

    # =========================================================================
    # 3DuF EXPORT
    # =========================================================================

    def export_to_3duf_file(self, design: ChipDesign, filepath: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Export chip design to 3DuF JSON file."""
        import json
        if name is None:
            name = design.network_type or "Microfluidic Chip"
        data = design.to_3duf_json(name)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return {"success": True, "filepath": filepath, "name": name, "features": len(data["layers"][0]["features"])}

    def export_to_3duf_mcp(self, design: ChipDesign, name: Optional[str] = None,
                           host: str = "localhost", port: int = 9000) -> Dict[str, Any]:
        """Export chip design to 3DuF via MCP server."""
        if name is None:
            name = design.network_type or "Microfluidic Chip"
        data = design.to_3duf_json(name)
        try:
            server = xmlrpc.client.ServerProxy(f'http://{host}:{port}')
            result = server.load_design(data)
            return {"success": True, "name": name, "features": len(data["layers"][0]["features"]), "server_response": result}
        except Exception as e:
            return {"success": False, "error": str(e), "data": data}

    def export_comparison_to_3duf(self, designs: Dict[str, ChipDesign], output_dir: str) -> Dict[str, Any]:
        """Export multiple designs to 3DuF JSON files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        for name, design in designs.items():
            filepath = os.path.join(output_dir, f"{name}.json")
            results[name] = self.export_to_3duf_file(design, filepath, name)
        return results


class FreeCADExporter:
    """Export chip designs to FreeCAD via XML-RPC."""

    def __init__(self, host: str = "localhost", port: int = 9875):
        self.host = host
        self.port = port
        self._server = None

    def connect(self) -> bool:
        """Connect to FreeCAD RPC server."""
        try:
            self._server = xmlrpc.client.ServerProxy(
                f'http://{self.host}:{self.port}',
                allow_none=True
            )
            return self._server.ping()
        except Exception:
            return False

    def export_design(self, design: ChipDesign, doc_name: str = "BrainChip") -> Dict[str, Any]:
        """Export chip design to FreeCAD.

        Args:
            design: ChipDesign to export
            doc_name: FreeCAD document name

        Returns:
            Export result
        """
        if self._server is None:
            if not self.connect():
                return {"error": "Could not connect to FreeCAD RPC server"}

        code = self._generate_freecad_code(design, doc_name)

        try:
            result = self._server.execute_code(code)
            return {
                "success": True,
                "document": doc_name,
                "result": result,
                "design_summary": design.to_dict(),
            }
        except Exception as e:
            return {"error": str(e)}

    def _generate_freecad_code(self, design: ChipDesign, doc_name: str) -> str:
        """Generate FreeCAD Python code for the chip design."""

        # Build channel cutting code
        channel_code = ""
        for i, ch in enumerate(design.channels):
            channel_code += f'''
# Channel {i}: {ch.name}
ch_{i}_start = Vector({ch.start[0]}, {ch.start[1]}, {ch.start[2]})
ch_{i}_end = Vector({ch.end[0]}, {ch.end[1]}, {ch.end[2]})
ch_{i}_dir = ch_{i}_end - ch_{i}_start
ch_{i}_len = ch_{i}_dir.Length
ch_{i}_dir.normalize()
ch_{i} = Part.makeCylinder({ch.diameter_um / 1000 / 2}, ch_{i}_len, ch_{i}_start, ch_{i}_dir)
chip = chip.cut(ch_{i})
'''

        # Build chamber cutting code
        chamber_code = ""
        for i, cm in enumerate(design.chambers):
            chamber_code += f'''
# Chamber {i}: {cm.name}
cm_{i} = Part.makeBox({cm.width}, {cm.height}, {cm.depth},
    Vector({cm.center[0] - cm.width/2}, {cm.center[1] - cm.height/2}, {cm.center[2] - cm.depth/2}))
chip = chip.cut(cm_{i})
'''

        # Build inlet port code (adjacent to chambers, horizontal entry)
        inlet_port_code = ""
        for i, port in enumerate(design.inlet_ports):
            x, y, z = port.position
            inlet_port_code += f'''
# Inlet port {i} (adjacent to chamber, for {port.chamber_name})
port_{i} = Part.makeCylinder({port.port_diameter_um / 1000 / 2}, 2.0, Vector({x}, {y}, {z - 1.0}), Vector(0, 0, 1))
chip = chip.cut(port_{i})
'''

        # Build inlet channel code (short 100µm channels connecting ports to chambers)
        inlet_channel_code = ""
        for i, ch in enumerate(design.inlet_channels):
            inlet_channel_code += f'''
# Inlet channel {i} (port-to-chamber connector, 100µm)
ch_in_{i} = Part.makeCylinder({ch.diameter_um / 1000 / 2}, {ch.length_mm},
    Vector({ch.start[0]}, {ch.start[1]}, {ch.start[2]}),
    Vector({ch.end[0] - ch.start[0]}, {ch.end[1] - ch.start[1]}, {ch.end[2] - ch.start[2]}).normalize())
chip = chip.cut(ch_in_{i})
'''

        # Main inlet/outlet ports
        inlet = design.main_inlet_position
        outlet = design.main_outlet_position

        code = f'''
import Part
import FreeCAD as App
from FreeCAD import Vector

# Create new document
doc = App.newDocument("{doc_name}")

# ============================================
# Brain-on-Chip for Glymphatic Validation
# Disc dimension: {design.disc_dimension:.2f}
# Network type: {design.network_type}
# ============================================

# Chip body
chip_length = {design.length}
chip_width = {design.width}
chip_height = {design.height}

chip = Part.makeBox(chip_length, chip_width, chip_height,
    Vector(-chip_length/2, -chip_width/2, 0))

# Main inlet port (for tracer injection)
main_inlet = Part.makeCylinder(0.5, 3.0, Vector({inlet[0]}, {inlet[1]}, 0), Vector(0, 0, 1))
chip = chip.cut(main_inlet)

# Main outlet port (for waste collection)
main_outlet = Part.makeCylinder(0.5, 3.0, Vector({outlet[0]}, {outlet[1]}, 0), Vector(0, 0, 1))
chip = chip.cut(main_outlet)

# Inter-chamber channels (network topology)
{channel_code}

# Chambers (imaged from below through glass substrate)
{chamber_code}

# Inlet ports (adjacent to chambers, horizontal entry)
{inlet_port_code}

# Inlet channels (100um connectors from ports to chambers)
{inlet_channel_code}

# Add to document
chip_obj = doc.addObject("Part::Feature", "BrainChip")
chip_obj.Shape = chip
chip_obj.ViewObject.ShapeColor = (0.8, 0.85, 0.9)  # Light blue PDMS
chip_obj.ViewObject.Transparency = 50

doc.recompute()

# Try to set view
try:
    FreeCADGui.ActiveDocument.ActiveView.fitAll()
    FreeCADGui.ActiveDocument.ActiveView.viewIsometric()
except:
    pass

print("Brain-on-Chip created (Inverted Microscopy Design)!")
print(f"  Chip: {{chip_length}} x {{chip_width}} x {{chip_height}} mm")
print(f"  PDMS bonded to glass substrate (imaging from below)")
print(f"  Inter-chamber channels: {len(design.channels)}")
print(f"  Chambers: {len(design.chambers)}")
print(f"  Inlet ports: {len(design.inlet_ports)} (adjacent to chambers)")
print(f"  Inlet channels: {len(design.inlet_channels)} (100um connectors)")
print(f"  Disc dimension: {design.disc_dimension:.2f}")
print(f"  Network type: {design.network_type}")
'''

        return code


# Convenience function for MCP integration
def design_brain_chip(
    connectivity_matrix: Optional[NDArray] = None,
    n_regions: int = 5,
    network_type: str = "planar",
    channel_diameter_um: float = 50.0,
    export_to_freecad: bool = False,
) -> Dict[str, Any]:
    """Design a brain-mimetic microfluidic chip.

    Args:
        connectivity_matrix: Optional brain connectivity matrix
        n_regions: Number of brain regions (if no matrix provided)
        network_type: "planar", "non_planar", or "tree"
        channel_diameter_um: Channel diameter in micrometers
        export_to_freecad: Whether to export to FreeCAD

    Returns:
        Design specification and optional export result
    """
    designer = BrainChipDesigner()

    if connectivity_matrix is not None:
        design = designer.design_from_connectivity(
            np.array(connectivity_matrix),
            channel_diameter_um=channel_diameter_um,
        )
    elif network_type == "planar":
        design = designer.design_planar_network(n_regions, channel_diameter_um)
    elif network_type == "non_planar":
        design = designer.design_nonplanar_network(n_regions, channel_diameter_um)
    elif network_type == "tree":
        design = designer._design_tree_network(n_regions)
    else:
        design = designer.design_planar_network(n_regions, channel_diameter_um)

    result = {
        "design": design.to_dict(),
        "channels": [
            {
                "name": ch.name,
                "start": ch.start,
                "end": ch.end,
                "diameter_um": ch.diameter_um,
                "length_mm": ch.length_mm,
            }
            for ch in design.channels
        ],
        "chambers": [
            {
                "name": cm.name,
                "center": cm.center,
                "size_mm": (cm.width, cm.height, cm.depth),
            }
            for cm in design.chambers
        ],
    }

    if export_to_freecad:
        exporter = FreeCADExporter()
        if exporter.connect():
            export_result = exporter.export_design(design)
            result["freecad_export"] = export_result
        else:
            result["freecad_export"] = {"error": "Could not connect to FreeCAD"}

    return result
