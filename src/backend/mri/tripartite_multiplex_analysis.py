"""Tripartite multiplex network analysis with quantum operator framework.

Integrates:
1. Tripartite P3 cover algorithm (from merge2docs)
2. Quantum network operators (SymPy framework)
3. Three-layer disc dimension analysis

Three-layer brain network model:
- Layer A (tripartite=0): Signal/Electrical (synaptic connectivity)
- Layer B (tripartite=1): Photonic/Optical (biophoton propagation)
- Layer C (tripartite=2): Lymphatic/Fluid (glymphatic clearance)

P3 paths: A → B → C represent information flow through layers
- Signal activates photonic response → triggers lymphatic clearance
- Minimal dominating set = Critical hubs controlling multi-layer dynamics

References:
- merge2docs tripartite_cover.py (P3 domination algorithm)
- quantum_network_operators.py (SymPy operator framework)
- docs/papers/BIOLOGICAL_COMPUTATIONAL_TRACTABILITY_PRINCIPLE.md (Fellows' principle)
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import networkx as nx
from sympy import Matrix

from .disc_dimension_analysis import (
    DiscDimensionPredictor,
    MultiplexDiscAnalyzer
)
from .quantum_network_operators import (
    QuantumNetworkState,
    ObstructionOperator,
    QTRMLevelTransitionOperator,
    disc_dimension_via_eigenspectrum
)


class TripartiteMultiplexAnalyzer:
    """Analyze three-layer multiplex networks with quantum operators.

    Combines:
    - P3 cover algorithm for layer connectivity
    - Quantum operators for obstruction detection
    - Disc dimension analysis per layer
    - QTRM transitions between layers

    Example:
        >>> analyzer = TripartiteMultiplexAnalyzer()
        >>> result = await analyzer.analyze_tripartite_multiplex(
        ...     G_signal, G_photonic, G_lymph,
        ...     cross_AB_edges, cross_BC_edges
        ... )
        >>> print(f"Critical hubs: {result['p3_dominating_set']}")
        >>> print(f"Layer A disc: {result['layer_A']['disc_consensus']}")
    """

    def __init__(self):
        self.disc_predictor = DiscDimensionPredictor()

    async def analyze_tripartite_multiplex(
        self,
        G_A: nx.Graph,  # Signal/electrical layer
        G_B: nx.Graph,  # Photonic/optical layer
        G_C: nx.Graph,  # Lymphatic/fluid layer
        cross_AB_edges: List[Tuple[Any, Any]],
        cross_BC_edges: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Comprehensive analysis of three-layer multiplex network.

        Args:
            G_A: Layer A graph (signal/electrical)
            G_B: Layer B graph (photonic/optical)
            G_C: Layer C graph (lymphatic/fluid)
            cross_AB_edges: Edges connecting A ↔ B
            cross_BC_edges: Edges connecting B ↔ C

        Returns:
            Dict with:
            - layer_A/B/C: Per-layer disc dimension analysis
            - p3_dominating_set: Critical nodes controlling all layers
            - quantum_operators: Obstruction operators per layer
            - qtrm_transitions: Layer transition operators (A→B, B→C)
            - effective_dimension: Three-layer d_eff calculation
            - tractability: Fellows' principle validation
        """
        results = {}

        # 1. Per-layer disc dimension analysis
        results['layer_A'] = await self.disc_predictor.predict_consensus(G_A)
        results['layer_B'] = await self.disc_predictor.predict_consensus(G_B)
        results['layer_C'] = await self.disc_predictor.predict_consensus(G_C)

        # 2. Build tripartite graph for P3 cover
        G_tripartite = self._build_tripartite_graph(
            G_A, G_B, G_C,
            cross_AB_edges, cross_BC_edges
        )

        # 3. Find P3 dominating set
        p3_result = await self._compute_p3_dominating_set(G_tripartite)
        results['p3_dominating_set'] = p3_result['a_cover_set']
        results['p3_paths'] = p3_result['p3_paths']
        results['p3_coverage'] = p3_result['coverage_percentage']

        # 4. Quantum operator analysis
        results['quantum_operators'] = await self._analyze_quantum_operators(
            G_A, G_B, G_C
        )

        # 5. QTRM layer transitions
        results['qtrm_transitions'] = self._compute_qtrm_transitions(
            G_A, G_B, G_C
        )

        # 6. Effective dimension (three layers)
        results['effective_dimension'] = self._compute_tripartite_effective_dimension(
            G_A, G_B, G_C,
            cross_AB_edges, cross_BC_edges
        )

        # 7. Tractability analysis
        results['tractability'] = await self._analyze_tractability(
            G_A, G_B, G_C,
            cross_AB_edges, cross_BC_edges
        )

        return results

    def _build_tripartite_graph(
        self,
        G_A: nx.Graph,
        G_B: nx.Graph,
        G_C: nx.Graph,
        cross_AB_edges: List[Tuple[Any, Any]],
        cross_BC_edges: List[Tuple[Any, Any]]
    ) -> nx.Graph:
        """Build tripartite graph from three layers.

        Node labeling:
        - A nodes: tripartite=0
        - B nodes: tripartite=1
        - C nodes: tripartite=2

        Edges:
        - Intra-layer: A-A, B-B, C-C
        - Inter-layer: A-B, B-C (P3 structure)

        Returns:
            Tripartite NetworkX graph
        """
        G_tri = nx.Graph()

        # Add layer A nodes
        for node in G_A.nodes():
            G_tri.add_node(f"A_{node}", tripartite=0, layer='A', original_id=node)

        # Add layer B nodes
        for node in G_B.nodes():
            G_tri.add_node(f"B_{node}", tripartite=1, layer='B', original_id=node)

        # Add layer C nodes
        for node in G_C.nodes():
            G_tri.add_node(f"C_{node}", tripartite=2, layer='C', original_id=node)

        # Add intra-layer edges (within each layer)
        for u, v in G_A.edges():
            G_tri.add_edge(f"A_{u}", f"A_{v}", edge_type='intra_A')

        for u, v in G_B.edges():
            G_tri.add_edge(f"B_{u}", f"B_{v}", edge_type='intra_B')

        for u, v in G_C.edges():
            G_tri.add_edge(f"C_{u}", f"C_{v}", edge_type='intra_C')

        # Add inter-layer edges (A → B, B → C)
        for u, v in cross_AB_edges:
            G_tri.add_edge(f"A_{u}", f"B_{v}", edge_type='cross_AB')

        for u, v in cross_BC_edges:
            G_tri.add_edge(f"B_{u}", f"C_{v}", edge_type='cross_BC')

        return G_tri

    async def _compute_p3_dominating_set(
        self,
        G_tripartite: nx.Graph
    ) -> Dict[str, Any]:
        """Compute P3 dominating set: minimal A nodes covering C through B.

        Uses greedy algorithm:
        1. For each node a ∈ A, count how many c ∈ C it can reach via B
        2. Greedily select a with maximum coverage
        3. Remove covered c nodes
        4. Repeat until target coverage reached

        Returns:
            Dict with a_cover_set, p3_paths, coverage_percentage
        """
        def _compute():
            # Extract tripartite sets
            A_nodes = {n for n, d in G_tripartite.nodes(data=True)
                      if d.get('tripartite') == 0}
            B_nodes = {n for n, d in G_tripartite.nodes(data=True)
                      if d.get('tripartite') == 1}
            C_nodes = {n for n, d in G_tripartite.nodes(data=True)
                      if d.get('tripartite') == 2}

            # Build adjacency structure for efficient P3 path queries
            # Instead of materializing all paths, build a lookup: a -> {c reachable via P3}
            a_to_c_reachable = {}
            for a in A_nodes:
                reachable_c = set()
                b_neighbors = set(G_tripartite.neighbors(a)) & B_nodes
                for b in b_neighbors:
                    c_neighbors = set(G_tripartite.neighbors(b)) & C_nodes
                    reachable_c.update(c_neighbors)
                a_to_c_reachable[a] = reachable_c

            # Greedy cover algorithm (memory-efficient)
            a_cover_set = set()
            c_covered = set()
            c_uncovered = C_nodes.copy()

            while c_uncovered and len(a_cover_set) < len(A_nodes):
                # Find a ∈ A with maximum coverage of uncovered c
                best_a = None
                best_coverage = 0

                for a in A_nodes - a_cover_set:
                    # Count how many uncovered c this a can reach
                    reachable_c = a_to_c_reachable[a] & c_uncovered

                    if len(reachable_c) > best_coverage:
                        best_a = a
                        best_coverage = len(reachable_c)

                if best_a is None:
                    break

                # Add best_a to cover set
                a_cover_set.add(best_a)

                # Mark c nodes as covered
                newly_covered = a_to_c_reachable[best_a] & c_uncovered
                c_covered.update(newly_covered)
                c_uncovered -= newly_covered

            # Reconstruct p3_paths for result (only for nodes in cover)
            p3_paths = []
            for a in a_cover_set:
                b_neighbors = set(G_tripartite.neighbors(a)) & B_nodes
                for b in b_neighbors:
                    c_neighbors = set(G_tripartite.neighbors(b)) & C_nodes
                    for c in c_neighbors & c_covered:
                        p3_paths.append((a, b, c))

            coverage_percentage = len(c_covered) / len(C_nodes) if C_nodes else 0

            return {
                'a_cover_set': a_cover_set,
                'b_intermediates': set([b for a, b, c in p3_paths if a in a_cover_set]),
                'c_covered': c_covered,
                'c_uncovered': c_uncovered,
                'p3_paths': p3_paths,
                'coverage_percentage': coverage_percentage
            }

        return await asyncio.to_thread(_compute)

    async def _analyze_quantum_operators(
        self,
        G_A: nx.Graph,
        G_B: nx.Graph,
        G_C: nx.Graph
    ) -> Dict[str, Any]:
        """Analyze obstructions using quantum operators.

        For each layer, construct ObstructionOperator and detect K₅, K₃,₃.

        Returns:
            Dict with operators and eigenvalues per layer
        """
        def _analyze():
            results = {}

            for layer_name, G in [('A', G_A), ('B', G_B), ('C', G_C)]:
                # K₅ obstruction
                k5_op = ObstructionOperator('K5')
                k5_result = k5_op.detect(G)

                # K₃,₃ obstruction
                k33_op = ObstructionOperator('K33')
                k33_result = k33_op.detect(G)

                # Eigenspectrum analysis
                spectral = disc_dimension_via_eigenspectrum(G)

                results[layer_name] = {
                    'k5_obstruction': k5_result,
                    'k33_obstruction': k33_result,
                    'eigenspectrum': spectral,
                    'has_obstructions': k5_result['has_obstruction'] or k33_result['has_obstruction']
                }

            return results

        return await asyncio.to_thread(_analyze)

    def _compute_qtrm_transitions(
        self,
        G_A: nx.Graph,
        G_B: nx.Graph,
        G_C: nx.Graph
    ) -> Dict[str, Any]:
        """Compute QTRM transition operators between layers.

        Models layer transitions as unitary operators:
        - U_AB: A → B transition
        - U_BC: B → C transition
        - U_AC: Composed A → B → C

        Returns:
            Dict with transition operators and unitarity checks
        """
        # Transition A → B
        U_AB = QTRMLevelTransitionOperator(source=0, target=1, coupling=0.5)

        # Transition B → C
        U_BC = QTRMLevelTransitionOperator(source=1, target=2, coupling=0.5)

        # Composed transition A → C
        U_AC = QTRMLevelTransitionOperator(source=0, target=2, coupling=0.7)

        return {
            'U_AB': {
                'source': 'A (signal)',
                'target': 'B (photonic)',
                'unitary': U_AB.is_unitary(),
                'coupling': 0.5
            },
            'U_BC': {
                'source': 'B (photonic)',
                'target': 'C (lymphatic)',
                'unitary': U_BC.is_unitary(),
                'coupling': 0.5
            },
            'U_AC': {
                'source': 'A (signal)',
                'target': 'C (lymphatic)',
                'unitary': U_AC.is_unitary(),
                'coupling': 0.7,
                'note': 'Composed A→B→C transition'
            }
        }

    def _compute_tripartite_effective_dimension(
        self,
        G_A: nx.Graph,
        G_B: nx.Graph,
        G_C: nx.Graph,
        cross_AB_edges: List[Tuple[Any, Any]],
        cross_BC_edges: List[Tuple[Any, Any]]
    ) -> Dict[str, float]:
        """Compute effective dimension for three-layer multiplex.

        Formula (extended from ernie2 Q6):
        d_eff = d_layer + log₂(L) + C_coupling

        For L=3 layers:
        d_eff = 2 + log₂(3) + C_coupling = 2 + 1.585 + C ≈ 4.1-4.6

        Returns:
            Dict with d_layer, log2_L, C_coupling, d_eff
        """
        # Assume each layer on 2-sphere
        d_layer = 2

        # Number of layers
        L = 3

        # Coupling complexity
        E_A = G_A.number_of_edges()
        E_B = G_B.number_of_edges()
        E_C = G_C.number_of_edges()
        E_AB = len(cross_AB_edges)
        E_BC = len(cross_BC_edges)

        E_intra = E_A + E_B + E_C
        E_cross = E_AB + E_BC
        E_total = E_intra + E_cross

        # Coupling entropy (simplified)
        if E_total > 0:
            p_intra = E_intra / E_total
            p_cross = E_cross / E_total

            C_coupling = 0
            if p_intra > 0:
                C_coupling -= p_intra * np.log2(p_intra)
            if p_cross > 0:
                C_coupling -= p_cross * np.log2(p_cross)
        else:
            C_coupling = 0.5

        d_eff = d_layer + np.log2(L) + C_coupling

        return {
            'd_layer': d_layer,
            'log2_L': float(np.log2(L)),
            'C_coupling': float(C_coupling),
            'd_eff': float(d_eff),
            'interpretation': 'information-theoretic, NOT topological',
            'E_intra': E_intra,
            'E_cross': E_cross,
            'E_total': E_total
        }

    async def _analyze_tractability(
        self,
        G_A: nx.Graph,
        G_B: nx.Graph,
        G_C: nx.Graph,
        cross_AB_edges: List[Tuple[Any, Any]],
        cross_BC_edges: List[Tuple[Any, Any]]
    ) -> Dict[str, Any]:
        """Analyze Fellows' Biological Computational Tractability Principle.

        Tests:
        1. Per-layer disc ≤ 3 (FPT-tractable obstructions)
        2. Layer separation E_intra/E_total > 0.9
        3. No dense inter-layer coupling

        Returns:
            Dict with tractability validation results
        """
        results = {}

        # Test 1: Per-layer disc dimension
        disc_A = (await self.disc_predictor.predict_consensus(G_A))['disc_consensus']
        disc_B = (await self.disc_predictor.predict_consensus(G_B))['disc_consensus']
        disc_C = (await self.disc_predictor.predict_consensus(G_C))['disc_consensus']

        results['per_layer_disc'] = {
            'A': disc_A,
            'B': disc_B,
            'C': disc_C,
            'all_tractable': disc_A <= 3 and disc_B <= 3 and disc_C <= 3
        }

        # Test 2: Layer separation
        E_A = G_A.number_of_edges()
        E_B = G_B.number_of_edges()
        E_C = G_C.number_of_edges()
        E_AB = len(cross_AB_edges)
        E_BC = len(cross_BC_edges)

        E_intra = E_A + E_B + E_C
        E_cross = E_AB + E_BC
        E_total = E_intra + E_cross

        separation_ratio = E_intra / E_total if E_total > 0 else 1.0

        results['layer_separation'] = {
            'E_intra': E_intra,
            'E_cross': E_cross,
            'E_total': E_total,
            'separation_ratio': separation_ratio,
            'tractable': separation_ratio > 0.9
        }

        # Test 3: Dense coupling check
        max_cross_edges = len(G_A.nodes()) * len(G_B.nodes()) + len(G_B.nodes()) * len(G_C.nodes())
        cross_density = E_cross / max_cross_edges if max_cross_edges > 0 else 0

        results['coupling_density'] = {
            'E_cross': E_cross,
            'max_possible': max_cross_edges,
            'density': cross_density,
            'sparse': cross_density < 0.1
        }

        # Overall tractability
        results['overall_tractable'] = (
            results['per_layer_disc']['all_tractable'] and
            results['layer_separation']['tractable'] and
            results['coupling_density']['sparse']
        )

        return results


# Convenience function

async def analyze_three_layer_brain_network(
    G_signal: nx.Graph,
    G_photonic: nx.Graph,
    G_lymphatic: nx.Graph,
    cross_signal_photonic: List[Tuple[Any, Any]],
    cross_photonic_lymphatic: List[Tuple[Any, Any]]
) -> Dict[str, Any]:
    """Analyze three-layer brain network (signal + photonic + lymphatic).

    Comprehensive analysis combining:
    - P3 dominating set (critical hubs)
    - Quantum operators (obstruction detection)
    - Disc dimension per layer
    - QTRM transitions
    - Fellows' tractability validation

    Args:
        G_signal: Signal/electrical layer
        G_photonic: Photonic/optical layer
        G_lymphatic: Lymphatic/fluid layer
        cross_signal_photonic: Signal ↔ Photonic edges
        cross_photonic_lymphatic: Photonic ↔ Lymphatic edges

    Returns:
        Complete tripartite multiplex analysis
    """
    analyzer = TripartiteMultiplexAnalyzer()
    return await analyzer.analyze_tripartite_multiplex(
        G_signal, G_photonic, G_lymphatic,
        cross_signal_photonic, cross_photonic_lymphatic
    )
