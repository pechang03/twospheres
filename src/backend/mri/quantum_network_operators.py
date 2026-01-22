"""Quantum operator formalism for network topology and disc dimension analysis.

Uses SymPy symbolic framework from entangled-pair-quantum-eraser to:
1. Model topological obstructions (K₅, K₃,₃) as quantum operators
2. Represent network states as quantum state vectors
3. Analyze disc dimension via eigenvalue decomposition
4. Improve QTRM/QEC routing with unitary operator transitions

Mathematical Framework:
- Graph G → State vector |ψ_G⟩ in Hilbert space H_N
- Obstruction → Operator Ô acting on H_N
- Disc dimension → Eigenspectrum of adjacency/Laplacian operator
- QTRM transitions → Unitary operators U_i: F_i → F_{i+1}
- QEC error correction → Projection operators P_i

References:
- entangled-pair-quantum-eraser/lab6entangled.py (SymPy quantum framework)
- docs/papers/ernie2_synthesis_unified_framework.md (disc dimension theory)
- merge2docs QTRM models (functor hierarchy routing)
"""

from sympy import *
from sympy.physics.quantum import TensorProduct, Dagger
from sympy.matrices import Matrix, eye, zeros
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any


class QuantumNetworkState:
    """Represent network graph as quantum state vector.

    Maps graph G with N nodes to state vector |ψ_G⟩ in C^N.

    Encoding schemes:
    1. Adjacency: |ψ⟩ = Σ A_ij |i⟩⊗|j⟩ (tensor product of connections)
    2. Laplacian: |ψ⟩ = eigenvector of graph Laplacian
    3. Walk: |ψ⟩ = probability distribution of random walk

    Example:
        >>> G = nx.karate_club_graph()
        >>> state = QuantumNetworkState(G, encoding='laplacian')
        >>> psi = state.to_symbolic()  # SymPy Matrix
        >>> dim = state.intrinsic_dimension()  # Effective dimension
    """

    def __init__(self, G: nx.Graph, encoding: str = 'adjacency'):
        """Initialize quantum state from graph.

        Args:
            G: NetworkX graph
            encoding: 'adjacency', 'laplacian', or 'walk'
        """
        self.G = G
        self.N = G.number_of_nodes()
        self.encoding = encoding
        self._state_vector = None

    def to_symbolic(self) -> Matrix:
        """Convert graph to symbolic SymPy state vector.

        Returns:
            SymPy Matrix representing |ψ_G⟩
        """
        if self.encoding == 'adjacency':
            return self._adjacency_encoding()
        elif self.encoding == 'laplacian':
            return self._laplacian_encoding()
        elif self.encoding == 'walk':
            return self._walk_encoding()
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _adjacency_encoding(self) -> Matrix:
        """Encode as adjacency matrix flattened to vector."""
        A = nx.adjacency_matrix(self.G).todense()
        # Flatten and normalize
        psi = Matrix(A.flatten().T)
        norm = sqrt(sum([x**2 for x in psi]))
        return psi / norm if norm != 0 else psi

    def _laplacian_encoding(self) -> Matrix:
        """Encode as smallest eigenvector of graph Laplacian.

        The Fiedler vector (2nd smallest eigenvalue) encodes graph structure.
        Its sign pattern reveals clusters and obstructions.
        """
        L = nx.laplacian_matrix(self.G).todense()
        # Convert to SymPy for symbolic computation
        L_sym = Matrix(L)

        # For small graphs, compute eigenvectors symbolically
        # For large graphs, use numerical approximation
        if self.N <= 10:
            eigenvals = L_sym.eigenvals()
            eigenvects = L_sym.eigenvects()
            # Get Fiedler vector (2nd smallest eigenvalue)
            sorted_eigs = sorted(eigenvects, key=lambda x: x[0])
            if len(sorted_eigs) > 1:
                fiedler = sorted_eigs[1][2][0]  # eigenvector
                return fiedler

        # Numerical fallback
        eigenvals, eigenvects = np.linalg.eigh(L)
        fiedler_vec = eigenvects[:, 1]  # 2nd eigenvector
        return Matrix(fiedler_vec)

    def _walk_encoding(self) -> Matrix:
        """Encode as stationary distribution of random walk."""
        # Transition matrix
        A = nx.adjacency_matrix(self.G).todense()
        degrees = np.array(A.sum(axis=1)).flatten()

        # Avoid division by zero
        degrees[degrees == 0] = 1
        D_inv = np.diag(1.0 / degrees)
        P = D_inv @ A  # Transition matrix

        # Stationary distribution (eigenvector for eigenvalue 1)
        eigenvals, eigenvects = np.linalg.eig(P.T)
        stationary_idx = np.argmin(np.abs(eigenvals - 1))
        stationary = np.real(eigenvects[:, stationary_idx])
        stationary = stationary / np.sum(stationary)

        return Matrix(stationary)

    def intrinsic_dimension(self) -> float:
        """Estimate intrinsic dimension from state vector.

        Uses participation ratio: D_eff = (Σ|ψ_i|²)² / Σ|ψ_i|⁴

        Returns:
            Effective dimension (1 = localized, N = uniform)
        """
        psi = self.to_symbolic()

        # Convert to numerical for computation
        psi_num = np.array(psi, dtype=float).flatten()

        sum_sq = np.sum(psi_num**2)
        sum_fourth = np.sum(psi_num**4)

        if sum_fourth > 0:
            return float(sum_sq**2 / sum_fourth)
        else:
            return 1.0


class ObstructionOperator:
    """Quantum operator representing topological obstruction.

    Models K₅, K₃,₃, and other forbidden minors as operators on network space.

    Operator properties:
    - Hermitian: O† = O (observable)
    - Eigenvalues: Real (obstruction strength)
    - Eigenvectors: States with/without obstruction

    Example:
        >>> G = nx.complete_graph(5)  # K₅
        >>> op = ObstructionOperator('K5')
        >>> O = op.to_symbolic(G)
        >>> eigenvals = O.eigenvals()  # Detect obstruction
    """

    def __init__(self, obstruction_type: str = 'K5'):
        """Initialize obstruction operator.

        Args:
            obstruction_type: 'K5', 'K33', or 'custom'
        """
        self.type = obstruction_type

    def to_symbolic(self, G: nx.Graph) -> Matrix:
        """Construct symbolic operator for given graph.

        Args:
            G: NetworkX graph to analyze

        Returns:
            SymPy Matrix operator Ô
        """
        if self.type == 'K5':
            return self._k5_operator(G)
        elif self.type == 'K33':
            return self._k33_operator(G)
        else:
            raise ValueError(f"Unknown obstruction type: {self.type}")

    def _k5_operator(self, G: nx.Graph) -> Matrix:
        """Construct K₅ detection operator.

        K₅ signature: 5 nodes with all (5 choose 2) = 10 edges
        Operator: Projects onto subspace spanned by 5-cliques
        """
        N = G.number_of_nodes()
        O = zeros(N, N)

        # Find all 5-cliques
        from networkx.algorithms.clique import find_cliques
        cliques = [c for c in find_cliques(G) if len(c) >= 5]

        # For each K₅, add projection operator
        for clique in cliques:
            if len(clique) >= 5:
                # Take first 5 nodes
                nodes = list(clique)[:5]

                # Projection: |clique⟩⟨clique|
                for i in nodes:
                    for j in nodes:
                        O[i, j] += Rational(1, 5)

        return O

    def _k33_operator(self, G: nx.Graph) -> Matrix:
        """Construct K₃,₃ detection operator.

        K₃,₃ signature: Complete bipartite graph (3 nodes × 3 nodes)
        """
        N = G.number_of_nodes()
        O = zeros(N, N)

        # Check if graph is bipartite
        try:
            from networkx.algorithms.bipartite import is_bipartite, sets as bipartite_sets

            if is_bipartite(G):
                sets = bipartite_sets(G)
                set1, set2 = sets

                # Look for complete bipartite subgraphs
                for subset1 in self._powerset(set1, 3):
                    for subset2 in self._powerset(set2, 3):
                        # Check if complete bipartite
                        is_complete = all(
                            G.has_edge(u, v)
                            for u in subset1 for v in subset2
                        )

                        if is_complete:
                            # Add projection operator
                            nodes = list(subset1) + list(subset2)
                            for i in nodes:
                                for j in nodes:
                                    O[i, j] += Rational(1, 6)
        except:
            pass

        return O

    @staticmethod
    def _powerset(iterable, size):
        """Generate all subsets of given size."""
        from itertools import combinations
        return list(combinations(iterable, size))

    def detect(self, G: nx.Graph) -> Dict[str, Any]:
        """Detect obstruction via operator eigenvalues.

        Args:
            G: Graph to analyze

        Returns:
            Dict with:
            - has_obstruction: bool
            - eigenvalues: List of operator eigenvalues
            - strength: Maximum eigenvalue (obstruction strength)
        """
        O = self.to_symbolic(G)

        # Compute eigenvalues
        try:
            eigenvals = list(O.eigenvals().keys())
            # Convert to float for comparison
            eigenvals_num = [float(ev) for ev in eigenvals]

            max_eigenval = max(eigenvals_num)

            # Obstruction detected if max eigenvalue > threshold
            threshold = 0.1
            has_obstruction = max_eigenval > threshold

            return {
                'has_obstruction': has_obstruction,
                'eigenvalues': eigenvals,
                'strength': max_eigenval,
                'type': self.type
            }
        except:
            # Numerical fallback
            O_num = np.array(O, dtype=float)
            eigenvals = np.linalg.eigvals(O_num)
            max_eigenval = float(np.max(np.real(eigenvals)))

            return {
                'has_obstruction': max_eigenval > 0.1,
                'eigenvalues': eigenvals.tolist(),
                'strength': max_eigenval,
                'type': self.type
            }


class QTRMLevelTransitionOperator:
    """Unitary operator for QTRM functor level transitions.

    Models transitions F_i → F_{i+1} in functor hierarchy as quantum gates.

    Operator properties:
    - Unitary: U†U = I (preserves norm)
    - Composition: U_{i+1} ∘ U_i = U_{combined}
    - Invertible: Information-preserving transitions

    Applications:
    - Abstraction level routing (F₀ → F₁ → ... → F₆)
    - QEC error correction bridges
    - Information flow analysis

    Example:
        >>> # Transition from quantum (F₀) to syntactic (F₁)
        >>> U_01 = QTRMLevelTransitionOperator(source=0, target=1)
        >>> state_F0 = QuantumNetworkState(G, encoding='laplacian')
        >>> state_F1 = U_01.apply(state_F0)
    """

    def __init__(self, source: int, target: int, coupling: float = 0.5):
        """Initialize QTRM transition operator.

        Args:
            source: Source functor level (0-6)
            target: Target functor level (0-6)
            coupling: Coupling strength (0 = no mixing, 1 = full mixing)
        """
        self.source = source
        self.target = target
        self.coupling = coupling

    def to_symbolic(self, dim: int) -> Matrix:
        """Construct symbolic unitary operator.

        Uses rotation operator: U(θ) = exp(iθ·H)
        where H is Hermitian generator (like Hamiltonian)

        Args:
            dim: Hilbert space dimension

        Returns:
            Unitary matrix U
        """
        # Rotation angle from coupling strength
        theta = symbols('theta', real=True)
        theta_val = self.coupling * pi / 2

        # Generator: Off-diagonal coupling
        H = zeros(dim, dim)
        for i in range(dim - 1):
            H[i, i+1] = 1
            H[i+1, i] = 1

        # Exponentiate: U = exp(i·theta·H)
        # For small dim, use symbolic
        if dim <= 4:
            U = (I * theta_val * H).exp()
            return U
        else:
            # Numerical approximation
            H_num = np.array(H, dtype=complex)
            U_num = scipy.linalg.expm(1j * theta_val * H_num)
            return Matrix(U_num)

    def apply(self, state: QuantumNetworkState) -> Matrix:
        """Apply transition operator to network state.

        Args:
            state: Input quantum network state

        Returns:
            Transformed state vector
        """
        psi_in = state.to_symbolic()
        dim = psi_in.shape[0]

        U = self.to_symbolic(dim)
        psi_out = U * psi_in

        return psi_out

    def is_unitary(self) -> bool:
        """Verify unitarity: U†U = I.

        Returns:
            True if operator preserves norm
        """
        try:
            dim = 4  # Test dimension
            U = self.to_symbolic(dim)
            U_dag = Dagger(U)

            product = U_dag * U
            identity = eye(dim)

            # Check if close to identity
            diff = product - identity
            norm = sum([abs(x) for x in diff])

            return norm < 0.01
        except:
            return False


class QECProjectionOperator:
    """Projection operator for QEC (Quantum Error Correction) bridges.

    Models error correction in QTRM transitions: F_i ↔ F_j bridges.

    Operator properties:
    - Projection: P² = P (idempotent)
    - Hermitian: P† = P
    - Eigenvalues: 0 or 1 (project onto error/correct subspaces)

    Error model:
    - Errors = States that violate level constraints
    - Correction = Project onto valid subspace

    Example:
        >>> # Correct errors in F₁ → F₂ transition
        >>> P_12 = QECProjectionOperator(level_pair=(1, 2))
        >>> state_noisy = ...  # State with errors
        >>> state_corrected = P_12.correct(state_noisy)
    """

    def __init__(self, level_pair: Tuple[int, int]):
        """Initialize QEC projection operator.

        Args:
            level_pair: (F_i, F_j) functor levels
        """
        self.level_pair = level_pair

    def to_symbolic(self, dim: int, error_subspace_dim: int) -> Matrix:
        """Construct projection operator onto correct subspace.

        P = I - |error⟩⟨error|

        Args:
            dim: Total Hilbert space dimension
            error_subspace_dim: Dimension of error subspace

        Returns:
            Projection matrix P
        """
        # Identity - projection onto error subspace
        P = eye(dim)

        # Simple model: Errors in first k dimensions
        for i in range(error_subspace_dim):
            P[i, i] = 0

        return P

    def correct(self, state: Matrix) -> Matrix:
        """Apply error correction to state.

        Args:
            state: Potentially noisy state vector

        Returns:
            Corrected state (projected onto valid subspace)
        """
        dim = state.shape[0]
        error_dim = max(1, dim // 4)  # Assume 25% error rate

        P = self.to_symbolic(dim, error_dim)
        corrected = P * state

        # Renormalize
        norm = sqrt(sum([x**2 for x in corrected]))
        if norm > 0:
            corrected = corrected / norm

        return corrected


# Utility functions

def network_to_operator(G: nx.Graph, operator_type: str = 'adjacency') -> Matrix:
    """Convert graph to quantum operator (matrix).

    Args:
        G: NetworkX graph
        operator_type: 'adjacency', 'laplacian', 'modularity'

    Returns:
        SymPy Matrix operator
    """
    if operator_type == 'adjacency':
        A = nx.adjacency_matrix(G).todense()
        return Matrix(A)
    elif operator_type == 'laplacian':
        L = nx.laplacian_matrix(G).todense()
        return Matrix(L)
    elif operator_type == 'modularity':
        # Modularity matrix: B_ij = A_ij - k_i·k_j/2m
        A = nx.adjacency_matrix(G).todense()
        degrees = np.array(A.sum(axis=1)).flatten()
        m = G.number_of_edges()

        B = A - np.outer(degrees, degrees) / (2*m) if m > 0 else A
        return Matrix(B)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def disc_dimension_via_eigenspectrum(G: nx.Graph) -> Dict[str, Any]:
    """Estimate disc dimension from graph operator eigenspectrum.

    Theory: Eigenvalue distribution reveals intrinsic dimensionality
    - 1D: Linear eigenvalue spacing
    - 2D: √n eigenvalue scaling
    - 3D+: n^(1/d) scaling

    Args:
        G: NetworkX graph

    Returns:
        Dict with:
        - disc_dim_estimate: Estimated disc dimension
        - eigenvalues: List of eigenvalues
        - spectral_gap: Gap between largest eigenvalues
    """
    L = nx.laplacian_matrix(G).todense()
    eigenvals = np.linalg.eigvalsh(L)
    eigenvals = sorted(eigenvals)

    # Remove zero eigenvalue (always present in Laplacian)
    eigenvals_nonzero = [ev for ev in eigenvals if abs(ev) > 1e-10]

    if len(eigenvals_nonzero) < 2:
        return {
            'disc_dim_estimate': 1,
            'eigenvalues': eigenvals.tolist(),
            'spectral_gap': 0
        }

    # Spectral gap = difference between two largest eigenvalues
    spectral_gap = eigenvals_nonzero[-1] - eigenvals_nonzero[-2]

    # Estimate dimension from eigenvalue scaling
    # log(λ_max) ~ d·log(n) for d-dimensional graphs
    n = G.number_of_nodes()
    lambda_max = eigenvals_nonzero[-1]

    if n > 1 and lambda_max > 0:
        dim_estimate = np.log(lambda_max) / np.log(n)
        disc_dim = max(1, int(np.round(dim_estimate)))
    else:
        disc_dim = 1

    return {
        'disc_dim_estimate': disc_dim,
        'eigenvalues': eigenvals.tolist(),
        'spectral_gap': float(spectral_gap)
    }
