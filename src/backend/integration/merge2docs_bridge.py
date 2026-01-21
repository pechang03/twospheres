"""Bridge to merge2docs services using A2A pattern.

Provides clean integration with merge2docs services without code duplication.
Uses GraphDependencies and the agentic workflow system for secure service access.

Security: No direct algorithm imports - all access goes through service layer.
"""

import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Detect merge2docs path
MERGE2DOCS_PATH = Path(__file__).parent.parent.parent.parent.parent / "merge2docs"

if not MERGE2DOCS_PATH.exists():
    logger.warning(f"merge2docs not found at {MERGE2DOCS_PATH}")
    MERGE2DOCS_AVAILABLE = False
else:
    # Add merge2docs to Python path for GraphDependencies access
    merge2docs_src = str(MERGE2DOCS_PATH / "src")
    if merge2docs_src not in sys.path:
        sys.path.insert(0, merge2docs_src)
    MERGE2DOCS_AVAILABLE = True
    logger.info(f"merge2docs bridge enabled: {MERGE2DOCS_PATH}")


# Global GraphDependencies instance
_deps = None

def get_deps():
    """Get or create GraphDependencies instance for A2A communication."""
    global _deps
    if _deps is None:
        if not MERGE2DOCS_AVAILABLE:
            logger.warning("Cannot initialize GraphDependencies: merge2docs not available")
            return None

        try:
            from src.backend.graphs.a2a_dependencies import create_a2a_context, GraphDependencies

            # Import services to trigger registration
            # This ensures @register_service decorators execute
            try:
                from src.backend.services.algorithm_service import AlgorithmService
                logger.info("AlgorithmService imported for registration")
            except ImportError as e:
                logger.warning(f"Could not import AlgorithmService: {e}")

            ctx = create_a2a_context(session_id="twosphere_mcp")
            _deps = GraphDependencies(a2a_context=ctx)
            logger.info("GraphDependencies initialized for merge2docs bridge")
        except Exception as e:
            logger.error(f"Failed to initialize GraphDependencies: {e}")
            return None

    return _deps


async def call_algorithm_service(
    algorithm_name: str,
    graph_data: Dict[str, Any],
    **parameters
) -> Optional[Any]:
    """Call AlgorithmService via A2A pattern.

    Args:
        algorithm_name: Algorithm to execute (e.g., 'cluster_editing_gpu')
        graph_data: Graph data structure (dict with 'nodes' and 'edges')
        **parameters: Algorithm-specific parameters

    Returns:
        Algorithm result or None if failed

    Example:
        >>> result = await call_algorithm_service(
        ...     'cluster_editing_gpu',
        ...     graph_data={'nodes': [...], 'edges': [...]},
        ...     k=5
        ... )
    """
    import networkx as nx

    deps = get_deps()
    if not deps:
        return None

    try:
        # Convert graph_data dict to NetworkX graph
        # AlgorithmService expects a NetworkX graph object
        if isinstance(graph_data, dict) and 'nodes' in graph_data and 'edges' in graph_data:
            G = nx.Graph()
            G.add_nodes_from(graph_data['nodes'])
            G.add_edges_from(graph_data['edges'])
            logger.debug(f"Converted dict to NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        elif isinstance(graph_data, nx.Graph):
            G = graph_data
        else:
            logger.error(f"Invalid graph_data format: {type(graph_data)}")
            return None

        result = await deps.call_service_method(
            service_name="AlgorithmService",
            method_name="apply_algorithm",
            algorithm_name=algorithm_name,
            graph=G,
            **parameters
        )

        if result.success:
            return result.data
        else:
            logger.error(f"Algorithm service failed: {result.error}")
            return None
    except Exception as e:
        logger.error(f"Failed to call AlgorithmService: {e}")
        return None


async def call_monte_carlo_service(
    simulation_type: str = "RISK_ANALYSIS",
    n_simulations: int = 10000,
    data: Optional[Dict[str, Any]] = None,
    **parameters
) -> Optional[Any]:
    """Call MonteCarloService via A2A pattern.

    Args:
        simulation_type: Type of simulation (RISK_ANALYSIS, TIMING_ESTIMATION, etc.)
        n_simulations: Number of Monte Carlo samples
        data: Simulation data
        **parameters: Additional parameters

    Returns:
        Simulation result or None if failed

    Example:
        >>> result = await call_monte_carlo_service(
        ...     simulation_type="RISK_ANALYSIS",
        ...     n_simulations=5000,
        ...     data={"complexity_factor": {"mean": 1.0, "std": 0.3}}
        ... )
    """
    deps = get_deps()
    if not deps:
        return None

    try:
        # Import enums from MonteCarloService
        from src.backend.services.monte_carlo_service import (
            SimulationType, SimulationParams, DistributionType
        )

        # Convert string to enum
        sim_type = getattr(SimulationType, simulation_type, SimulationType.RISK_ANALYSIS)

        # Create simulation parameters
        sim_params = SimulationParams(
            n_simulations=n_simulations,
            distribution_type=parameters.get('distribution_type', DistributionType.NORMAL),
            **{k: v for k, v in parameters.items() if k != 'distribution_type'}
        )

        result = await deps.call_service_method(
            service_name="MonteCarloService",
            method_name="run_simulation",
            simulation_type=sim_type,
            params=sim_params,
            data=data
        )

        if result.success:
            return result.data
        else:
            logger.error(f"Monte Carlo service failed: {result.error}")
            return None
    except Exception as e:
        logger.error(f"Failed to call MonteCarloService: {e}")
        return None


# Backwards compatibility - deprecated functions that return None
# These guide users to use the async service call functions instead

def get_merge2docs_algorithm(algorithm_name: str) -> Optional[Any]:
    """DEPRECATED: Use call_algorithm_service() instead.

    Direct algorithm imports bypass the service layer and A2A pattern.
    """
    logger.warning(
        f"get_merge2docs_algorithm() is deprecated. "
        f"Use: await call_algorithm_service('{algorithm_name}', graph_data, **params)"
    )
    return None


def get_bayesian_optimizer():
    """DEPRECATED: Bayesian optimization not yet exposed as service.

    For now, use direct import from merge2docs if needed:
        from backend.algorithms.bayesian_compression_weight_optimizer import (
            BayesianCompressionWeightOptimizer
        )
    """
    logger.warning(
        "get_bayesian_optimizer() is deprecated. "
        "Bayesian optimization service integration pending."
    )
    return None


def get_monte_carlo_optimizer():
    """DEPRECATED: Use call_monte_carlo_service() instead.

    Monte Carlo functionality is now accessed through MonteCarloService.
    """
    logger.warning(
        "get_monte_carlo_optimizer() is deprecated. "
        "Use: await call_monte_carlo_service(simulation_type='...', data={...})"
    )
    return None


def get_algorithm_service():
    """DEPRECATED: Use call_algorithm_service() instead.

    Services are accessed through GraphDependencies, not direct instantiation.
    """
    logger.warning(
        "get_algorithm_service() is deprecated. "
        "Use: await call_algorithm_service(algorithm_name, graph_data, **params)"
    )
    return None


def get_bayesian_link_evaluator():
    """DEPRECATED: Service wrapper not yet available.

    For now, use direct import from merge2docs if needed.
    """
    logger.warning(
        "get_bayesian_link_evaluator() is deprecated. "
        "Service wrapper not yet available."
    )
    return None


# Convenience exports
__all__ = [
    'MERGE2DOCS_AVAILABLE',
    'MERGE2DOCS_PATH',
    'get_deps',
    'call_algorithm_service',
    'call_monte_carlo_service',
    # Deprecated - kept for backwards compatibility
    'get_merge2docs_algorithm',
    'get_bayesian_optimizer',
    'get_monte_carlo_optimizer',
    'get_algorithm_service',
    'get_bayesian_link_evaluator',
]
