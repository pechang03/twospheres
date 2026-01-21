"""Integration layer for external services and algorithms.

Uses A2A pattern through GraphDependencies for secure service access.
"""

try:
    from .merge2docs_bridge import (
        # New A2A async functions (preferred)
        get_deps,
        call_algorithm_service,
        call_monte_carlo_service,
        # Deprecated functions (kept for backwards compatibility)
        get_merge2docs_algorithm,
        get_bayesian_optimizer,
        get_monte_carlo_optimizer,
        get_algorithm_service,
        get_bayesian_link_evaluator,
        MERGE2DOCS_AVAILABLE,
    )
except ImportError:
    # Fallback if merge2docs not available
    MERGE2DOCS_AVAILABLE = False
    get_deps = None
    call_algorithm_service = None
    call_monte_carlo_service = None
    get_merge2docs_algorithm = None
    get_bayesian_optimizer = None
    get_monte_carlo_optimizer = None
    get_algorithm_service = None
    get_bayesian_link_evaluator = None

from .tensor_routing_client import TensorRoutingClient, route_query

__all__ = [
    'MERGE2DOCS_AVAILABLE',
    # New A2A functions (preferred)
    'get_deps',
    'call_algorithm_service',
    'call_monte_carlo_service',
    # Deprecated functions (kept for backwards compatibility)
    'get_merge2docs_algorithm',
    'get_bayesian_optimizer',
    'get_monte_carlo_optimizer',
    'get_algorithm_service',
    'get_bayesian_link_evaluator',
    'TensorRoutingClient',
    'route_query',
]
