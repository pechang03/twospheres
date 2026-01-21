"""Bridge to merge2docs algorithms and services.

Provides clean integration with merge2docs algorithms without code duplication.
Follows Option C: Use merge2docs algorithms from the start for all new work.
"""

import sys
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
    # Add merge2docs to Python path
    sys.path.insert(0, str(MERGE2DOCS_PATH / "src"))
    MERGE2DOCS_AVAILABLE = True
    logger.info(f"merge2docs bridge enabled: {MERGE2DOCS_PATH}")


def get_merge2docs_algorithm(algorithm_name: str) -> Optional[Any]:
    """Get algorithm class from merge2docs by name.

    Args:
        algorithm_name: Name of algorithm (e.g., 'cluster_editing_gpu')

    Returns:
        Algorithm class or None if not available

    Example:
        >>> ClusterEditing = get_merge2docs_algorithm('cluster_editing_gpu')
        >>> if ClusterEditing:
        >>>     algo = ClusterEditing()
    """
    if not MERGE2DOCS_AVAILABLE:
        logger.warning(f"Cannot load {algorithm_name}: merge2docs not available")
        return None

    try:
        from backend.algorithms.factory import AlgorithmFactory
        return AlgorithmFactory.create(algorithm_name)
    except Exception as e:
        logger.error(f"Failed to load algorithm {algorithm_name}: {e}")
        return None


def get_bayesian_optimizer():
    """Get Bayesian optimization framework from merge2docs.

    Returns:
        BayesianCompressionWeightOptimizer class or None

    Example:
        >>> Optimizer = get_bayesian_optimizer()
        >>> if Optimizer:
        >>>     optimizer = Optimizer()
        >>>     result = await optimizer.optimize_weights(...)
    """
    if not MERGE2DOCS_AVAILABLE:
        logger.warning("Bayesian optimizer not available: merge2docs not found")
        return None

    try:
        from backend.algorithms.bayesian_compression_weight_optimizer import (
            BayesianCompressionWeightOptimizer
        )
        return BayesianCompressionWeightOptimizer
    except ImportError as e:
        logger.error(f"Failed to import Bayesian optimizer: {e}")
        return None


def get_monte_carlo_optimizer():
    """Get Monte Carlo optimization framework from merge2docs.

    Returns:
        EnhancedMonteCarloROptimization class or None

    Example:
        >>> MCOptimizer = get_monte_carlo_optimizer()
        >>> if MCOptimizer:
        >>>     mc = MCOptimizer()
        >>>     result = mc.run_optimization(...)
    """
    if not MERGE2DOCS_AVAILABLE:
        logger.warning("Monte Carlo optimizer not available: merge2docs not found")
        return None

    try:
        from backend.algorithms.enhanced_monte_carlo_r_optimization import (
            EnhancedMonteCarloROptimization
        )
        return EnhancedMonteCarloROptimization
    except ImportError as e:
        logger.error(f"Failed to import Monte Carlo optimizer: {e}")
        return None


def get_algorithm_service():
    """Get AlgorithmService for unified algorithm access.

    Returns:
        AlgorithmService class or None

    Example:
        >>> AlgoService = get_algorithm_service()
        >>> if AlgoService:
        >>>     service = AlgoService()
        >>>     result = await service.apply_algorithm('cluster_editing_gpu', graph, params)
    """
    if not MERGE2DOCS_AVAILABLE:
        logger.warning("Algorithm service not available: merge2docs not found")
        return None

    try:
        from backend.services.algorithm_service import AlgorithmService
        return AlgorithmService
    except ImportError as e:
        logger.error(f"Failed to import AlgorithmService: {e}")
        return None


def get_bayesian_link_evaluator():
    """Get Bayesian link evaluation utilities.

    Returns:
        BayesianLinkEvaluator class or None
    """
    if not MERGE2DOCS_AVAILABLE:
        return None

    try:
        from backend.algorithms.bayesian_link_utils import BayesianLinkEvaluator
        return BayesianLinkEvaluator
    except ImportError as e:
        logger.error(f"Failed to import BayesianLinkEvaluator: {e}")
        return None


# Convenience exports
__all__ = [
    'MERGE2DOCS_AVAILABLE',
    'MERGE2DOCS_PATH',
    'get_merge2docs_algorithm',
    'get_bayesian_optimizer',
    'get_monte_carlo_optimizer',
    'get_algorithm_service',
    'get_bayesian_link_evaluator',
]
