# Phase 4 Refactoring Plan: Use merge2docs Algorithms

## Overview

Phase 4 implementation currently duplicates functionality already available in `../merge2docs/src/backend/algorithms`. This document outlines the refactoring strategy to eliminate duplication and leverage existing, battle-tested algorithms.

## Code Duplication Analysis

### 1. Alignment Sensitivity (Monte Carlo Simulation)

**Current:** `src/backend/optics/alignment_sensitivity.py` (369 lines)
- Custom Monte Carlo implementation for coupling efficiency
- Statistical tolerance analysis

**merge2docs Equivalent:**
- `enhanced_monte_carlo_r_optimization.py` - Advanced Monte Carlo with LID-aware integration
- `kozyra_monte_carlo_extensions.py` - Monte Carlo extensions
- Features: Statistical confidence intervals, risk assessment, convergence detection

**Refactoring Strategy:**
- Replace custom Monte Carlo with `EnhancedMonteCarloROptimization` from merge2docs
- Adapt for optical coupling efficiency objective function
- Leverage existing uncertainty quantification and risk assessment

### 2. Global Optimization Framework

**Current:** `src/backend/optics/global_optimization.py` (434 lines)
- Simplified NSGA-II implementation (incomplete)
- Basic Bayesian optimization placeholder
- Differential evolution wrapper

**merge2docs Equivalent:**
- `bayesian_compression_weight_optimizer.py` - Full Bayesian optimization with GP
- `bayesian_link_utils.py` - Bayesian framework and utilities
- `enhanced_monte_carlo_r_optimization.py` - Multi-objective optimization support

**Refactoring Strategy:**
- Use `BayesianCompressionWeightOptimizer` as base class
- Extend for LOC-specific objectives (Strehl ratio, MTF, coupling efficiency)
- Leverage existing GP infrastructure with Matern/RBF kernels
- For NSGA-II, recommend using `pymoo` library (already in merge2docs ecosystem)

### 3. Whole-Brain Network Analysis

**Current:** `src/backend/mri/whole_brain_network.py` (330 lines)
- Custom NetworkX-based analysis
- Graph metrics computation

**merge2docs Equivalent:**
- `cluster_editing.py`, `cluster_editing_gpu.py` - Advanced graph clustering
- `shared_clustering.py` - Clustering utilities
- `shared_graph_utils.py` - Graph utilities
- `gnn_euler_embedding.py` - GNN-based graph embeddings
- `similarity.py` - Similarity graph construction

**Refactoring Strategy:**
- **Keep custom implementation** - Brain network analysis has domain-specific requirements
- **Import graph utilities** from merge2docs for:
  - Community detection algorithms
  - Advanced centrality measures
  - Graph embedding methods (for future enhancements)
- Use `AlgorithmService` for complex graph operations

### 4. Multi-Organ OOC Simulation

**Current:** `src/backend/services/multi_organ_ooc.py` (385 lines)
- Custom ODE solver for pharmacokinetics
- Organ compartment models

**merge2docs Equivalent:**
- No direct equivalent (domain-specific biomedical simulation)
- Could leverage:
  - `model_tensor.py` - Tensor-based modeling
  - General optimization infrastructure for parameter fitting

**Refactoring Strategy:**
- **Keep custom implementation** - Highly domain-specific
- Consider using merge2docs optimization for:
  - Parameter estimation (Bayesian optimization for organ parameters)
  - Flow rate optimization
  - Multi-objective optimization (competing organ objectives)

## Implementation Plan

### Phase 1: Setup Integration Layer (Priority 1)

Create `src/backend/integration/merge2docs_bridge.py`:

```python
"""Bridge to merge2docs algorithms and services."""

import sys
from pathlib import Path

# Add merge2docs to path
MERGE2DOCS_PATH = Path(__file__).parent.parent.parent.parent / "merge2docs"
sys.path.insert(0, str(MERGE2DOCS_PATH / "src"))

# Import merge2docs algorithms
from backend.algorithms.enhanced_monte_carlo_r_optimization import (
    EnhancedMonteCarloROptimization,
    OptimizationMetrics
)
from backend.algorithms.bayesian_compression_weight_optimizer import (
    BayesianCompressionWeightOptimizer,
    CompressionWeights
)
from backend.algorithms.bayesian_link_utils import BayesianLinkEvaluator
from backend.services.algorithm_service import AlgorithmService

# Export for use in twosphere-mcp
__all__ = [
    'EnhancedMonteCarloROptimization',
    'BayesianCompressionWeightOptimizer',
    'BayesianLinkEvaluator',
    'AlgorithmService',
    'OptimizationMetrics',
    'CompressionWeights'
]
```

### Phase 2: Refactor Alignment Sensitivity (Priority 1)

**Before:**
- Custom Monte Carlo in `alignment_sensitivity.py`

**After:**
- Thin wrapper around merge2docs Monte Carlo
- Custom objective function for coupling efficiency
- Delegate sampling and statistics to merge2docs

**File:** `src/backend/optics/alignment_sensitivity_v2.py`

```python
from backend.integration.merge2docs_bridge import EnhancedMonteCarloROptimization

class AlignmentSensitivityAnalyzer:
    def __init__(self, ...):
        self.mc_optimizer = EnhancedMonteCarloROptimization()

    async def run_monte_carlo(self, tolerance_spec, n_samples):
        # Define coupling efficiency objective
        def objective(params):
            return self.compute_coupling_efficiency(
                lateral_offset_um=params['lateral'],
                angular_offset_deg=params['angular']
            )

        # Use merge2docs Monte Carlo engine
        results = await self.mc_optimizer.optimize(
            objective_function=objective,
            parameter_bounds={
                'lateral': (-tolerance_spec.lateral_tolerance_um,
                           tolerance_spec.lateral_tolerance_um),
                'angular': (-tolerance_spec.angular_tolerance_deg,
                           tolerance_spec.angular_tolerance_deg)
            },
            n_samples=n_samples
        )
        return results
```

### Phase 3: Refactor Global Optimization (Priority 1)

**Before:**
- Incomplete NSGA-II, placeholder Bayesian optimization

**After:**
- Full Bayesian optimization from merge2docs
- Proper multi-objective support

**File:** `src/backend/optics/global_optimization_v2.py`

```python
from backend.integration.merge2docs_bridge import (
    BayesianCompressionWeightOptimizer,
    BayesianLinkEvaluator
)

class LOCSystemOptimizer(BayesianCompressionWeightOptimizer):
    """LOC system optimizer using merge2docs Bayesian framework."""

    def __init__(self, variables, objectives):
        # Adapt merge2docs weight optimizer for LOC objectives
        super().__init__()
        self.variables = variables
        self.objectives = objectives

    async def optimize_bayesian(self, ...):
        # Use parent class GP-based optimization
        return await super().optimize_weights(...)
```

### Phase 4: Enhance Network Analysis (Priority 2)

**Before:**
- Basic NetworkX operations

**After:**
- Import advanced clustering from merge2docs
- Use GPU-accelerated algorithms where applicable

**File:** `src/backend/mri/whole_brain_network.py` (update)

```python
from backend.integration.merge2docs_bridge import AlgorithmService

class WholeBrainNetworkAnalyzer:
    def __init__(self, ...):
        self.algo_service = AlgorithmService()

    async def detect_communities_advanced(self, graph):
        # Use merge2docs cluster editing for community detection
        result = await self.algo_service.apply_algorithm(
            algorithm_name='cluster_editing_gpu',
            graph=graph,
            parameters={'enable_crown_rule': True}
        )
        return result
```

### Phase 5: Use yada-services MCP (Priority 2)

Instead of direct imports, use yada-services MCP server for algorithm access:

**Update:** `bin/twosphere_mcp.py`

```python
# Add tool to query yada-services
Tool(
    name="query_merge2docs_algorithm",
    description="Execute merge2docs algorithm via yada-services",
    inputSchema={
        "type": "object",
        "properties": {
            "algorithm_name": {
                "type": "string",
                "description": "Algorithm to execute (e.g., 'cluster_editing_gpu')"
            },
            "graph_data": {
                "type": "object",
                "description": "Graph adjacency data"
            },
            "parameters": {
                "type": "object",
                "description": "Algorithm-specific parameters"
            }
        }
    }
)
```

## Benefits

### Code Reduction
- **Before:** ~1,188 lines of algorithm code in P4
- **After:** ~300 lines (thin wrappers + domain-specific logic)
- **Savings:** ~75% code reduction

### Quality Improvements
- Leverage battle-tested merge2docs algorithms
- GPU acceleration where available
- Advanced features: LID-aware optimization, risk assessment, convergence detection
- Better statistical rigor (confidence intervals, uncertainty quantification)

### Maintainability
- Single source of truth for algorithms
- Bug fixes in merge2docs benefit both projects
- Consistent API across projects

### Future-Proofing
- Easy to adopt new merge2docs algorithms
- Access to full service discovery ecosystem
- Integration with ERNIE-FPT for complex analysis

## Migration Checklist

- [ ] Create `src/backend/integration/merge2docs_bridge.py`
- [ ] Test bridge imports and path setup
- [ ] Refactor `alignment_sensitivity.py` → `alignment_sensitivity_v2.py`
- [ ] Refactor `global_optimization.py` → `global_optimization_v2.py`
- [ ] Update tests to use new implementations
- [ ] Add integration tests with merge2docs
- [ ] Update MCP tools to use refactored modules
- [ ] Deprecate old implementations
- [ ] Update documentation

## Testing Strategy

1. **Unit tests:** Verify wrappers correctly delegate to merge2docs
2. **Integration tests:** End-to-end tests with real merge2docs algorithms
3. **Regression tests:** Ensure refactored code produces equivalent results
4. **Performance tests:** Validate GPU acceleration works
5. **Compatibility tests:** Ensure merge2docs version compatibility

## Risks and Mitigation

### Risk 1: merge2docs API Changes
**Mitigation:** Version pin merge2docs, use bridge layer to isolate changes

### Risk 2: Path/Import Issues
**Mitigation:** Robust path detection, fallback to local implementation

### Risk 3: Missing Dependencies
**Mitigation:** Document merge2docs requirements, add to requirements.txt

### Risk 4: Performance Regression
**Mitigation:** Benchmark before/after, use GPU acceleration

## Timeline

- **Week 1:** Setup bridge layer, test imports
- **Week 2:** Refactor alignment sensitivity + tests
- **Week 3:** Refactor global optimization + tests
- **Week 4:** Enhance network analysis, integration tests
- **Week 5:** Documentation, deprecate old code

## Recommendation

**Proceed with refactoring** - The benefits (code reduction, quality, maintainability) far outweigh the migration cost. Start with bridge layer and alignment sensitivity as proof of concept.
