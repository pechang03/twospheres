# Phase 5 (F₅ Meta/Planning) Architecture

## Overview

Phase 5 demonstrates full meta/planning integration using merge2docs algorithms and services through the A2A (Agent-to-Agent) secure pattern. **Critical**: We do NOT duplicate algorithms or knowledge - all access goes through the merge2docs service layer.

## Architecture Principles

### 1. No Code Duplication
- ✅ Use merge2docs algorithms via services (AlgorithmService, MonteCarloService)
- ✅ Use A2A pattern through GraphDependencies
- ❌ NO direct algorithm imports
- ❌ NO subprocess/string interpolation

### 2. Service Layer Access
```
twosphere-mcp → GraphDependencies (A2A) → merge2docs/services → merge2docs/algorithms
```

### 3. Agentic Workflow System
Uses `../merge2docs/src/backend/graphs` which provides:
- Agent-to-Agent (A2A) dependency injection
- Service registration and discovery
- Secure cross-project calls

## Implementation Stack

### Layer 1: Integration Bridge (`src/backend/integration/`)

**merge2docs_bridge.py** - A2A pattern bridge (SECURE):
```python
# NEW: Async service calls via GraphDependencies
async def call_algorithm_service(algorithm_name, graph_data, **params):
    deps = get_deps()  # GraphDependencies instance
    result = await deps.call_service_method(
        service_name="AlgorithmService",
        method_name="apply_algorithm",
        algorithm_name=algorithm_name,
        graph=graph_data,
        **params
    )
    return result.data if result.success else None

async def call_monte_carlo_service(simulation_type, n_simulations, data, **params):
    deps = get_deps()
    result = await deps.call_service_method(
        service_name="MonteCarloService",
        method_name="run_simulation",
        simulation_type=sim_type,
        params=sim_params,
        data=data
    )
    return result.data if result.success else None
```

**tensor_routing_client.py** - Query routing to domain cells:
```python
client = TensorRoutingClient()  # Connects to port 8091 (ameme_2_services)
routing = await client.route_query(
    "Optimize ring resonator at 1550nm",
    domain_hint='physics'
)
# Returns: {domain: 'physics', fi_level: 'F5', tools: [...]}
```

**Note**: Tensor routing service runs at `../merge2docs/bin/ameme_2_services` on port 8091.

### Layer 2: Phase 5 Services (`src/backend/services/`)

**fiber_coupling_designer.py** - Fiber-to-chip coupling optimization:
- Uses MonteCarloService via bridge for robustness analysis
- Queries ernie2_swarm for optics expertise
- No algorithm duplication

**resonator_optimizer.py** - Optical resonator design with full stack:
1. Tensor routing (F₅ planning)
2. ernie2_swarm expert queries
3. Bayesian optimization (via merge2docs)
4. Cross-validation

### Layer 3: MCP Tools (`../merge2docs/bin/yada_services_secure.py`)

Provides secure MCP interface to merge2docs services:

**execute_algorithm** - Graph algorithms:
```python
async def handle_execute_algorithm(algorithm_name, graph_data, parameters):
    deps = get_deps()
    result = await deps.call_service_method(
        service_name="AlgorithmService",
        method_name="apply_algorithm",
        algorithm_name=algorithm_name,
        graph=graph_data,
        parameters=parameters or {}
    )
    return result
```

**execute_monte_carlo_optimization** - Statistical optimization:
```python
async def handle_execute_monte_carlo_optimization(...):
    deps = get_deps()
    from src.backend.services.monte_carlo_service import SimulationType, SimulationParams

    sim_params = SimulationParams(n_simulations=n_samples, ...)
    result = await deps.call_service_method(
        service_name="MonteCarloService",
        method_name="run_simulation",
        simulation_type=SimulationType.RISK_ANALYSIS,
        params=sim_params,
        data=data
    )
    return result
```

**execute_bayesian_optimization** - Gaussian Process optimization:
```python
async def handle_execute_bayesian_optimization(...):
    # Note: No BayesianService wrapper yet, uses direct import
    from backend.algorithms.bayesian_compression_weight_optimizer import (
        BayesianCompressionWeightOptimizer
    )
    optimizer = BayesianCompressionWeightOptimizer()
    # Integration pending full service wrapper
```

## Security Pattern (A2A)

From `../merge2docs/src/backend/graphs/a2a_dependencies.py`:

```python
@dataclass
class GraphDependencies:
    """Dependency injection for secure service access."""

    async def call_service_method(
        self, service_name: str, method_name: str, **kwargs
    ) -> CrossProjectCallResult:
        # 1. Get service from registry (LazyServiceRegistry fallback)
        service = self.get_service(service_name) or lazy_registry.get_service(service_name)

        # 2. Get method
        method = getattr(service, method_name)

        # 3. Track delegation in A2A context
        self.a2a_context.add_message(...)

        # 4. Call method (async or sync)
        if asyncio.iscoroutinefunction(method):
            result = await method(**kwargs)
        else:
            result = await loop.run_in_executor(None, lambda: method(**kwargs))

        # 5. Track metrics
        self.a2a_context.track_request(execution_time_ms=...)

        return CrossProjectCallResult(success=True, data=result, ...)
```

**Security guarantees:**
- ✅ No subprocess.run() with user input
- ✅ No f-string code interpolation
- ✅ All service calls through validated registry
- ✅ A2A telemetry and usage tracking

## Service Wrappers in merge2docs

### AlgorithmService
**Location**: `../merge2docs/src/backend/services/algorithm_service.py`

**Registered algorithms**:
- Graph algorithms: cluster_editing, cluster_editing_gpu, vertex_cover, fvs, treewidth, rb_domination
- FPT routing for intelligent algorithm selection
- Parameter routing and preprocessing

**Usage**:
```python
result = await call_algorithm_service(
    'cluster_editing_gpu',
    graph_data={'nodes': [...], 'edges': [...]},
    k=5,
    use_kernelization=True
)
```

### MonteCarloService
**Location**: `../merge2docs/src/backend/services/monte_carlo_service.py`

**Simulation types**:
- RISK_ANALYSIS: Design review risk assessment
- TIMING_ESTIMATION: Project timeline forecasting
- FEATURE_OPTIMIZATION: r-IDS feature selection
- ROUTER_CONFIDENCE: Routing reliability
- PROJECT_FORECASTING: Completion estimation

**Usage**:
```python
result = await call_monte_carlo_service(
    simulation_type="RISK_ANALYSIS",
    n_simulations=5000,
    data={"complexity_factor": {"mean": 1.0, "std": 0.3}}
)
```

## Data Flow Example

**Optimize resonator:**
```
1. resonator_optimizer.py
   ↓ call TensorRoutingClient
2. merge2docs tensor routing (port 8091)
   ↓ returns: {domain: 'physics', fi_level: 'F5', tools: [...]}
3. query_expert_collections (ernie2_swarm)
   ↓ returns expert insights
4. call_algorithm_service (if graph algorithms needed)
   ↓ GraphDependencies.call_service_method()
5. AlgorithmService.apply_algorithm()
   ↓ AlgorithmFactory.create_algorithm()
6. Result returned through A2A context
   ↓ with telemetry and metrics
7. resonator_optimizer returns ResonatorDesign
```

## Migration Guide

### Before (Direct imports - WRONG):
```python
from backend.algorithms.bayesian_compression_weight_optimizer import BayesianCompressionWeightOptimizer
optimizer = BayesianCompressionWeightOptimizer()
result = optimizer.optimize(...)
```

### After (A2A pattern - CORRECT):
```python
from backend.integration import call_algorithm_service, call_monte_carlo_service

# For graph algorithms
result = await call_algorithm_service(
    'cluster_editing_gpu',
    graph_data,
    k=5
)

# For Monte Carlo optimization
result = await call_monte_carlo_service(
    simulation_type="RISK_ANALYSIS",
    n_simulations=5000,
    data={...}
)
```

## Testing

Run Phase 5 integration tests:
```bash
cd /Users/petershaw/code/aider/twosphere-mcp
pytest tests/test_phase5_integration.py -v
```

Test yada_services_secure MCP tools:
```bash
cd /Users/petershaw/code/aider/merge2docs
python bin/yada_services_secure.py  # Start MCP server
# In another terminal:
# Test execute_algorithm, execute_monte_carlo_optimization, etc.
```

## Benefits

1. **No duplication**: 0 lines of duplicated algorithm code
2. **Security**: A2A pattern prevents code injection
3. **Telemetry**: Full request tracking and usage metrics
4. **Flexibility**: Easy to swap services or add new ones
5. **Maintainability**: Single source of truth for algorithms

## Future Work

- [ ] Create BayesianOptimizationService wrapper (currently uses direct import)
- [ ] Add more simulation types to MonteCarloService
- [ ] Integrate GNN services for graph embedding
- [ ] Expand tensor routing domain coverage
- [ ] Add MCP tools for tensor routing queries

## References

- **A2A Design**: `../merge2docs/docs/designs/design-2.2.110-a2a-refactor/design.md`
- **Algorithm Service**: `../merge2docs/src/backend/services/algorithm_service.py`
- **Monte Carlo Service**: `../merge2docs/src/backend/services/monte_carlo_service.py`
- **GraphDependencies**: `../merge2docs/src/backend/graphs/a2a_dependencies.py`
- **Refactoring Plan**: `./docs/P4_REFACTORING_PLAN.md`
