# QEC Tensor Bootstrap Integration Tests

Comprehensive integration tests for the QEC tensor bootstrap pipeline, including service connectivity, corpus download, tensor dimensionality validation, and PRIME-DE integration.

## Overview

This test suite validates:

- **Service Connectivity**: k=6 services in ernie2_swarm FPT analysis
- **Corpus Download**: 56MB streaming corpus verification
- **Bootstrap Time**: Performance constraint ≤60 seconds
- **Tensor Dimensions**: Specification [6, 100, 3] validation
- **PRIME-DE Integration**: Real neuroimaging data loading
- **Error Handling**: Resilience and timeout behavior
- **Mathematical Validators**: Category theory properties

## Test Coverage

### Test Classes (34 tests total)

| Class | Tests | Description |
|-------|-------|-------------|
| `TestServiceConnectivity` | 5 | k=6 service health checks |
| `TestQECTensorService` | 3 | Corpus info, cell listing, metadata |
| `TestCorpusDownload` | 3 | 56MB streaming, size verification |
| `TestBootstrapPipeline` | 5 | Configuration, functors, hierarchy |
| `TestTensorDimensions` | 4 | Shape [6, 100, 3] validation |
| `TestPRIMEDEIntegration` | 3 | Neuroimaging data loading |
| `TestErrorHandling` | 3 | Timeout, 404, malformed JSON |
| `TestPerformanceConstraints` | 2 | Bootstrap time, concurrency |
| `TestValidators` | 2 | Bipartite, treewidth analysis |
| `TestEndToEndIntegration` | 2 | Workflow, service discovery |
| `TestDocumentation` | 2 | TEST_PLAN, DESIGN references |

## Services Tested (k=6)

```
1. QEC Tensor Service          (:8092)
2. PRIME-DE NIfTI API          (:8009)
3. yada-services-secure        (:8003)
4. ameme_2_services/merge2docs (:8091)
5. BrainDecode API             (:8010)
6. Ernie2 Swarm                (:8011)
```

## Quick Start

### Run All Integration Tests

```bash
pytest tests/integration/test_qec_integration.py -v -m integration
```

### Run Only Quick Tests (Exclude Slow)

```bash
pytest tests/integration/test_qec_integration.py -v -m "not slow"
```

### Run Only Slow Tests (>5 seconds)

```bash
pytest tests/integration/test_qec_integration.py -v -m slow --tb=short
```

### Run Specific Test Class

```bash
pytest tests/integration/test_qec_integration.py::TestBootstrapPipeline -v
pytest tests/integration/test_qec_integration.py::TestTensorDimensions -v
```

### Run Specific Test

```bash
pytest tests/integration/test_qec_integration.py::TestBootstrapPipeline::test_qec_config_initialization -v
```

### Verbose Output with Logging

```bash
pytest tests/integration/test_qec_integration.py -vv -s --log-cli-level=DEBUG
```

## Test Markers

### `@pytest.mark.integration`

Tests that require live services. Can be skipped with:

```bash
pytest -m "not integration"
```

### `@pytest.mark.slow`

Long-running tests (>5 seconds, includes corpus download). Can be skipped with:

```bash
pytest -m "not slow"
```

## Expected Behavior

### Service Connectivity Tests

Tests gracefully skip if services are unavailable:

```
TestServiceConnectivity::test_qec_tensor_service_health SKIPPED
  [Service not available] QEC Tensor Service not available at 8092
```

### Bootstrap Pipeline Tests

All passing without live services:

```
TestBootstrapPipeline::test_qec_config_initialization PASSED
  ✅ QECTensorConfig initialized correctly
     Tensor dimensions: 6 × 100 × 3
```

### Corpus Download Tests

These tests `SKIP` if QEC service is not available:

```
TestCorpusDownload::test_corpus_size_verification SKIPPED
  [Service not available] QEC service not available
```

## Configuration

### pytest.ini

Registered markers and asyncio configuration:

```ini
[pytest]
markers =
    integration: requires live services
    slow: long-running tests (>5 seconds)
    unit: unit tests
    performance: performance benchmarks

asyncio_mode = auto
```

### Service Configuration

```python
@dataclass
class ServiceConfig:
    qec_tensor = "http://localhost:8092"
    prime_de = "http://localhost:8009"
    yada_services = "http://localhost:8003"
    ameme_2_services = "http://localhost:8091"
    braindecode_api = "http://localhost:8010"
    ernie2_swarm = "http://localhost:8011"
```

## Fixtures

### `service_config`

Configuration for all k=6 services:

```python
@pytest.fixture(scope="session")
def service_config():
    return ServiceConfig()
```

### `tensor_dimensions`

Expected tensor dimensions [6, 100, 3]:

```python
@pytest.fixture
def tensor_dimensions():
    return {
        "functors": 6,
        "regions": 100,
        "scales": 3
    }
```

### `bootstrap_timeout`

Bootstrap time constraint (≤60 seconds):

```python
@pytest.fixture
def bootstrap_timeout():
    return 60.0
```

### `corpus_size_range`

Expected corpus size (56MB ±10%):

```python
@pytest.fixture
def corpus_size_range():
    return {"min_mb": 50, "max_mb": 70}
```

## Key Test Scenarios

### 1. Service Connectivity

Verifies all k=6 services respond to health checks:

```python
async def test_qec_tensor_service_health(self, service_config):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{service_config.qec_tensor}/health",
            timeout=5.0
        )
        assert response.status_code == 200
```

### 2. Tensor Dimensions

Validates [6, 100, 3] specification:

```python
async def test_tensor_dimension_spec(self, tensor_dimensions):
    config = QECTensorConfig()
    assert len(config.brain_functors) == tensor_dimensions["functors"]  # 6
    assert config.brain_regions == tensor_dimensions["regions"]           # 100
    assert len(config.brain_scales) == tensor_dimensions["scales"]        # 3
```

### 3. Corpus Download

Verifies 56MB download completes in ≤60 seconds:

```python
async def test_corpus_size_verification(
    self,
    service_config,
    corpus_size_range,
    bootstrap_timeout
):
    start_time = time.time()
    response = await client.get(f"{service_config.qec_tensor}/qec/corpus/download")
    elapsed = time.time() - start_time

    size_mb = len(response.content) / (1024 * 1024)
    assert corpus_size_range["min_mb"] <= size_mb <= corpus_size_range["max_mb"]
    assert elapsed <= bootstrap_timeout  # ≤60 seconds
```

### 4. Functor Hierarchy

Tests F_i category theory properties:

```python
async def test_functor_hierarchy(self):
    # Higher abstraction teaches lower
    assert can_teach("anatomy", "function")

    # Reflexivity
    for functor in hierarchy:
        assert can_teach(functor, functor)

    # No upward teaching
    assert not can_teach("behavior", "anatomy")
```

### 5. PRIME-DE Integration

Tests neuroimaging data loading:

```python
async def test_prime_de_metadata_query(self, service_config):
    response = await client.post(
        f"{service_config.prime_de}/api/get_nifti_path",
        json={
            "dataset": "BORDEAUX24",
            "subject": "m01",
            "suffix": "T1w"
        }
    )
    assert response.status_code == 200
```

## Performance Constraints

| Constraint | Target | Status |
|-----------|--------|--------|
| Bootstrap time | ≤60s | Validated |
| Corpus size | 56MB ±10% | Validated |
| Concurrent queries | k=6 services | Tested |
| Corpus streaming | No buffering | Verified |

## Error Handling

Tests validate graceful degradation:

- **Service timeout**: Handled with `asyncio.TimeoutError`
- **404 responses**: Logged but don't fail tests
- **Malformed JSON**: Requests with invalid fields return 4xx
- **Connection errors**: Tests skip gracefully

## Mathematical Validators

Tests validate design properties through yada-services:

```python
async def test_bipartite_design_mapping(self, service_config):
    response = await client.post(
        f"{service_config.yada_services}/api/validate/bipartite",
        json={"requirements": [...]}
    )
    # Validates design → task mapping

async def test_treewidth_coupling_analysis(self, service_config):
    response = await client.post(
        f"{service_config.yada_services}/api/validate/treewidth",
        json={"structure": "brain_tensor"}
    )
    # Validates low coupling (treewidth=2)
```

## Documentation References

- **TEST_PLAN**: `docs/designs/yada-hierarchical-brain-model/TEST_PLAN_QEC_TENSOR.md`
- **DESIGN**: `docs/designs/yada-hierarchical-brain-model/DESIGN.md`
- **SPEC**: `docs/designs/yada-hierarchical-brain-model/MERGE2DOCS_ENDPOINTS_SPEC.md`

## Contributing

### Adding New Integration Tests

1. Add test class to `test_qec_integration.py`
2. Use appropriate markers (`@pytest.mark.integration`, `@pytest.mark.slow`)
3. Include docstrings with test purpose
4. Use fixtures for service configuration
5. Handle service unavailability gracefully (use `pytest.skip()`)

### Example Test Template

```python
@pytest.mark.integration
class TestNewFeature:
    """Test description."""

    @pytest.mark.asyncio
    async def test_new_feature(self, service_config):
        """Test specific behavior."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{service_config.service}/endpoint",
                    timeout=5.0
                )
                assert response.status_code == 200
                logger.info("✅ Feature works")
            except httpx.ConnectError:
                pytest.skip("Service not available")
```

## Debugging

### View Detailed Logs

```bash
pytest tests/integration/test_qec_integration.py -vv -s --log-cli-level=DEBUG
```

### Run Single Test with Full Output

```bash
pytest tests/integration/test_qec_integration.py::TestBootstrapPipeline::test_qec_config_initialization -vv -s
```

### View Test Collection

```bash
pytest tests/integration/test_qec_integration.py --collect-only -q
```

## Health Score Impact

Per TEST_PLAN_QEC_TENSOR.md:

- **Phase 1 (Validation)**: 0.72 → 0.75 (+3%)
- **Phase 2 (Mock Tests)**: 0.75 → 0.78 (+3%)
- **Phase 3 (Integration)**: 0.78 → 0.82 (+4%)
- **Target**: 0.90 (Production ready)

## See Also

- `/tests/backend/services/test_qec_functor_validation.py` - Unit tests (26 tests)
- `/tests/backend/mri/` - MRI signal processing tests
- `/tests/backend/services/test_ernie2_integration.py` - Ernie2 integration

---

**Status**: ✅ PHASE 3 INTEGRATION TESTS
**Created**: 2026-01-21
**Tests**: 34 total
**Coverage**: Service connectivity, corpus download, tensor dimensions, PRIME-DE integration, error handling, performance validation
