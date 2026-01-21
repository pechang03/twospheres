"""Integration tests for QEC tensor bootstrap.

This package contains end-to-end integration tests for the QEC tensor
bootstrap pipeline, including:

- Service connectivity tests (k=6 services)
- Corpus download and streaming verification
- Bootstrap pipeline tests
- Tensor dimension validation [6, 100, 3]
- PRIME-DE integration tests
- Performance and constraint validation
- Mathematical validator tests

Test markers:
    @pytest.mark.integration: Requires live services
    @pytest.mark.slow: Long-running tests (>5s)

Run integration tests:
    pytest tests/integration/ -v -m integration
    pytest tests/integration/ -v -m slow --tb=short

Documentation:
    docs/designs/yada-hierarchical-brain-model/TEST_PLAN_QEC_TENSOR.md
"""

__all__ = [
    "test_qec_integration",
]
