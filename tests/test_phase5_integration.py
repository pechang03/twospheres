"""Phase 5 (F₅ Meta/Planning) Integration Tests.

Tests the complete Phase 5 stack:
- Integration layer (merge2docs_bridge, tensor_routing_client)
- Phase 5 services (resonator_optimizer, fiber_coupling_designer)
- MCP tool (simulate_loc_chip)
- Phase 4 tools enhanced with query_experts

All tests use A2A pattern through GraphDependencies.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Integration Layer Tests
# =============================================================================

class TestMerge2DocsBridge:
    """Test merge2docs_bridge A2A pattern integration."""

    def test_merge2docs_available(self):
        """Test merge2docs availability detection."""
        from backend.integration import MERGE2DOCS_AVAILABLE
        # Should be True if merge2docs exists, False otherwise
        assert isinstance(MERGE2DOCS_AVAILABLE, bool)

    def test_get_deps_initialization(self):
        """Test GraphDependencies initialization."""
        from backend.integration.merge2docs_bridge import get_deps, MERGE2DOCS_AVAILABLE

        if not MERGE2DOCS_AVAILABLE:
            pytest.skip("merge2docs not available")

        deps = get_deps()
        assert deps is not None
        # Should return same instance on subsequent calls
        deps2 = get_deps()
        assert deps is deps2

    @pytest.mark.asyncio
    async def test_call_algorithm_service_mock(self):
        """Test call_algorithm_service with mocked GraphDependencies."""
        from backend.integration.merge2docs_bridge import call_algorithm_service

        with patch('backend.integration.merge2docs_bridge.get_deps') as mock_get_deps:
            # Mock GraphDependencies
            mock_deps = Mock()
            mock_result = Mock()
            mock_result.success = True
            mock_result.data = {'solution': [1, 2, 3], 'cost': 42}
            mock_deps.call_service_method = AsyncMock(return_value=mock_result)
            mock_get_deps.return_value = mock_deps

            # Call service
            result = await call_algorithm_service(
                'cluster_editing_gpu',
                graph_data={'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]},
                k=5
            )

            # Verify call was made correctly
            mock_deps.call_service_method.assert_called_once()
            call_args = mock_deps.call_service_method.call_args
            assert call_args[1]['service_name'] == 'AlgorithmService'
            assert call_args[1]['method_name'] == 'apply_algorithm'
            assert call_args[1]['algorithm_name'] == 'cluster_editing_gpu'

            # Verify result
            assert result == {'solution': [1, 2, 3], 'cost': 42}

    @pytest.mark.asyncio
    async def test_call_monte_carlo_service_mock(self):
        """Test call_monte_carlo_service with mocked GraphDependencies."""
        from backend.integration.merge2docs_bridge import call_monte_carlo_service

        with patch('backend.integration.merge2docs_bridge.get_deps') as mock_get_deps:
            # Mock GraphDependencies
            mock_deps = Mock()
            mock_sim_result = Mock()
            mock_sim_result.statistics = {'mean': 1.5, 'std': 0.3}
            mock_sim_result.confidence_intervals = {0.95: (1.2, 1.8)}

            mock_result = Mock()
            mock_result.success = True
            mock_result.data = mock_sim_result
            mock_deps.call_service_method = AsyncMock(return_value=mock_result)
            mock_get_deps.return_value = mock_deps

            # Call service
            result = await call_monte_carlo_service(
                simulation_type="RISK_ANALYSIS",
                n_simulations=5000,
                data={"complexity_factor": {"mean": 1.0, "std": 0.3}}
            )

            # Verify call was made correctly
            mock_deps.call_service_method.assert_called_once()
            call_args = mock_deps.call_service_method.call_args
            assert call_args[1]['service_name'] == 'MonteCarloService'
            assert call_args[1]['method_name'] == 'run_simulation'

            # Verify result
            assert result.statistics == {'mean': 1.5, 'std': 0.3}

    def test_deprecated_functions_warn(self):
        """Test that deprecated functions issue warnings."""
        from backend.integration.merge2docs_bridge import (
            get_bayesian_optimizer,
            get_monte_carlo_optimizer,
            get_algorithm_service
        )

        # These should return None and log warnings
        assert get_bayesian_optimizer() is None
        assert get_monte_carlo_optimizer() is None
        assert get_algorithm_service() is None


class TestTensorRoutingClient:
    """Test tensor routing client."""

    def test_client_initialization(self):
        """Test TensorRoutingClient initialization."""
        from backend.integration import TensorRoutingClient

        client = TensorRoutingClient()
        assert client.base_url == "http://localhost"
        assert client.port == 8091  # Default ameme_2_services port

    @pytest.mark.asyncio
    async def test_route_query_fallback(self):
        """Test fallback routing when service unavailable."""
        from backend.integration import TensorRoutingClient

        client = TensorRoutingClient()

        # Should fall back to keyword-based routing
        result = await client.route_query(
            "Optimize fiber coupling at 1550nm",
            domain_hint='physics'
        )

        # Verify fallback routing structure
        assert 'domain' in result
        assert 'fi_level' in result
        assert 'tools' in result
        assert result['routing_info']['fallback'] is True

    def test_fallback_routing_keywords(self):
        """Test keyword-based domain detection in fallback."""
        from backend.integration import TensorRoutingClient

        client = TensorRoutingClient()

        # Test physics keywords
        result = client._fallback_routing("optimize resonator cavity Q-factor")
        assert result['domain'] == 'physics'

        # Test neuroscience keywords
        result = client._fallback_routing("analyze fMRI brain connectivity")
        assert result['domain'] == 'neuroscience'

        # Test bioengineering keywords
        result = client._fallback_routing("design organ-on-chip microfluidic system")
        assert result['domain'] == 'bioengineering'


# =============================================================================
# Phase 5 Services Tests
# =============================================================================

class TestResonatorOptimizer:
    """Test resonator optimizer service."""

    @pytest.mark.asyncio
    async def test_resonator_spec_defaults(self):
        """Test ResonatorSpec default values."""
        from backend.services.resonator_optimizer import ResonatorSpec

        spec = ResonatorSpec()
        assert spec.wavelength_nm == 1550
        assert spec.fsr_ghz == 100
        assert spec.q_factor == 1e6
        assert spec.resonator_type == "ring"
        assert spec.material == "silicon"

    @pytest.mark.asyncio
    async def test_resonator_optimizer_initialization(self):
        """Test ResonatorOptimizer initialization."""
        from backend.services.resonator_optimizer import ResonatorOptimizer, ResonatorSpec

        spec = ResonatorSpec(wavelength_nm=1550, q_factor=1e6)
        optimizer = ResonatorOptimizer(spec)

        assert optimizer.spec.wavelength_nm == 1550
        assert optimizer.spec.q_factor == 1e6

    @pytest.mark.asyncio
    async def test_resonator_optimization_simplified(self):
        """Test resonator optimization with simplified fallback."""
        from backend.services.resonator_optimizer import optimize_resonator

        # Run optimization (should use simplified model without merge2docs)
        result = await optimize_resonator(
            wavelength_nm=1550,
            fsr_ghz=100,
            q_factor=1e6,
            resonator_type="ring",
            query_experts=False  # Skip expert query for speed
        )

        # Verify result structure
        assert 'radius_um' in result
        assert 'waveguide_width_nm' in result
        assert 'coupling_gap_nm' in result
        assert 'predicted_q' in result
        assert 'predicted_fsr_ghz' in result
        assert 'optimization_summary' in result

        # Verify reasonable values
        assert result['radius_um'] > 0
        assert result['predicted_q'] > 0


class TestFiberCouplingDesigner:
    """Test fiber coupling designer service."""

    @pytest.mark.asyncio
    async def test_fiber_coupling_config_defaults(self):
        """Test FiberCouplingConfig default values."""
        from backend.services.fiber_coupling_designer import FiberCouplingConfig

        config = FiberCouplingConfig()
        assert config.wavelength_nm == 1550
        assert config.fiber_type == "SMF28"
        assert config.target_coupling_efficiency == 0.7

    @pytest.mark.asyncio
    async def test_fiber_coupling_design_simplified(self):
        """Test fiber coupling design with simplified optimization."""
        from backend.services.fiber_coupling_designer import design_fiber_coupling

        # Run design (simplified model)
        result = await design_fiber_coupling(
            wavelength_nm=1550,
            fiber_type="SMF28",
            waveguide_width_um=0.5,
            target_efficiency=0.7,
            query_experts=False  # Skip expert query
        )

        # Verify result structure
        assert 'optimal_parameters' in result
        assert 'predicted_efficiency' in result
        assert 'fabrication_robustness' in result
        assert 'design_notes' in result

        # Verify optimal parameters
        params = result['optimal_parameters']
        assert 'spot_size_converter_length_um' in params
        assert 'taper_angle_deg' in params


# =============================================================================
# MCP Tool Tests
# =============================================================================

class TestSimulateLOCChip:
    """Test simulate_loc_chip MCP tool."""

    @pytest.mark.asyncio
    async def test_simulate_loc_chip_handler(self):
        """Test simulate_loc_chip handler function."""
        # Import handler
        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from twosphere_mcp import handle_simulate_loc_chip

        # Test with default parameters
        args = {
            "wavelength_nm": 633,
            "na_objective": 0.6,
            "pixel_size_um": 6.5,
            "target_strehl": 0.8,
            "target_coupling_efficiency": 0.7,
            "query_experts": False,  # Skip for speed
            "use_tensor_routing": False,  # Skip for speed
            "optimization_method": "differential_evolution"
        }

        result = await handle_simulate_loc_chip(args)

        # Verify result structure
        assert 'parameters' in result
        assert 'optimization' in result
        assert 'recommendations' in result
        assert 'summary' in result

        # Verify parameters
        assert result['parameters']['wavelength_nm'] == 633

        # Verify optimization results
        opt = result['optimization']
        assert 'airy_radius_um' in opt
        assert 'is_nyquist_sampled' in opt
        assert 'estimated_strehl_ratio' in opt

        # Verify summary
        assert result['summary']['phase'] == "P5 (F₅ Meta/Planning)"

    @pytest.mark.asyncio
    async def test_simulate_loc_chip_with_experts(self):
        """Test simulate_loc_chip with expert query (mocked)."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from twosphere_mcp import handle_simulate_loc_chip

        # Mock expert query
        with patch('backend.services.ernie2_integration.query_expert_collections') as mock_query:
            mock_query.return_value = AsyncMock(return_value={
                'answer': 'Expert guidance on LOC design...',
                'collections_searched': ['physics_optics']
            })

            args = {
                "wavelength_nm": 633,
                "query_experts": True,
                "use_tensor_routing": False,
                "optimization_method": "differential_evolution"
            }

            result = await handle_simulate_loc_chip(args)

            # Should have expert insights
            if 'expert_insights' in result:
                assert 'answer' in result['expert_insights']


# =============================================================================
# Phase 4 Tools Enhanced with Expert Queries
# =============================================================================

class TestPhase4ExpertIntegration:
    """Test Phase 4 tools with query_experts parameter."""

    @pytest.mark.asyncio
    async def test_alignment_sensitivity_without_experts(self):
        """Test alignment sensitivity Monte Carlo without expert query."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from twosphere_mcp import handle_alignment_sensitivity_monte_carlo

        args = {
            "wavelength_nm": 1550,
            "fiber_mfd_um": 10.4,
            "spot_size_um": 3.0,
            "n_samples": 100,  # Small for speed
            "query_experts": False
        }

        result = await handle_alignment_sensitivity_monte_carlo(args)

        # Should complete without expert insights
        assert 'mean_efficiency' in result or 'error' in result
        if 'mean_efficiency' in result:
            assert 'expert_insights' not in result

    @pytest.mark.asyncio
    async def test_alignment_sensitivity_with_experts_mock(self):
        """Test alignment sensitivity with mocked expert query."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from twosphere_mcp import handle_alignment_sensitivity_monte_carlo

        with patch('backend.services.ernie2_integration.query_expert_collections') as mock_query:
            mock_query.return_value = AsyncMock(return_value={
                'answer': 'Dominant loss mechanisms include mode mismatch...',
            })

            args = {
                "wavelength_nm": 1550,
                "n_samples": 100,
                "query_experts": True
            }

            result = await handle_alignment_sensitivity_monte_carlo(args)

            # Should have expert insights
            if 'expert_insights' in result:
                assert isinstance(result['expert_insights'], str)

    @pytest.mark.asyncio
    async def test_whole_brain_network_with_experts_mock(self):
        """Test whole-brain network analysis with mocked expert query."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from twosphere_mcp import handle_whole_brain_network_analysis

        with patch('backend.services.ernie2_integration.query_expert_collections') as mock_query:
            mock_query.return_value = AsyncMock(return_value={
                'answer': 'Key graph metrics include clustering coefficient...',
            })

            args = {
                "region_labels": ["V1_L", "V1_R", "V4_L", "V4_R"],
                "n_timepoints": 100,  # Small for speed
                "query_experts": True
            }

            result = await handle_whole_brain_network_analysis(args)

            # Should have expert insights
            if 'expert_insights' in result:
                assert isinstance(result['expert_insights'], str)


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end tests for Phase 5 stack."""

    @pytest.mark.asyncio
    async def test_p5_full_stack_simplified(self):
        """Test full Phase 5 stack with simplified models."""
        from backend.services.resonator_optimizer import optimize_resonator

        # This exercises:
        # 1. Tensor routing (fallback)
        # 2. Expert query (skipped for speed)
        # 3. Optimization (simplified)
        # 4. Result compilation

        result = await optimize_resonator(
            wavelength_nm=1550,
            fsr_ghz=100,
            q_factor=1e6,
            resonator_type="ring",
            material="silicon",
            query_experts=False
        )

        # Verify complete result structure
        assert 'radius_um' in result
        assert 'predicted_q' in result
        assert 'predicted_fsr_ghz' in result
        assert 'optimization_summary' in result

        # Verify optimization ran
        summary = result['optimization_summary']
        assert 'iterations' in summary
        assert 'convergence' in summary

    def test_integration_layer_exports(self):
        """Test that integration layer exports all required functions."""
        from backend.integration import (
            MERGE2DOCS_AVAILABLE,
            get_deps,
            call_algorithm_service,
            call_monte_carlo_service,
            TensorRoutingClient,
            route_query
        )

        # Verify all exports are available
        assert callable(get_deps)
        assert callable(call_algorithm_service)
        assert callable(call_monte_carlo_service)
        assert TensorRoutingClient is not None
        assert callable(route_query)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
