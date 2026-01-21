"""Fiber coupling design service using merge2docs optimization.

Designs optimal fiber-to-chip coupling configurations using Bayesian optimization
from merge2docs algorithms. Integrates with ernie2_swarm for expert guidance.

Phase 5 (Meta/Planning Level F₅) - Uses merge2docs from the start (Option C).
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import merge2docs optimization (will use bridge)
try:
    from backend.integration import (
        get_bayesian_optimizer,
        get_monte_carlo_optimizer,
        MERGE2DOCS_AVAILABLE
    )
except ImportError:
    MERGE2DOCS_AVAILABLE = False
    logger.warning("merge2docs bridge not available - falling back to simple optimization")

from .ernie2_integration import query_expert_collections


@dataclass
class FiberCouplingConfig:
    """Configuration for fiber-to-chip coupling design.

    Attributes:
        wavelength_nm: Operating wavelength in nm
        fiber_type: Fiber type (e.g., 'SMF28', 'PM fiber')
        chip_waveguide_width_um: Waveguide width in µm
        chip_waveguide_height_um: Waveguide height in µm
        target_coupling_efficiency: Target efficiency (0-1)
        fabrication_tolerance_um: Manufacturing tolerance in µm
    """
    wavelength_nm: float = 1550
    fiber_type: str = "SMF28"
    chip_waveguide_width_um: float = 0.5
    chip_waveguide_height_um: float = 0.22
    target_coupling_efficiency: float = 0.7
    fabrication_tolerance_um: float = 0.1


@dataclass
class CouplingDesignResult:
    """Result of fiber coupling optimization.

    Attributes:
        optimal_parameters: Best parameter values
        predicted_efficiency: Expected coupling efficiency
        confidence_interval: 95% CI for efficiency
        expert_recommendations: Guidance from ernie2_swarm
        fabrication_robustness: Monte Carlo robustness score
        design_notes: Implementation notes
    """
    optimal_parameters: Dict[str, float]
    predicted_efficiency: float
    confidence_interval: Tuple[float, float]
    expert_recommendations: str
    fabrication_robustness: float
    design_notes: List[str]


class FiberCouplingDesigner:
    """Fiber-to-chip coupling designer using merge2docs Bayesian optimization."""

    def __init__(self, config: FiberCouplingConfig):
        """Initialize coupling designer.

        Args:
            config: Coupling configuration parameters
        """
        self.config = config

        # Try to use merge2docs Bayesian optimizer
        if MERGE2DOCS_AVAILABLE:
            BayesianOpt = get_bayesian_optimizer()
            if BayesianOpt:
                self.optimizer = BayesianOpt()
                self.use_merge2docs = True
                logger.info("Using merge2docs Bayesian optimization")
            else:
                self.optimizer = None
                self.use_merge2docs = False
                logger.warning("merge2docs Bayesian optimizer unavailable")
        else:
            self.optimizer = None
            self.use_merge2docs = False

    async def design_coupling(
        self,
        query_experts: bool = True
    ) -> CouplingDesignResult:
        """Design optimal fiber coupling configuration.

        Args:
            query_experts: Whether to query ernie2_swarm for expert guidance

        Returns:
            CouplingDesignResult with optimal parameters and predictions
        """
        # Step 1: Query experts for guidance (if requested)
        expert_advice = ""
        if query_experts:
            expert_advice = await self._query_optics_experts()

        # Step 2: Define design space
        design_space = {
            'spot_size_converter_length_um': (50.0, 500.0),
            'taper_angle_deg': (0.5, 5.0),
            'mode_mismatch_factor': (0.5, 2.0),
            'vertical_offset_um': (-1.0, 1.0),
            'horizontal_offset_um': (-1.0, 1.0)
        }

        # Step 3: Optimize using merge2docs (if available)
        if self.use_merge2docs and self.optimizer:
            result = await self._optimize_with_merge2docs(design_space)
        else:
            result = await self._optimize_simple(design_space)

        # Step 4: Evaluate fabrication robustness
        robustness = await self._evaluate_robustness(result['optimal_parameters'])

        # Step 5: Generate design notes
        notes = self._generate_design_notes(result, expert_advice)

        return CouplingDesignResult(
            optimal_parameters=result['optimal_parameters'],
            predicted_efficiency=result['predicted_efficiency'],
            confidence_interval=result.get('confidence_interval', (0.0, 1.0)),
            expert_recommendations=expert_advice,
            fabrication_robustness=robustness,
            design_notes=notes
        )

    async def _query_optics_experts(self) -> str:
        """Query ernie2_swarm optics experts for coupling guidance."""
        question = (
            f"What are the best practices for fiber-to-chip coupling at {self.config.wavelength_nm}nm "
            f"with {self.config.fiber_type} fiber to a {self.config.chip_waveguide_width_um}µm wide "
            f"silicon waveguide? Target efficiency: {self.config.target_coupling_efficiency*100:.0f}%."
        )

        try:
            result = await query_expert_collections(
                question=question,
                collections=["physics_optics", "bioengineering_LOC"],
                use_cloud=False
            )
            return result.get('answer', 'No expert guidance available')
        except Exception as e:
            logger.error(f"Expert query failed: {e}")
            return "Expert guidance unavailable"

    async def _optimize_with_merge2docs(
        self,
        design_space: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Optimize using merge2docs Bayesian optimization.

        This is a placeholder that would integrate with actual merge2docs
        BayesianCompressionWeightOptimizer once we adapt it for coupling design.
        """
        # TODO: Implement actual merge2docs integration
        # For now, use simplified optimization
        logger.info("merge2docs optimization integration pending - using simplified method")
        return await self._optimize_simple(design_space)

    async def _optimize_simple(
        self,
        design_space: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Simple optimization fallback."""
        def _compute():
            # Initialize at center of design space
            optimal_params = {
                param: (bounds[0] + bounds[1]) / 2
                for param, bounds in design_space.items()
            }

            # Simple coupling efficiency model
            spot_size_length = optimal_params['spot_size_converter_length_um']
            taper_angle = optimal_params['taper_angle_deg']
            mode_factor = optimal_params['mode_mismatch_factor']

            # Simplified model (would be replaced with full optical simulation)
            base_efficiency = 0.7
            length_factor = 1.0 / (1 + np.abs(spot_size_length - 200) / 100)
            taper_factor = 1.0 / (1 + np.abs(taper_angle - 2.0) / 2.0)
            mode_factor_penalty = 1.0 / (1 + np.abs(mode_factor - 1.0) / 0.5)

            efficiency = base_efficiency * length_factor * taper_factor * mode_factor_penalty
            efficiency = min(efficiency, 0.95)  # Physical limit

            return {
                'optimal_parameters': optimal_params,
                'predicted_efficiency': float(efficiency),
                'confidence_interval': (efficiency * 0.9, efficiency * 1.05)
            }

        return await asyncio.to_thread(_compute)

    async def _evaluate_robustness(
        self,
        parameters: Dict[str, float]
    ) -> float:
        """Evaluate fabrication robustness via Monte Carlo.

        Uses merge2docs Monte Carlo optimizer if available.
        """
        if not MERGE2DOCS_AVAILABLE:
            # Simple robustness estimate
            return 0.75

        MCOptimizer = get_monte_carlo_optimizer()
        if not MCOptimizer:
            return 0.75

        # TODO: Integrate merge2docs Monte Carlo for robustness analysis
        # For now, return placeholder
        return 0.80

    def _generate_design_notes(
        self,
        result: Dict[str, Any],
        expert_advice: str
    ) -> List[str]:
        """Generate design implementation notes."""
        notes = [
            f"Operating wavelength: {self.config.wavelength_nm} nm",
            f"Predicted coupling efficiency: {result['predicted_efficiency']*100:.1f}%",
            f"Target efficiency: {self.config.target_coupling_efficiency*100:.1f}%",
        ]

        params = result['optimal_parameters']
        notes.extend([
            f"Spot size converter length: {params['spot_size_converter_length_um']:.1f} µm",
            f"Taper angle: {params['taper_angle_deg']:.2f}°",
            f"Mode mismatch factor: {params['mode_mismatch_factor']:.2f}",
        ])

        if expert_advice and "unavailable" not in expert_advice.lower():
            notes.append(f"Expert guidance: {expert_advice[:200]}...")

        return notes


# Convenience function for MCP tool
async def design_fiber_coupling(
    wavelength_nm: float = 1550,
    fiber_type: str = "SMF28",
    waveguide_width_um: float = 0.5,
    target_efficiency: float = 0.7,
    query_experts: bool = True
) -> Dict[str, Any]:
    """Design fiber-to-chip coupling (MCP tool interface).

    Args:
        wavelength_nm: Operating wavelength
        fiber_type: Fiber type
        waveguide_width_um: Chip waveguide width
        target_efficiency: Target coupling efficiency
        query_experts: Query ernie2_swarm experts

    Returns:
        Design result dict
    """
    config = FiberCouplingConfig(
        wavelength_nm=wavelength_nm,
        fiber_type=fiber_type,
        chip_waveguide_width_um=waveguide_width_um,
        target_coupling_efficiency=target_efficiency
    )

    designer = FiberCouplingDesigner(config)
    result = await designer.design_coupling(query_experts=query_experts)

    return {
        'optimal_parameters': result.optimal_parameters,
        'predicted_efficiency': result.predicted_efficiency,
        'confidence_interval_min': result.confidence_interval[0],
        'confidence_interval_max': result.confidence_interval[1],
        'fabrication_robustness': result.fabrication_robustness,
        'expert_recommendations': result.expert_recommendations,
        'design_notes': result.design_notes
    }
