"""Glymphatic-fMRI Integration Module.

Bridges brain functional connectivity (fMRI) with glymphatic clearance simulation.

Key insight: Brain network topology from fMRI correlates with waste clearance
efficiency through the glymphatic system. This module connects:
1. WholeBrainNetworkAnalyzer → functional connectivity matrices
2. ClearanceNetworkAnalyzer → disc dimension & clearance prediction
3. GlymphaticFlowSimulator → perivascular flow dynamics

Based on PH-7 hypothesis: Networks with disc dimension ~2.5 (near-planar)
provide optimal waste clearance routing.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum

from .clearance_network import ClearanceNetworkAnalyzer, disc_dimension_clearance_model
from .glymphatic_flow import (
    GlymphaticFlowSimulator,
    PerivascularSpace,
)


class BrainState(Enum):
    """Brain state affecting glymphatic function."""
    AWAKE = "awake"
    SLEEP = "sleep"
    ANESTHESIA = "anesthesia"


# Sleep increases interstitial space by ~60% (Xie et al. 2013)
STATE_CLEARANCE_MULTIPLIER = {
    BrainState.AWAKE: 1.0,
    BrainState.SLEEP: 1.6,
    BrainState.ANESTHESIA: 1.8,
}


@dataclass
class RegionalClearance:
    """Clearance metrics for a brain region."""
    region_name: str
    connectivity_strength: float  # Mean connection weight
    local_clustering: float  # Local clustering coefficient
    centrality: float  # Betweenness centrality
    perivascular_flow_rate: float  # µL/min
    clearance_efficiency: float  # 0-1 scale
    predicted_clearance_time_hr: float  # Hours for 90% clearance


@dataclass
class AmyloidAccumulation:
    """Amyloid-β accumulation dynamics for a region."""
    region_name: str
    initial_concentration: float  # Arbitrary units (normalized)
    production_rate: float  # Units/hour
    clearance_rate: float  # 1/hour (first-order kinetics)
    steady_state_concentration: float  # Production/clearance
    half_life_hr: float  # Time to clear 50%
    risk_score: float  # 0-1, higher = more accumulation risk


@dataclass
class GlymphaticAnalysisResult:
    """Complete glymphatic-fMRI analysis result."""
    # Network topology
    disc_dimension: float
    is_planar: bool
    network_efficiency: float

    # Clearance prediction
    global_clearance_efficiency: float
    state: BrainState
    state_adjusted_efficiency: float

    # Regional analysis
    regional_clearances: List[RegionalClearance] = field(default_factory=list)
    high_risk_regions: List[str] = field(default_factory=list)

    # Amyloid dynamics (if computed)
    amyloid_dynamics: Optional[List[AmyloidAccumulation]] = None

    # Metadata
    connectivity_method: str = ""
    n_regions: int = 0
    analysis_timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "network_topology": {
                "disc_dimension": self.disc_dimension,
                "is_planar": self.is_planar,
                "network_efficiency": self.network_efficiency,
            },
            "clearance_prediction": {
                "global_clearance_efficiency": self.global_clearance_efficiency,
                "brain_state": self.state.value,
                "state_adjusted_efficiency": self.state_adjusted_efficiency,
                "interpretation": self._interpret_clearance(),
            },
            "regional_analysis": {
                "n_regions": self.n_regions,
                "regional_clearances": [
                    {
                        "region": r.region_name,
                        "connectivity_strength": r.connectivity_strength,
                        "local_clustering": r.local_clustering,
                        "centrality": r.centrality,
                        "flow_rate_uL_min": r.perivascular_flow_rate,
                        "clearance_efficiency": r.clearance_efficiency,
                        "clearance_time_hr": r.predicted_clearance_time_hr,
                    }
                    for r in self.regional_clearances
                ],
                "high_risk_regions": self.high_risk_regions,
            },
            "amyloid_dynamics": (
                [
                    {
                        "region": a.region_name,
                        "production_rate": a.production_rate,
                        "clearance_rate": a.clearance_rate,
                        "steady_state": a.steady_state_concentration,
                        "half_life_hr": a.half_life_hr,
                        "risk_score": a.risk_score,
                    }
                    for a in self.amyloid_dynamics
                ]
                if self.amyloid_dynamics
                else None
            ),
            "metadata": {
                "connectivity_method": self.connectivity_method,
                "analysis_timestamp": self.analysis_timestamp,
            },
        }

    def _interpret_clearance(self) -> str:
        """Interpret clearance efficiency."""
        eff = self.state_adjusted_efficiency
        if eff >= 0.85:
            return "Excellent - optimal glymphatic function"
        elif eff >= 0.65:
            return "Good - healthy clearance expected"
        elif eff >= 0.45:
            return "Moderate - some clearance impairment"
        elif eff >= 0.25:
            return "Reduced - significant impairment, monitor for accumulation"
        else:
            return "Poor - high risk of metabolite accumulation"


class GlymphaticFMRIIntegrator:
    """Integrates fMRI connectivity analysis with glymphatic clearance simulation.

    This class bridges:
    - Brain network topology (from fMRI functional connectivity)
    - Disc dimension theory (Paul et al. 2023)
    - Perivascular flow physics (Stokes flow)
    - Waste clearance prediction
    """

    # Default perivascular space parameters by vessel type
    PVS_PARAMS = {
        "artery": {"vessel_radius_um": 50, "gap_um": 20, "length_mm": 5},
        "arteriole": {"vessel_radius_um": 15, "gap_um": 8, "length_mm": 2},
        "capillary": {"vessel_radius_um": 4, "gap_um": 3, "length_mm": 0.5},
        "venule": {"vessel_radius_um": 20, "gap_um": 10, "length_mm": 2},
    }

    # Typical pressure gradients (Pa/m) in brain vasculature
    PRESSURE_GRADIENTS = {
        "artery": 100,
        "arteriole": 50,
        "capillary": 20,
        "venule": 30,
    }

    def __init__(
        self,
        optimal_disc_dim: float = 2.5,
        clearance_beta: float = 0.3,
    ):
        """Initialize integrator.

        Args:
            optimal_disc_dim: Optimal disc dimension for clearance (default 2.5)
            clearance_beta: Sensitivity parameter for disc-clearance model
        """
        self.optimal_disc_dim = optimal_disc_dim
        self.clearance_beta = clearance_beta
        self.clearance_analyzer = ClearanceNetworkAnalyzer(
            optimal_disc_dim=optimal_disc_dim,
            beta=clearance_beta
        )

    async def analyze_from_connectivity_matrix(
        self,
        connectivity_matrix: NDArray,
        region_labels: List[str],
        brain_state: BrainState = BrainState.AWAKE,
        connectivity_method: str = "fMRI",
        compute_amyloid: bool = False,
    ) -> GlymphaticAnalysisResult:
        """Analyze glymphatic function from fMRI connectivity matrix.

        Args:
            connectivity_matrix: NxN functional connectivity matrix
            region_labels: List of brain region names
            brain_state: Current brain state (awake/sleep/anesthesia)
            connectivity_method: Method used to compute connectivity
            compute_amyloid: Whether to compute amyloid-β dynamics

        Returns:
            GlymphaticAnalysisResult with complete analysis
        """
        import datetime

        n_regions = connectivity_matrix.shape[0]

        # Step 1: Network topology analysis
        network_analysis = self.clearance_analyzer.analyze_network(
            np.array(connectivity_matrix)
        )

        disc_dim = network_analysis["clearance_prediction"]["disc_dimension"]
        is_planar = network_analysis["network_metrics"]["is_planar"]
        base_efficiency = network_analysis["clearance_prediction"]["clearance_efficiency"]

        # Step 2: State-adjusted efficiency
        state_multiplier = STATE_CLEARANCE_MULTIPLIER[brain_state]
        adjusted_efficiency = min(base_efficiency * state_multiplier, 1.0)

        # Step 3: Regional clearance analysis
        regional_clearances = await self._compute_regional_clearance(
            connectivity_matrix,
            region_labels,
            network_analysis,
            brain_state,
        )

        # Step 4: Identify high-risk regions (low clearance efficiency)
        high_risk_threshold = 0.4
        high_risk_regions = [
            r.region_name
            for r in regional_clearances
            if r.clearance_efficiency < high_risk_threshold
        ]

        # Step 5: Amyloid dynamics (optional)
        amyloid_dynamics = None
        if compute_amyloid:
            amyloid_dynamics = await self._compute_amyloid_dynamics(
                regional_clearances,
                brain_state,
            )

        return GlymphaticAnalysisResult(
            disc_dimension=disc_dim,
            is_planar=is_planar,
            network_efficiency=float(network_analysis["network_metrics"]["clustering_coefficient"]),
            global_clearance_efficiency=base_efficiency,
            state=brain_state,
            state_adjusted_efficiency=adjusted_efficiency,
            regional_clearances=regional_clearances,
            high_risk_regions=high_risk_regions,
            amyloid_dynamics=amyloid_dynamics,
            connectivity_method=connectivity_method,
            n_regions=n_regions,
            analysis_timestamp=datetime.datetime.now().isoformat(),
        )

    async def analyze_from_fmri_results(
        self,
        fmri_analysis: Dict[str, Any],
        brain_state: BrainState = BrainState.AWAKE,
        compute_amyloid: bool = False,
    ) -> GlymphaticAnalysisResult:
        """Analyze glymphatic function from WholeBrainNetworkAnalyzer results.

        Args:
            fmri_analysis: Output from WholeBrainNetworkAnalyzer.run_complete_analysis()
            brain_state: Current brain state
            compute_amyloid: Whether to compute amyloid-β dynamics

        Returns:
            GlymphaticAnalysisResult
        """
        connectivity_matrix = np.array(fmri_analysis["connectivity_matrix"])
        region_labels = fmri_analysis.get("region_labels", [
            f"region_{i}" for i in range(connectivity_matrix.shape[0])
        ])
        connectivity_method = fmri_analysis.get("connectivity_method", "fMRI")

        return await self.analyze_from_connectivity_matrix(
            connectivity_matrix,
            region_labels,
            brain_state,
            connectivity_method,
            compute_amyloid,
        )

    async def _compute_regional_clearance(
        self,
        connectivity_matrix: NDArray,
        region_labels: List[str],
        network_analysis: Dict[str, Any],
        brain_state: BrainState,
    ) -> List[RegionalClearance]:
        """Compute clearance metrics for each brain region."""
        n_regions = connectivity_matrix.shape[0]
        results = []

        # Get global disc dimension for baseline
        global_disc_dim = network_analysis["clearance_prediction"]["disc_dimension"]

        # Create flow simulator for current state
        simulator = GlymphaticFlowSimulator(state=brain_state.value)

        for i, region in enumerate(region_labels):
            # Regional connectivity strength
            conn_strength = float(np.mean(connectivity_matrix[i, :]))

            # Local clustering (from network analysis if available, else compute)
            # Approximate local clustering from connectivity
            neighbors = np.where(connectivity_matrix[i, :] > np.mean(connectivity_matrix))[0]
            if len(neighbors) >= 2:
                submatrix = connectivity_matrix[np.ix_(neighbors, neighbors)]
                local_clustering = float(np.mean(submatrix > 0))
            else:
                local_clustering = 0.0

            # Betweenness centrality approximation (based on connectivity)
            # Regions with many strong connections have higher centrality
            centrality = float(np.sum(connectivity_matrix[i, :] > np.mean(connectivity_matrix)) / n_regions)

            # Estimate local disc dimension (deviation from global)
            # High clustering → lower local disc dimension → better local clearance
            local_disc_deviation = (1 - local_clustering) * 0.5
            local_disc_dim = global_disc_dim + local_disc_deviation

            # Regional clearance efficiency (base from topology)
            base_regional_efficiency = disc_dimension_clearance_model(
                local_disc_dim,
                self.optimal_disc_dim,
                beta=self.clearance_beta
            )
            # Apply state multiplier (sleep increases clearance by ~60%)
            state_multiplier = STATE_CLEARANCE_MULTIPLIER[brain_state]
            regional_efficiency = min(base_regional_efficiency * state_multiplier, 1.0)

            # Perivascular flow simulation
            # Use arteriole parameters as representative
            pvs = PerivascularSpace(
                vessel_radius_um=self.PVS_PARAMS["arteriole"]["vessel_radius_um"],
                gap_thickness_um=self.PVS_PARAMS["arteriole"]["gap_um"],
                length_mm=self.PVS_PARAMS["arteriole"]["length_mm"],
            )
            flow_result = simulator.compute_steady_flow(
                pvs,
                self.PRESSURE_GRADIENTS["arteriole"]
            )
            flow_rate = flow_result["flow_rate_uL_min"]

            # Clearance time estimation
            # Based on region volume (~1 cm³) and flow rate
            region_volume_uL = 1000  # 1 cm³ = 1000 µL
            if flow_rate > 0 and regional_efficiency > 0:
                clearance_time_hr = (region_volume_uL / flow_rate / regional_efficiency) / 60
            else:
                clearance_time_hr = float('inf')

            results.append(RegionalClearance(
                region_name=region,
                connectivity_strength=conn_strength,
                local_clustering=local_clustering,
                centrality=centrality,
                perivascular_flow_rate=flow_rate,
                clearance_efficiency=float(regional_efficiency),
                predicted_clearance_time_hr=min(clearance_time_hr, 168),  # Cap at 1 week
            ))

        return results

    async def _compute_amyloid_dynamics(
        self,
        regional_clearances: List[RegionalClearance],
        brain_state: BrainState,
    ) -> List[AmyloidAccumulation]:
        """Compute amyloid-β accumulation dynamics for each region.

        Model: dAβ/dt = production - clearance_rate × Aβ
        Steady state: Aβ_ss = production / clearance_rate

        Clearance rate depends on regional clearance efficiency and brain state.
        """
        results = []

        # Base amyloid production rate (arbitrary normalized units)
        BASE_PRODUCTION_RATE = 1.0  # units/hour

        # Base clearance rate coefficient
        BASE_CLEARANCE_COEFF = 0.1  # 1/hour

        state_factor = STATE_CLEARANCE_MULTIPLIER[brain_state]

        for rc in regional_clearances:
            # Regional production varies with activity (connectivity)
            # High connectivity → high activity → high production
            regional_production = BASE_PRODUCTION_RATE * (0.5 + 0.5 * rc.connectivity_strength)

            # Clearance rate depends on local efficiency and brain state
            clearance_rate = BASE_CLEARANCE_COEFF * rc.clearance_efficiency * state_factor

            # Steady state concentration
            if clearance_rate > 0:
                steady_state = regional_production / clearance_rate
                half_life = np.log(2) / clearance_rate
            else:
                steady_state = float('inf')
                half_life = float('inf')

            # Risk score: higher for high production, low clearance
            # Normalized to 0-1 range
            risk_score = min(1.0, steady_state / 20.0)  # 20 is "high risk" threshold

            results.append(AmyloidAccumulation(
                region_name=rc.region_name,
                initial_concentration=1.0,  # Normalized baseline
                production_rate=regional_production,
                clearance_rate=clearance_rate,
                steady_state_concentration=min(steady_state, 100.0),
                half_life_hr=min(half_life, 168.0),
                risk_score=float(risk_score),
            ))

        return results

    async def compare_sleep_wake_clearance(
        self,
        connectivity_matrix: NDArray,
        region_labels: List[str],
    ) -> Dict[str, Any]:
        """Compare clearance efficiency between sleep and wake states.

        Demonstrates the ~60% improvement in clearance during sleep
        (Xie et al. 2013: "Sleep drives metabolite clearance from the adult brain").

        Args:
            connectivity_matrix: NxN functional connectivity matrix
            region_labels: Brain region names

        Returns:
            Comparison of awake vs sleep clearance metrics
        """
        awake_result = await self.analyze_from_connectivity_matrix(
            connectivity_matrix,
            region_labels,
            brain_state=BrainState.AWAKE,
            compute_amyloid=True,
        )

        sleep_result = await self.analyze_from_connectivity_matrix(
            connectivity_matrix,
            region_labels,
            brain_state=BrainState.SLEEP,
            compute_amyloid=True,
        )

        # Compute improvement metrics
        efficiency_improvement = (
            (sleep_result.state_adjusted_efficiency - awake_result.state_adjusted_efficiency)
            / awake_result.state_adjusted_efficiency
            * 100
        ) if awake_result.state_adjusted_efficiency > 0 else 0

        # Average clearance time improvement
        awake_times = [r.predicted_clearance_time_hr for r in awake_result.regional_clearances]
        sleep_times = [r.predicted_clearance_time_hr for r in sleep_result.regional_clearances]

        valid_awake = [t for t in awake_times if t < 168]
        valid_sleep = [t for t in sleep_times if t < 168]
        avg_awake_time = float(np.mean(valid_awake)) if valid_awake else 168.0
        avg_sleep_time = float(np.mean(valid_sleep)) if valid_sleep else 168.0

        time_improvement = (
            (avg_awake_time - avg_sleep_time) / avg_awake_time * 100
            if avg_awake_time > 0 else 0
        )

        # Amyloid accumulation risk reduction
        if awake_result.amyloid_dynamics and sleep_result.amyloid_dynamics:
            awake_risk = np.mean([a.risk_score for a in awake_result.amyloid_dynamics])
            sleep_risk = np.mean([a.risk_score for a in sleep_result.amyloid_dynamics])
            risk_reduction = (awake_risk - sleep_risk) / awake_risk * 100 if awake_risk > 0 else 0
        else:
            awake_risk = sleep_risk = risk_reduction = 0

        return {
            "comparison_summary": {
                "efficiency_improvement_percent": float(efficiency_improvement),
                "clearance_time_improvement_percent": float(time_improvement),
                "amyloid_risk_reduction_percent": float(risk_reduction),
            },
            "awake_state": {
                "clearance_efficiency": awake_result.state_adjusted_efficiency,
                "avg_clearance_time_hr": float(avg_awake_time),
                "mean_amyloid_risk": float(awake_risk),
                "high_risk_regions": awake_result.high_risk_regions,
            },
            "sleep_state": {
                "clearance_efficiency": sleep_result.state_adjusted_efficiency,
                "avg_clearance_time_hr": float(avg_sleep_time),
                "mean_amyloid_risk": float(sleep_risk),
                "high_risk_regions": sleep_result.high_risk_regions,
            },
            "clinical_implications": self._interpret_sleep_wake_comparison(
                float(efficiency_improvement), float(time_improvement), float(risk_reduction)
            ),
            "reference": "Xie et al. 2013 Science: Sleep increases interstitial space by ~60%",
        }

    def _interpret_sleep_wake_comparison(
        self,
        efficiency_improvement: float,
        time_improvement: float,
        risk_reduction: float,
    ) -> str:
        """Interpret sleep vs wake comparison results."""
        if efficiency_improvement >= 50:
            return (
                "Strong sleep-dependent clearance benefit. "
                "Consistent with healthy glymphatic function. "
                "Adequate sleep is critical for waste clearance."
            )
        elif efficiency_improvement >= 30:
            return (
                "Moderate sleep-dependent clearance benefit. "
                "Some impairment may be present. "
                "Consider optimizing sleep quality."
            )
        elif efficiency_improvement >= 10:
            return (
                "Reduced sleep-dependent clearance benefit. "
                "Potential glymphatic dysfunction. "
                "Further investigation recommended."
            )
        else:
            return (
                "Minimal sleep-dependent clearance benefit. "
                "Significant glymphatic impairment likely. "
                "Clinical evaluation recommended."
            )


# Convenience function for direct use
async def analyze_brain_clearance(
    connectivity_matrix: NDArray,
    region_labels: List[str],
    brain_state: str = "awake",
    compute_amyloid: bool = False,
) -> Dict[str, Any]:
    """Analyze brain clearance from connectivity matrix.

    Convenience wrapper for GlymphaticFMRIIntegrator.

    Args:
        connectivity_matrix: NxN functional connectivity matrix
        region_labels: List of brain region names
        brain_state: "awake", "sleep", or "anesthesia"
        compute_amyloid: Whether to compute amyloid-β dynamics

    Returns:
        Complete analysis results as dictionary
    """
    state_map = {
        "awake": BrainState.AWAKE,
        "sleep": BrainState.SLEEP,
        "anesthesia": BrainState.ANESTHESIA,
    }

    integrator = GlymphaticFMRIIntegrator()
    result = await integrator.analyze_from_connectivity_matrix(
        np.array(connectivity_matrix),
        region_labels,
        brain_state=state_map.get(brain_state, BrainState.AWAKE),
        compute_amyloid=compute_amyloid,
    )

    return result.to_dict()
