# BEAD: QEC Time-Series Integration for Brain Tensor

**Status**: 🟢 READY (merge2docs integration complete)
**Priority**: HIGH
**Phase**: 4-5 (parallel with test completion)
**Effort**: 12 hours

**Dependencies**:
- merge2docs QEC time-series analysis (✅ available: commit 5cabe8c6)
- Brain tensor Phase 1-3 complete (✅ done: 86% health score)
- PRIME-DE loader operational (✅ done: 93.5% tests passing)

---

## Goal

Integrate merge2docs QEC time-series analysis capabilities into the brain tensor system for:
1. Syndrome evolution tracking (clinical applications)
2. fMRI temporal dynamics analysis (research)
3. Cross-functor causality discovery (theory validation)
4. Feedback loop identification (intervention targets)

---

## Background

### merge2docs QEC Time-Series Analysis

**New Capabilities** (commit 5cabe8c6):
- `TimeSeriesAnalysisService` - Unified time-series analysis
- `track_qec_syndrome_evolution()` - Syndrome tracking
- Granger causality analysis - Causal relationships (not just correlation!)
- Feedback Vertex Set (FVS) - Regulatory control point identification
- Convergence monitoring - Real-time performance tracking

**Key Benefit**: **Causality, not just correlation!**
- Traditional connectivity: Pearson correlation (symmetric, no directionality)
- QEC approach: Granger causality (directed, temporal precedence)
- Result: Discover **which regions/functors drive others**

### Brain Tensor Applications

**Perfect alignment** with existing beads:
- BEAD-P6-2: Syndrome detection → Enhanced with temporal tracking
- BEAD-P6-3: Cross-functor training → Validated by causality discovery
- BEAD-P9-1: Temporal dynamics → Directly supported by time-series tools
- **New**: Feedback loop analysis for clinical interventions

---

## Architecture

### Integration Points

```
┌──────────────────────────────────────┐
│  Brain Tensor System (twosphere-mcp) │
├──────────────────────────────────────┤
│  • PRIME-DE fMRI data (368 regions)  │
│  • 6 functors × 380 regions × 3 scales│
│  • Syndrome detection (cross-functor)│
└────────────┬─────────────────────────┘
             │
             │ Import
             ↓
┌──────────────────────────────────────┐
│  merge2docs QEC Time-Series Analysis │
├──────────────────────────────────────┤
│  • TimeSeriesAnalysisService         │
│  • Granger causality                 │
│  • Feedback Vertex Set (FVS)         │
│  • Syndrome evolution tracking       │
└──────────────────────────────────────┘
```

### Data Flow

```python
# 1. Load fMRI data (T × 368 regions)
timeseries = PRIMEDELoader.load_subject("bordeaux24", "m01", "bold")

# 2. Extract region-specific time-series
region_ts = timeseries[:, region_idx]  # Shape: (T,)

# 3. Analyze with QEC time-series tools
ts_service = TimeSeriesAnalysisService()
result = ts_service.track_qec_syndrome_evolution(region_ts)

# 4. Store results in brain tensor
store_syndrome_analysis(region_id, result)
```

---

## Implementation

### Task 1: Setup Integration (2 hours)

**Goal**: Import merge2docs QEC time-series tools

**Steps**:
1. Add merge2docs to Python path
2. Import `TimeSeriesAnalysisService`
3. Write adapter layer for brain tensor data

**Code**:
```python
# src/backend/services/qec_timeseries_adapter.py

import sys
from pathlib import Path

# Add merge2docs to path
MERGE2DOCS_PATH = Path(__file__).parent.parent.parent.parent / "merge2docs"
sys.path.insert(0, str(MERGE2DOCS_PATH))

from src.backend.services.time_series_analysis_service import TimeSeriesAnalysisService

class BrainQECTimeSeriesAdapter:
    """Adapter for merge2docs QEC time-series analysis."""

    def __init__(self):
        self.ts_service = TimeSeriesAnalysisService()

    def convert_brain_to_qec_format(
        self,
        brain_timeseries: np.ndarray
    ) -> Dict:
        """Convert brain tensor data to QEC syndrome format.

        Args:
            brain_timeseries: (T, 368) or (T,) array

        Returns:
            QEC-compatible syndrome dict
        """
        # QEC expects syndrome evolution across correction cycles
        # Brain has temporal evolution across time points
        # Mapping: time point → correction cycle

        return {
            "syndromes": brain_timeseries,
            "metadata": {
                "source": "brain_tensor",
                "format": "fmri_timeseries"
            }
        }

    def analyze_region_evolution(
        self,
        region_timeseries: np.ndarray
    ) -> Dict:
        """Analyze temporal evolution of a single brain region."""
        qec_format = self.convert_brain_to_qec_format(region_timeseries)

        # Use QEC syndrome tracking
        result = self.ts_service.track_qec_syndrome_evolution(
            qec_format["syndromes"]
        )

        return result
```

**Success Criteria**:
- ✅ Import works without errors
- ✅ Adapter converts brain data to QEC format
- ✅ Basic analysis runs on BORDEAUX24 data

---

### Task 2: Syndrome Evolution Tracking (3 hours)

**Goal**: Track brain syndrome evolution over time

**Enhancement to BEAD-P6-2**:
```python
# src/backend/analysis/syndrome_evolution_tracker.py

from src.backend.services.qec_timeseries_adapter import BrainQECTimeSeriesAdapter

class SyndromeEvolutionTracker:
    """Track syndrome evolution using QEC time-series analysis."""

    def __init__(self, db_connection):
        self.db = db_connection
        self.adapter = BrainQECTimeSeriesAdapter()

    async def track_syndrome_over_time(
        self,
        region_id: int,
        start_time: int,
        end_time: int
    ) -> Dict:
        """Track how syndrome patterns evolve over time.

        Clinical use case: Monitor patient recovery after stroke
        - Does syndrome converge (recovery)?
        - What's the convergence rate?
        - Are there feedback loops preventing recovery?
        """
        # Get syndrome scores over time
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT
                t.timepoint,
                AVG(tc.syndrome_score) as avg_syndrome
            FROM tensor_cells tc
            JOIN timepoints t ON t.cell_id = tc.cell_id
            WHERE tc.region_id = %s
              AND t.timepoint BETWEEN %s AND %s
            GROUP BY t.timepoint
            ORDER BY t.timepoint
        """, (region_id, start_time, end_time))

        timeseries = np.array([row["avg_syndrome"] for row in cursor.fetchall()])

        # Analyze evolution
        analysis = self.adapter.analyze_region_evolution(timeseries)

        return {
            "region_id": region_id,
            "converged": analysis["converged"],
            "convergence_rate": analysis["mean_convergence_rate"],
            "complexity": analysis["complexity"],
            "interpretation": self._clinical_interpretation(analysis),
            "recommended_intervention": self._suggest_intervention(analysis)
        }

    def _clinical_interpretation(self, analysis: Dict) -> str:
        """Translate QEC metrics to clinical language."""
        if analysis["converged"]:
            rate = analysis["mean_convergence_rate"]
            if rate > 0.8:
                return "Rapid recovery: Syndrome resolving quickly"
            elif rate > 0.5:
                return "Moderate recovery: Steady improvement"
            else:
                return "Slow recovery: Consider intervention"
        else:
            complexity = analysis["complexity"]
            if complexity > 0.8:
                return "Chaotic dynamics: Unstable syndrome pattern"
            else:
                return "Non-convergent: Chronic syndrome likely"

    def _suggest_intervention(self, analysis: Dict) -> str:
        """Suggest interventions based on syndrome evolution."""
        if not analysis["converged"]:
            return "Consider neuromodulation or therapy to break feedback loops"
        else:
            return "Continue current treatment - showing positive response"
```

**Success Criteria**:
- ✅ Syndrome tracking working on BORDEAUX24
- ✅ Convergence detection functional
- ✅ Clinical interpretations generated

---

### Task 3: Causal Functor Discovery (4 hours)

**Goal**: Use Granger causality to discover which functors teach others

**Validation of BEAD-P6-3**:
```python
# src/backend/analysis/functor_causality_discovery.py

from src.backend.services.qec_timeseries_adapter import BrainQECTimeSeriesAdapter

class FunctorCausalityDiscovery:
    """Discover causal relationships between functors using data.

    Instead of assuming hierarchy, DISCOVER it empirically!
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.adapter = BrainQECTimeSeriesAdapter()

        self.functor_hierarchy = [
            "anatomy", "function", "electro",
            "genetics", "behavior", "pathology"
        ]

    async def discover_teaching_relationships(
        self,
        region_id: int
    ) -> Dict[Tuple[str, str], float]:
        """Discover which functors Granger-cause others.

        Theory: If anatomy teaches function, then:
        - Past anatomy values predict future function values
        - This is Granger causality!

        Returns:
            Dict mapping (source, target) → causality strength
        """
        # Get time-series for each functor
        functor_timeseries = await self._get_functor_timeseries(region_id)

        causality_graph = {}

        for source in self.functor_hierarchy:
            for target in self.functor_hierarchy:
                if source == target:
                    continue

                # Test Granger causality: does source → target?
                causality_score = self._test_granger_causality(
                    functor_timeseries[source],
                    functor_timeseries[target]
                )

                if causality_score > 0.5:  # Significant causality
                    causality_graph[(source, target)] = causality_score

        return causality_graph

    async def validate_theoretical_hierarchy(
        self,
        region_id: int
    ) -> Dict:
        """Check if discovered causality matches theoretical hierarchy.

        Expected (theory):
        - anatomy → function (structure determines computation)
        - function → electro (computation drives dynamics)
        - genetics → anatomy (genes build structure)

        Discovered (data):
        - Empirical causality from Granger analysis

        Returns:
            Comparison of theory vs. empirical findings
        """
        discovered = await self.discover_teaching_relationships(region_id)

        # Theoretical expectations
        expected = {
            ("anatomy", "function"): "Structure → Computation",
            ("function", "electro"): "Computation → Dynamics",
            ("genetics", "anatomy"): "Genes → Structure",
            ("electro", "behavior"): "Dynamics → Task relevance"
        }

        comparison = {
            "confirmed": [],
            "unexpected": [],
            "missing": []
        }

        # Check expected relationships
        for (source, target), description in expected.items():
            if (source, target) in discovered:
                comparison["confirmed"].append({
                    "relationship": f"{source} → {target}",
                    "description": description,
                    "strength": discovered[(source, target)]
                })
            else:
                comparison["missing"].append({
                    "relationship": f"{source} → {target}",
                    "description": description,
                    "note": "Not detected in data - may be indirect or weak"
                })

        # Check unexpected relationships
        for (source, target) in discovered:
            if (source, target) not in expected:
                comparison["unexpected"].append({
                    "relationship": f"{source} → {target}",
                    "strength": discovered[(source, target)],
                    "interpretation": "Novel causal relationship - warrants investigation"
                })

        return comparison

    def _test_granger_causality(
        self,
        source_ts: np.ndarray,
        target_ts: np.ndarray,
        max_lag: int = 5
    ) -> float:
        """Test if source Granger-causes target.

        Uses merge2docs time-series analysis service.
        """
        # Delegate to QEC time-series tools
        result = self.adapter.ts_service.test_granger_causality(
            source_ts,
            target_ts,
            max_lag=max_lag
        )

        return result.get("causality_score", 0.0)
```

**Success Criteria**:
- ✅ Granger causality working
- ✅ Theory vs. empirical comparison
- ✅ Novel relationships discovered

---

### Task 4: Feedback Loop Detection (3 hours)

**Goal**: Use FVS to identify regulatory control points

**New capability for clinical interventions**:
```python
# src/backend/analysis/feedback_loop_analysis.py

from src.backend.services.qec_timeseries_adapter import BrainQECTimeSeriesAdapter

class BrainFeedbackLoopAnalyzer:
    """Identify feedback loops and control points in brain networks.

    Clinical application: Find minimal intervention targets
    - Feedback loops maintain chronic conditions
    - Breaking key loops can restore healthy dynamics
    - FVS finds minimal set of regions to target
    """

    def __init__(self, db_connection):
        self.db = db_connection
        self.adapter = BrainQECTimeSeriesAdapter()

    async def identify_feedback_loops(
        self,
        subject: str
    ) -> Dict:
        """Identify feedback loops in brain connectivity.

        Method:
        1. Build directed graph from Granger causality
        2. Find cycles (feedback loops)
        3. Compute Feedback Vertex Set (FVS)
        4. FVS = minimal intervention targets
        """
        # Build causal connectivity graph
        causal_graph = await self._build_causal_connectivity(subject)

        # Detect cycles (feedback loops)
        cycles = self._find_cycles(causal_graph)

        # Compute FVS (minimal breaking set)
        fvs = self.adapter.ts_service.find_feedback_vertices(causal_graph)

        return {
            "subject": subject,
            "feedback_loops": len(cycles),
            "control_points": fvs,
            "interpretation": self._interpret_control_points(fvs),
            "intervention_strategy": self._suggest_interventions(fvs)
        }

    async def _build_causal_connectivity(
        self,
        subject: str
    ) -> Dict[int, List[int]]:
        """Build directed graph from Granger causality.

        Edge A → B means: Activity in region A Granger-causes activity in B
        """
        loader = PRIMEDELoader()
        data = await loader.load_subject("bordeaux24", subject, "bold")

        # timeseries: (T, 368)
        timeseries = data["timeseries"]

        graph = {}

        # Test causality between all region pairs (expensive!)
        # For demo, use correlation + threshold for speed
        # In production, use Granger causality

        correlation = np.corrcoef(timeseries.T)  # (368, 368)

        for i in range(368):
            graph[i] = []
            for j in range(368):
                if i != j and abs(correlation[i, j]) > 0.7:
                    # High correlation suggests potential causality
                    # (should use Granger test for true causality)
                    graph[i].append(j)

        return graph

    def _interpret_control_points(self, fvs: List[int]) -> str:
        """Interpret FVS in clinical terms."""
        if len(fvs) == 0:
            return "No feedback loops detected - healthy dynamics"
        elif len(fvs) <= 5:
            return f"Minimal control points: Target {len(fvs)} regions for maximum impact"
        else:
            return f"Complex feedback structure: {len(fvs)} control points identified"

    def _suggest_interventions(self, fvs: List[int]) -> List[Dict]:
        """Suggest interventions for control points."""
        interventions = []

        for region_id in fvs:
            interventions.append({
                "region_id": region_id,
                "region_name": self._get_region_name(region_id),
                "methods": [
                    "Transcranial Magnetic Stimulation (TMS)",
                    "Neurofeedback training",
                    "Pharmacological modulation"
                ],
                "rationale": "Breaking feedback loop at this control point"
            })

        return interventions
```

**Success Criteria**:
- ✅ Feedback loops identified
- ✅ FVS computed correctly
- ✅ Intervention strategies generated

---

## Testing

### Unit Tests

```python
# tests/backend/analysis/test_qec_timeseries_integration.py

class TestQECTimeSeriesIntegration:

    def test_adapter_conversion(self):
        """Test brain data → QEC format conversion."""
        adapter = BrainQECTimeSeriesAdapter()

        # Mock fMRI timeseries
        brain_ts = np.random.randn(100, 368)

        qec_format = adapter.convert_brain_to_qec_format(brain_ts)

        assert "syndromes" in qec_format
        assert qec_format["syndromes"].shape == brain_ts.shape

    @pytest.mark.asyncio
    async def test_syndrome_tracking(self):
        """Test syndrome evolution tracking."""
        tracker = SyndromeEvolutionTracker(db_connection)

        # Track syndrome for test region
        result = await tracker.track_syndrome_over_time(
            region_id=1,
            start_time=0,
            end_time=100
        )

        assert "converged" in result
        assert "convergence_rate" in result
        assert "interpretation" in result

    @pytest.mark.asyncio
    async def test_functor_causality(self):
        """Test Granger causality discovery."""
        discoverer = FunctorCausalityDiscovery(db_connection)

        causality = await discoverer.discover_teaching_relationships(
            region_id=1
        )

        assert isinstance(causality, dict)
        # Check for expected relationships
        assert ("anatomy", "function") in causality or len(causality) == 0

    @pytest.mark.asyncio
    async def test_feedback_loops(self):
        """Test feedback loop identification."""
        analyzer = BrainFeedbackLoopAnalyzer(db_connection)

        result = await analyzer.identify_feedback_loops("m01")

        assert "feedback_loops" in result
        assert "control_points" in result
        assert isinstance(result["control_points"], list)
```

**Success Criteria**:
- ✅ All unit tests pass
- ✅ Integration tests with BORDEAUX24 pass
- ✅ Performance acceptable (<10s per subject)

---

## Performance Considerations

### Computational Cost

**Granger Causality Analysis**:
- 368 regions × 368 regions = 135,424 pairwise tests
- Each test: ~10ms (with caching)
- Total: ~23 minutes per subject (parallelizable)

**Optimization**:
```python
# Use sparse connectivity for speed
# Only test pairs with high correlation (pre-filter)

def fast_causal_discovery(timeseries, correlation_threshold=0.5):
    """Fast causality discovery using correlation pre-filter."""

    # Step 1: Compute correlation (fast, O(n²))
    correlation = np.corrcoef(timeseries.T)

    # Step 2: Pre-filter candidates
    candidates = np.where(np.abs(correlation) > correlation_threshold)

    # Step 3: Test only candidates (much smaller set)
    causal_pairs = {}
    for i, j in zip(*candidates):
        if i != j:
            causality = test_granger_causality(
                timeseries[:, i],
                timeseries[:, j]
            )
            if causality > 0.5:
                causal_pairs[(i, j)] = causality

    return causal_pairs
```

**Expected Performance**:
- Pre-filter: 1s
- Candidate testing: ~2-5 minutes (100-1000 pairs instead of 135k)
- Total: **<10 minutes per subject** ✅

---

## Documentation

### User Guide Addition

```markdown
# Time-Series Analysis with QEC Integration

## Overview

The brain tensor system integrates QEC (Quantum Error Correction) time-series
analysis from merge2docs to provide advanced temporal dynamics analysis.

## Key Features

### 1. Syndrome Evolution Tracking

Monitor how brain syndromes evolve over time:

```python
from src.backend.analysis.syndrome_evolution_tracker import SyndromeEvolutionTracker

tracker = SyndromeEvolutionTracker(db_connection)
result = await tracker.track_syndrome_over_time(
    region_id=42,
    start_time=0,
    end_time=150
)

print(f"Converged: {result['converged']}")
print(f"Recovery rate: {result['convergence_rate']:.2%}")
print(f"Clinical interpretation: {result['interpretation']}")
```

### 2. Causal Functor Discovery

Discover which functors drive others:

```python
from src.backend.analysis.functor_causality_discovery import FunctorCausalityDiscovery

discoverer = FunctorCausalityDiscovery(db_connection)
causality = await discoverer.discover_teaching_relationships(region_id=42)

for (source, target), strength in causality.items():
    print(f"{source} → {target}: {strength:.2f}")
```

### 3. Feedback Loop Analysis

Identify control points for interventions:

```python
from src.backend.analysis.feedback_loop_analysis import BrainFeedbackLoopAnalyzer

analyzer = BrainFeedbackLoopAnalyzer(db_connection)
result = await analyzer.identify_feedback_loops("m01")

print(f"Feedback loops detected: {result['feedback_loops']}")
print(f"Control points: {result['control_points']}")
print(f"Suggested interventions:")
for intervention in result['intervention_strategy']:
    print(f"  - {intervention['region_name']}: {intervention['methods'][0]}")
```

## Benefits

1. **Causality, Not Correlation**: Uses Granger causality to find true causal relationships
2. **Clinical Applications**: Syndrome tracking for patient monitoring
3. **Theory Validation**: Empirically test functor hierarchy assumptions
4. **Intervention Guidance**: Identify optimal neuromodulation targets
```

---

## Health Score Impact

**Immediate**: +2% (87% → 89%)
- Integration with merge2docs time-series tools
- Enhanced syndrome detection capabilities
- Causal analysis (beyond correlation)

**Long-term**: +5% (enables advanced research features)
- Foundation for Phase 9 ML integration
- Clinical validation studies
- Novel neuroscience insights

---

## Timeline

**Week 4** (parallel with Phase 4 test completion):
- Day 1-2: Setup integration (Task 1)
- Day 3-4: Syndrome tracking (Task 2)
- Day 5-6: Causal discovery (Task 3)
- Day 7: Feedback loops (Task 4)

**Total**: 12 hours over 1 week

---

## Success Metrics

1. **Integration**: ✅ Import merge2docs tools successfully
2. **Syndrome Tracking**: ✅ Detect convergence in test data
3. **Causality Discovery**: ✅ Confirm theoretical hierarchy in >80% of regions
4. **Feedback Loops**: ✅ Identify <10 control points per subject
5. **Performance**: ✅ <10 minutes analysis per subject
6. **Health Score**: ✅ Reach 89%

---

## Future Extensions

1. **Real-time Monitoring**: Stream analysis of ongoing fMRI sessions
2. **Closed-loop Intervention**: Use FVS for real-time neuromodulation
3. **Multi-subject Meta-analysis**: Discover common control points across population
4. **Disease-specific Patterns**: Build syndrome libraries for different conditions

---

**Status**: 🟢 READY FOR IMPLEMENTATION
**Dependencies**: ✅ All met (merge2docs integration available)
**Expected Impact**: Transforms brain tensor from static to dynamic analysis
**Clinical Value**: Enables intervention planning and patient monitoring
