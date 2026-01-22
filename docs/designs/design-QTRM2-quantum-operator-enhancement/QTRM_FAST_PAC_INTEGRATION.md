# Fast PAC-Based Obstruction Detection for QTRM Router

**Date**: 2026-01-22
**Status**: Design Specification
**Connection**: Fast obstruction detection ↔ QTRM strategy routing

---

## Executive Summary

Integrate **fast PAC-based obstruction detection** into QTRM (Quantum Theory Router Models) routing system using the same k-common neighbor approach as Q-mamba integration.

### Key Insight

**Topological obstructions guide functor level routing**:
- K₅ detected → disc ≥ 3 → Complex problem → Route to `gpu_semi_exact` or `hybrid`
- No obstructions → disc ≤ 2 → Simple problem → Route to `sage_direct`
- Fast PAC queries: O(n² × D) ≈ O(n²) for D=16

### Current QTRM Architecture

**UltimateQTRMRouter** (from `ultimate_qtrm_router.py`):
```python
class UltimateQTRMRouter(nn.Module):
    """75 features → 4 strategy classes"""

    STRATEGY_MAPPING = {
        'sage_direct': 0,        # Simple problems
        'mathematical_glue': 1,  # Medium complexity
        'hybrid': 2,             # Complex problems
        'gpu_semi_exact': 3      # Very complex
    }

    FEATURE_GROUPS = {
        'base': (0, 16),
        'llm': (16, 24),
        'rl': (24, 32),
        'repair': (32, 36),
        'rids': (36, 44),
        'lid': (44, 48),
        'fpt': (48, 52),
        'mc': (52, 56),
        'quantum': (56, 75),  # ← 19 quantum features (FFT-based)
    }
```

**Current quantum features** (19 dims):
- P3 dominating set features
- GHMN graph hyperbolic manifold network features
- Domain-specific features
- Artifact features
- FFT-based quantum mixing via `QuantumFFTLayer`

**Limitations**:
- No topological obstruction awareness
- Heuristic routing based on learned patterns
- No connection to disc dimension theory

---

## Enhanced QTRM with Fast PAC Obstruction Detection

### New Quantum Features (Add 5 dimensions)

**Feature group expansion**: `quantum: (56, 80)` (19 → 24 dims)

**New features** (5 additional):
1. **has_k5_obstruction**: bool (0/1) - K₅ detected via PAC k-common neighbor
2. **has_k33_obstruction**: bool (0/1) - K₃,₃ detected via PAC
3. **obstruction_strength**: float (0.0-1.0) - Severity of obstruction
4. **disc_dimension_estimate**: int (2, 3, 4) - Via obstruction detection
5. **is_planar**: bool (0/1) - Kuratowski planarity (no K₅ or K₃,₃)

### Obstruction-Aware Routing Logic

**Strategy selection guided by topology**:

```python
if has_k5_obstruction or has_k33_obstruction:
    # Topological complexity detected
    if obstruction_strength > 0.7:
        # Severe obstruction → Very complex
        recommended_strategy = 'gpu_semi_exact'  # Class 3
    else:
        # Moderate obstruction → Complex
        recommended_strategy = 'hybrid'  # Class 2
else:
    # No obstruction → Simple or medium
    if graph.number_of_nodes() < 100:
        recommended_strategy = 'sage_direct'  # Class 0
    else:
        recommended_strategy = 'mathematical_glue'  # Class 1
```

**Connection to disc dimension**:
```
disc ≤ 2 (planar)    → sage_direct or mathematical_glue
disc = 3 (K₅ or K₃,₃) → hybrid
disc ≥ 4             → gpu_semi_exact
```

---

## Implementation

### Phase 1: Add Obstruction Features (Immediate)

**File**: `merge2docs/src/rl_proof_router/quantum_features_enhanced.py`

```python
"""Enhanced quantum features with fast PAC obstruction detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'twosphere-mcp'))

from src.backend.mri.fast_obstruction_detection import (
    FastObstructionDetector,
    disc_dimension_via_obstructions
)

class EnhancedQuantumFeatureExtractor:
    """Quantum feature extraction with topological obstruction detection."""

    def __init__(self, use_pac: bool = True):
        self.use_pac = use_pac
        self.obstruction_detector = FastObstructionDetector(use_pac=use_pac)

    def extract_quantum_features(
        self,
        graph: nx.Graph,
        problem_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract 24 quantum features (19 original + 5 new obstruction features).

        Returns:
            features: np.ndarray of shape (24,)
                [0:19] - Original quantum features (P3, GHMN, domain, artifacts)
                [19] - has_k5_obstruction (0/1)
                [20] - has_k33_obstruction (0/1)
                [21] - obstruction_strength (0.0-1.0)
                [22] - disc_dimension_estimate (2, 3, or 4)
                [23] - is_planar (0/1)
        """
        # 1. Original quantum features (from QuantumModule)
        original_features = self._extract_original_quantum_features(graph, problem_data)

        # 2. NEW: Fast PAC obstruction detection
        obstruction_result = self.obstruction_detector.detect_both(graph)

        # 3. NEW: Disc dimension estimation
        disc_result = disc_dimension_via_obstructions(graph, use_pac=self.use_pac)

        # 4. Combine features
        new_features = np.array([
            float(obstruction_result['has_k5']),            # [19]
            float(obstruction_result['has_k33']),           # [20]
            obstruction_result['k5_result']['strength'],    # [21]
            float(disc_result['disc_dim_estimate']),        # [22]
            float(disc_result['is_planar']),                # [23]
        ])

        return np.concatenate([original_features, new_features])

    def _extract_original_quantum_features(
        self,
        graph: nx.Graph,
        problem_data: Dict[str, Any]
    ) -> np.ndarray:
        """Extract original 19 quantum features (unchanged)."""
        # Use existing QuantumModule logic
        from ultimate_integrated_model import QuantumModule
        quantum_module = QuantumModule()
        return quantum_module.extract_features(graph, problem_data)
```

**Changes to UltimateQTRMRouter**:
```python
class UltimateQTRMRouter(nn.Module):
    """Updated with 24 quantum features (was 19)."""

    FEATURE_GROUPS = {
        'base': (0, 16),
        'llm': (16, 24),
        'rl': (24, 32),
        'repair': (32, 36),
        'rids': (36, 44),
        'lid': (44, 48),
        'fpt': (48, 52),
        'mc': (52, 56),
        'quantum': (56, 80),  # ← Updated: 19 → 24 dims
    }

    # Total features: 80 (was 75)
```

**Backward compatibility**: Existing 75-feature models can be fine-tuned with 5 new features initialized to zero.

### Phase 2: Obstruction-Aware Loss Function (1-2 weeks)

**Add topological consistency loss**:

```python
class ObstructionAwareLoss(nn.Module):
    """Loss function with topological consistency penalty."""

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for consistency term

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        has_k5: torch.Tensor,
        has_k33: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with topological consistency penalty.

        Args:
            logits: (batch, 4) - Model predictions
            labels: (batch,) - True strategy labels
            has_k5: (batch,) - K₅ obstruction flags
            has_k33: (batch,) - K₃,₃ obstruction flags

        Returns:
            loss: Scalar tensor
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)

        # Topological consistency penalty
        # If K₅ or K₃,₃ detected, penalize low-complexity strategies
        has_obstruction = (has_k5 + has_k33) > 0  # Any obstruction

        # Get predicted strategy
        pred_strategy = torch.argmax(logits, dim=-1)

        # Penalty: If obstruction but predicted simple strategy (0 or 1)
        simple_strategy = (pred_strategy <= 1).float()  # sage_direct or mathematical_glue
        consistency_penalty = has_obstruction.float() * simple_strategy

        # Combined loss
        total_loss = ce_loss + self.alpha * consistency_penalty.mean()

        return total_loss
```

**Benefits**:
- ✅ Guides model to route complex problems (K₅/K₃,₃) to complex strategies
- ✅ Enforces topological consistency during training
- ✅ Reduces routing errors on obstructed graphs

### Phase 3: Attention Mechanism for Obstruction Features (1 month)

**Add cross-attention between obstruction features and strategy selection**:

```python
class ObstructionAttention(nn.Module):
    """Attention mechanism highlighting obstruction features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(5, hidden_dim)  # 5 obstruction features
        self.value = nn.Linear(5, hidden_dim)

    def forward(
        self,
        encoded_features: torch.Tensor,
        obstruction_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention to obstruction features.

        Args:
            encoded_features: (batch, hidden_dim) - Encoded feature vector
            obstruction_features: (batch, 5) - Obstruction features

        Returns:
            attended: (batch, hidden_dim) - Attention-weighted features
        """
        Q = self.query(encoded_features)  # (batch, hidden_dim)
        K = self.key(obstruction_features)  # (batch, hidden_dim)
        V = self.value(obstruction_features)  # (batch, hidden_dim)

        # Attention weights
        attention = torch.matmul(Q, K.T) / np.sqrt(Q.size(-1))
        attention_weights = F.softmax(attention, dim=-1)

        # Weighted sum
        attended = torch.matmul(attention_weights, V)

        return attended + encoded_features  # Residual connection
```

---

## Performance Comparison

### Feature Extraction Time

| Method | N=368 | Complexity |
|--------|-------|------------|
| **Original quantum features** | ~50ms | O(n log n) FFT |
| **+ Fast PAC obstruction** | +300ms | O(n² × D), D=16 |
| **Total (enhanced)** | ~350ms | Acceptable |

**Trade-off**: Slightly slower feature extraction (+6x), but **much better routing accuracy**.

### Expected Routing Improvements

| Metric | Original | Enhanced | Improvement |
|--------|----------|----------|-------------|
| **Overall accuracy** | 92% (v2 baseline) | **95%+** | +3% |
| **Complex problem routing** | 85% | **92%+** | +7% |
| **False simple routing** | 8% | **<3%** | -5% |
| **Inference time** | 5ms | 350ms (with PAC) | Acceptable |

**Key improvement**: **Fewer routing errors on topologically complex problems** (K₅/K₃,₃ obstructions).

---

## Integration with Existing QTRM Pipeline

### Training Data Augmentation

**Add obstruction labels to training data**:

```python
def augment_training_data_with_obstructions(
    dataset: List[Dict[str, Any]],
    use_pac: bool = True
) -> List[Dict[str, Any]]:
    """Add obstruction features to existing QTRM training data."""

    detector = FastObstructionDetector(use_pac=use_pac)
    augmented = []

    for sample in dataset:
        graph = sample['graph']

        # Detect obstructions
        result = detector.detect_both(graph)
        disc_result = disc_dimension_via_obstructions(graph, use_pac=use_pac)

        # Add new features
        sample['has_k5'] = result['has_k5']
        sample['has_k33'] = result['has_k33']
        sample['obstruction_strength'] = result['k5_result']['strength']
        sample['disc_dimension'] = disc_result['disc_dim_estimate']
        sample['is_planar'] = disc_result['is_planar']

        augmented.append(sample)

    return augmented
```

### Inference Pipeline

**Enhanced routing with obstruction awareness**:

```python
class EnhancedQTRMRouter:
    """QTRM router with fast PAC obstruction detection."""

    def __init__(self, model_path: str, use_pac: bool = True):
        self.model = UltimateQTRMRouter.load(model_path)
        self.feature_extractor = EnhancedQuantumFeatureExtractor(use_pac=use_pac)

    def route_query(
        self,
        graph: nx.Graph,
        problem_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route query with obstruction-aware strategy selection.

        Returns:
            {
                'strategy': str (sage_direct, mathematical_glue, hybrid, gpu_semi_exact),
                'confidence': float,
                'has_obstruction': bool,
                'disc_dimension': int,
                'reasoning': str
            }
        """
        # 1. Extract 80 features (75 original + 5 obstruction)
        features = self.feature_extractor.extract_quantum_features(graph, problem_data)

        # 2. Model prediction
        with torch.no_grad():
            logits = self.model(torch.tensor(features).unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, pred_class].item()

        # 3. Map to strategy
        strategy = list(self.model.STRATEGY_MAPPING.keys())[pred_class]

        # 4. Extract obstruction info
        has_k5 = bool(features[56 + 19])  # Index 75
        has_k33 = bool(features[56 + 20])  # Index 76
        disc_dim = int(features[56 + 22])   # Index 78

        # 5. Sanity check: If obstruction detected but simple strategy predicted, warn
        if (has_k5 or has_k33) and pred_class <= 1:
            logger.warning(
                f"Obstruction detected (K₅={has_k5}, K₃,₃={has_k33}) "
                f"but simple strategy predicted ({strategy}). "
                f"Consider overriding to 'hybrid' or 'gpu_semi_exact'."
            )

        # 6. Generate reasoning
        reasoning = self._generate_reasoning(
            strategy, has_k5, has_k33, disc_dim, confidence
        )

        return {
            'strategy': strategy,
            'confidence': confidence,
            'has_obstruction': has_k5 or has_k33,
            'disc_dimension': disc_dim,
            'reasoning': reasoning
        }

    def _generate_reasoning(
        self,
        strategy: str,
        has_k5: bool,
        has_k33: bool,
        disc_dim: int,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for routing decision."""

        if has_k5:
            obstruction = "K₅ obstruction detected (complete graph of 5 vertices)"
        elif has_k33:
            obstruction = "K₃,₃ obstruction detected (complete bipartite graph)"
        else:
            obstruction = "No topological obstructions"

        return (
            f"Strategy: {strategy} (confidence: {confidence:.2%})\n"
            f"Disc dimension: {disc_dim}\n"
            f"Topology: {obstruction}\n"
            f"Reasoning: "
            f"{'Complex topological structure requires GPU acceleration' if has_k5 or has_k33 else 'Simple topology suitable for direct solving'}"
        )
```

---

## Validation Plan

### Unit Tests

```python
def test_obstruction_feature_extraction():
    """Verify obstruction features are correctly extracted."""
    # K₅ graph (known obstruction)
    G_k5 = nx.complete_graph(5)

    extractor = EnhancedQuantumFeatureExtractor(use_pac=True)
    features = extractor.extract_quantum_features(G_k5, {})

    assert features[75] == 1.0, "has_k5 should be True"
    assert features[78] >= 3.0, "disc_dimension should be >= 3"
    assert features[79] == 0.0, "is_planar should be False"

def test_routing_with_obstructions():
    """Verify routing decisions respect topological complexity."""
    # K₅ graph should route to complex strategy
    G_k5 = nx.complete_graph(5)

    router = EnhancedQTRMRouter(model_path='models/ultimate_qtrm_router.pth')
    result = router.route_query(G_k5, {})

    assert result['strategy'] in ['hybrid', 'gpu_semi_exact'], \
        "K₅ graph should route to complex strategy"
```

### Integration Tests

1. **Augment existing QTRM training data** with obstruction features
2. **Fine-tune ultimate_qtrm_router.pth** with 5 new features
3. **Compare routing accuracy** before/after on held-out test set
4. **Measure inference time** overhead (target: <500ms per query)

---

## Comparison: QTRM vs Q-Mamba Integration

| Aspect | **QTRM Router** | **QEC-ComoRAG-YadaMamba** |
|--------|----------------|--------------------------|
| **Use case** | Strategy routing (F₀-F₆) | Reasoning correction loop |
| **Obstruction role** | Guides strategy selection | Detects reasoning impasse |
| **d_model connection** | Functor level complexity | State dimension (d=4) |
| **Integration** | Add 5 new quantum features | Enhance V₄ syndrome detection |
| **Expected improvement** | +3-5% routing accuracy | +1-2 fewer correction cycles |

**Common approach**:
- Fast PAC k-common neighbor queries (O(n² × D))
- FastMap R^D backbone (D=16)
- Topological obstruction → complexity signal
- <500ms for brain-sized graphs (N=368)

---

## Conclusion

Fast PAC-based obstruction detection enhances QTRM routing by:

1. **Adding topological awareness** - K₅/K₃,₃ detection guides strategy selection
2. **Reducing routing errors** - Complex problems (disc ≥ 3) route to GPU strategies
3. **Minimal overhead** - +300ms feature extraction vs 10+ seconds for symbolic eigenvalues
4. **Backward compatible** - Existing 75-feature models can be fine-tuned

**Parallel to Q-mamba integration**:
- Q-mamba: V₄ syndrome ↔ obstruction detection
- QTRM: Strategy routing ↔ obstruction detection
- Both use **same fast PAC approach** (FastMap k-common neighbor)

**Next steps**:
1. Augment QTRM training data with obstruction labels
2. Fine-tune ultimate_qtrm_router.pth with 5 new features
3. Validate +3-5% routing accuracy improvement

---

## References

- **QTRM router**: `merge2docs/src/rl_proof_router/ultimate_qtrm_router.py`
- **Fast obstruction detection**: `twosphere-mcp/src/backend/mri/fast_obstruction_detection.py`
- **Q-mamba integration**: `twosphere-mcp/docs/designs/design-QTRM2-quantum-operator-enhancement/QMAMBA_INTEGRATION.md`
- **Cluster editing**: `merge2docs/src/backend/algorithms/cluster_editing.py`
- **Ultimate QTRM v2**: 92% baseline accuracy (bead: `2026-01-18-ultimate-qtrm-v2-95-percent.md`)

---

**Status**: Design complete, ready for Phase 1 implementation (add 5 obstruction features to quantum feature group).
