# Paul et al. 2023 - Universal Obstructions and Graph Parameters

**Citation**: Paul, C., Protopapas, E., & Thilikos, D. M. (2023). Graph Parameters, Universal Obstructions, and WQO. arXiv:2304.03688v1

## Relevance to This Project

This paper provides the theoretical foundation for our obstruction-based disc dimension estimation in `src/backend/mri/fast_obstruction_detection.py`.

## Key Results

### Parametric Obstructions for Treewidth (Section 6.1)

The paper proves:

```
pobs≤m(tw) = {K₅, K₃,₃}
```

This means detecting K₅ and K₃,₃ minors is *exactly* the right approach for bounding treewidth-class parameters asymptotically.

### The Chain of Obstructions

For treewidth (tw):

1. **Universal obstruction**: Grid sequence Γ = ⟨Γₖ⟩ (the k×k grids)
2. **Class obstruction**: cobs(tw) = {P} where P = planar graphs
3. **Parametric obstruction**: pobs(tw) = obs(P) = **{K₅, K₃,₃}**

### Why This Matters for Brain Networks

Our `disc_dimension_via_obstructions()` function returns:
- `disc_dim_estimate = 2` if planar (no K₅ or K₃,₃)
- `disc_dim_estimate ≥ 3` if non-planar

This is a valid lower bound because of Kuratowski's theorem and the universal obstruction framework.

## Key Theorems

### Theorem 44 (Main Characterization)

> Assuming (Gall, ≤) is a wqo, the following are equivalent:
> 1. (Gall, ≤) is an ω²-wqo
> 2. Every ≤-monotone parameter has a finite universal obstruction
> 3. Every ≤-monotone parameter has a finite class obstruction
> 4. Every ≤-monotone parameter has a finite parametric obstruction
> 5. Every ≤-monotone parameter is rational

### Theorem 36 (Existence under WQO)

> If (Gall, ≤) is a wqo, then every ≤-monotone parameter has a universal obstruction.

Robertson-Seymour proved ≤m (minor relation) is wqo. The ω²-wqo question for minors remains open (Thomas 1989 proved it for bounded treewidth).

## Implementation Notes

### FastObstructionDetector Methods

| Method | Theoretical Basis |
|--------|-------------------|
| `detect_k5()` | Checks for K₅ ∈ pobs(tw) |
| `detect_k33()` | Checks for K₃,₃ ∈ pobs(tw) |
| `detect_both()` | Complete pobs(tw) check |

### Complexity

The paper notes (Section 4) that if ≤ is FPT-decidable and parametric obstructions are finite, then the parameter is FPT-approximable. Our PAC k-common neighbor approach provides practical O(n²) detection.

## Other Parameters Covered

The paper also provides universal obstructions for:

- **Pathwidth** (Section 6.1): Universal = complete ternary trees, pobs = {K₃}
- **Biconnected pathwidth**: pobs = {{K₄, S₃, 2·K₃}, {K₄, K₂,₃}}
- **Edge-degree** (Section 6.2): pobs = {{3·K₁}, {θ₂, 2·P₃}}
- **Cutwidth**: pobs = {{3·K₁}, {θ₂, 2·P₃}, {θ₂, K₁,₄}}

## Future Work

Consider extending obstruction detection to:
1. Pathwidth obstructions (ternary tree detection)
2. Immersion-monotone parameters for directed brain connectivity
3. ω²-wqo implications for multiplex network parameters

## Related Issues

- twosphere-mcp-af4: Document this theoretical connection
- See also: `DISC_DIMENSION_OBSTRUCTION_SETS_CORRECTION.md`
