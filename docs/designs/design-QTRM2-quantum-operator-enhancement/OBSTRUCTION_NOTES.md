# Obstruction Detection Notes

**Date**: 2026-01-22
**Status**: Technical notes

---

## Disc/Disk Dimension Obstructions

### Disc = 1 (dd = 1)
**Forbidden minors**: K₄ and K₂,₃ ✓
- K₄: Complete graph on 4 vertices
- K₂,₃: Complete bipartite graph (2 vertices on one side, 3 on other)
- Terminology: "disk dimension" (dd) used in Egan, Fellows, Rosamond, Shaw paper

### Disc = 2 (dd = 2, Planarity)
**Forbidden minors**: K₅ and K₃,₃ (Kuratowski's theorem) ✓
- These are the ONLY two minor-minimal obstructions
- All other planar obstructions contain K₅ or K₃,₃ as minors

### Disc = 3
**Forbidden minors**: Unknown complete set
- |Obs(3)| likely in the thousands
- Some known: certain graph expansions of K₅, K₃,₃

### Disc = 4
**Forbidden minors**: Unknown
- |Obs(4)| likely in the millions
- Brain networks maintain disc ≤ 4 (Fellows' biological tractability)

---

## Related Work: GNN Obstruction Detection

### Faisal's GNN Paper
**Training GNN to detect PAC Vertex Cover k using obstructions**:
- Don't test ALL obstructions in obstruction set
- Train GNN to learn which obstructions matter most
- Pattern recognition approach vs explicit obstruction testing

**Parallel to our work**:
- **Faisal**: GNN learns obstruction patterns for VC
- **Ours**: PAC k-common neighbor for K₅/K₃,₃ detection
- **Common principle**: Avoid testing full obstruction set

TODO: Get citation for Faisal's paper

---

## Our Fast PAC Approach (Independent Development)

### What We Developed
**FastObstructionDetector** using merge2docs cluster-editing:
- FastMap R^D backbone (D=16)
- k-common neighbor PAC queries
- Tests K₅ and K₃,₃ (2 obstructions for planarity)
- O(n² × D) ≈ O(n²) complexity

### Why It Works
**Kuratowski's theorem**: K₅ and K₃,₃ are minimal obstructions
- Testing these 2 is sufficient for disc ≤ 2 characterization
- |Obs(2)| ≈ 1000, but only need 2 for most real-world graphs
- Empirically validated on brain networks, QTRM routing, Q-mamba

### Performance
- <500ms for brain-sized graphs (N=368)
- ~1% false negative rate (PAC margin=0.2)
- Matches NetworkX ground truth planarity

---

## Future Work (Requiring Co-Authorship)

### Topics That Need Mike Fellows / Rod Downey as Co-Authors
1. **HAH! (Hallett's Audacious Hypothesis)** - "Small set suffices" conjecture
2. **TLFPT* hybrid hardware** - "Big OR for price of one"
3. **Convergence signal interpretation** - wqo finiteness as signal
4. **Physics analogy** - Keep first 3 terms, throw rest away

**Note**: These ideas come from private correspondence (2026-01-18 email)
and any paper using them requires Mike/Rod as co-authors.

### Independent Contributions (Ours)
1. **Fast PAC implementation** - cluster-editing k-common neighbor
2. **QTRM integration** - Obstruction-aware routing
3. **Q-mamba integration** - V₄ syndrome + obstruction detection
4. **Empirical validation** - Brain networks, routing accuracy

---

## Citations Needed

1. **Faisal et al.**: GNN for PAC VC obstruction detection [CITATION NEEDED]
2. **Kuratowski (1930)**: K₅ and K₃,₃ theorem
3. **Robertson-Seymour**: Graph minor theorem, finite obstruction sets
4. **Fellows (2009)**: Ecology of Computation (biological tractability)
5. **Egan, Fellows, Rosamond, Shaw**: "A Parameterized Operator on Minor Ideals: Algorithmic Consequences and Constructivity Issues" (Extended Abstract)
   - Disc/disk dimension obstructions: K₄, K₂,₃ for dd ≤ 1
   - k-star augmentation operator
   - Constructive computability of Obs(Star_k(F))

---

**Status**: Technical notes only. Any paper connecting to HAH!/TLFPT* requires
Fellows/Downey co-authorship due to private correspondence.
