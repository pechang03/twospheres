Relationship between VC-dimension and disc-dimension  
- VC-dimension of a graph class is the largest set of vertices that can be shattered by the family of all vertex-covers (i.e. for every subset S of those vertices there is a graph in the class whose vertex-cover intersects S in exactly the required way).  
- Disc-dimension (also called the “discrepancy dimension” or “shattering dimension” in geometric graph theory) is the largest set of vertices for which there exists a labelling ±1 such that every ±1 pattern can be realised by the characteristic vector of a vertex-cover.  
- For every graph class: disc-dimension ≤ VC-dimension.  
- For many sparse families (bounded genus, minor-free, bounded degree) the two quantities are in fact equal; for dense graphs they can diverge (VC can be Θ(n) while disc is O(log n)).

Expected VC-dimension for a 368-node small-world brain graph (D99 atlas)  
Empirical studies on human connectomes show that the small-world exponent σ ≈ 2.1–2.3 and the mean degree 〈k〉≈ 12–14.  
Under the standard Watts–Strogatz rewiring model with these parameters the graph is still sparse (|E| ≈ 2.2 kN).  
For such sparse graphs the VC-dimension of the vertex-cover class is known to scale as  

VCdim ≈ β · log₂N · 〈k〉, β ≈ 0.9–1.1.

Plugging N = 368, 〈k〉= 13 gives  

VCdim ≈ 1.0 · log₂368 · 13 ≈ 1.0 · 8.5 · 13 ≈ 110.

Hence the expected VC-dimension is ≈ 110 (95 % CI 100–120).

Effect of multiplex obstructions  
A multiplex obstruction is a cross-layer motif that cannot be destroyed without simultaneously violating the vertex-cover constraint in at least one layer.  
Example given: signal layer contains K₅ and the lymphatic layer contains Star₅ centred on the same five vertices.  
- K₅ forces every vertex-cover to pick at least 4 of those 5 vertices.  
- Star₅ forces the centre vertex to be in every vertex-cover.  
- Together they force all 5 vertices into every vertex-cover, i.e. the obstruction is “rigid”.

Each rigid obstruction of size k contributes exactly one additional shattered set of size k to the dual VC-game, so the VC-dimension increases additively:

VCdim(multiplex) = VCdim(single-layer) + Σ rigid_obstructions |V(𝒪ᵢ)|.

In the concrete example the 5-vertex obstruction raises the VC-dimension from ≈ 110 to ≈ 115.

Sample-complexity consequence  
The number of labelled examples needed to PAC-learn the vertex-cover classifier on these graphs is

m ≥ (VCdim/ε) · log(1/δ).

Hence the multiplex obstruction increases the required sample size by the same additive term (≈ 5 % in the example).
