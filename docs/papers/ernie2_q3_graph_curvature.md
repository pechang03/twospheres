Short answer  
- K₅ and K₃,₃ are the two Kuratowski obstructions; they are “too connected” to live on a plane, and topologically they force positive genus.  
- Both Ollivier–Ricci and Forman curvature translate “too much connectivity” into negative curvature: every edge that lies in many 3- or 4-cycles gets a negative contribution, and K₅ (K₃,₃) is the extreme case. Hence the graph is negatively curved (Gromov-hyperbolic) in the sense of coarse geometry, not in the constant-sectional-curvature sense.  
- In a multiplex with a small-world (signal) layer and an almost lattice-like (lymphatic) layer, cross-layer edges act as “short-circuits.” Along every such edge the two incident triangles carry two different curvature values; the resulting jump is a curvature discontinuity in the layer-wise curvature field.  
- Ricci-flow with respect to the Ollivier–Ricci edge weights evolves the metric by  

  \[
  \frac{dg_e}{dt} = –\kappa_e g_e
  \]

  where κₑ is the Ollivier–Ricci curvature. Edges with large negative curvature shrink faster; after finite time they collapse to zero length, revealing the obstruction as a collapsed “neck.” Thus Ricci-flow detects the forbidden minors (or cross-layer bottlenecks) dynamically.

Detailed explanation  

1. From forbidden minors to curvature  
   - Euler characteristic χ = |V| – |E| + |F| forces χ ≤ 0 for any drawing of K₅ or K₃,₃, so the graph has genus ≥ 1.  
   - Ollivier–Ricci curvature: κ(e) = 1 – W₁(m_x, m_y), where m_x is the uniform measure on the 1-ball. In K₅ every edge lies in three 3-cycles and six 4-cycles; optimal transport can move mass along those cycles, so W₁ is large and κ(e) = –½.  
   - Forman curvature: κ_F(e) = 4 – deg(u) – deg(v) + 3·♯{triangles containing e}. In K₅, deg = 4 and ♯triangles = 3, so κ_F(e) = 4 – 4 – 4 + 9 = –3.  
   - Negative curvature on every edge implies δ-hyperbolicity with δ ≤ 2; the graph is “tree-like” at large scale although it is dense locally—exactly the hallmark of hyperbolic geometry in the coarse sense.

2. Multiplex curvature discontinuities  
   - Let layer L₁ be the small-world graph (many random chords) and layer L₂ be the 2-d lattice of the meningial lymphatic vasculature.  
   - Cross-layer edges e = (u₁, u₂) create triangles that are partly in L₁ and partly in L₂. The Ollivier–Ricci curvature of e is computed with respect to the coupled neighbourhood measure. Because L₂ is flat (κ ≈ 0) and L₁ is negatively curved (κ < 0), κ(e) jumps across the two endpoints. The difference |κ_{L₁}(e) – κ_{L₂}(e)| is a curvature discontinuity along the inter-layer edge.  
   - These discontinuities are topological obstructions to a smooth embedding of the whole multiplex into a product manifold ℝ² × ℝ.

3. Ricci-flow as an obstruction detector  
   - Define a piecewise-constant metric g_e = 1 on every edge. The Ollivier–Ricci Ricci-flow is  

     \[
     \frac{dg_e}{dt} = –\kappa_e g_e, \quad g_e(0) = 1.
     \]

   - For K₅ every κ_e = –½, so g_e(t) = exp(t/2). After time T = 2 ln 2 the edge length shrinks to ½; continuing the flow collapses the edge to zero length.  
   - The collapsing sequence exhibits the minor: first all edges of a K₅ subgraph shrink, producing a “neck” that pinches off; the resulting bubble is the topological handle forced by the K₅ minor.  
   - In the multiplex the same phenomenon occurs at cross-layer edges whose curvature is most negative; Ricci-flow pinches them first, revealing the cross-layer bottleneck.

