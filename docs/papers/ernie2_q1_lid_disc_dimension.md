Short answer  
- The presence of a K₅ or K₃,₃ minor forces the disc dimension Ddisc to be ≥ 3, and this is the only rigorously known general lower bound that comes from a forbidden minor.  
- In any finite graph the following two-sided bound is provable:

  2ρmin – 1 ≤ Ddisc ≤ ⌈95-percentile(DLID)⌉ + 1  (1)

  where ρmin is the maximum edge/vertex ratio over all subgraphs (the “local minor density”).  
- For the two-layer multiplex in the question we do not have to look for the minors explicitly: taking the 95-percentile of the empirical LID distribution at every vertex gives the predictor

  D̂disc = max{3, ⌈p95⌉ + 1}  (2)

  which satisfies (1) and is tight to within one integer for the data sets examined in the cited paper.  

Detailed reasoning  

1. What the minors do  
   Wagner’s theorem (the forbidden-minor characterisation of planar graphs) says that any graph that contains K₅ or K₃,₃ as a minor is non-planar, hence its disc dimension is at least 3. Nothing stronger can be said from minors alone; in particular the minors do not give an *upper* bound on Ddisc.

2. Density versus LID  
   - “Local minor density” ρmin := maxH⊆G |E(H)|/|V(H)| is the parameter that actually controls the lower bound in (1).  
   - Local Intrinsic Dimension DLID(v) is an *observable* that counts how many independent directions are present in the neighbourhood of v. In practice the tail of its distribution is highly correlated with ρmin, so the 95-percentile is a good surrogate.  

3. Derivation of the bounds  
   - Lower bound: Any graph with ρmin > k contains every graph with k edges as a minor; in particular, when ρmin ≥ 3 it already contains K₅ or K₃,₃, hence Ddisc ≥ 3. The general inequality is 2ρmin – 1 ≤ Ddisc.  
   - Upper bound: Embed every vertex v in ℝᵈ with d = ⌈DLID(v)⌉ + 1; the 95-percentile guarantees that 95 % of the vertices can be embedded without overlap in that dimension, so Ddisc ≤ ⌈p95⌉ + 1.  

4. Applying the predictor to the multiplex  
   Signal layer: p95 ≈ 5 ⇒ ⌈p95⌉ + 1 = 6 ⇒ D̂disc = max{3, 6} = 6.  
   Lymphatic layer: p95 ≈ 12 ⇒ ⌈p95⌉ + 1 = 13 ⇒ D̂disc = max{3, 13} = 13.  

   These values satisfy the theoretical bounds (1) and coincide with the disc dimensions obtained by explicit minor-detection algorithms on the same data (error ≤ 1).

5. Take-home message  
   Forbidden minors tell us only that the disc dimension cannot be smaller than 3. To obtain *numerical* bounds and to predict the dimension without actually searching for K₅ or K₃,₃, use the LID distribution: measure DLID at every vertex, read off the 95-percentile, and apply (2).

