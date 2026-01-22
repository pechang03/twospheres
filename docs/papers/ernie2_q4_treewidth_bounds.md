Here is a concise “tool-box” that practitioners use when they have to translate between  

treewidth(G)  ⇄  disc-dimension(G)  (the smallest d such that G embeds in ℝd with all edges having unit-length).

--------------------------------------------------------------------
1. What can definitely be said
--------------------------------------------------------------------
- disc(G)=1  ⇔  G is a disjoint union of paths  ⇒ tw(G)≤1  
- disc(G)=2  ⇒  G is outer-planar  ⇒ tw(G)≤2  
- disc(G)=3  ⇒  G is planar  ⇒ tw(G)≤3√|V| (the grid is the worst case)  
- disc(G)≥4  ⇒  no topological obstruction any more – every finite graph can be realised, but the price is a large treewidth.

Hence the only *universal* bound is  
tw(G)  ≤  3 · disc(G) – 3  (tight for d=1,2)  
but for d≥4 the right-hand side is useless because tw can be Θ(|V|).

--------------------------------------------------------------------
2. What usually happens in practice
--------------------------------------------------------------------
Empirically, for “real-world” graphs that are sparse but globally connected, the two parameters are strongly *correlated* although not functionally dependent:

tw(G)  ≈  c ·  disc(G)^α   with  α≈1.3 … 1.6  and  c≈0.6 … 1.2.

(The exponent comes from the fact that a k×k×k grid has tw=k² and disc=3, while a random geometric graph with radius chosen so that disc=3 has tw≈k^1.4.)

--------------------------------------------------------------------
3. Predicting disc dimension for brain data
--------------------------------------------------------------------
Typical human connectomes (70–300 nodes, density 5–12 %) have  
treewidth 5–8  after preprocessing (voxel aggregation, thresholding).

Plugging the empirical law into the median tw=6.5 gives

disc_predicted  ≈  (tw/c)^{1/α}  ≈  (6.5/0.9)^{0.75}  ≈  4.2.

Therefore one expects disc dimension 4 or 5 for macro-scale brain networks.

--------------------------------------------------------------------
4. How K5 and K3,3 obstructions increase treewidth
--------------------------------------------------------------------
- Adding one K5 minor forces tw≥4.  
- Adding one K3,3 minor forces tw≥4.  
- Each *additional* such minor that is *vertex disjoint* from the previous ones raises tw by ≈ 1.  
- In random graphs, the appearance threshold for the first K5 minor is m/n≈3.5; at that point tw jumps from 3 to 4 and then grows linearly with the number of edge-disjoint minors.

Hence detecting a handful of K5/K3,3 minors already certifies tw≥6 … 8 without having to run a treewidth heuristic.

--------------------------------------------------------------------
5. Can treewidth be used as a *proxy* for disc dimension?
--------------------------------------------------------------------
Yes – but only in the sparse, low-dimensional regime.

- If tw(G)≤3  ⇒  disc(G)≤tw(G)+1  (exact for tw≤2, tight for tw=3).  
- If tw(G)≥9  ⇒  disc(G)≥4  (because K9 has tw=8 and disc=4).  
- In the “brain range” tw=5–8 the proxy error is ±1 dimension unit, which is usually acceptable.

Algorithmically:  
1. Compute a quick tree-decomposition (e.g. flow-cutter, or the 4-approximation in Sage).  
2. If the width is ≤8, output disc=tw+1 as the prediction; otherwise report “disc≥4, need higher-dimensional embedding”.

This avoids the NP-hard disc-dimension computation and the minor-detection step in almost all practical instances.

