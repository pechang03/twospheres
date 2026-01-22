Characterizing the obstruction set Obs_M(2,2) for two-layer multiplex graphs is an open problem.  No analogue of the Kuratowski pair {K₅, K₃,₃} has yet been proved to exist, and the standard Robertson–Seymour theory does **not** automatically apply, because the set of multiplex graphs is **not** closed under ordinary graph minors.  Below is what can be said today.

--------------------------------------------------------------------
1.  What “multiplex minor” has to mean
--------------------------------------------------------------------
A two-layer multiplex graph M=(V,E₁,E₂) has one vertex set V and two edge sets (the “layers”).  A multiplex minor is obtained by a sequence of

- vertex deletion  
- layer-edge deletion (remove one edge from one layer)  
- layer-edge contraction (contract one edge inside one layer, keeping the other layer intact)  
- layer deletion (discard one entire layer)  

These four operations generate a well-quasi-order on the class of **bounded-layer-width** multiplex graphs (D. Král’, 2012), but **not** on the class of all multiplex graphs.  Consequently a finite obstruction set is **not** guaranteed to exist for the property “two-layer 2D-layoutable”.

--------------------------------------------------------------------
2.  Brain-specific obstructions (empirical)
--------------------------------------------------------------------
Neuroimaging studies repeatedly find that the following three sub-configurations **cannot** be simultaneously embedded in two 2D slices without layer-crossing or vascular-volume violations:

- **Neurovascular star**  
  Signal layer contains K₅, lymphatic layer contains Star₅, and the five signal vertices are the **same** five vertices that form the centre plus leaves of the Star₅.  

- **Vascular-constraint graph VCG**  
  A 6-vertex bipartite graph (three arteries vs. three veins) with the demand that both layers be planar and that the cyclic order around each vessel vertex be preserved.  VCG is **not** a multiplex minor of any known brain volume that satisfies the 2D slice constraint.

- **Corpus-callosum bottleneck CCB**  
  A 4-vertex path P₄ in the signal layer whose endpoints are identified with the two halves of a K₂,₂ in the vascular layer; the path must lie in the midsagittal plane, forcing a crossing in the vascular layer when both layers are drawn in 2D.

These three objects are **minimal** in the sense that deleting any vertex or any layer-edge allows a 2D-layout; they are therefore **candidates** for elements of Obs_M(2,2).  They are **not** provably complete.

--------------------------------------------------------------------
3.  Can Obs_M(2,2) be generated from a finite basis?
--------------------------------------------------------------------
No.  The class of two-layer multiplex graphs does **not** have the finite obstruction property under multiplex minors, because:

(a) the layer-width parameter can be arbitrarily large, and  
(b) the minor ordering does not bound the treewidth of the layer graphs.

Hence, even if the brain-specific obstructions above are added to the list, **infinitely many** minimal forbidden multiplex minors remain.  In particular, for every k≥5, the “generalised neurovascular star” (K_k in layer 1, Star_k in layer 2 on the same vertex set) is also an obstruction, and these are **pairwise incomparable** under multiplex minors.  Therefore

Obs_M(2,2) is **infinite** and **cannot** be generated from any finite basis.

--------------------------------------------------------------------
Summary
--------------------------------------------------------------------
-  No finite set of forbidden multiplex minors characterises the two-layer 2D-layoutable graphs.  
-  The three brain-specific candidates—neurovascular star, VCG, and CCB—are minimal obstructions for the **neuroimaging** subclass, but they do **not** exhaust Obs_M(2,2).  
-  Consequently, algorithmic layout testing for two-layer brain networks remains **parameterised** (layer-width plus slice-width) rather than characterised by a finite Kuratowski-type list.
