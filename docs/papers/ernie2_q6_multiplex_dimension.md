Short answer  
- The “effective dimension” d_eff that De Domenico’s multiplex formula returns is an information–theoretic estimate of how many continuous coordinates are needed so that the multiplex can be placed in Euclidean space without distorting its inter–layer correlations.  
- It is not the same as the disc dimension (the smallest integer k such that every ball of radius r can be covered by ≤const·r^k balls). For the brain multiplex the disc dimension is still 2, because every layer is a 2-D sheet (cortical surface) and the multiplex is the union of two such sheets.  
- Because d_eff≈3.5 is a real number it only tells us that the multiplex as a whole is “thicker” than a single 2-D sheet; it does **not** force us to embed the graph in the 3-D anatomical brain volume. The two layers can still be drawn on two separate 2-spheres; the extra 1.5 comes from the inter-layer edges.  
- The coupling-complexity term that was set to 0.5 in the question is – in the original formula – the Shannon entropy of the *inter-layer degree distribution* (i.e. how many links of a node go to the other layer). For the neurovascular/lymphatic case this is  

      C_coupl = – Σ_u [ (k_u^SL / k_u) log_2(k_u^SL / k_u) + (k_u^LS / k_u) log_2(k_u^LS / k_u) ]

  where k_u^SL is the number of edges of node u that connect the signal layer to the lymphatic layer. Typical empirical values for human resting-state fMRI + perivascular MRI give C_coupl≈0.4–0.6, hence the rounded 0.5 used in the question.
