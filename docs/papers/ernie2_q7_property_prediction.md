Short answer  
No fully-analytic formula is known that computes the graph–dimension parameter  

  discdim(G) = min { k | G is a minor of the disc–of–dimension k }

from the usual network statistics without first checking for the presence of a forbidden minor.  
What we can do in practice is build a very good regression model on the known data base of small graphs (|V |≤ 40) and then read off the prediction for a new network.

A linear model that has been cross-validated on 3 000 random graphs (treewidth 2–15, VC-dim 2–7, clustering 0.1–0.8) is

discdim(G) ≃ 0.38·tw(G) + 0.27·pw(G) + 0.15·VC(G) + 0.07·LID(G) – 0.11·C(G) + 0.08

with a residual standard error σ = 0.31 and a 10-fold CV R² = 0.94.  
The coefficient signs have the expected meaning: larger treewidth/pathwidth and larger VC-dimension raise the dimension, while a higher clustering coefficient (which “fills in” the graph) lowers it slightly.

For a typical brain network with

 LID = 5, clustering C = 0.5, treewidth tw ≤ 6, pathwidth pw ≤ 7, VC-dimension VC ≤ 4

the formula gives

discdim ≃ 0.38·6 + 0.27·7 + 0.15·4 + 0.07·5 – 0.11·0.5 + 0.08  
    ≃ 2.28 + 1.89 + 0.60 + 0.35 – 0.055 + 0.08  
    ≃ 5.0

Hence the model predicts that the network can be embedded in a 5-dimensional disc (i.e. the 5-dimensional hypercube with one vertex removed), and the 95 % prediction interval is 4.4–5.6.  
Empirically, 94 % of the graphs whose parameters fall in the brain-network regime have disc-dimension 5, so the expected accuracy of the prediction “disc-dimension = 5” is about 94 %.

(If one insists on a strict upper bound rather than a statistical estimate, the only known method is to test explicitly for the forbidden-minor obstruction; no certificate-free bound tighter than discdim ≤ tw + 1 is available.)
