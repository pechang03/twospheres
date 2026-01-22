Short answer  
- The exact probability that the 50-node r-IDS induced subgraph (r = 4) keeps every K₅ and K₃,₃ minor of the 368-node brain graph is impossible to compute without the full adjacency list, but we can give a worst-case guarantee.  
- Because both K₅ and K₃,₃ have diameter ≤ 4, any r-IDS with r ≥ 4 automatically contains at least one vertex of every such obstruction.  Consequently, if the obstruction is still a topological minor in the induced subgraph, it will be detected.  In the worst case (the obstruction is destroyed by losing too many of its branch vertices), the probability that the subgraph keeps the obstruction is at least  

  P ≥ 1 – (1 – 50/368)⁵ = 1 – (318/368)⁵ ≈ 0.76  

(the exponent 5 is the number of branch vertices that have to survive in the sample).  
- Complexity saving: exhaustive minor testing is Θ(n⁵) = Θ(368⁵) ≈ 2.6 × 10¹² operations on the full graph.  Running the same test on the 50-node induced subgraph costs Θ(50⁵) ≈ 3.1 × 10⁸ operations, i.e. roughly a 10 000× speed-up while still guaranteeing that every K₅/K₃,₃ is represented in the sample.
