#!/bin/bash
#
# Query ernie2_swarm about disc dimension obstructions and network properties
# without needing to empirically find the obstructions first
#

ERNIE2_PATH="/Users/petershaw/code/aider/merge2docs/bin/ernie2_swarm.py"
OUTPUT_DIR="docs/papers"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Querying ernie2_swarm about disc dimension and network properties..."
echo ""

# Query 1: LID and Disc Dimension
echo "=== Query 1: LID and Disc Dimension Relationship ==="
python "$ERNIE2_PATH" \
  --question "How do forbidden graph minors (K5 and K3,3) affect Local Intrinsic Dimension (LID) in brain connectivity networks? For a multiplex brain network with signal layer (small-world: clustering=0.5, path_length=3, LID≈5) and lymphatic layer (Euclidean: clustering=0.3, path_length=5, LID≈12), what theoretical bounds exist between disc dimension and LID? Can we predict disc dimension from LID distribution without explicit minor detection?" \
  --collection docs_library_neuroscience_MRI \
  --collection docs_library_mathematics \
  --fpt-enhanced \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q1_lid_disc_dimension.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q1_lid_disc_dimension.txt"
echo ""

# Query 2: VC Dimension and Sample Complexity
echo "=== Query 2: VC Dimension and Embeddability ==="
python "$ERNIE2_PATH" \
  --question "What is the relationship between VC dimension and disc dimension for graphs? For brain connectivity graphs with 368 nodes (D99 atlas) and small-world topology, what is the expected VC dimension? How do multiplex obstructions (cross-layer structures like K5 in signal layer combined with Star5 in lymphatic layer) affect VC dimension and sample complexity for learning graph structure?" \
  --collection docs_library_mathematics \
  --collection docs_library_neuroscience_MRI \
  --fpt-enhanced \
  --math-auto \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q2_vc_dimension.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q2_vc_dimension.txt"
echo ""

# Query 3: Graph Curvature Near Obstructions
echo "=== Query 3: Curvature Signatures of Obstructions ==="
python "$ERNIE2_PATH" \
  --question "How do forbidden graph minors (K5, K3,3) affect local graph curvature using Ollivier-Ricci and Forman curvature? Does K5 create negative curvature regions (hyperbolic geometry)? In multiplex brain networks where signal layer has small-world structure and lymphatic layer has Euclidean-like structure, do cross-layer obstructions create curvature discontinuities? Can Ricci flow detect these obstructions?" \
  --collection docs_library_mathematics \
  --collection docs_library_physics_differential_geometry \
  --fpt-enhanced \
  --math-auto \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q3_graph_curvature.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q3_graph_curvature.txt"
echo ""

# Query 4: Treewidth and Disc Dimension
echo "=== Query 4: Treewidth Bounds from Disc Dimension ==="
python "$ERNIE2_PATH" \
  --question "What bounds exist between treewidth and disc dimension for graphs? If disc(G)=d, what can we say about treewidth(G)? For brain networks with expected treewidth 5-8, what disc dimension is predicted? How do K5 and K3,3 obstructions increase treewidth? Can we use treewidth as a proxy for disc dimension without explicit obstruction detection?" \
  --collection docs_library_mathematics \
  --fpt-enhanced \
  --fpt-complexity-analysis \
  --math-auto \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q4_treewidth_bounds.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q4_treewidth_bounds.txt"
echo ""

# Query 5: r-IDS Sampling Coverage
echo "=== Query 5: r-IDS for Obstruction Detection ==="
python "$ERNIE2_PATH" \
  --question "For a brain connectivity graph with 368 nodes, if we use r-IDS (r-Independent Dominating Set) with r=4 and target_size=50 to sample 50 representative regions, what is the probability that the induced subgraph contains all K5 and K3,3 obstructions present in the full graph? Since K5 has diameter ≤4 and K3,3 has diameter ≤3, and r-IDS guarantees all nodes are within distance r of the sample (domination property), can we bound the coverage probability? What is the algorithmic complexity reduction compared to exhaustive minor detection?" \
  --collection docs_library_mathematics \
  --fpt-enhanced \
  --rids-cache-warmup \
  --rids-r-values 4 \
  --fpt-complexity-analysis \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q5_rids_sampling.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q5_rids_sampling.txt"
echo ""

# Query 6: Multiplex Dimension Formula
echo "=== Query 6: Multiplex Effective Dimension ==="
python "$ERNIE2_PATH" \
  --question "For a multiplex brain network with L=2 layers (signal and lymphatic), each with disc dimension d_layer=2 (embedded on 2-sphere), De Domenico's formula gives effective dimension d_eff = d_layer + log2(L) + coupling_complexity ≈ 2 + 1 + 0.5 = 3.5. How does this relate to actual disc dimension? Does d_eff > 3 imply we need 3D embedding? What is the coupling complexity term for neurovascular coupling between signal (functional connectivity) and lymphatic (perivascular fluid flow)?" \
  --collection docs_library_neuroscience_MRI \
  --collection docs_library_mathematics \
  --fpt-enhanced \
  --math-auto \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q6_multiplex_dimension.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q6_multiplex_dimension.txt"
echo ""

# Query 7: Unified Property-Based Prediction
echo "=== Query 7: Obstruction-Free Disc Dimension Prediction ==="
python "$ERNIE2_PATH" \
  --question "Can we predict disc dimension of a graph from measurable properties (LID, VC dimension, Ricci curvature, clustering coefficient, betweenness centrality, treewidth, pathwidth, chromatic number) WITHOUT explicitly finding forbidden minors? Derive a formula or decision boundary: disc_dimension = f(LID, VC_dim, curvature, clustering, betweenness, treewidth, ...). For brain networks with these typical properties: LID≈5, clustering≈0.5, treewidth≈6, what disc dimension is predicted? What is the expected accuracy?" \
  --collection docs_library_mathematics \
  --collection docs_library_neuroscience_MRI \
  --fpt-enhanced \
  --math-auto \
  --fpt-complexity-analysis \
  --adaptive-synthesis \
  --style technical \
  > "$OUTPUT_DIR/ernie2_q7_property_prediction.txt"

echo "Results saved to: $OUTPUT_DIR/ernie2_q7_property_prediction.txt"
echo ""

echo "=========================================="
echo "All queries complete!"
echo "Results in: $OUTPUT_DIR/"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review results in docs/papers/"
echo "2. Synthesize insights into unified framework"
echo "3. Validate on synthetic graphs"
echo "4. Apply to PRIME-DE empirical data"
