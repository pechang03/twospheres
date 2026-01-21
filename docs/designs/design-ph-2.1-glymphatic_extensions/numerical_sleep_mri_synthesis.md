Below is a compact "Minion brief" that can be passed to three specialists (numerical-methods, sleep-physiology, MRI-physics) with just the information they need to act.  
The boxes contain the key technical items each specialist should deliver.

---------------------------------------------------
1. NUMERICS – surface shallow-water + 1-D perivascular network
---------------------------------------------------
Surface mesh: start from a recursive icosahedral refinement  
(10–12 levels → 2.6 M faces, edge-length h ≈ 0.7 mm).  
Map the fractal radius R(θ,φ)=R_WM+ε·f(θ,φ) onto each vertex;  
store curvature κ and dual-area |Ωᵢ| at every cell.

Scheme choice  
- Use nodal Discontinuous Galerkin (DG) on curved triangles with  
  RK4–SSP time-stepping (τ≈0.5 ms for CFL=0.3).  
- Approximate the shallow-water terms by "hydrostatic" wave equation  
  (valid because Rossby ≪ 1, Mach ≪ 1 for glymphatic speeds).  
- Divergence and gradient → covariant derivatives computed with  
  metric tensor gᵢⱼ obtained from R(θ,φ).  
- Add Laplace-Beltrami artificial viscosity (ν=10⁻⁶ m² s⁻¹) only in  
  cells where the curvature variation ε|∇²f| exceeds threshold.

1-D perivascular network  
- Extract the arterial tree from the same mesh: skeletonise the  
  Voronoi edges that run in the "valleys" of f(θ,φ) (fractal  
  dimension D≈1.37).  
- Solve on each 1-D segment l the Poiseuille–Starling equation  
  q = –(δ³/12μ) ∂p/∂s + L_p(p_i–p_tissue)  
  with DG–1D (degree 2 polynomials) and couple to tissue pressure  
  via mortar flux: ∫_edge Nᵢ p_tissue dΓ = ∫_edge Nᵢ p_1D dΓ.  
- Neural coupling ∑q_e = α m_v enters as a source term in the  
  tissue continuity equation.  
- Use implicit-explicit (IMEX) splitting: treat δ³/μ explicitly,  
  tissue compliance implicitly; keeps time-step ∝ τ.

---------------------------------------------------
2. SLEEP-WAKE – how to switch the glymphatic "valves"
---------------------------------------------------
Parameter to toggle: perivascular gap δ(t)  
- Awake δ_awake = 7 µm, asleep δ_asleep = 9 µm (≈60 % area increase).  
- Smooth transition over 2–3 min with tanh(t/30 s).  
- Leave tissue compliance d(x,t) and metabolic demand m_v unchanged  
  (they vary <10 % between states in rodent data).

Timescale of state changes  
- Light sleep → deep NREM: 30–90 s (K-complexes).  
- Wake → sleep: 2–3 min (EEG theta power drop).  
- Model therefore keeps δ(t) piece-wise constant over 30-s windows and  
  updates only at the start of each MRI frame.

---------------------------------------------------
3. 4-D MRI – what to pull out of the data
---------------------------------------------------
Spatial features (surface maps, 1 mm resolution)  
- Water-content "slopes" ∇MRI along principal sulci vs gyri;  
  expect 20–30 % steeper slope in sulci during sleep.  
- Relative uptake in peri-arterial "rim" (mask: 0–2 mm from  
  pial surface) vs parenchyma; compute ratio R=rim/parenchyma.  
  R_sleep/R_wake ≈1.5 is a key metric.

Temporal signatures  
- Cardiac-band (1–1.5 Hz) amplitude of water-content signal;  
  integrate power in 0.8–2 Hz band and normalise by mean.  
- Ultra-slow drift (<0.1 Hz) slope during first 15 min of sleep;  
  fit A·exp(–t/τ) + B, extract τ (expected 6–8 min).

Phase relationships (cross-correlation)  
- Between arterial-CSF flow (measured with phase-contrast) and  
  water-content rise in peri-arterial mask; look for phase lead  
  of CSF ≈ π/4 (≈0.2 s at 1 Hz).  
- Between contralateral sulci: time-shift of water-content peak  
  gives effective wave speed; target 2–4 mm s⁻1.

