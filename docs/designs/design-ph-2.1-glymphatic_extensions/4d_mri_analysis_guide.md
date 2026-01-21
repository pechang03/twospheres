…moving and at what frequencies  
--------------------------------------------------------------------
A. Frequency bands of interest  
- Cardiac band: 0.5–2 Hz (fundamental ≈ primate heart-rate/60).  
- Respiratory band: 0.1–0.3 Hz.  
- Ultra-slow glymphatic drift: 0.001–0.01 Hz (≈0.06–0.6 cpm).  

B. Extraction pipeline  
1. Motion-correct every dynamic with a rigid 3-D registration to the time-averaged image (FSL mcflirt, 6-dof).  
2. Band-pass each voxel’s time-series with zero-phase Butterworth 4th-order filters for the three bands above.  
3. Hilbert-transform to obtain instantaneous phase φ_card(x,t), φ_resp(x,t), φ_drift(x,t).  

C. Directionality proxy (fractal two-sphere predicts reversal of cardiac phase between arterial and venous sides)  
- Define an arterial ROI = cortical surface ≤ 5 mm from middle cerebral artery tree (segmented from TOF-MRA).  
- Define a venous ROI = cortical surface ≤ 5 mm from superior sagittal sinus.  
- Compute the mean phase difference  
    Δφ = ⟨φ_card⟩_arterial – ⟨φ_card⟩_venous.  
- Model acceptance criterion:  
    awake Δφ ≈ +π/2 (flow inward),  
    sleep Δφ ≈ –π/2 (flow outward).  
  A sign change across states is mandatory before the code is allowed to fit a non-zero ε (pulsation amplitude).  

D. Power-ratio between bands  
- Integrate power spectral density in each band and form  
    R_cv = P_card / P_drift,  
    R_rv = P_resp / P_drift.  
- Empirical priors for primates under 1.5 % isoflurane:  
    R_cv ≥ 3, R_rv ≥ 1.2.  
  Values below these bounds flag poor SNR or inadequate cardiac gating.  

--------------------------------------------------------------------
3. PARAMETER EXTRACTION – turning MRI intensities into δ, ε, mᵥ
--------------------------------------------------------------------
A. Perivascular gap width δ (µm)  
- Use the PVS map A_PVS(x) created in 1B.  
- Inside every A_PVS=1 voxel, measure the T2 relaxation time and convert to water-content fraction w via the linear calibration  
    w = (T2 – T2_cortex) / (T2_CSF – T2_cortex),  
    with T2_cortex = 80 ms, T2_CSF = 2000 ms.  
- The model gives the analytic relation between w and δ:  
    w = 1 – exp(–δ/120 µm).  
  Solve voxel-wise for δ; report median and IQR across the whole brain.  
- Required preprocessing: B0 field-map correction to remove through-plane de-phasing that biases T2 by up to 8 %.  

B. Pulsation amplitude ε (dimensionless)  
- From the cardiac-band filtered data, compute the peak-to-peak signal change  
    ΔS_card = max_t(S(t)) – min_t(S(t))  
  in the arterial ROI.  
- Convert to relative volumetric strain using the empirical factor 0.35 % signal change per 1 % volume change (phantom-validated).  
- ε is defined as the ratio of peak strain to mean strain; typical primate values 0.15–0.25 are acceptable.  

C. Metabolic demand field mᵥ(x) (arbitrary units)  
- Acquire a simultaneous BOLD sequence (TR = 1 s, TE = 25 ms).  
- Compute the fractional BOLD signal change during 2 min of hypercapnia (5 % CO₂).  
- The model assumes mᵥ ∝ fractional BOLD change / baseline CMRO₂.  
- Smooth to 1 mm resolution to match the mesh used by the two-sphere solver.  

D. Critical pre-processing checklist  
- Motion correction with 0.1 mm translational tolerance.  
- B0 distortion correction (FSL topup with reverse-phase-encoding blip-up/down pair).  
- Receive-field bias correction (ANTs N4, shrink factor 4).  
- Gradient-non-linearity correction (vendor-provided spherical-harmonic coefficients).  
Failure to apply any of the above introduces systematic errors that exceed the 5 % tolerance allowed for δ and ε.  

--------------------------------------------------------------------
4. SLEEP-WAKE VALIDATION – testing the 60 % interstitial expansion
--------------------------------------------------------------------
A. Water-content ratio  
- In the same animal, record 30 min awake (eyes-open,

