# Optical Resonators for Photonic Lab-on-Chip

Advanced resonant structures for sensing, filtering, and on-chip laser sources.

---

## Ring Resonator

The simplest optical resonator: a circular TIR path in PDMS.

```
         ╭───────────╮
        ╱             ╲
       │    Air/PDMS   │  ← Light circulates via TIR
       │    boundary   │
        ╲             ╱
         ╰─────┬─────╯
               ↑
         Evanescent coupling
               │
    ═══════════●═══════════  ← Bus waveguide (input/output fiber)
```

### Resonance Condition

Light constructively interferes when round-trip path = integer wavelengths:

- **Round trip path:** L = 2πr
- **Resonance:** n × L = m × λ (m = integer)
- **Resonant wavelengths:** λ_m = 2πrn / m

### Key Parameters

| Parameter | Formula | Example (r=100µm, n=1.4, λ=900nm) |
|-----------|---------|----------------------------------|
| Free Spectral Range | FSR = λ²/(2πrn) | 0.92 nm |
| Finesse | F = FSR/FWHM | 10-1000 (loss dependent) |
| Q-factor | Q = λ/FWHM | 10³-10⁵ |

### Sensing Application

Refractive index change in ring shifts resonance:
- **Sensitivity:** Δλ/Δn = λ/n_g ≈ 640 nm/RIU
- **Detection limit:** ~10⁻⁶ RIU with pm-resolution spectrometer
- **Advantage:** Light circulates many times → long effective interaction length

---

## Mach-Zehnder Interferometer

Two-arm interferometer for differential sensing.

```
                    Reference arm (air or sealed)
                   ╱                              ╲
  Fiber → Splitter                                  Combiner → Detector
                   ╲                              ╱
                    Sample arm (chamber/channel)
```

### Interference

- **Phase difference:** Δφ = (2π/λ) × (n₁L₁ - n₂L₂)
- **Output intensity:** I = I₀ × cos²(Δφ/2)
- **Sensitivity:** For L=5mm, λ=900nm: Δφ = 35 rad per Δn=0.001

### Design Variants

| Configuration | Description | Use Case |
|--------------|-------------|----------|
| **Balanced** | Equal arm lengths, differential n | Common-mode rejection |
| **Unbalanced** | Built-in path difference | Quadrature detection |
| **Thermal-compensated** | Both arms in same thermal zone | Stable baseline |

---

## Dye Laser Ring

Ring resonator filled with laser dye for on-chip coherent light source.

### Structure

```
    Side view                      Top view

    Pump fiber (vertical)              ↓ pump fiber
         │                            ╭───────╮
         ▼                           │    ●    │  ← dye-filled ring
    ══════════  PDMS surface         │  gain   │
         │                            ╰────┬───╯
    ─────●─────  ring (dye)                │
                                     ══════●══════  signal fiber
                                           (evanescent output)
```

### Pumping Methods

| Method | Description | Efficiency |
|--------|-------------|------------|
| **Direct illumination** | Pump fiber above ring | High, easy alignment |
| **Evanescent pumping** | Pump fiber crosses ring | Lower, but all-waveguide |
| **Fiber-lens** | Curved fiber as cylindrical lens | Good coverage of ring |

### Evanescent Pumping Detail

```
    Cross-section at fiber-ring crossing
    
           pump fiber
              ●══════════
        ──────┼──────  ← evanescent field (~200nm decay)
    ═══════════════════  ring channel (dye)
```

- Pump fiber passes perpendicular to ring plane
- Evanescent field excites dye molecules within ~200nm
- Gap must be < 500nm for useful coupling
- No free-space optics needed — fully integrated

### Dye Options

| Dye | Pump λ | Emission λ | Notes |
|-----|--------|------------|-------|
| Rhodamine 6G | 532nm | 580-620nm | High quantum yield |
| Rhodamine B | 532nm | 610-650nm | Red-shifted |
| Coumarin 460 | 405nm | 450-490nm | Blue emission |
| DCM | 532nm | 620-700nm | Large Stokes shift |

---

## Helix Resonator

3D spiral path for extended resonator length in compact footprint.

```
    Side view                    Top view (projection)
    
    ╭───╮                            ╭───────╮
    │   │  ↑                        │       │
    ╰─╮ │  │ pitch p               │   r   │
      │ │  │                        │       │
    ╭─╯ │  ↓                        ╰───────╯
    │   │
    ╰───╯
```

### Path Length

- **Per turn:** L = √[(2πr)² + p²]
- **N turns:** L_total = N × √[(2πr)² + p²]

### Two Tuning Parameters

| Parameter | Affects | Tuning Method |
|-----------|---------|---------------|
| **Radius r** | Circumference contribution | Lateral pressure, design choice |
| **Pitch p** | Vertical contribution | PDMS stretch, layer spacing |

### Advantages

- More path length per footprint than flat ring
- Decouple radial and axial tuning
- Different dispersion (waveguide geometry varies)
- Natural CW/CCW mode separation → gyroscope potential

### Fabrication

- Multi-layer PDMS with offset channels + angled vias
- 3D-printed mold for true helical channel
- Typical: 3-10 turns, r=100-500µm, p=50-200µm

---

## Möbius Resonator

Topological resonator with twisted path — requires 2 circuits to return to start.

```
    Simple ring:                  Möbius ring:
    
         ╭───────╮                    ╭───╮
        │       │                   ╱     ╲
        │   ●   │                  │   ╳   │  ← twist
        │       │                   ╲     ╱
         ╰───────╯                    ╰───╯
    
    1 circuit = complete           1 circuit = "other side"
    Path = 2πr                     2 circuits to return
                                   Path = 4πr
```

### Topological Properties

| Property | Simple Ring | Möbius Ring |
|----------|-------------|-------------|
| Path to return | 2πr | **4πr** |
| FSR | λ²/(2πrn) | **λ²/(4πrn)** (half) |
| Mode density | m modes | **2m modes** same bandwidth |
| Topology | Trivial | Non-orientable |

### Two Parameters (2r Tuning)

1. **Geometric radius r** — physical size
2. **Topological factor** — twist doubles effective path

Effective resonator length = 2 × geometric circumference

### Unique Physics

| Effect | Description |
|--------|-------------|
| **Berry phase** | Geometric phase from traversing twisted path |
| **Polarization rotation** | Polarization state evolves around twist |
| **Mode splitting** | CW/CCW modes experience different topology |
| **Topological protection** | Some modes robust against perturbation |

### Fabrication

```
    Two-layer Möbius construction:
    
    Top layer:      ╭───────────╮
                    │     ╲     │
                    │      ╲    │  ← twisted via
                    │       ╲   │
    Bottom layer:   │        ╲──┴───────────╮
                    ╰───────────────────────╯
```

- Two-layer PDMS with diagonal via connecting top path to bottom
- 3D-printed mold for continuous twisted channel
- Critical: smooth transition at twist to minimize scattering loss

---

## Comparison Summary

| Resonator | Path | FSR | Tuning DOF | Fabrication |
|-----------|------|-----|------------|-------------|
| **Ring** | 2πr | λ²/(2πrn) | 1 (r) | Single layer, easy |
| **MZI** | L₁, L₂ | N/A (broadband) | 2 (arm lengths) | Single layer |
| **Helix** | N√[(2πr)²+p²] | Varies | 2 (r, p) | Multi-layer |
| **Möbius** | 4πr | λ²/(4πrn) | 2 (r, topology) | Multi-layer, precision |

---

## Sensing Parameters

Design data for NIR absorption-based sensing in PDMS ring resonators.

### Optical Power Requirements

| Application | Power | Notes |
|-------------|-------|-------|
| **Passive sensing** | 0.2–1 mW | Safe for PDMS, no thermal drift |
| **Shot-noise limit** | ~70 µW | SNR=1 for Δα=10⁻⁴ cm⁻¹ |
| **Dye laser threshold (cw)** | 0.8–1.2 mW | Rhodamine 6G @ 532nm pump |
| **Dye laser threshold (pulsed)** | 3–5 µJ/pulse | 5ns pulses, 10Hz rep rate |
| **Dye laser operation** | 1–5 mW pump | 6-8% slope efficiency |

### Optimal Ring Radius by Analyte

For PDMS n=1.406, n_g=1.47, evanescent fraction Γ≈0.25 (4µm strip waveguide).

| Analyte | Fundamental | Overtone | λ (nm) | R (µm) | Q | LOD |
|---------|-------------|----------|--------|--------|---|-----|
| **CO₂** | 4.26 µm | 4ν₂+ν₁ | 884 | 50-80 | 6×10⁴ | 200 ppm |
| **CO₂** | 2.7 µm | 2ν₂+ν₃ | 2050 | 80 | 6×10⁴ | 200 ppm |
| **Lactate** | 3.4 µm (C-H) | 3νC-H | 905 | 60 | 8×10⁴ | 120 ppm |
| **Glucose** | 2.9 µm (O-H) | 3νO-H | 833 | 50 | 9×10⁴ | 90 ppm |
| **Glucose** | 3.4 µm (C-H) | 2νC-H | 1695 | 60 | 8×10⁴ | 110 ppm |
| **Dissolved O₂** | 1.27 µm | 0-0 (a¹Δg) | 1270 | 40 | 1×10⁵ | 15 µM |

**Note:** Mid-IR bands (>2.3µm) blocked by PDMS absorption — use NIR overtones.

### NIR Overtone Bands (800-1000nm Si-detector window)

These bands allow use of inexpensive Si CCD/CMOS detectors:

| Molecule | λ (nm) | ε (M⁻¹cm⁻¹) | Band Assignment |
|----------|--------|-------------|------------------|
| CO₂ | 884 | 0.05 | 4ν₂+ν₁, 2ν₁+ν₃ |
| Lactate | 905 | 0.2 | 3νC-H |
| Glucose | 833 | 0.15 | 3νO-H + νC-O |

### Sensitivity Formula

```
S = (λ·Q)/(2π²·R·n_g) · Γ · (∂n/∂c)
```

Where:
- λ = wavelength
- Q = quality factor
- R = ring radius
- n_g = group index (~1.47 for PDMS)
- Γ = evanescent field fraction (~0.25)
- ∂n/∂c = refractive index change per concentration

### Design Guidelines

1. **FSR ≥ 2nm** for single Vernier filter coverage
2. **Bending loss < 0.01 dB/round-trip** — sets minimum R
3. **R = 40-80 µm** optimal range for NIR biosensing
4. **Q ~ 10⁴-10⁵** achievable in PDMS

---

## Thermal Considerations for Dye Lasers

Pump power dissipation must be managed to prevent:
- PDMS thermal degradation (>150°C)
- Dye photobleaching
- Refractive index drift (dn/dT ≈ -4×10⁻⁴/°C)

### Heat Load Estimate

```
P_heat = P_pump × (1 - η_laser - η_fluorescence)
       ≈ P_pump × 0.85  (for η_laser ≈ 7%, η_fluor ≈ 8%)
```

For 5 mW pump: ~4.3 mW heat to dissipate

### Cooling Strategies

| Method | Heat removal | Complexity |
|--------|--------------|------------|
| **Passive (PDMS conduction)** | ~1-2 mW | None |
| **Microfluidic flow** | ~10-50 mW | Medium |
| **Peltier under chip** | ~100+ mW | High |
| **Pulsed operation** | Duty cycle limited | Low |

**Rule of thumb:** For cw operation >2 mW pump, use microfluidic cooling (flow dye solution through ring).

---

## OOC Biosensor Applications

Integration with Organ-on-Chip platforms for real-time metabolic monitoring.

### Key Biomarkers (from biosearch analysis)

| Marker | Threshold | Biological Significance |
|--------|-----------|-------------------------|
| **pH** | < 6.8 | Warburg effect (glycolysis) |
| **Lactate** | > 10 mM | Glycolysis upregulation |
| **O₂** | < 2% | Hypoxia → EMT risk |
| **CO₂** | Elevated | Metabolic activity indicator |
| **Glucose** | Depletion | High consumption = proliferation |

### Sensor Configuration

```
    ┌─────────────────────────────────────────┐
    │  OOC Chamber (37.1°C, humid)            │
    │  ┌─────────────────────────────────┐    │
    │  │  Cell culture region            │    │
    │  └──────────────┬──────────────────┘    │
    │                 │ sample flow           │
    │  ┌──────────────▼──────────────────┐    │
    │  │  Ring resonator array           │    │
    │  │  ○ CO₂ (884nm, R=60µm)         │    │
    │  │  ○ Lactate (905nm, R=60µm)     │    │
    │  │  ○ Glucose (833nm, R=50µm)     │    │
    │  │  ○ O₂ (1270nm, R=40µm)         │    │
    │  └─────────────────────────────────┘    │
    │         ↑               ↓               │
    │    Input fiber    Output to spectrometer│
    └─────────────────────────────────────────┘
```

### Detection Limits vs Clinical Relevance

| Analyte | Ring LOD | Clinical Range | Sufficient? |
|---------|----------|----------------|-------------|
| CO₂ | 200 ppm | 5% (~50,000 ppm) | ✓ Yes |
| Lactate | 120 ppm (~1 mM) | 1-20 mM | ✓ Yes |
| Glucose | 90 ppm (~0.5 mM) | 4-8 mM | ✓ Yes |
| O₂ | 15 µM | 0-250 µM | ✓ Yes |

All detection limits are well within clinical relevance for OOC monitoring.

---

## Integration with twosphere

These resonator concepts connect to existing `vortex_ring.py` work:
- Trefoil knot neural pathway modeling
- Frenet-Serret frame evolution along curved paths
- Topological structures in MRI spherical geometry

Potential simulation additions:
- `RingResonator` class with FSR/Q calculation
- `MobiusResonator` with Berry phase modeling
- Ray tracing through 3D helical/twisted paths
- Thermal dissipation modeling for dye lasers
