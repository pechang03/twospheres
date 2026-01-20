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

## Integration with twosphere

These resonator concepts connect to existing `vortex_ring.py` work:
- Trefoil knot neural pathway modeling
- Frenet-Serret frame evolution along curved paths
- Topological structures in MRI spherical geometry

Potential simulation additions:
- `RingResonator` class with FSR/Q calculation
- `MobiusResonator` with Berry phase modeling
- Ray tracing through 3D helical/twisted paths
