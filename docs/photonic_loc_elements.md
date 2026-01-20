# Photonic Lab-on-Chip (LOC) Elements

PDMS-based optical elements for microfluidic photonics. Exploits refractive index contrast between PDMS (n≈1.41), air (n=1.0), water (n≈1.33), and other fluids.

## Key Refractive Indices (NIR ~900nm)

| Material | n | Notes |
|----------|---|-------|
| PDMS | 1.403 | dn/dT ≈ -4×10⁻⁴/°C |
| Air | 1.000 | |
| Water | 1.33 | |
| Glycerol | 1.47 | Higher than PDMS |
| Galinstan | — | ~85% reflective (metallic) |
| Optical fiber | 1.46 | Silica core |

**Critical angles (PDMS to...):**
- Air: **45.2°**
- Water: **70.5°**
- Glycerol: no TIR (n_glycerol > n_PDMS)

---

## Passive Elements (Fixed Geometry)

### Mirrors (TIR-based)

| Element | Structure | Function |
|---------|-----------|----------|
| **Flat air mirror** | 45° PDMS/air interface | 90° beam redirect |
| **Concave air mirror** | Curved PDMS/air cavity | Focusing reflector |
| **Convex air mirror** | Curved PDMS/air bump | Diverging reflector |
| **Parabolic air mirror** | Parabolic PDMS/air surface | Collimate point source, or focus parallel beam |
| **Elliptical air mirror** | Elliptical PDMS/air surface | Image one point to another |
| **Cylindrical air mirror** | Cylindrical PDMS/air surface | Line focus |
| **Toroidal air mirror** | Toroidal PDMS/air surface | Astigmatism correction |

**Advantages of curved TIR mirrors:**
- Lossless (TIR ≈ 100% vs metallic ~85-95%)
- Combined redirect + focus = fewer elements
- Any conic section fabricable via soft lithography

### Lenses

| Element | Structure | Function |
|---------|-----------|----------|
| **Plano-convex lens** | Curved PDMS/air cavity | Focusing |
| **Meniscus lens** | Convex-concave PDMS/air | SA-optimized focusing (79% SA reduction) |
| **Cylindrical lens** | Linear PDMS/air channel | Line focus |
| **Compound lens** | Multiple air cavities | Aberration correction |

### Other Passive Elements

| Element | Structure | Function |
|---------|-----------|----------|
| **Beam splitter** | Angled PDMS/water interface (~70°) | Partial reflection |
| **Prism/wedge** | Triangular air or water pocket | Beam steering |
| **Waveguide** | PDMS ridge with air cladding | TIR confinement |
| **Diffraction grating** | Periodic air/PDMS channels | Wavelength separation |
| **Fabry-Pérot cavity** | Two parallel air gaps | Wavelength selection |

---

## Tunable Elements (Fluid-Switched)

| Element | Empty (Air) | Filled | Effect |
|---------|-------------|--------|--------|
| **Switchable mirror** | TIR (reflect) | Water (transmit) | On/off optical switch |
| **Variable splitter** | 100% reflect | Glycerol (partial) | Adjustable split ratio |
| **Tunable lens** | f=short | Water (f=longer) | Focus adjustment |
| **Optical shutter** | Transmit | Liquid metal | Block beam |
| **Index-match bypass** | Lens active | n-matched fluid | Disable optic |

---

## Specialty Elements

| Element | Material | Application |
|---------|----------|-------------|
| **Liquid metal mirror** | Galinstan in chamber | High reflectivity, reconfigurable |
| **Absorber/filter** | Dye solution | Wavelength filtering |
| **Scattering cell** | Particle suspension | Diffuser, mixer sensor |
| **Evanescent coupler** | Thin PDMS wall between channels | Proximity-based switching |

---

## Fiber Coupling Techniques

### Index-Matched Fiber Interface
Fill channel before lens with liquid PDMS (n≈1.41):
- Eliminates air gap Fresnel reflections (~3.5% loss per surface)
- Smooth transition: fiber (1.46) → liquid PDMS (1.41) → solid PDMS → lens
- Reduces distortion vs fiber→air→PDMS

### Periodic Fiber Supports with Air Cladding
```
   ┌─────┐     ┌─────┐     ┌─────┐
───┤PDMS ├─air─┤PDMS ├─air─┤PDMS ├───  ← periodic posts
   └──┬──┘     └──┬──┘     └──┬──┘
      │           │           │
   ═══●═══════════●═══════════●═══     ← fiber
```
- Air gaps: TIR containment
- Posts: mechanical alignment every few mm
- Easy fiber insertion/removal
- Low propagation loss

---

## OOC Platform Considerations

**Environment requirements:**
- Temperature: 37.1°C (body temp for cell culture)
- Humidity: High (prevent PDMS dehydration)
- On-chip heaters + temp sensors for PID control

**Thermal effects on optics:**
- PDMS dn/dT ≈ -4×10⁻⁴/°C
- At 37°C vs 25°C: Δn ≈ -0.005 → focal shift ~1%

**Detection options:**
| Method | Pros | Cons |
|--------|------|------|
| Array spectrometer (AS7341) | Compact, fast | Fixed wavelengths |
| Prism + mono RPi NoIR cam | Full spectrum, cheap | Needs alignment |
| Raspberry Pi camera | Visual imaging | Not NIR-sensitive without NoIR |

---

## Fabrication Constraints

### Demolding Limit: Max 180° per Layer

Soft lithography requires mold release — features cannot wrap >180°:

```
Can't make (single layer):     Can make:
   ╭───────╮                      ╭───────╮
  │  360°  │  ← mold trapped     │  180°  │  ← mold lifts out
  │  ring  │                     │  half  │
   ╰───────╯                      ╰───┬───╯
                                     ↓ demolding direction
```

**Solutions for full rings:**
1. **Two-layer bonding**: Top half + bottom half, plasma bond
   - Alignment: ~5-10µm with 4-point fiducials + microscope
   - PDMS is soft and sticky — one shot to place, no repositioning
   - Much worse than <1µm single-layer lithography precision
2. **Racetrack resonator**: Two 180° bends + straights (single layer, preferred)
3. **Spiral entry**: Open ring with tangential waveguide entry
4. **3D printing**: Direct-write for complex 3D (not soft lithography)

**Design principle**: Keep critical features in single layer whenever possible.

### Surface Roughness

Typical PDMS from SU-8 mold:
- RMS roughness: 10-100 nm
- Correlation length: 0.5-5 µm
- Scattering loss: ~0.1-1 dB/cm at NIR

### Minimum Feature Sizes

| Feature | Minimum | Notes |
|---------|---------|-------|
| Channel width | ~10 µm | Limited by photolithography |
| Channel depth | ~5 µm | Aspect ratio <10:1 for demolding |
| Radius of curvature | ~20 µm | Smaller = more roughness |
| Gap (evanescent) | ~200 nm | Requires e-beam or interference litho |

---

## Design Rules of Thumb

1. **TIR mirrors** are lossless — prefer over metallic when geometry allows
2. **Curved air cavities** combine reflection + focusing in one element
3. **Index-match** fiber interfaces to reduce Fresnel losses
4. **Air cladding** provides best waveguide confinement (largest Δn)
5. **Glycerol fill** disables TIR elements (n > PDMS)
6. **Liquid metal** for reconfigurable high-reflectivity mirrors
7. **Account for thermal shift** in precision applications (~1%/12°C)
8. **Max 180°** features per mold layer (demolding constraint)
9. **Aspect ratio <10:1** for reliable demolding
