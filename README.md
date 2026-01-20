# TwoSphere MCP Server

An MCP server for optical physics simulations and MRI spherical geometry analysis, integrating pyoptools with the MRISpheres/twospheres research project.

## 🔬 Features

### MRI Analysis Tools
1. **Two-Sphere Visualization** - 3D visualization of paired brain regions
2. **Vortex Ring Analysis** - Trefoil knot and vortex tube modeling for connectivity patterns
3. **FFT Correlation** - Frequency-domain correlation between brain regions
4. **Distance Metrics** - Geometric interpretation of functional connectivity

### Optical Simulation Tools (pyoptools)
5. **Ray Tracing** - Optical system simulation
6. **Wavefront Analysis** - Wavefront calculations and visualization
7. **Lens Design** - Optical element placement and optimization

## 📦 Quick Start

### Installation

```bash
# Create dedicated conda environment
conda create -n twosphere python=3.11
conda activate twosphere

# Install dependencies
pip install -r requirements.txt

# Link to MRISpheres research
ln -s ~/MRISpheres/twospheres data/twospheres
```

### Configuration

Create `.env_twosphere`:

```bash
# MRISpheres data path
TWOSPHERES_PATH=~/MRISpheres/twospheres

# Server settings
TWOSPHERE_PORT=8006

# Optional: OpenAI for advanced analysis
OPENAI_API_KEY=your_key_here
```

## 🚀 Usage

### As MCP Server (stdio)

```bash
python bin/twosphere_mcp.py
```

### As HTTP Server

```bash
python bin/twosphere_http_server.py --port 8006
```

### With ernie2_swarm

```bash
# Query optical physics with twosphere integration
python ../merge2docs/bin/ernie2_swarm_mcp.py \
    -q "Explain how spherical harmonics relate to MRI signal analysis" \
    -c docs_library_Physics \
    -c docs_library_Neuroscience
```

## 🧪 Example: Two-Sphere Correlation

```python
from src.mri_analysis import TwoSphereModel, compute_fft_correlation

# Create paired brain region model
model = TwoSphereModel(radius=1.0)
model.add_sphere(center=[0, 1, 0], label="Region_A")
model.add_sphere(center=[0, -1, 0], label="Region_B")

# Compute frequency-domain correlation
correlation = compute_fft_correlation(
    region_a_signal,
    region_b_signal,
    sampling_rate=1000
)
print(f"Cross-correlation peak: {correlation.peak_value}")
```

## 📁 Project Structure

```
twosphere-mcp/
├── bin/
│   ├── twosphere_mcp.py        # MCP server (stdio)
│   └── twosphere_http_server.py # HTTP/SSE server
├── src/
│   ├── simulations/            # pyoptools wrappers
│   │   ├── ray_tracing.py
│   │   └── wavefront.py
│   └── mri_analysis/           # MRISpheres integration
│       ├── two_sphere.py
│       ├── vortex_ring.py
│       └── fft_correlation.py
├── data/
│   └── twospheres -> ~/MRISpheres/twospheres
├── tests/
├── docs/
├── examples/
├── requirements.txt
└── .env_twosphere
```

## 🔗 Integration with MRISpheres

This MCP server wraps and extends the research code from `~/MRISpheres/twospheres/`:

- `two_spheres.py` → `src/mri_analysis/two_sphere.py`
- `VortexRingInterpolation.py` → `src/mri_analysis/vortex_ring.py`
- `signal_processing.py` → `src/mri_analysis/fft_correlation.py`

## 📚 Research Context

Based on the paper: "Integrating Correlation and Distance Analysis in Alzheimer's Disease"

The two-sphere model provides a geometric framework for:
- Functional connectivity analysis via frequency-domain correlation
- Distance metrics between brain regions
- Visualization of neural pathway changes in AD progression

## 🔧 Dependencies

- **pyoptools** >= 0.3.7 - Optical system simulation
- **numpy** - Numerical computing
- **scipy** - Signal processing and interpolation
- **matplotlib** - Visualization
- **mcp** - Model Context Protocol server

## 📄 License

Research code - see MRISpheres project for licensing.
