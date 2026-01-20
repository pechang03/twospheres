# Agent Instructions

This project uses the **yada-task-management** MCP server for task tracking with PHY (physics) task-ids.

## Task ID Format

Physics tasks use the `PHY` prefix: `PHY.major.minor`
- `PHY.1.x` - MRI analysis (two-sphere models, brain region geometry)
- `PHY.2.x` - Optical simulations (ray tracing, wavefront analysis)
- `PHY.3.x` - Signal processing (FFT correlation, coherence)
- `PHY.4.x` - Vortex/topology (trefoil knots, Frenet-Serret frames)

Example: `PHY.2.3` = Optical simulations category, task #3

## Quick Reference

```bash
# Get next tasks (physics domain)
# Via yada-task-management MCP:
{"name": "get_next_tasks", "input": {"advice": true, "resources": true}}

# Add a physics task
{"name": "add_task", "input": {
  "task_id": "PHY.2.1",
  "title": "Implement Zernike coefficient fitting",
  "description": "Fit measured wavefront to Zernike basis",
  "priority": "medium",
  "category": "optical_simulations"
}}

# Complete task
{"name": "complete_task", "input": {
  "task_id": "PHY.2.1",
  "notes": "Implemented via WavefrontAnalyzer.fit_zernike()",
  "check_tests": true
}}
```

## Database Location

Tasks are stored in the shared merge2docs database:
```bash
YADA_DB=../merge2docs/yada-work.db
```

## Landing the Plane (Session Completion)

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create PHY tasks for follow-up
2. **Run quality gates** (if code changed):
   ```bash
   pytest tests/
   python bin/twosphere_mcp.py --validate  # if available
   ```
3. **Update task status** - Complete finished PHY tasks
4. **PUSH TO REMOTE**:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Hand off** - Provide context for next session

## Physics Domain Skills

### MRI Analysis (`src/mri_analysis/`)
- `TwoSphereModel` - Paired brain region geometry
- Distance/overlap calculations for functional connectivity
- Integration with MRISpheres research data

### Optical Simulations (`src/simulations/`)
- `RayTracer` - pyoptools wrapper (graceful fallback if unavailable)
- `WavefrontAnalyzer` - Zernike polynomial decomposition
- Aberration characterization (defocus, astigmatism, coma, spherical)

### Signal Processing
- `compute_fft_correlation` - Frequency-domain cross-correlation
- `coherence` - Normalized correlation vs frequency
- `phase_correlation` - Phase difference analysis

### Topology (Vortex Ring)
- `VortexRing` - Trefoil knot parametric curves
- Frenet-Serret frame computation for tube surfaces
- Neural pathway modeling via knot geometry

## Research Context

Based on: "Integrating Correlation and Distance Analysis in Alzheimer's Disease"
- Two-sphere model provides geometric framework for functional connectivity
- Vortex structures model neural pathway topology
- FFT correlation reveals frequency-specific brain region coupling
