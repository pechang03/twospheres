# MRI Analysis Tools Specification

## Tool 1: `extract_phase_coherence`

### Purpose
Extract phase relationships between brain regions across frequency bands to validate glymphatic flow direction.

### MCP Schema
```json
{
  "name": "extract_phase_coherence",
  "description": "Extract phase coherence between arterial and venous ROIs across cardiac, respiratory, and ultra-slow frequency bands. Validates glymphatic flow direction.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string", "description": "Path to 4D water content NIfTI"},
      "roi_arterial": {"type": "string", "description": "NIfTI mask for arterial regions (near MCA)"},
      "roi_venous": {"type": "string", "description": "NIfTI mask for venous regions (near SSS)"},
      "tr_seconds": {"type": "number", "description": "Repetition time in seconds"},
      "bands": {
        "type": "object",
        "default": {
          "cardiac": [0.5, 2.0],
          "respiratory": [0.1, 0.3],
          "ultraslow": [0.001, 0.01]
        }
      },
      "window_seconds": {"type": "number", "default": 60},
      "overlap_fraction": {"type": "number", "default": 0.75}
    },
    "required": ["input_nifti", "roi_arterial", "roi_venous", "tr_seconds"]
  }
}
```

### Response Schema
```json
{
  "phase_difference": {
    "cardiac": {"mean": "float", "std": "float", "unit": "radians"},
    "respiratory": {"mean": "float", "std": "float"},
    "ultraslow": {"mean": "float", "std": "float"}
  },
  "coherence": {
    "cardiac": "float 0-1",
    "respiratory": "float 0-1",
    "ultraslow": "float 0-1"
  },
  "power_ratio": {
    "cardiac_to_ultraslow": "float",
    "respiratory_to_ultraslow": "float"
  },
  "flow_direction": {
    "indicator": "+1 (inward) or -1 (outward)",
    "confidence": "float 0-1",
    "interpretation": "string"
  },
  "quality_flags": {
    "sufficient_snr": "boolean",
    "cardiac_detected": "boolean",
    "respiratory_detected": "boolean"
  }
}
```

### Implementation Sketch
```python
# src/mri_analysis/phase_analysis.py

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import nibabel as nib

def extract_phase_coherence(
    input_nifti: str,
    roi_arterial: str,
    roi_venous: str,
    tr_seconds: float,
    bands: dict = None,
    window_seconds: float = 60,
    overlap_fraction: float = 0.75
) -> dict:
    """
    Extract phase coherence between arterial and venous ROIs.
    
    Flow direction criterion:
    - Awake: Δφ ≈ +π/2 (arterial leads → inward flow)
    - Sleep: Δφ ≈ -π/2 (venous leads → outward/clearance flow)
    """
    # Load data
    img = nib.load(input_nifti)
    data = img.get_fdata()  # (x, y, z, t)
    
    art_mask = nib.load(roi_arterial).get_fdata() > 0
    ven_mask = nib.load(roi_venous).get_fdata() > 0
    
    # Extract mean time series
    art_ts = data[art_mask].mean(axis=0)
    ven_ts = data[ven_mask].mean(axis=0)
    
    fs = 1.0 / tr_seconds
    
    if bands is None:
        bands = {
            "cardiac": [0.5, 2.0],
            "respiratory": [0.1, 0.3],
            "ultraslow": [0.001, 0.01]
        }
    
    results = {
        "phase_difference": {},
        "coherence": {},
        "power_ratio": {}
    }
    
    for band_name, (low, high) in bands.items():
        # Bandpass filter
        nyq = fs / 2
        if high >= nyq:
            high = nyq * 0.95
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        
        art_filt = filtfilt(b, a, art_ts)
        ven_filt = filtfilt(b, a, ven_ts)
        
        # Hilbert transform for instantaneous phase
        art_analytic = hilbert(art_filt)
        ven_analytic = hilbert(ven_filt)
        
        art_phase = np.angle(art_analytic)
        ven_phase = np.angle(ven_analytic)
        
        # Phase difference (circular)
        phase_diff = np.angle(np.exp(1j * (art_phase - ven_phase)))
        
        results["phase_difference"][band_name] = {
            "mean": float(np.mean(phase_diff)),
            "std": float(np.std(phase_diff)),
            "unit": "radians"
        }
        
        # Coherence (magnitude squared coherence)
        coherence = np.abs(np.mean(np.exp(1j * phase_diff))) ** 2
        results["coherence"][band_name] = float(coherence)
    
    # Power ratios
    art_power = {k: np.var(filtfilt(*butter(4, [v[0]/(fs/2), min(v[1], fs/2*0.95)/(fs/2)], btype='band'), art_ts)) 
                 for k, v in bands.items()}
    
    results["power_ratio"] = {
        "cardiac_to_ultraslow": art_power.get("cardiac", 1) / (art_power.get("ultraslow", 1) + 1e-10),
        "respiratory_to_ultraslow": art_power.get("respiratory", 1) / (art_power.get("ultraslow", 1) + 1e-10)
    }
    
    # Flow direction interpretation
    cardiac_phase = results["phase_difference"].get("cardiac", {}).get("mean", 0)
    
    if cardiac_phase > np.pi/4:
        flow_dir = +1
        interpretation = "Arterial leads venous → inward flow (awake pattern)"
    elif cardiac_phase < -np.pi/4:
        flow_dir = -1
        interpretation = "Venous leads arterial → outward clearance (sleep pattern)"
    else:
        flow_dir = 0
        interpretation = "Ambiguous phase relationship"
    
    results["flow_direction"] = {
        "indicator": flow_dir,
        "confidence": float(results["coherence"].get("cardiac", 0)),
        "interpretation": interpretation
    }
    
    # Quality flags
    results["quality_flags"] = {
        "sufficient_snr": results["power_ratio"]["cardiac_to_ultraslow"] >= 3,
        "cardiac_detected": results["coherence"].get("cardiac", 0) > 0.3,
        "respiratory_detected": results["coherence"].get("respiratory", 0) > 0.2
    }
    
    return results
```

---

## Tool 2: `extract_pvs_from_mri`

### Purpose
Segment perivascular spaces from MRI to estimate δ parameter.

### MCP Schema
```json
{
  "name": "extract_pvs_from_mri",
  "description": "Segment perivascular spaces using Frangi vesselness filter. Estimates gap width δ for glymphatic modeling.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string"},
      "method": {"enum": ["frangi", "threshold", "hybrid"], "default": "frangi"},
      "frangi_scales_mm": {"type": "array", "default": [0.2, 0.4, 0.6, 0.8]},
      "frangi_alpha": {"type": "number", "default": 0.5},
      "frangi_beta": {"type": "number", "default": 0.5},
      "frangi_gamma": {"type": "number", "default": 15},
      "vessel_mask": {"type": "string", "description": "Optional mask to exclude large vessels"},
      "min_size_voxels": {"type": "integer", "default": 10},
      "output_dir": {"type": "string"}
    },
    "required": ["input_nifti"]
  }
}
```

### Response Schema
```json
{
  "pvs_mask_path": "string",
  "pvs_centerlines_path": "string (VTK polydata)",
  "metrics": {
    "total_volume_mm3": "float",
    "total_length_mm": "float",
    "estimated_delta_um": {
      "median": "float",
      "iqr": [25th, 75th],
      "method": "sqrt(volume/length)"
    },
    "count_components": "int",
    "mean_radius_mm": "float"
  },
  "spatial_distribution": {
    "periventricular_fraction": "float",
    "cortical_fraction": "float",
    "basal_ganglia_fraction": "float"
  }
}
```

### Implementation Sketch
```python
# src/mri_analysis/pvs_extraction.py

import numpy as np
import nibabel as nib
from skimage.filters import frangi
from skimage.morphology import skeletonize_3d
from scipy.ndimage import label

def extract_pvs_from_mri(
    input_nifti: str,
    method: str = "frangi",
    frangi_scales_mm: list = [0.2, 0.4, 0.6, 0.8],
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 15,
    vessel_mask: str = None,
    min_size_voxels: int = 10,
    output_dir: str = None
) -> dict:
    """
    Extract perivascular spaces using Frangi vesselness.
    
    δ estimation: δ = sqrt(V_PVS / L_PVS)
    where V = total PVS volume, L = centerline length
    """
    img = nib.load(input_nifti)
    data = img.get_fdata()
    voxel_size = img.header.get_zooms()[:3]
    
    # Convert scales from mm to voxels
    scales_voxels = [s / np.mean(voxel_size) for s in frangi_scales_mm]
    
    if method == "frangi":
        # Frangi vesselness filter (detects tubular structures)
        vesselness = frangi(
            data,
            sigmas=scales_voxels,
            alpha=frangi_alpha,
            beta=frangi_beta,
            gamma=frangi_gamma,
            black_ridges=False
        )
        
        # Threshold
        threshold = np.percentile(vesselness[vesselness > 0], 90)
        pvs_mask = vesselness > threshold
    
    # Exclude large vessels if mask provided
    if vessel_mask:
        vessel_data = nib.load(vessel_mask).get_fdata() > 0
        pvs_mask = pvs_mask & ~vessel_data
    
    # Remove small components
    labeled, n_components = label(pvs_mask)
    for i in range(1, n_components + 1):
        if np.sum(labeled == i) < min_size_voxels:
            pvs_mask[labeled == i] = False
    
    # Compute metrics
    voxel_vol = np.prod(voxel_size)
    total_volume = np.sum(pvs_mask) * voxel_vol
    
    # Skeletonize for centerline length
    skeleton = skeletonize_3d(pvs_mask)
    total_length = np.sum(skeleton) * np.mean(voxel_size)
    
    # Estimate delta
    if total_length > 0:
        delta_mm = np.sqrt(total_volume / total_length)
        delta_um = delta_mm * 1000
    else:
        delta_um = 0
    
    # Save outputs
    results = {
        "metrics": {
            "total_volume_mm3": float(total_volume),
            "total_length_mm": float(total_length),
            "estimated_delta_um": {
                "median": float(delta_um),
                "iqr": [delta_um * 0.8, delta_um * 1.2],  # Approximate
                "method": "sqrt(volume/length)"
            },
            "count_components": int(n_components),
            "mean_radius_mm": float(delta_mm / 2) if delta_mm > 0 else 0
        }
    }
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mask
        mask_path = os.path.join(output_dir, "pvs_mask.nii.gz")
        nib.save(nib.Nifti1Image(pvs_mask.astype(np.uint8), img.affine), mask_path)
        results["pvs_mask_path"] = mask_path
        
        # Save centerlines as VTK (simplified)
        centerlines_path = os.path.join(output_dir, "pvs_centerlines.vtk")
        _save_skeleton_vtk(skeleton, voxel_size, img.affine, centerlines_path)
        results["pvs_centerlines_path"] = centerlines_path
    
    return results


def _save_skeleton_vtk(skeleton, voxel_size, affine, output_path):
    """Save skeleton as VTK polydata."""
    import pyvista as pv
    
    points = np.argwhere(skeleton)
    # Convert to world coordinates
    points_world = nib.affines.apply_affine(affine, points)
    
    cloud = pv.PolyData(points_world)
    cloud.save(output_path)
```

---

## Tool 3: `preprocess_4d_mri`

### MCP Schema
```json
{
  "name": "preprocess_4d_mri",
  "description": "Preprocess 4D MRI for glymphatic analysis: motion correction, B0 unwarp, bias correction",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input_nifti": {"type": "string"},
      "steps": {
        "type": "array",
        "items": {"enum": ["motion_correct", "b0_unwarp", "bias_correct", "temporal_filter", "register"]},
        "default": ["motion_correct", "bias_correct"]
      },
      "reference_volume": {"type": "integer", "default": 0},
      "b0_fieldmap": {"type": "string", "description": "Optional fieldmap for B0 correction"},
      "template": {"type": "string", "description": "Template for registration"},
      "temporal_filter_hz": {"type": "array", "description": "[low, high] for bandpass"},
      "output_dir": {"type": "string"}
    },
    "required": ["input_nifti", "output_dir"]
  }
}
```

### Response Schema
```json
{
  "preprocessed_nifti": "path",
  "motion_parameters": "path to .par file",
  "mean_framewise_displacement": "float mm",
  "excluded_volumes": "array of indices",
  "transform_to_template": "path if registered",
  "qc_report": "path to HTML"
}
```

---

## Tool 4: `compute_sulcal_depth`

### MCP Schema
```json
{
  "name": "compute_sulcal_depth",
  "description": "Compute signed distance from cortical surface (positive in sulci, negative on gyri)",
  "inputSchema": {
    "type": "object",
    "properties": {
      "surface_white": {"type": "string", "description": "White matter surface (GIFTI)"},
      "surface_pial": {"type": "string", "description": "Pial surface (GIFTI)"},
      "volume_reference": {"type": "string", "description": "Reference volume for output grid"},
      "output_nifti": {"type": "string"}
    },
    "required": ["surface_white", "surface_pial", "output_nifti"]
  }
}
```

### Response Schema
```json
{
  "sulcal_depth_nifti": "path",
  "statistics": {
    "mean_sulcal_depth_mm": "float",
    "max_sulcal_depth_mm": "float",
    "sulcal_volume_mm3": "float (where depth > 1mm)",
    "gyral_volume_mm3": "float (where depth < -1mm)"
  }
}
```
