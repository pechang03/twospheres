#!/usr/bin/env python3
"""PRIME-DE MCP HTTP Server - Primate neuroimaging data queries.

Provides access to PRIME-DE (PRIMatE Data Exchange) neuroimaging datasets.
Data stored externally in ~/data/prime_de (configurable via PRIME_DE_DATA_DIR).

Usage:
    python bin/prime_de_http_server.py --port 8009

Endpoints:
    GET  /health              - Health check
    GET  /api/datasets        - List available datasets
    POST /api/subjects        - List subjects in dataset
    POST /api/subject_info    - Get subject metadata and file list
    POST /api/get_nifti_path  - Get path to NIfTI file
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import uvicorn

# Configuration
DATA_DIR = Path(os.getenv("PRIME_DE_DATA_DIR", Path.home() / "data" / "prime_de"))


def get_available_datasets() -> Dict[str, Dict]:
    """Scan data directory for available datasets."""
    datasets = {}
    
    if not DATA_DIR.exists():
        return datasets
    
    for item in DATA_DIR.iterdir():
        if item.is_dir():
            desc_file = item / "dataset_description.json"
            if desc_file.exists():
                try:
                    with open(desc_file) as f:
                        desc = json.load(f)
                except:
                    desc = {}
                
                # Count subjects
                subjects = list(item.glob("sub-*"))
                
                datasets[item.name] = {
                    "name": desc.get("Name", item.name),
                    "path": str(item),
                    "bids_version": desc.get("BIDSVersion", "unknown"),
                    "license": desc.get("License", "unknown"),
                    "subject_count": len(subjects),
                    "has_derivatives": (item / "derivatives").exists(),
                }
            elif list(item.glob("*.tar.gz")):
                # Tarball dataset (not extracted)
                tarballs = list(item.glob("*.tar.gz"))
                datasets[item.name] = {
                    "name": item.name,
                    "path": str(item),
                    "format": "tarball",
                    "parts": len(tarballs),
                    "extracted": False,
                }
    
    return datasets


def get_subjects(dataset_path: Path) -> List[Dict]:
    """Get list of subjects in a BIDS dataset."""
    subjects = []
    
    for subj_dir in sorted(dataset_path.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        
        subj_id = subj_dir.name
        
        # Check available modalities
        modalities = {}
        
        anat_dir = subj_dir / "anat"
        if anat_dir.exists():
            modalities["anat"] = {
                "T1w": len(list(anat_dir.glob("*T1w.nii*"))),
                "T2w": len(list(anat_dir.glob("*T2w.nii*"))),
            }
        
        func_dir = subj_dir / "func"
        if func_dir.exists():
            modalities["func"] = {
                "bold": len(list(func_dir.glob("*bold.nii*"))),
            }
        
        dwi_dir = subj_dir / "dwi"
        if dwi_dir.exists():
            modalities["dwi"] = {
                "dwi": len(list(dwi_dir.glob("*dwi.nii*"))),
            }
        
        subjects.append({
            "id": subj_id,
            "path": str(subj_dir),
            "modalities": modalities,
        })
    
    return subjects


def get_subject_files(subject_path: Path) -> Dict[str, List[str]]:
    """Get all NIfTI files for a subject organized by modality."""
    files = {"anat": [], "func": [], "dwi": [], "other": []}
    
    for nii_file in subject_path.rglob("*.nii*"):
        rel_path = str(nii_file.relative_to(subject_path))
        
        if "/anat/" in str(nii_file) or rel_path.startswith("anat/"):
            files["anat"].append(rel_path)
        elif "/func/" in str(nii_file) or rel_path.startswith("func/"):
            files["func"].append(rel_path)
        elif "/dwi/" in str(nii_file) or rel_path.startswith("dwi/"):
            files["dwi"].append(rel_path)
        else:
            files["other"].append(rel_path)
    
    return files


# =============================================================================
# API Routes
# =============================================================================

async def health_check(request):
    """Health check endpoint."""
    datasets = get_available_datasets()
    
    return JSONResponse({
        "status": "healthy",
        "service": "prime-de-mcp",
        "version": "0.1.0",
        "data_dir": str(DATA_DIR),
        "datasets_available": len(datasets),
        "datasets": list(datasets.keys()),
    })


async def list_datasets(request):
    """List all available datasets."""
    datasets = get_available_datasets()
    
    return JSONResponse({
        "data_dir": str(DATA_DIR),
        "datasets": datasets,
    })


async def list_subjects(request):
    """List subjects in a dataset.
    
    POST body:
    {
        "dataset": "BORDEAUX24"
    }
    """
    try:
        body = await request.json()
        dataset_name = body.get("dataset")
        
        if not dataset_name:
            return JSONResponse({"error": "dataset required"}, status_code=400)
        
        dataset_path = DATA_DIR / dataset_name
        if not dataset_path.exists():
            return JSONResponse(
                {"error": f"Dataset not found: {dataset_name}"}, 
                status_code=404
            )
        
        subjects = get_subjects(dataset_path)
        
        return JSONResponse({
            "dataset": dataset_name,
            "subject_count": len(subjects),
            "subjects": subjects,
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def subject_info(request):
    """Get detailed subject info including file list.
    
    POST body:
    {
        "dataset": "BORDEAUX24",
        "subject": "sub-m01"
    }
    """
    try:
        body = await request.json()
        dataset_name = body.get("dataset")
        subject_id = body.get("subject")
        
        if not dataset_name or not subject_id:
            return JSONResponse(
                {"error": "dataset and subject required"}, 
                status_code=400
            )
        
        # Normalize subject ID
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"
        
        subject_path = DATA_DIR / dataset_name / subject_id
        if not subject_path.exists():
            return JSONResponse(
                {"error": f"Subject not found: {subject_id}"}, 
                status_code=404
            )
        
        files = get_subject_files(subject_path)
        
        # Load sidecar JSON if available
        metadata = {}
        for json_file in subject_path.rglob("*.json"):
            key = json_file.stem
            try:
                with open(json_file) as f:
                    metadata[key] = json.load(f)
            except:
                pass
        
        return JSONResponse({
            "dataset": dataset_name,
            "subject": subject_id,
            "path": str(subject_path),
            "files": files,
            "metadata": metadata,
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_nifti_path(request):
    """Get absolute path to a NIfTI file.
    
    POST body:
    {
        "dataset": "BORDEAUX24",
        "subject": "sub-m01",
        "modality": "anat",
        "suffix": "T1w"
    }
    """
    try:
        body = await request.json()
        dataset_name = body.get("dataset")
        subject_id = body.get("subject")
        modality = body.get("modality", "anat")
        suffix = body.get("suffix", "T1w")
        
        if not dataset_name or not subject_id:
            return JSONResponse(
                {"error": "dataset and subject required"}, 
                status_code=400
            )
        
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"
        
        subject_path = DATA_DIR / dataset_name / subject_id
        
        # Find matching file
        pattern = f"*{suffix}.nii*"
        search_path = subject_path / modality if modality else subject_path
        
        matches = list(search_path.glob(pattern))
        if not matches:
            # Try recursive search
            matches = list(subject_path.rglob(pattern))
        
        if not matches:
            return JSONResponse(
                {"error": f"No {suffix} file found for {subject_id}"}, 
                status_code=404
            )
        
        # Return first match
        nifti_path = matches[0]
        
        return JSONResponse({
            "dataset": dataset_name,
            "subject": subject_id,
            "modality": modality,
            "suffix": suffix,
            "path": str(nifti_path),
            "filename": nifti_path.name,
            "exists": nifti_path.exists(),
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def get_derivatives(request):
    """List available derivatives for a dataset.
    
    POST body:
    {
        "dataset": "BORDEAUX24"
    }
    """
    try:
        body = await request.json()
        dataset_name = body.get("dataset")
        
        if not dataset_name:
            return JSONResponse({"error": "dataset required"}, status_code=400)
        
        deriv_path = DATA_DIR / dataset_name / "derivatives"
        if not deriv_path.exists():
            return JSONResponse({
                "dataset": dataset_name,
                "has_derivatives": False,
                "pipelines": [],
            })
        
        pipelines = []
        for item in deriv_path.iterdir():
            if item.is_dir():
                desc_file = item / "dataset_description.json"
                desc = {}
                if desc_file.exists():
                    try:
                        with open(desc_file) as f:
                            desc = json.load(f)
                    except:
                        pass
                
                pipelines.append({
                    "name": item.name,
                    "path": str(item),
                    "description": desc.get("Name", item.name),
                })
        
        return JSONResponse({
            "dataset": dataset_name,
            "has_derivatives": True,
            "pipelines": pipelines,
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# Application Setup
# =============================================================================

routes = [
    Route("/health", health_check, methods=["GET"]),
    Route("/api/datasets", list_datasets, methods=["GET"]),
    Route("/api/subjects", list_subjects, methods=["POST"]),
    Route("/api/subject_info", subject_info, methods=["POST"]),
    Route("/api/get_nifti_path", get_nifti_path, methods=["POST"]),
    Route("/api/derivatives", get_derivatives, methods=["POST"]),
]

app = Starlette(routes=routes)


def main():
    parser = argparse.ArgumentParser(description="PRIME-DE MCP HTTP Server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("PRIME_DE_PORT", 8009)),
        help="Port to run server on (default: 8009)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    datasets = get_available_datasets()
    
    print(f"🐒 PRIME-DE MCP HTTP Server starting on {args.host}:{args.port}")
    print(f"\nData directory: {DATA_DIR}")
    print(f"\nAvailable datasets ({len(datasets)}):")
    for name, info in datasets.items():
        subj = info.get('subject_count', info.get('parts', '?'))
        print(f"  📁 {name}: {subj} subjects/parts")
    
    print(f"\nEndpoints:")
    print(f"  Health:      GET  http://{args.host}:{args.port}/health")
    print(f"  Datasets:    GET  http://{args.host}:{args.port}/api/datasets")
    print(f"  Subjects:    POST http://{args.host}:{args.port}/api/subjects")
    print(f"  Subject:     POST http://{args.host}:{args.port}/api/subject_info")
    print(f"  NIfTI path:  POST http://{args.host}:{args.port}/api/get_nifti_path")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
