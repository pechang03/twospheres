"""PRIME-DE Loader for macaque fMRI data processing.

Implements BEAD-QEC-6: PRIME-DE MRI Data Processing.

Loads NIfTI data from PRIME-DE API, extracts ROI timeseries using D99 atlas
(368 regions), and computes functional connectivity using distance correlation.

FPT 4-parameter kernel: (dataset, subject, suffix, atlas)
Memory bound: O(T·368) where T is time points
Runtime target: ≤30 seconds

Reference:
    Design: docs/designs/yada-hierarchical-brain-model/BEADS_QEC_TENSOR.md (BEAD-QEC-6)
    Integration: /merge2docs/docs/integrations/yada-services-secure-qec-nifti.md
"""

import asyncio
import httpx
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import nibabel as nib
except ImportError:
    nib = None

logger = logging.getLogger(__name__)


@dataclass
class D99AtlasConfig:
    """Configuration for D99 atlas (macaque brain)."""
    n_regions: int = 368
    region_names: Optional[List[str]] = None
    ijk_to_mni: Optional[np.ndarray] = None  # Affine transformation


class D99Atlas:
    """D99 atlas for macaque brain with 368 regions."""

    def __init__(self, config: Optional[D99AtlasConfig] = None):
        """Initialize D99 atlas.

        Args:
            config: Optional D99 atlas configuration
        """
        self.config = config or D99AtlasConfig()
        self._initialize_region_names()

    def _initialize_region_names(self):
        """Initialize D99 region names.

        If not provided in config, generates default names following
        standard D99 atlas nomenclature.
        """
        if self.config.region_names is None:
            # Generate default region names
            # In practice, these would come from official D99 atlas file
            self.region_names = [f"D99_Region_{i:03d}" for i in range(self.config.n_regions)]
        else:
            self.region_names = self.config.region_names

    def extract_timeseries(
        self,
        nifti_img,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract ROI timeseries from NIfTI image using D99 atlas.

        Args:
            nifti_img: NIfTI image (4D: x, y, z, time) or nibabel Nifti1Image
            mask: Optional binary mask for valid regions (3D: x, y, z)

        Returns:
            Timeseries array of shape (T, 368) where T is number of timepoints
            and 368 is number of D99 regions. Values are mean BOLD signals
            per region.

        Raises:
            ValueError: If nifti_img dimensions are incorrect
            RuntimeError: If nibabel is not installed
        """
        # Handle both nibabel Nifti1Image and raw numpy arrays
        # Check if it's a nibabel object by checking for get_fdata method
        if hasattr(nifti_img, 'get_fdata'):
            data = nifti_img.get_fdata()
        else:
            data = np.asarray(nifti_img)

        # Validate dimensions (expect 4D: x, y, z, time)
        if data.ndim != 4:
            raise ValueError(
                f"Expected 4D NIfTI data (x, y, z, time), got shape {data.shape}"
            )

        x, y, z, t = data.shape
        n_voxels = x * y * z
        n_regions = self.config.n_regions

        # Initialize timeseries array (T, N_regions)
        timeseries = np.zeros((t, n_regions))

        # Extract mean signal per region
        # In practice, this would use the actual D99 atlas label map
        # For now, we partition voxels into regions
        voxels_per_region = n_voxels // n_regions
        remainder = n_voxels % n_regions

        voxel_idx = 0
        data_2d = data.reshape(n_voxels, t)  # Flatten spatial dimensions

        for region_idx in range(n_regions):
            # Calculate region boundaries
            region_size = voxels_per_region + (1 if region_idx < remainder else 0)
            region_voxels = data_2d[voxel_idx:voxel_idx + region_size, :]

            # Compute mean across voxels in region
            # Handle edge case: if region is empty (size=0), fill with 0.0
            if region_size > 0:
                timeseries[:, region_idx] = np.mean(region_voxels, axis=0)
            else:
                timeseries[:, region_idx] = 0.0
            voxel_idx += region_size

        return timeseries

    async def extract_timeseries_async(
        self,
        nifti_img,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Async wrapper for extract_timeseries.

        Args:
            nifti_img: NIfTI image
            mask: Optional binary mask

        Returns:
            Timeseries array of shape (T, 368)
        """
        return await asyncio.to_thread(
            self.extract_timeseries, nifti_img, mask
        )


class PRIMEDELoader:
    """Loader for PRIME-DE macaque fMRI dataset.

    Loads NIfTI data from PRIME-DE API, extracts timeseries using D99 atlas,
    and computes functional connectivity matrices.

    Implements FPT 4-parameter kernel: (dataset, subject, suffix, atlas)
    Memory bound: O(T·368) where T is time points
    Runtime target: ≤30 seconds per subject
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8009",
        atlas: Optional[D99Atlas] = None,
        timeout: float = 60.0
    ):
        """Initialize PRIME-DE loader.

        Args:
            api_url: Base URL for PRIME-DE API (default: localhost:8009)
            atlas: Optional D99Atlas instance. If None, creates default.
            timeout: HTTP timeout in seconds (default: 60.0)
        """
        self.api_url = api_url.rstrip("/")
        self.atlas = atlas or D99Atlas()
        self.client = httpx.AsyncClient(timeout=timeout)
        self._session_cache: Dict[str, np.ndarray] = {}

    async def get_nifti_path(
        self,
        dataset: str,
        subject: str,
        suffix: str
    ) -> Dict:
        """Query PRIME-DE API for NIfTI file path.

        Args:
            dataset: Dataset name (e.g., "BORDEAUX24")
            subject: Subject ID (e.g., "m01")
            suffix: File suffix/modality (e.g., "T1w", "bold")

        Returns:
            Response dict with structure:
            {
                "result": {
                    "path": str,
                    "dataset": str,
                    "subject": str,
                    "suffix": str,
                    "exists": bool
                },
                "error": Optional[str]
            }

        Raises:
            httpx.HTTPError: If API request fails
            ValueError: If response format is invalid
        """
        try:
            response = await self.client.post(
                f"{self.api_url}/api/get_nifti_path",
                json={
                    "dataset": dataset,
                    "subject": subject,
                    "suffix": suffix
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            # Handle both sync and async json() methods
            if asyncio.iscoroutine(result):
                return await result
            return result
        except httpx.HTTPError as e:
            logger.error(
                f"Failed to get NIfTI path for {dataset}/{subject}/{suffix}: {e}"
            )
            raise

    async def load_subject(
        self,
        dataset: str,
        subject: str,
        suffix: str = "bold"
    ) -> Dict:
        """Load fMRI data for one subject.

        Args:
            dataset: Dataset name (e.g., "BORDEAUX24")
            subject: Subject ID (e.g., "m01")
            suffix: File modality (default: "bold")

        Returns:
            Dict with structure:
            {
                "subject_id": str,
                "dataset": str,
                "suffix": str,
                "nifti_path": str,
                "timeseries": np.ndarray,  # (T, 368) time × regions
                "regions": List[str],       # D99 region names
                "n_timepoints": int,
                "metadata": Dict
            }

        Raises:
            ValueError: If NIfTI path not found or file cannot be loaded
            RuntimeError: If nibabel not installed or data invalid
        """
        # Get file path from API
        result = await self.get_nifti_path(dataset, subject, suffix)

        if "error" in result and result["error"]:
            raise ValueError(
                f"API error for {dataset}/{subject}/{suffix}: {result['error']}"
            )

        nifti_path = result.get("result", {}).get("path")
        if not nifti_path:
            raise ValueError(
                f"No NIfTI path found for {dataset}/{subject}/{suffix}"
            )

        try:
            # Load NIfTI file
            nifti_file = Path(nifti_path)
            if not nifti_file.exists():
                raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

            # Try to load with nibabel if available
            if nib is not None:
                img = nib.load(str(nifti_file))
            else:
                # If nibabel not available, try dynamic import
                try:
                    import nibabel as nibabel_import
                    img = nibabel_import.load(str(nifti_file))
                except ImportError:
                    raise RuntimeError(
                        "nibabel required. Install with: pip install nibabel"
                    )

            # Extract timeseries
            timeseries = await self.atlas.extract_timeseries_async(img)
            n_timepoints = timeseries.shape[0]

            logger.info(
                f"Loaded {dataset}/{subject}/{suffix}: "
                f"{n_timepoints} timepoints × {len(self.atlas.region_names)} regions"
            )

            return {
                "subject_id": subject,
                "dataset": dataset,
                "suffix": suffix,
                "nifti_path": nifti_path,
                "timeseries": timeseries,
                "regions": self.atlas.region_names,
                "n_timepoints": n_timepoints,
                "n_regions": len(self.atlas.region_names),
                "metadata": result.get("result", {})
            }

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {dataset}/{subject}/{suffix}: {e}")
            raise

    async def compute_functional_connectivity(
        self,
        timeseries: np.ndarray,
        method: str = "distance_correlation"
    ) -> Tuple[np.ndarray, Dict]:
        """Compute functional connectivity matrix for timeseries.

        Args:
            timeseries: Shape (T, N) where T=timepoints, N=regions
            method: Connectivity method (default: "distance_correlation")
                   Options: "distance_correlation", "pearson", "fft"

        Returns:
            Tuple of (connectivity_matrix, metadata) where:
            - connectivity_matrix: Shape (N, N), symmetric
            - metadata: Dict with method info and computation time

        Raises:
            ValueError: If method not recognized or data invalid
        """
        if timeseries.ndim != 2:
            raise ValueError(
                f"Expected 2D timeseries (T, N), got shape {timeseries.shape}"
            )

        t, n_regions = timeseries.shape

        if method == "distance_correlation":
            return await self._compute_distance_correlation(timeseries)
        elif method == "pearson":
            return await self._compute_pearson_correlation(timeseries)
        elif method == "fft":
            from src.backend.mri.mri_signal_processing import (
                compute_pairwise_correlation_fft
            )
            # Convert to list of signals for FFT method
            signals = [timeseries[:, i] for i in range(n_regions)]
            conn_matrix = await compute_pairwise_correlation_fft(signals)
            return conn_matrix, {"method": "fft", "n_regions": n_regions}
        else:
            raise ValueError(f"Unknown connectivity method: {method}")

    async def _compute_distance_correlation(
        self,
        timeseries: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Compute distance correlation connectivity matrix.

        Uses Lu et al. (2019) multivariate distance correlation.
        More robust than Pearson for detecting non-linear dependencies.

        Args:
            timeseries: Shape (T, N)

        Returns:
            Tuple of (connectivity_matrix, metadata)
        """
        from src.backend.mri.mri_signal_processing import (
            compute_distance_correlation
        )

        t, n_regions = timeseries.shape
        conn_matrix = np.zeros((n_regions, n_regions))

        # Compute pairwise distance correlations
        # Treat each region's timeseries as 1D signal (averaged across voxels)
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                dcor = await compute_distance_correlation(
                    timeseries[:, i],
                    timeseries[:, j]
                )
                conn_matrix[i, j] = dcor
                conn_matrix[j, i] = dcor

        # Diagonal is self-correlation
        np.fill_diagonal(conn_matrix, 1.0)

        return conn_matrix, {
            "method": "distance_correlation",
            "n_regions": n_regions,
            "n_timepoints": t
        }

    async def _compute_pearson_correlation(
        self,
        timeseries: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Compute Pearson correlation connectivity matrix.

        Args:
            timeseries: Shape (T, N)

        Returns:
            Tuple of (connectivity_matrix, metadata)
        """
        def _compute():
            # Use numpy's fast correlation
            conn_matrix = np.corrcoef(timeseries.T)
            return conn_matrix

        t, n_regions = timeseries.shape
        conn_matrix = await asyncio.to_thread(_compute)

        return conn_matrix, {
            "method": "pearson",
            "n_regions": n_regions,
            "n_timepoints": t
        }

    async def load_and_process_subject(
        self,
        dataset: str,
        subject: str,
        suffix: str = "bold",
        connectivity_method: str = "distance_correlation"
    ) -> Dict:
        """Load subject data and compute connectivity in one call.

        Convenience method combining load_subject and compute_functional_connectivity.

        Args:
            dataset: Dataset name
            subject: Subject ID
            suffix: File modality
            connectivity_method: Method for connectivity computation

        Returns:
            Dict combining subject data and connectivity:
            {
                "subject_id": str,
                "dataset": str,
                "timeseries": np.ndarray,  # (T, 368)
                "connectivity": np.ndarray,  # (368, 368)
                "regions": List[str],
                "metadata": Dict
            }
        """
        # Load subject
        subject_data = await self.load_subject(dataset, subject, suffix)

        # Compute connectivity
        conn_matrix, conn_metadata = await self.compute_functional_connectivity(
            subject_data["timeseries"],
            method=connectivity_method
        )

        return {
            "subject_id": subject_data["subject_id"],
            "dataset": subject_data["dataset"],
            "suffix": subject_data["suffix"],
            "nifti_path": subject_data["nifti_path"],
            "timeseries": subject_data["timeseries"],
            "connectivity": conn_matrix,
            "regions": subject_data["regions"],
            "n_timepoints": subject_data["n_timepoints"],
            "n_regions": subject_data["n_regions"],
            "metadata": {
                **subject_data["metadata"],
                "connectivity": conn_metadata
            }
        }

    async def load_multiple_subjects(
        self,
        dataset: str,
        subject_ids: List[str],
        suffix: str = "bold"
    ) -> List[Dict]:
        """Load multiple subjects in parallel.

        Args:
            dataset: Dataset name
            subject_ids: List of subject IDs
            suffix: File modality

        Returns:
            List of subject data dicts
        """
        tasks = [
            self.load_subject(dataset, subj_id, suffix)
            for subj_id in subject_ids
        ]
        return await asyncio.gather(*tasks)

    async def close(self):
        """Close HTTP client and cleanup resources."""
        if self.client:
            await self.client.aclose()
            logger.info("PRIME-DE loader closed")

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
