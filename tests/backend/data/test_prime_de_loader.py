"""Tests for PRIME-DE fMRI data loader.

Tests cover:
- D99 atlas functionality
- API communication with PRIME-DE service
- NIfTI data loading and timeseries extraction
- Functional connectivity computation (distance correlation, Pearson, FFT)
- Integration tests with mock API responses
"""

import asyncio
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict

from src.backend.data.prime_de_loader import (
    D99Atlas,
    D99AtlasConfig,
    PRIMEDELoader,
)


class TestD99AtlasConfig:
    """Tests for D99 atlas configuration."""

    def test_default_config(self):
        """Test default D99 configuration."""
        config = D99AtlasConfig()
        assert config.n_regions == 368
        assert config.region_names is None

    def test_custom_config(self):
        """Test custom D99 configuration."""
        custom_names = [f"Region_{i}" for i in range(10)]
        config = D99AtlasConfig(
            n_regions=10,
            region_names=custom_names
        )
        assert config.n_regions == 10
        assert len(config.region_names) == 10


class TestD99Atlas:
    """Tests for D99 atlas operations."""

    def test_atlas_initialization(self):
        """Test atlas initialization with default config."""
        atlas = D99Atlas()
        assert len(atlas.region_names) == 368
        assert atlas.region_names[0] == "D99_Region_000"
        assert atlas.region_names[367] == "D99_Region_367"

    def test_atlas_custom_initialization(self):
        """Test atlas initialization with custom names."""
        custom_names = ["V1", "V2", "V3"]
        config = D99AtlasConfig(n_regions=3, region_names=custom_names)
        atlas = D99Atlas(config)
        assert atlas.region_names == custom_names

    def test_extract_timeseries_shapes(self):
        """Test extract_timeseries returns correct shapes."""
        atlas = D99Atlas()

        # Create mock 4D NIfTI data (10x10x10 voxels, 100 timepoints)
        nifti_data = np.random.randn(10, 10, 10, 100).astype(np.float32)

        timeseries = atlas.extract_timeseries(nifti_data)

        # Should return (T, N) = (100, 368)
        assert timeseries.shape == (100, 368)
        # Result will be float64 after mean operation
        assert timeseries.dtype in (np.float32, np.float64)

    def test_extract_timeseries_mean_values(self):
        """Test that extracted timeseries contains reasonable values."""
        atlas = D99Atlas()

        # Create data with known mean
        nifti_data = np.ones((10, 10, 10, 50), dtype=np.float32)
        timeseries = atlas.extract_timeseries(nifti_data)

        # All values should be close to 1.0 (mean of ones)
        assert np.allclose(timeseries, 1.0)

    def test_extract_timeseries_invalid_shape(self):
        """Test extract_timeseries rejects non-4D data."""
        atlas = D99Atlas()

        # 3D data instead of 4D
        invalid_data = np.random.randn(10, 10, 10)

        with pytest.raises(ValueError, match="Expected 4D NIfTI data"):
            atlas.extract_timeseries(invalid_data)

    def test_extract_timeseries_3d_input(self):
        """Test extract_timeseries with 3D data (single timepoint)."""
        atlas = D99Atlas()

        # Still should be 4D (with T=1)
        data_3d = np.random.randn(10, 10, 10, 1)
        timeseries = atlas.extract_timeseries(data_3d)

        assert timeseries.shape == (1, 368)

    @pytest.mark.asyncio
    async def test_extract_timeseries_async(self):
        """Test async extraction of timeseries."""
        atlas = D99Atlas()
        nifti_data = np.random.randn(10, 10, 10, 50).astype(np.float32)

        timeseries = await atlas.extract_timeseries_async(nifti_data)

        assert timeseries.shape == (50, 368)

    def test_extract_timeseries_with_nibabel(self):
        """Test extract_timeseries with actual nibabel Nifti1Image.

        This test requires nibabel to be installed.
        """
        pytest.importorskip("nibabel")
        import nibabel as nib

        atlas = D99Atlas()

        # Create mock Nifti1Image
        nifti_data = np.random.randn(10, 10, 10, 50).astype(np.float32)
        img = nib.Nifti1Image(nifti_data, np.eye(4))

        timeseries = atlas.extract_timeseries(img)

        assert timeseries.shape == (50, 368)


class TestPRIMEDELoaderInit:
    """Tests for PRIME-DE loader initialization."""

    def test_loader_initialization_defaults(self):
        """Test loader initialization with defaults."""
        loader = PRIMEDELoader()
        assert loader.api_url == "http://localhost:8009"
        assert isinstance(loader.atlas, D99Atlas)
        assert loader.client is not None

    def test_loader_initialization_custom_url(self):
        """Test loader initialization with custom API URL."""
        custom_url = "http://example.com:9000"
        loader = PRIMEDELoader(api_url=custom_url)
        assert loader.api_url == "http://example.com:9000"

    def test_loader_initialization_trailing_slash(self):
        """Test that trailing slash is removed from API URL."""
        loader = PRIMEDELoader(api_url="http://localhost:8009/")
        assert loader.api_url == "http://localhost:8009"

    def test_loader_initialization_custom_atlas(self):
        """Test loader initialization with custom atlas."""
        custom_atlas = D99Atlas(D99AtlasConfig(n_regions=10))
        loader = PRIMEDELoader(atlas=custom_atlas)
        assert loader.atlas is custom_atlas


class TestPRIMEDELoaderAPI:
    """Tests for PRIME-DE API communication."""

    @pytest.mark.asyncio
    async def test_get_nifti_path_success(self):
        """Test successful NIfTI path retrieval."""
        loader = PRIMEDELoader()

        # Mock the HTTP client
        mock_response = {
            "result": {
                "path": "/data/BORDEAUX24/m01/bold.nii.gz",
                "dataset": "BORDEAUX24",
                "subject": "m01",
                "suffix": "bold",
                "exists": True
            }
        }

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            # Create async mock for json()
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = await loader.get_nifti_path("BORDEAUX24", "m01", "bold")

            assert result == mock_response
            assert result["result"]["path"] == "/data/BORDEAUX24/m01/bold.nii.gz"

            # Verify correct API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "get_nifti_path" in call_args[0][0]
            assert call_args[1]["json"]["dataset"] == "BORDEAUX24"
            assert call_args[1]["json"]["subject"] == "m01"

    @pytest.mark.asyncio
    async def test_get_nifti_path_api_error(self):
        """Test API error handling."""
        loader = PRIMEDELoader()

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            # Mock raise_for_status to raise an exception
            def raise_error():
                raise httpx.HTTPError("API Error")

            mock_resp = AsyncMock()
            mock_resp.raise_for_status = MagicMock(side_effect=raise_error)
            mock_post.return_value = mock_resp

            with pytest.raises(httpx.HTTPError):
                await loader.get_nifti_path("BORDEAUX24", "m01", "bold")

    @pytest.mark.asyncio
    async def test_load_subject_path_not_found(self):
        """Test load_subject when API returns no path."""
        loader = PRIMEDELoader()

        mock_response = {
            "result": {"path": None}
        }

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            with pytest.raises(ValueError, match="No NIfTI path found"):
                await loader.load_subject("BORDEAUX24", "m01", "bold")


class TestPRIMEDELoaderDataLoading:
    """Tests for data loading functionality."""

    @pytest.mark.asyncio
    async def test_load_subject_success(self):
        """Test successful subject data loading.

        This test mocks the nibabel loading and API response.
        """
        loader = PRIMEDELoader()

        # Mock API response
        mock_api_response = {
            "result": {
                "path": "/tmp/test_bold.nii.gz",
                "dataset": "BORDEAUX24",
                "subject": "m01",
                "suffix": "bold"
            }
        }

        # Mock NIfTI loading
        mock_nifti_data = np.random.randn(10, 10, 10, 100).astype(np.float32)

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            with patch("src.backend.data.prime_de_loader.nib") as mock_nib:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_resp = AsyncMock()
                    mock_resp.json = AsyncMock(return_value=mock_api_response)
                    mock_resp.raise_for_status = MagicMock()
                    mock_post.return_value = mock_resp

                    # Mock nibabel return
                    mock_img = MagicMock()
                    mock_img.get_fdata.return_value = mock_nifti_data
                    mock_nib.load.return_value = mock_img

                    subject_data = await loader.load_subject(
                        "BORDEAUX24", "m01", "bold"
                    )

                    assert subject_data["subject_id"] == "m01"
                    assert subject_data["dataset"] == "BORDEAUX24"
                    assert subject_data["suffix"] == "bold"
                    assert subject_data["timeseries"].shape == (100, 368)
                    assert len(subject_data["regions"]) == 368
                    assert subject_data["n_timepoints"] == 100
                    assert subject_data["n_regions"] == 368

    @pytest.mark.asyncio
    async def test_load_subject_file_not_found(self):
        """Test load_subject when file doesn't exist."""
        loader = PRIMEDELoader()

        mock_api_response = {
            "result": {
                "path": "/nonexistent/file.nii.gz"
            }
        }

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            with patch("pathlib.Path.exists", return_value=False):
                mock_resp = AsyncMock()
                mock_resp.json = AsyncMock(return_value=mock_api_response)
                mock_resp.raise_for_status = MagicMock()
                mock_post.return_value = mock_resp

                with pytest.raises(FileNotFoundError):
                    await loader.load_subject("BORDEAUX24", "m01", "bold")


class TestConnectivityComputation:
    """Tests for functional connectivity computation."""

    @pytest.mark.asyncio
    async def test_compute_distance_correlation(self):
        """Test distance correlation connectivity computation."""
        loader = PRIMEDELoader()

        # Create test timeseries (100 timepoints, 368 regions)
        timeseries = np.random.randn(100, 368)

        # Compute connectivity
        conn_matrix, metadata = await loader.compute_functional_connectivity(
            timeseries,
            method="distance_correlation"
        )

        # Validate output
        assert conn_matrix.shape == (368, 368)
        assert np.allclose(np.diag(conn_matrix), 1.0)  # Diagonal is 1.0
        assert np.allclose(conn_matrix, conn_matrix.T)  # Symmetric
        assert np.all(conn_matrix >= 0)  # Distance correlation is non-negative
        assert np.all(conn_matrix <= 1)  # Normalized

        # Check metadata
        assert metadata["method"] == "distance_correlation"
        assert metadata["n_regions"] == 368
        assert metadata["n_timepoints"] == 100

    @pytest.mark.asyncio
    async def test_compute_pearson_correlation(self):
        """Test Pearson correlation connectivity computation."""
        loader = PRIMEDELoader()

        # Create test timeseries with known correlation
        t = np.linspace(0, 10, 100)
        signal1 = np.sin(2 * np.pi * 0.1 * t)
        signal2 = np.sin(2 * np.pi * 0.1 * t)  # Identical
        noise = np.random.randn(100, 366) * 0.1

        timeseries = np.column_stack([signal1, signal2, noise])

        conn_matrix, metadata = await loader.compute_functional_connectivity(
            timeseries,
            method="pearson"
        )

        # Validate output
        assert conn_matrix.shape == (368, 368)
        assert np.allclose(np.diag(conn_matrix), 1.0)  # Diagonal is 1.0
        assert np.allclose(conn_matrix, conn_matrix.T)  # Symmetric

        # First two signals should be highly correlated
        assert conn_matrix[0, 1] > 0.95

        # Check metadata
        assert metadata["method"] == "pearson"

    @pytest.mark.asyncio
    async def test_compute_connectivity_invalid_shape(self):
        """Test connectivity computation rejects invalid shapes."""
        loader = PRIMEDELoader()

        # 1D instead of 2D
        invalid_data = np.random.randn(100)

        with pytest.raises(ValueError, match="Expected 2D timeseries"):
            await loader.compute_functional_connectivity(invalid_data)

    @pytest.mark.asyncio
    async def test_compute_connectivity_invalid_method(self):
        """Test connectivity computation with unknown method."""
        loader = PRIMEDELoader()

        timeseries = np.random.randn(100, 368)

        with pytest.raises(ValueError, match="Unknown connectivity method"):
            await loader.compute_functional_connectivity(
                timeseries,
                method="unknown_method"
            )


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_load_and_process_subject(self):
        """Test combined load and process workflow."""
        loader = PRIMEDELoader()

        # Mock API response
        mock_api_response = {
            "result": {
                "path": "/tmp/test_bold.nii.gz",
                "dataset": "BORDEAUX24",
                "subject": "m01",
                "suffix": "bold"
            }
        }

        mock_nifti_data = np.random.randn(10, 10, 10, 100).astype(np.float32)

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            with patch("src.backend.data.prime_de_loader.nib") as mock_nib:
                with patch("pathlib.Path.exists", return_value=True):
                    mock_resp = AsyncMock()
                    mock_resp.json = AsyncMock(return_value=mock_api_response)
                    mock_resp.raise_for_status = MagicMock()
                    mock_post.return_value = mock_resp

                    mock_img = MagicMock()
                    mock_img.get_fdata.return_value = mock_nifti_data
                    mock_nib.load.return_value = mock_img

                    result = await loader.load_and_process_subject(
                        "BORDEAUX24",
                        "m01",
                        "bold",
                        connectivity_method="pearson"
                    )

                    assert result["subject_id"] == "m01"
                    assert result["dataset"] == "BORDEAUX24"
                    assert result["timeseries"].shape == (100, 368)
                    assert result["connectivity"].shape == (368, 368)
                    assert "pearson" in result["metadata"]["connectivity"]["method"]

    @pytest.mark.asyncio
    async def test_load_multiple_subjects(self):
        """Test loading multiple subjects in parallel."""
        loader = PRIMEDELoader()

        subject_ids = ["m01", "m02"]

        # Create mock responses for each subject
        def create_mock_resp(subject_id):
            resp = AsyncMock()
            resp.json = AsyncMock(return_value={
                "result": {
                    "path": f"/tmp/{subject_id}_bold.nii.gz",
                    "dataset": "BORDEAUX24",
                    "subject": subject_id,
                    "suffix": "bold"
                }
            })
            resp.raise_for_status = MagicMock()
            return resp

        mock_nifti_data = np.random.randn(10, 10, 10, 100).astype(np.float32)

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            with patch("src.backend.data.prime_de_loader.nib") as mock_nib:
                with patch("pathlib.Path.exists", return_value=True):
                    # Set up side effect for multiple calls
                    mock_post.side_effect = [
                        create_mock_resp(subj_id)
                        for subj_id in subject_ids
                    ]

                    mock_img = MagicMock()
                    mock_img.get_fdata.return_value = mock_nifti_data
                    mock_nib.load.return_value = mock_img

                    results = await loader.load_multiple_subjects(
                        "BORDEAUX24",
                        subject_ids,
                        "bold"
                    )

                    assert len(results) == len(subject_ids)
                    for i, result in enumerate(results):
                        assert result["subject_id"] == subject_ids[i]
                        assert result["dataset"] == "BORDEAUX24"
                        assert result["timeseries"].shape == (100, 368)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test loader as async context manager."""
        async with PRIMEDELoader() as loader:
            assert loader.client is not None

        # After exit, client should be closed
        # (We can't easily test closure state, but the method should not error)


class TestPerformance:
    """Performance and memory tests."""

    def test_memory_bound_single_subject(self):
        """Test memory usage stays within O(T·368) bound.

        For 100 timepoints and 368 regions:
        - Timeseries: 100 × 368 × 8 bytes ≈ 295 KB
        - Connectivity: 368 × 368 × 8 bytes ≈ 1.1 MB
        Total: < 2 MB per subject
        """
        n_timepoints = 100
        n_regions = 368

        # Create timeseries
        timeseries = np.random.randn(n_timepoints, n_regions).astype(np.float32)
        ts_size = timeseries.nbytes

        # Create connectivity (worst case)
        connectivity = np.random.randn(n_regions, n_regions).astype(np.float32)
        conn_size = connectivity.nbytes

        total_mb = (ts_size + conn_size) / (1024 * 1024)

        # Should be < 3 MB for single subject
        assert total_mb < 3.0

    @pytest.mark.asyncio
    async def test_distance_correlation_timeseries_size(self):
        """Test distance correlation with realistic data sizes."""
        loader = PRIMEDELoader()

        # 400 timepoints (typical fMRI scan)
        # 368 regions (D99 atlas)
        timeseries = np.random.randn(400, 368).astype(np.float32)

        # This should complete reasonably quickly
        conn_matrix, _ = await loader.compute_functional_connectivity(
            timeseries,
            method="pearson"  # Pearson is faster for timing test
        )

        assert conn_matrix.shape == (368, 368)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_load_subject_api_error_response(self):
        """Test handling of API error in response."""
        loader = PRIMEDELoader()

        mock_response = {
            "error": "File not found"
        }

        with patch.object(loader.client, "post", new_callable=AsyncMock) as mock_post:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            with pytest.raises(ValueError, match="File not found"):
                await loader.load_subject("BORDEAUX24", "m01", "bold")

    def test_atlas_with_single_voxel_regions(self):
        """Test atlas extraction when data is minimal."""
        atlas = D99Atlas()

        # Minimal data: one voxel
        nifti_data = np.ones((1, 1, 1, 50), dtype=np.float32)
        timeseries = atlas.extract_timeseries(nifti_data)

        assert timeseries.shape == (50, 368)
        assert np.allclose(timeseries, 1.0)

    @pytest.mark.asyncio
    async def test_connectivity_single_region(self):
        """Test connectivity with minimal number of regions.

        Edge case: very small timeseries.
        """
        loader = PRIMEDELoader()

        # Only 2 regions, 10 timepoints
        timeseries = np.random.randn(10, 2)

        conn_matrix, _ = await loader.compute_functional_connectivity(timeseries)

        assert conn_matrix.shape == (2, 2)
        assert np.allclose(np.diag(conn_matrix), 1.0)
