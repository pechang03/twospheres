"""Tests for Ernie2 swarm integration."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.backend.services.ernie2_integration import (
    Ernie2SwarmClient,
    query_expert_collections,
)


class TestErnie2SwarmClient:
    """Tests for Ernie2SwarmClient."""

    def test_initialization_default_path(self):
        """Test client initialization with default path."""
        # Mock path existence check
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()
            assert client.ernie2_path.name == "ernie2_swarm.py"
            assert not client.use_cloud

    def test_initialization_custom_path(self):
        """Test client initialization with custom path."""
        custom_path = "/custom/path/ernie2_swarm.py"
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient(ernie2_path=custom_path)
            assert str(client.ernie2_path) == custom_path

    def test_initialization_missing_script(self):
        """Test error when ernie2_swarm.py not found."""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="ernie2_swarm.py not found"):
                Ernie2SwarmClient()

    @pytest.mark.asyncio
    async def test_query_with_json_response(self):
        """Test query with JSON response."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()

            # Mock subprocess.run
            mock_result = MagicMock()
            mock_result.stdout = '{"answer": "Sensitivity: 1e-6 RIU", "refractive_index_sensitivity": 1e-6}'
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result):
                result = await client.query(
                    question="What sensitivity?",
                    collections=["neuroscience_MRI"]
                )

                assert "answer" in result
                assert result["collections_queried"] == ["neuroscience_MRI"]
                assert "parameters" in result
                assert result["parameters"]["refractive_index_sensitivity"] == 1e-6

    @pytest.mark.asyncio
    async def test_query_with_text_response(self):
        """Test query with plain text response."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()

            # Mock subprocess.run with plain text
            mock_result = MagicMock()
            mock_result.stdout = "The refractive index sensitivity should be around 1e-6. Use 633 nm wavelength."
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result):
                result = await client.query(
                    question="What parameters?",
                    collections=["physics_optics"]
                )

                assert "answer" in result
                assert "1e-6" in result["answer"]
                assert result["parameters"]["refractive_index_sensitivity"] == 1e-6
                assert result["parameters"]["wavelength_nm"] == 633.0

    @pytest.mark.asyncio
    async def test_query_timeout(self):
        """Test query timeout handling."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()

            # Mock timeout
            import subprocess
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(cmd="test", timeout=30)):
                result = await client.query(
                    question="Test",
                    collections=["test"]
                )

                assert "error" in result
                assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_query_with_cloud_model(self):
        """Test query using cloud model."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient(use_cloud=True)

            mock_result = MagicMock()
            mock_result.stdout = '{"answer": "Response from cloud"}'
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                result = await client.query(
                    question="Test cloud",
                    collections=["test"]
                )

                # Verify --cloud flag was used
                call_args = mock_run.call_args[0][0]
                assert "--cloud" in call_args
                assert "--local" not in call_args

    def test_extract_parameters_from_dict(self):
        """Test parameter extraction from dict."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()

            response = {
                "answer": "Test",
                "refractive_index_sensitivity": 1e-6,
                "wavelength_nm": 633,
                "path_length_mm": 10.0
            }

            params = client._extract_parameters(response)

            assert params["refractive_index_sensitivity"] == 1e-6
            assert params["wavelength_nm"] == 633
            assert params["path_length_mm"] == 10.0

    def test_extract_parameters_from_text(self):
        """Test parameter extraction from text."""
        with patch.object(Path, 'exists', return_value=True):
            client = Ernie2SwarmClient()

            text = "The sensitivity should be 1.5e-6 RIU. Use wavelength 850 nm."

            params = client._extract_parameters_from_text(text)

            assert params["refractive_index_sensitivity"] == 1.5e-6
            assert params["wavelength_nm"] == 850.0


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_query_expert_collections(self):
        """Test convenience function."""
        with patch.object(Path, 'exists', return_value=True):
            mock_result = MagicMock()
            mock_result.stdout = '{"answer": "Test response"}'
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result):
                result = await query_expert_collections(
                    question="Test",
                    collections=["neuroscience_MRI"],
                    use_cloud=False
                )

                assert "answer" in result
                assert result["collections_queried"] == ["neuroscience_MRI"]
