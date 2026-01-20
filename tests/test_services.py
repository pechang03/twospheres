"""Unit tests for backend services."""
import pytest

from src.backend.services import (
    LOCSimulator,
    MRIAnalysisOrchestrator,
    SensingService,
    ServiceValidationError,
)


@pytest.fixture
def valid_config():
    """Valid configuration for LOC and Sensing services."""
    return {
        "wavelength_nm": 800.0,
        "na_objective": 0.5,
        "pixel_size_um": 5.0,
        "dark_signal_e": 50.0,
    }


class TestLOCSimulator:
    """Tests for LOCSimulator service."""

    def test_valid_initialization(self, valid_config):
        """Test initialization with valid configuration."""
        simulator = LOCSimulator(valid_config)
        assert simulator.config == valid_config

    def test_missing_config_key(self):
        """Test initialization fails with missing config key."""
        config = {"wavelength_nm": 800.0}  # Missing other required keys
        with pytest.raises(ServiceValidationError, match="Missing required config key"):
            LOCSimulator(config)

    def test_wavelength_out_of_range(self, valid_config):
        """Test initialization fails with wavelength out of range."""
        config = valid_config.copy()
        config["wavelength_nm"] = 2000.0  # Too high
        with pytest.raises(ServiceValidationError, match="wavelength_nm must be in"):
            LOCSimulator(config)

    def test_na_out_of_range(self, valid_config):
        """Test initialization fails with NA out of range."""
        config = valid_config.copy()
        config["na_objective"] = 2.0  # Too high
        with pytest.raises(ServiceValidationError, match="na_objective must be in"):
            LOCSimulator(config)

    @pytest.mark.asyncio
    async def test_compute_speckle_valid(self, valid_config):
        """Test speckle computation with valid input."""
        simulator = LOCSimulator(valid_config)
        result = await simulator.compute_speckle(0.8)
        assert 0 <= result <= 1
        assert abs(result - 0.632) < 0.001  # 0.79 * 0.8

    @pytest.mark.asyncio
    async def test_compute_speckle_invalid_contrast(self, valid_config):
        """Test speckle computation fails with invalid contrast."""
        simulator = LOCSimulator(valid_config)
        with pytest.raises(ServiceValidationError, match="contrast_target must be in"):
            await simulator.compute_speckle(1.5)

    @pytest.mark.asyncio
    async def test_health_check(self, valid_config):
        """Test health check returns True for valid service."""
        simulator = LOCSimulator(valid_config)
        assert await simulator.health_check() is True


class TestSensingService:
    """Tests for SensingService."""

    def test_valid_initialization(self, valid_config):
        """Test initialization with valid configuration."""
        service = SensingService(valid_config)
        assert service.config == valid_config

    @pytest.mark.asyncio
    async def test_compute_intensity_valid(self, valid_config):
        """Test intensity computation with valid exposure time."""
        service = SensingService(valid_config)
        result = await service.compute_intensity(10.0)
        assert result > 0
        assert abs(result - 90.909) < 0.01  # 100 * 10 / 11

    @pytest.mark.asyncio
    async def test_compute_intensity_negative_exposure(self, valid_config):
        """Test intensity computation fails with negative exposure."""
        service = SensingService(valid_config)
        with pytest.raises(ServiceValidationError, match="exposure_ms must be"):
            await service.compute_intensity(-5.0)

    @pytest.mark.asyncio
    async def test_health_check(self, valid_config):
        """Test health check returns True for valid service."""
        service = SensingService(valid_config)
        assert await service.health_check() is True


class TestMRIAnalysisOrchestrator:
    """Tests for MRIAnalysisOrchestrator."""

    def test_valid_initialization(self):
        """Test initialization with configuration."""
        orchestrator = MRIAnalysisOrchestrator({"setting": "value"})
        assert orchestrator.config == {"setting": "value"}

    @pytest.mark.asyncio
    async def test_run_with_data(self):
        """Test running orchestrator with MRI volume data."""
        orchestrator = MRIAnalysisOrchestrator({})
        test_data = b"fake_mri_volume_data"
        result = await orchestrator.run(test_data)

        assert "status" in result
        assert result["status"] == "ok"
        assert "n_voxels" in result
        assert result["n_voxels"] == len(test_data)

    @pytest.mark.asyncio
    async def test_run_with_empty_data(self):
        """Test running orchestrator with empty data fails."""
        orchestrator = MRIAnalysisOrchestrator({})
        with pytest.raises(ServiceValidationError, match="mri_volume cannot be empty"):
            await orchestrator.run(b"")

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns True."""
        orchestrator = MRIAnalysisOrchestrator({})
        assert await orchestrator.health_check() is True
