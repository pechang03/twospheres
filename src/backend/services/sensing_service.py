"""
Convert exposure time to detected intensity.

Constructor arguments
---------------------
config : dict
    Must contain keys:
    - "wavelength_nm" (float) – vacuum wavelength in nanometres
    - "na_objective" (float) – numerical aperture of the objective
    - "pixel_size_um" (float) – camera pixel pitch
    - "dark_signal_e" (float) – average dark signal in electrons

Returns
-------
SensingService object with coroutines:
- `compute_intensity(exposure_ms)` → float
- `health_check()` → bool

Concurrency
-----------
Fully async; uses `to_thread` for CPU-heavy code if needed.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from ._service_base import BaseService, ServiceValidationError


class SensingService(BaseService):
    """
    Sensing service for converting exposure time to detected intensity.

    This service models the relationship between camera exposure time
    and detected optical intensity based on system parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize sensing service with optical configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with required keys:
            - wavelength_nm: Vacuum wavelength in nanometres (300-1100 nm)
            - na_objective: Numerical aperture of objective (0.1-1.45)
            - pixel_size_um: Camera pixel pitch in micrometres (0.5-50 µm)
            - dark_signal_e: Average dark signal in electrons (0-1000)

        Raises
        ------
        ServiceValidationError
            If required configuration keys are missing or values are out of range.
        """
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters against physical constraints."""
        required_keys = ["wavelength_nm", "na_objective", "pixel_size_um", "dark_signal_e"]
        for key in required_keys:
            if key not in self.config:
                raise ServiceValidationError(f"Missing required config key: {key}")

        wavelength = self.config["wavelength_nm"]
        if not (300 <= wavelength <= 1100):
            raise ServiceValidationError(
                f"wavelength_nm must be in [300, 1100], got {wavelength}"
            )

        na = self.config["na_objective"]
        if not (0.1 <= na <= 1.45):
            raise ServiceValidationError(
                f"na_objective must be in [0.1, 1.45], got {na}"
            )

        pixel_size = self.config["pixel_size_um"]
        if not (0.5 <= pixel_size <= 50):
            raise ServiceValidationError(
                f"pixel_size_um must be in [0.5, 50], got {pixel_size}"
            )

        dark_signal = self.config["dark_signal_e"]
        if not (0 <= dark_signal <= 1000):
            raise ServiceValidationError(
                f"dark_signal_e must be in [0, 1000], got {dark_signal}"
            )

    async def compute_intensity(self, exposure_ms: float) -> float:
        """
        Compute detected intensity for a given exposure time.

        Parameters
        ----------
        exposure_ms : float
            Camera exposure time in milliseconds.

        Returns
        -------
        float
            Computed detected intensity (arbitrary units).

        Raises
        ------
        ServiceValidationError
            If exposure_ms is negative.
        """
        if exposure_ms < 0:
            raise ServiceValidationError(
                f"exposure_ms must be non-negative, got {exposure_ms}"
            )
        return await asyncio.to_thread(self._blocking_compute_intensity, exposure_ms)

    def _blocking_compute_intensity(self, exposure_ms: float) -> float:
        """
        CPU-heavy intensity computation (placeholder implementation).

        Parameters
        ----------
        exposure_ms : float
            Exposure time in milliseconds.

        Returns
        -------
        float
            Computed intensity.
        """
        # Minimal dummy model - saturation curve
        return 100.0 * exposure_ms / (exposure_ms + 1.0)

    async def health_check(self) -> bool:
        """
        Check if the sensing service is healthy.

        Returns
        -------
        bool
            True if service is ready to accept requests.
        """
        try:
            # Verify configuration is valid
            self._validate_config()
            return True
        except ServiceValidationError:
            return False
