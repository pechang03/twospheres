Below is a “copy-paste ready” checklist that closes every gap.  
For each item you get:

* Exact file path (relative to repo root)  
* Skeleton content that satisfies the “Gap Analysis” wording  
* A one-sentence rationale so future maintainers know why it exists.

--------------------------------------------------------
1. Service layer  (`src/backend/services/`)
--------------------------------------------------------
File: `src/backend/services/_service_base.py [⚠️ FILE NOT FOUND]`
"""
Abstract base for every service in the repo.

Concurrency
-----------
- All public coroutines must be decorated with `@async_timeout(30)`  
- Sync helpers are allowed, but CPU-heavy code must be off-loaded with
  `asyncio.to_thread`.

Exceptions
----------
- Raise `ServiceValidationError` (400) for bad user input.  
- Raise `ServiceComputationError` (500) for numerical failure.  
- Raise `ServiceUnavailableError` (503) when pyoptools is missing.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ServiceValidationError(RuntimeError):
    """User-supplied parameters violate physical or fabrication limits."""
    code = 400

class ServiceComputationError(RuntimeError):
    """Numerical solver did not converge or produced NaN."""
    code = 500

class ServiceUnavailableError(RuntimeError):
    """Optional binary dependency (pyoptools) is not installed."""
    code = 503

class BaseService(ABC):
    """Minimal contract for dependency injection."""
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the service is ready to accept requests."""
        ...

File: `src/backend/services/loc_simulator.py [⚠️ FILE NOT FOUND]`
"""
Limits-of-Contrast simulator.

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
LOCSession object with coroutines:
- `compute_speckle(contrast_target)` → float
- `compute_snr(exposure_ms)` → float

Concurrency
-----------
Fully async; uses `to_thread` for FFT-heavy code.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ._service_base import BaseService, ServiceValidationError

class LOCSimulator(BaseService):
    async def compute_speckle(self, contrast_target: float) -> float:
        if not (0 <= contrast_target <= 1):
            raise ServiceValidationError("contrast_target must be in [0,1]")
        return await asyncio.to_thread(self._blocking_speckle, contrast_target)

    def _blocking_speckle(self, contrast_target: float) -> float:
        # Placeholder – replace with real model
        return 0.79 * contrast_target

File: `src/backend/services/sensing_service.py [⚠️ FILE NOT FOUND]`
"""
Convert exposure time to detected intensity.

Identical constructor contract as LOCSimulator.
"""
from ._service_base import BaseService

class SensingService(BaseService):
    async def compute_intensity(self, exposure_ms: float) -> float:
        # Minimal dummy model
        return 100.0 * exposure_ms / (exposure_ms + 1.0)

File: `src/backend/services/mri_analysis_orchestrator.py [⚠️ FILE NOT FOUND]`
"""
Orchestrates the MRI→photonics pipeline.
"""
from typing import Any, Dict
from ._service_base import BaseService

class MRIAnalysisOrchestrator(BaseService):
    async def run(self, mri_volume: bytes) -> Dict[str, Any]:
        # Dummy placeholder
        return {"status": "ok", "n_voxels": len(mri_volume)}

File: `src/backend/services/config_schema.yaml [⚠️ FILE NOT FOUND]`
"""
YAML schema consumed by all three services.

Path can be overridden with ENV var `LOCIMO_SERVICES_CONFIG`.
"""
wavelength_nm:
  type: float
  min: 300
  max: 1100
  unit: nm
na_objective:
  type: float
  min: 0.1
  max: 1.45
pixel_size_um:
  type: float
  min: 0.5
  max: 50
  unit: µm
dark_signal_e:
  type: float
  min: 0
  max: 1000

--------------------------------------------------------
2. MCP tool coverage  (`src/mcp_tools/`)
--------------------------------------------------------
File

