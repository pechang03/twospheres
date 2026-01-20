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

from abc import ABC, abstractmethod
from typing import Any, Dict


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
        """
        Initialize the service with configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing service-specific parameters.
        """
        self.config = config

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Return True if the service is ready to accept requests.

        Returns
        -------
        bool
            True if service is healthy and ready, False otherwise.
        """
        ...
