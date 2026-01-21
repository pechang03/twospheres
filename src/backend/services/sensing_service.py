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
from typing import Any, Dict, Optional, Tuple

import emcee
import numpy as np
from lmfit import Model, Parameters
from numpy.typing import NDArray

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


class InterferometricSensor(BaseService):
    """
    Interferometric biosensor for refractive index sensing via visibility analysis.

    This service implements visibility fitting using lmfit for automatic uncertainty
    propagation and derived parameter calculation. Based on the interference pattern
    from Mach-Zehnder interferometry (adapted from entangled-pair-quantum-eraser).

    Key Features
    ------------
    - lmfit.Model for non-linear least squares with error propagation
    - Automatic calculation of visibility V = A/(A + 2*C₀)
    - Bayesian uncertainty quantification (optional, via emcee)
    - Phase shift measurement for refractive index changes Δn

    The visibility V measures fringe contrast and is sensitive to optical path
    length differences, making it ideal for biosensing applications where
    biomarker binding causes refractive index shifts.

    References
    ----------
    See docs/designs/quantum_interferometry_integration.md for detailed analysis.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize interferometric sensor.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary with required keys:
            - wavelength_nm: Operating wavelength (300-1100 nm)
            - path_length_mm: Interferometer path length difference (0.1-1000 mm)
            - refractive_index_sensitivity: Minimum detectable Δn (1e-7 to 1e-3)

        Raises
        ------
        ServiceValidationError
            If configuration is invalid.
        """
        super().__init__(config)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate interferometric sensor configuration."""
        required_keys = ["wavelength_nm", "path_length_mm", "refractive_index_sensitivity"]
        for key in required_keys:
            if key not in self.config:
                raise ServiceValidationError(f"Missing required config key: {key}")

        wavelength = self.config["wavelength_nm"]
        if not (300 <= wavelength <= 1100):
            raise ServiceValidationError(
                f"wavelength_nm must be in [300, 1100], got {wavelength}"
            )

        path_length = self.config["path_length_mm"]
        if not (0.1 <= path_length <= 1000):
            raise ServiceValidationError(
                f"path_length_mm must be in [0.1, 1000], got {path_length}"
            )

        sensitivity = self.config["refractive_index_sensitivity"]
        if not (1e-7 <= sensitivity <= 1e-3):
            raise ServiceValidationError(
                f"refractive_index_sensitivity must be in [1e-7, 1e-3], got {sensitivity}"
            )

    def _interference_pattern(
        self,
        position: NDArray[np.float64],
        amplitude: float,
        background: float,
        phase: float,
        period: float
    ) -> NDArray[np.float64]:
        """
        Interference pattern model: I(x) = A·cos²(2π·x/Λ + φ) + C₀

        Parameters
        ----------
        position : NDArray
            Spatial position or time coordinate
        amplitude : float
            Fringe amplitude A
        background : float
            Background level C₀
        phase : float
            Phase offset φ (radians)
        period : float
            Fringe period Λ

        Returns
        -------
        NDArray
            Intensity pattern I(x)
        """
        return amplitude * np.cos(2 * np.pi * position / period + phase) ** 2 + background

    async def fit_visibility(
        self,
        position: NDArray[np.float64],
        intensity: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Fit interference pattern and compute visibility with uncertainty.

        This method uses lmfit.Model for automatic uncertainty propagation,
        which is superior to scipy.curve_fit because it:
        1. Automatically propagates uncertainties to derived parameters
        2. Supports parameter constraints and derived expressions
        3. Provides comprehensive fit statistics

        Parameters
        ----------
        position : NDArray
            Spatial position or time coordinate (N points)
        intensity : NDArray
            Measured intensity values (N points)
        weights : NDArray, optional
            Statistical weights for each point (typically 1/σᵢ)

        Returns
        -------
        visibility : float
            Fringe visibility V = A/(A + 2*C₀)
        visibility_uncertainty : float
            Standard error in visibility from error propagation
        fit_params : Dict[str, float]
            All fitted parameters with uncertainties
            Keys: amplitude, background, phase, period, and their _stderr values

        Raises
        ------
        ServiceValidationError
            If input arrays have mismatched shapes or insufficient data points
        """
        # Validation
        if len(position) != len(intensity):
            raise ServiceValidationError(
                f"position and intensity arrays must have same length, "
                f"got {len(position)} and {len(intensity)}"
            )

        if len(position) < 5:
            raise ServiceValidationError(
                f"Need at least 5 data points for fitting, got {len(position)}"
            )

        return await asyncio.to_thread(
            self._blocking_fit_visibility,
            position,
            intensity,
            weights
        )

    def _blocking_fit_visibility(
        self,
        position: NDArray[np.float64],
        intensity: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        CPU-heavy visibility fitting computation.

        This is the blocking implementation that performs the actual lmfit
        optimization. It's called via asyncio.to_thread to avoid blocking
        the event loop.
        """
        # Create lmfit Model
        model = Model(self._interference_pattern, independent_vars=['position'])

        # Initial parameter guesses
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        amplitude_guess = (intensity_max - intensity_min) / 2
        background_guess = intensity_min

        # Estimate period from Fourier transform
        fft = np.fft.rfft(intensity - np.mean(intensity))
        freqs = np.fft.rfftfreq(len(intensity), d=np.mean(np.diff(position)))
        peak_freq_idx = np.argmax(np.abs(fft[1:])) + 1  # Skip DC component
        period_guess = 1.0 / freqs[peak_freq_idx] if freqs[peak_freq_idx] != 0 else 1.0

        # Set up parameters with physical constraints
        params = Parameters()
        params.add('amplitude', value=amplitude_guess, min=0)
        params.add('background', value=background_guess, min=0)
        params.add('phase', value=0, min=-np.pi, max=np.pi)
        params.add('period', value=period_guess, min=0)

        # Add derived parameter: visibility V = A/(A + 2*C₀)
        params.add('visibility', expr='amplitude / (amplitude + 2*background)')

        # Perform fit
        result = model.fit(
            intensity,
            params=params,
            position=position,
            weights=weights
        )

        # Extract results
        visibility = result.params['visibility'].value
        visibility_stderr = result.params['visibility'].stderr

        # If stderr is None (fit failed to estimate uncertainties), use 0
        if visibility_stderr is None:
            visibility_stderr = 0.0

        # Build comprehensive results dictionary
        fit_params = {}
        for param_name in ['amplitude', 'background', 'phase', 'period', 'visibility']:
            param = result.params[param_name]
            fit_params[param_name] = param.value
            fit_params[f'{param_name}_stderr'] = param.stderr if param.stderr is not None else 0.0

        # Add fit quality metrics
        fit_params['chi_square'] = result.chisqr
        fit_params['reduced_chi_square'] = result.redchi
        fit_params['r_squared'] = 1 - result.residual.var() / np.var(intensity)

        return visibility, visibility_stderr, fit_params

    async def compute_refractive_index_shift(
        self,
        visibility_before: float,
        visibility_after: float,
        visibility_uncertainty: float = 0.0
    ) -> Tuple[float, float]:
        """
        Compute refractive index shift from visibility change.

        For a Mach-Zehnder interferometer with path length difference L,
        a phase shift Δφ = (2π/λ)·L·Δn causes a visibility change.

        Parameters
        ----------
        visibility_before : float
            Initial visibility (baseline, no analyte)
        visibility_after : float
            Final visibility (with bound analyte)
        visibility_uncertainty : float
            Combined uncertainty in visibility measurements

        Returns
        -------
        delta_n : float
            Refractive index shift Δn
        delta_n_uncertainty : float
            Uncertainty in Δn from error propagation

        Raises
        ------
        ServiceValidationError
            If visibilities are out of physical range [0, 1]
        """
        if not (0 <= visibility_before <= 1):
            raise ServiceValidationError(
                f"visibility_before must be in [0, 1], got {visibility_before}"
            )

        if not (0 <= visibility_after <= 1):
            raise ServiceValidationError(
                f"visibility_after must be in [0, 1], got {visibility_after}"
            )

        return await asyncio.to_thread(
            self._blocking_compute_refractive_index_shift,
            visibility_before,
            visibility_after,
            visibility_uncertainty
        )

    def _blocking_compute_refractive_index_shift(
        self,
        visibility_before: float,
        visibility_after: float,
        visibility_uncertainty: float
    ) -> Tuple[float, float]:
        """
        CPU-heavy computation of refractive index shift.

        Uses the relation: Δφ = (2π/λ)·L·Δn
        and ΔV ≈ -V₀·sin(φ₀)·Δφ for small perturbations
        """
        wavelength_m = self.config["wavelength_nm"] * 1e-9
        path_length_m = self.config["path_length_mm"] * 1e-3

        # Visibility change
        delta_visibility = visibility_after - visibility_before

        # For small phase shifts, ΔV ≈ -V₀·sin(φ₀)·Δφ
        # Assuming φ₀ ≈ π/2 for maximum sensitivity (sin(π/2) = 1)
        # Then: Δφ ≈ -ΔV/V₀
        if visibility_before > 1e-6:
            delta_phase = -delta_visibility / visibility_before
        else:
            delta_phase = 0.0

        # Convert phase shift to refractive index shift
        # Δφ = (2π/λ)·L·Δn  =>  Δn = λ·Δφ/(2π·L)
        delta_n = wavelength_m * delta_phase / (2 * np.pi * path_length_m)

        # Propagate uncertainty
        # σ(Δn) = (λ/(2π·L)) · σ(Δφ) = (λ/(2π·L·V₀)) · σ(ΔV)
        if visibility_before > 1e-6:
            delta_n_uncertainty = (
                wavelength_m * visibility_uncertainty /
                (2 * np.pi * path_length_m * visibility_before)
            )
        else:
            delta_n_uncertainty = np.inf

        return delta_n, delta_n_uncertainty

    async def fit_visibility_bayesian(
        self,
        position: NDArray[np.float64],
        intensity: NDArray[np.float64],
        intensity_uncertainty: Optional[NDArray[np.float64]] = None,
        nwalkers: int = 32,
        nsteps: int = 1000,
        burn_in: int = 100
    ) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
        """
        Fit interference pattern using Bayesian MCMC with emcee.

        This method provides full posterior distributions for all parameters
        instead of just point estimates. It's superior to lmfit when:
        1. Parameter uncertainties are non-Gaussian (asymmetric, multimodal)
        2. Strong parameter correlations exist
        3. Full posterior distributions are needed for decision-making
        4. Model comparison via Bayesian evidence is required

        Uses emcee (Foreman-Mackey et al. 2013) for affine-invariant ensemble
        sampling, which is more efficient than standard Metropolis-Hastings.

        Parameters
        ----------
        position : NDArray
            Spatial position or time coordinate (N points)
        intensity : NDArray
            Measured intensity values (N points)
        intensity_uncertainty : NDArray, optional
            Standard deviation of intensity measurements (N points)
            If None, assumes uniform uncertainty from data scatter
        nwalkers : int, optional
            Number of MCMC walkers (default: 32)
            Must be at least 2× number of parameters (4 params → 8 walkers minimum)
        nsteps : int, optional
            Number of MCMC steps per walker (default: 1000)
        burn_in : int, optional
            Number of initial steps to discard (default: 100)

        Returns
        -------
        summary_stats : Dict[str, float]
            Posterior medians and credible intervals
            Keys: amplitude_median, amplitude_16th, amplitude_84th, ...
            Also includes visibility_median, visibility_16th, visibility_84th
        samples : Dict[str, NDArray]
            Full posterior samples for each parameter
            Keys: amplitude, background, phase, period, visibility
            Each array has shape (nwalkers * (nsteps - burn_in),)

        Raises
        ------
        ServiceValidationError
            If input arrays have mismatched shapes or insufficient data

        Notes
        -----
        The 16th and 84th percentiles define the 68% credible interval,
        analogous to ±1σ for Gaussian distributions.
        """
        # Validation
        if len(position) != len(intensity):
            raise ServiceValidationError(
                f"position and intensity arrays must have same length, "
                f"got {len(position)} and {len(intensity)}"
            )

        if len(position) < 5:
            raise ServiceValidationError(
                f"Need at least 5 data points for Bayesian fitting, got {len(position)}"
            )

        return await asyncio.to_thread(
            self._blocking_fit_visibility_bayesian,
            position,
            intensity,
            intensity_uncertainty,
            nwalkers,
            nsteps,
            burn_in
        )

    def _blocking_fit_visibility_bayesian(
        self,
        position: NDArray[np.float64],
        intensity: NDArray[np.float64],
        intensity_uncertainty: Optional[NDArray[np.float64]],
        nwalkers: int,
        nsteps: int,
        burn_in: int
    ) -> Tuple[Dict[str, float], Dict[str, NDArray[np.float64]]]:
        """
        CPU-heavy Bayesian MCMC fitting computation.

        This implements the MCMC sampling using emcee. It's called via
        asyncio.to_thread to avoid blocking the event loop.
        """
        # Estimate uncertainty if not provided
        if intensity_uncertainty is None:
            # Use residual scatter from quick polynomial fit
            poly_coeffs = np.polyfit(position, intensity, deg=2)
            poly_fit = np.polyval(poly_coeffs, position)
            residuals = intensity - poly_fit
            intensity_uncertainty = np.full_like(intensity, np.std(residuals))

        # Define log-likelihood function
        def log_likelihood(params):
            amplitude, background, phase, period = params
            # Compute model prediction
            model = self._interference_pattern(
                position, amplitude, background, phase, period
            )
            # Chi-square statistic
            chi2 = np.sum(((intensity - model) / intensity_uncertainty) ** 2)
            return -0.5 * chi2

        # Define log-prior (uniform priors with physical constraints)
        def log_prior(params):
            amplitude, background, phase, period = params
            # Physical constraints
            if amplitude < 0 or background < 0:
                return -np.inf
            if not (-np.pi <= phase <= np.pi):
                return -np.inf
            if period <= 0:
                return -np.inf
            # Reasonable ranges (loose priors)
            if amplitude > 10 * np.max(intensity):
                return -np.inf
            if background > 10 * np.max(intensity):
                return -np.inf
            if period > 10 * (np.max(position) - np.min(position)):
                return -np.inf
            return 0.0  # Flat prior within bounds

        # Define log-posterior
        def log_posterior(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params)

        # Initial parameter guesses (same as lmfit)
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        amplitude_init = (intensity_max - intensity_min) / 2
        background_init = intensity_min

        # Estimate period from FFT
        fft = np.fft.rfft(intensity - np.mean(intensity))
        freqs = np.fft.rfftfreq(len(intensity), d=np.mean(np.diff(position)))
        peak_freq_idx = np.argmax(np.abs(fft[1:])) + 1
        period_init = 1.0 / freqs[peak_freq_idx] if freqs[peak_freq_idx] != 0 else 1.0

        # Initialize walkers around initial guess with small perturbations
        ndim = 4  # amplitude, background, phase, period
        initial_params = np.array([amplitude_init, background_init, 0.0, period_init])
        # Use larger perturbations (1%) to ensure linear independence
        pos = initial_params + 1e-2 * np.random.randn(nwalkers, ndim) * np.abs(initial_params)

        # Ensure all walkers satisfy prior constraints and are linearly independent
        for i in range(nwalkers):
            max_attempts = 100
            attempts = 0
            while not np.isfinite(log_prior(pos[i])) and attempts < max_attempts:
                # Use different perturbation scales for each parameter
                pos[i, 0] = amplitude_init * (0.8 + 0.4 * np.random.rand())  # amplitude: ±20%
                pos[i, 1] = background_init * (0.8 + 0.4 * np.random.rand())  # background: ±20%
                pos[i, 2] = np.random.uniform(-0.5, 0.5)  # phase: ±0.5 rad around 0
                pos[i, 3] = period_init * (0.8 + 0.4 * np.random.rand())  # period: ±20%
                attempts += 1
            if attempts >= max_attempts:
                # Fallback: use uniform sampling within bounds with more spread
                pos[i, 0] = amplitude_init * (0.5 + np.random.rand())
                pos[i, 1] = background_init * (0.5 + np.random.rand())
                pos[i, 2] = np.random.uniform(-np.pi, np.pi)
                pos[i, 3] = period_init * (0.5 + np.random.rand())

        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        # Skip initial state check for numerical stability with small perturbations
        sampler.run_mcmc(pos, nsteps, progress=False, skip_initial_state_check=True)

        # Get samples, discarding burn-in
        samples = sampler.get_chain(discard=burn_in, flat=True)

        # Extract parameter samples
        amplitude_samples = samples[:, 0]
        background_samples = samples[:, 1]
        phase_samples = samples[:, 2]
        period_samples = samples[:, 3]

        # Compute derived parameter: visibility V = A/(A + 2*C₀)
        visibility_samples = amplitude_samples / (amplitude_samples + 2 * background_samples)

        # Compute summary statistics (median and 68% credible interval)
        def compute_stats(param_samples, param_name):
            median = np.median(param_samples)
            percentile_16 = np.percentile(param_samples, 16)
            percentile_84 = np.percentile(param_samples, 84)
            return {
                f"{param_name}_median": median,
                f"{param_name}_16th": percentile_16,
                f"{param_name}_84th": percentile_84,
                f"{param_name}_std": np.std(param_samples)
            }

        # Build summary statistics dictionary
        summary_stats = {}
        summary_stats.update(compute_stats(amplitude_samples, "amplitude"))
        summary_stats.update(compute_stats(background_samples, "background"))
        summary_stats.update(compute_stats(phase_samples, "phase"))
        summary_stats.update(compute_stats(period_samples, "period"))
        summary_stats.update(compute_stats(visibility_samples, "visibility"))

        # Add MCMC diagnostics
        summary_stats["acceptance_fraction"] = float(np.mean(sampler.acceptance_fraction))
        summary_stats["autocorr_time"] = float(np.mean(sampler.get_autocorr_time(quiet=True)))
        summary_stats["nwalkers"] = nwalkers
        summary_stats["nsteps"] = nsteps
        summary_stats["burn_in"] = burn_in

        # Build samples dictionary
        samples_dict = {
            "amplitude": amplitude_samples,
            "background": background_samples,
            "phase": phase_samples,
            "period": period_samples,
            "visibility": visibility_samples
        }

        return summary_stats, samples_dict

    async def health_check(self) -> bool:
        """
        Check if the interferometric sensor is healthy.

        Returns
        -------
        bool
            True if service is ready to accept requests.
        """
        try:
            self._validate_config()
            return True
        except ServiceValidationError:
            return False
