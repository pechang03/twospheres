"""
Feedback control for optical resonators and interferometric sensors.

This module implements digital lock-in amplification and PID control for
stabilizing optical resonators and maintaining interferometer phase lock.

Key Features
------------
- Digital lock-in detection with in-phase (I) and quadrature (Q) channels
- PID controller for resonator wavelength/temperature tuning
- Phase-sensitive detection for noise rejection
- Integration with scipy.signal for filtering and demodulation

Applications
------------
1. Ring resonator feedback for LOC biosensing (bead twosphere-mcp-6ez)
2. Mach-Zehnder interferometer phase stabilization
3. Cavity ringdown spectroscopy lock-in detection
4. Homodyne/heterodyne detection for quantum optics

References
----------
See docs/designs/design-ph.0-quantum-primitive/spectroscopy_packages_integration.md
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal


class DigitalLockIn:
    """
    Digital lock-in amplifier for phase-sensitive detection.

    A lock-in amplifier extracts signals buried in noise by correlating
    the input with a reference frequency. It outputs in-phase (I) and
    quadrature (Q) components, which give both amplitude and phase:
        R = √(I² + Q²)  [amplitude]
        φ = arctan(Q/I)  [phase]

    This is critical for:
    - Detecting small refractive index changes in biosensors
    - Stabilizing optical cavities via feedback control
    - Rejecting 1/f noise and environmental drift

    Implementation uses scipy.signal.hilbert for quadrature demodulation,
    which is more efficient than explicit sin/cos multiplication + filtering.
    """

    def __init__(
        self,
        reference_frequency: float,
        sampling_rate: float,
        time_constant: float = 1.0
    ) -> None:
        """
        Initialize digital lock-in amplifier.

        Parameters
        ----------
        reference_frequency : float
            Reference frequency in Hz (carrier frequency)
        sampling_rate : float
            Data acquisition sampling rate in Hz
        time_constant : float, optional
            Low-pass filter time constant in seconds (default: 1.0)
            Determines noise bandwidth: BW ≈ 1/(4·τ)

        Raises
        ------
        ValueError
            If sampling_rate < 2·reference_frequency (violates Nyquist)
        """
        if sampling_rate < 2 * reference_frequency:
            raise ValueError(
                f"Sampling rate ({sampling_rate} Hz) must be at least twice "
                f"the reference frequency ({reference_frequency} Hz) to satisfy Nyquist criterion"
            )

        self.reference_frequency = reference_frequency
        self.sampling_rate = sampling_rate
        self.time_constant = time_constant

        # Design low-pass filter (Butterworth, 4th order)
        # Cutoff frequency: fc = 1/(2π·τ)
        cutoff_frequency = 1.0 / (2 * np.pi * time_constant)
        nyquist_frequency = sampling_rate / 2
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Ensure cutoff is within valid range (0, 1)
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.9
        elif normalized_cutoff <= 0:
            normalized_cutoff = 0.01

        self.sos = signal.butter(
            4,  # 4th order for steep rolloff
            normalized_cutoff,
            btype='low',
            output='sos'  # Second-order sections for numerical stability
        )

    def demodulate(
        self,
        signal_in: NDArray[np.float64],
        time: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Perform lock-in detection on input signal.

        Parameters
        ----------
        signal_in : NDArray
            Input signal to demodulate (length N)
        time : NDArray
            Time vector in seconds (length N)

        Returns
        -------
        i_component : NDArray
            In-phase component (length N)
        q_component : NDArray
            Quadrature component (length N)
        amplitude : NDArray
            Demodulated amplitude R = √(I² + Q²)
        phase : NDArray
            Demodulated phase φ = arctan(Q/I) in radians

        Notes
        -----
        The lock-in process:
        1. Multiply input by reference: s(t)·cos(ωt) and s(t)·sin(ωt)
        2. Low-pass filter to remove 2ω component
        3. Extract DC components → I and Q channels
        """
        # Generate reference signals
        reference_cos = np.cos(2 * np.pi * self.reference_frequency * time)
        reference_sin = np.sin(2 * np.pi * self.reference_frequency * time)

        # Mixer: multiply input by reference
        # s(t)·cos(ωt) = [A·cos(ωt+φ)]·cos(ωt) = (A/2)·[cos(φ) + cos(2ωt+φ)]
        mixed_i = signal_in * reference_cos
        mixed_q = signal_in * reference_sin

        # Low-pass filter to remove 2ω component, keep DC
        i_component = signal.sosfiltfilt(self.sos, mixed_i)
        q_component = signal.sosfiltfilt(self.sos, mixed_q)

        # Amplitude and phase
        # Factor of 2 accounts for cos product identity: (A/2)·cos(φ) → A·cos(φ)
        i_component *= 2
        q_component *= 2

        amplitude = np.sqrt(i_component**2 + q_component**2)
        phase = np.arctan2(q_component, i_component)

        return i_component, q_component, amplitude, phase

    def compute_error_signal(
        self,
        signal_in: NDArray[np.float64],
        time: NDArray[np.float64],
        setpoint_phase: float = 0.0
    ) -> NDArray[np.float64]:
        """
        Compute error signal for feedback control.

        This method extracts the phase deviation from a setpoint, which
        can be used as input to a PID controller for resonator locking.

        Parameters
        ----------
        signal_in : NDArray
            Input signal (e.g., resonator transmission)
        time : NDArray
            Time vector in seconds
        setpoint_phase : float, optional
            Target phase in radians (default: 0.0)

        Returns
        -------
        error_signal : NDArray
            Phase error signal: Δφ = φ_measured - φ_setpoint
            Units: radians

        Notes
        -----
        For a ring resonator, the phase error is proportional to the
        wavelength detuning: Δφ ≈ (2π/FSR)·Δλ
        """
        _, _, _, phase = self.demodulate(signal_in, time)
        error_signal = phase - setpoint_phase

        # Unwrap phase to avoid 2π jumps
        error_signal = np.unwrap(error_signal)

        return error_signal


class PIDController:
    """
    PID controller for optical resonator stabilization.

    A PID (Proportional-Integral-Derivative) controller generates a control
    signal u(t) to minimize the error e(t) between measured and setpoint:
        u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt

    For resonator locking:
    - e(t) = phase error from lock-in detector
    - u(t) = heater current or piezo voltage to tune resonance

    Anti-windup is implemented to prevent integral term saturation.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: Tuple[float, float] = (-np.inf, np.inf),
        integral_limit: float = np.inf
    ) -> None:
        """
        Initialize PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain
        ki : float
            Integral gain
        kd : float
            Derivative gain
        output_limits : Tuple[float, float], optional
            Min and max output values (default: unbounded)
        integral_limit : float, optional
            Anti-windup limit for integral term (default: unbounded)

        Notes
        -----
        Typical gain values for ring resonator thermo-optic tuning:
        - Kp ~ 0.1-1.0 mW/rad
        - Ki ~ 0.01-0.1 mW/(rad·s)
        - Kd ~ 0.001-0.01 mW·s/rad
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limit = integral_limit

        # State variables
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller and compute control signal.

        Parameters
        ----------
        error : float
            Current error value (measured - setpoint)
        dt : float
            Time step since last update (seconds)

        Returns
        -------
        control_signal : float
            PID output u(t)

        Notes
        -----
        This method should be called at regular intervals (constant dt)
        for best performance. Typical update rates: 1-10 kHz.
        """
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        # Derivative term (using backward difference)
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        # Compute output
        control_signal = p_term + i_term + d_term

        # Apply output limits
        control_signal = np.clip(control_signal, *self.output_limits)

        # Update state
        self.previous_error = error

        return control_signal

    def reset(self) -> None:
        """
        Reset PID controller state.

        Clears integral accumulator and previous error. Use this when
        re-acquiring lock or after large disturbances.
        """
        self.integral = 0.0
        self.previous_error = 0.0


class ResonatorFeedback:
    """
    Complete feedback system for optical resonator stabilization.

    Combines digital lock-in detection with PID control to maintain
    resonator on resonance despite environmental drift.

    Typical workflow:
    1. Acquire transmission signal from photodetector
    2. Lock-in amplifier extracts phase error
    3. PID controller generates heater/piezo control signal
    4. Apply control signal to resonator (via DAC)
    5. Repeat at ~1-10 kHz update rate

    This is essential for LOC biosensing applications (bead twosphere-mcp-6ez)
    where thermal drift can shift resonances by multiple FSRs.
    """

    def __init__(
        self,
        lock_in: DigitalLockIn,
        pid: PIDController,
        setpoint_phase: float = 0.0
    ) -> None:
        """
        Initialize resonator feedback system.

        Parameters
        ----------
        lock_in : DigitalLockIn
            Lock-in amplifier instance
        pid : PIDController
            PID controller instance
        setpoint_phase : float, optional
            Target phase in radians (default: 0.0)
            Typically set to 0 (on resonance) or ±π/2 (side-of-fringe)
        """
        self.lock_in = lock_in
        self.pid = pid
        self.setpoint_phase = setpoint_phase

    def process_step(
        self,
        signal_in: NDArray[np.float64],
        time: NDArray[np.float64]
    ) -> Tuple[float, NDArray[np.float64]]:
        """
        Process one feedback iteration.

        Parameters
        ----------
        signal_in : NDArray
            Measured transmission signal
        time : NDArray
            Time vector

        Returns
        -------
        control_output : float
            Control signal to apply to resonator (heater current, piezo voltage)
        error_signal : NDArray
            Phase error signal for monitoring/diagnostics

        Notes
        -----
        This method should be called at regular intervals. The dt for PID
        is computed from the time vector: dt = time[-1] - time[0]
        """
        # Extract phase error using lock-in detection
        error_signal = self.lock_in.compute_error_signal(signal_in, time, self.setpoint_phase)

        # Use mean error over the time window for PID input
        # (alternatively, could use final value: error_signal[-1])
        mean_error = np.mean(error_signal)

        # Compute time step
        dt = time[-1] - time[0] if len(time) > 1 else 0.0

        # Update PID controller
        control_output = self.pid.update(mean_error, dt)

        return control_output, error_signal

    def reset(self) -> None:
        """
        Reset feedback system state.

        Resets PID integral accumulator. Use when re-acquiring lock.
        """
        self.pid.reset()
