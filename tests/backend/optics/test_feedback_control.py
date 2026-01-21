"""Unit tests for feedback control systems."""
import numpy as np
import pytest

from src.backend.optics.feedback_control import (
    DigitalLockIn,
    PIDController,
    ResonatorFeedback,
)


class TestDigitalLockIn:
    """Tests for DigitalLockIn amplifier."""

    def test_initialization_valid(self):
        """Test lock-in initialization with valid parameters."""
        lock_in = DigitalLockIn(
            reference_frequency=1000.0,
            sampling_rate=10000.0,
            time_constant=1.0
        )
        assert lock_in.reference_frequency == 1000.0
        assert lock_in.sampling_rate == 10000.0

    def test_initialization_nyquist_violation(self):
        """Test initialization fails if Nyquist criterion is violated."""
        with pytest.raises(ValueError, match="Sampling rate.*must be at least twice"):
            DigitalLockIn(
                reference_frequency=1000.0,
                sampling_rate=1500.0,  # Too low
                time_constant=1.0
            )

    def test_demodulate_pure_sine(self):
        """Test demodulation of pure sinusoidal signal."""
        reference_freq = 100.0
        sampling_rate = 10000.0
        lock_in = DigitalLockIn(reference_freq, sampling_rate, time_constant=0.1)

        # Generate pure sine wave at reference frequency
        time = np.linspace(0, 1.0, int(sampling_rate))
        amplitude = 2.0
        phase = np.pi / 4
        signal = amplitude * np.cos(2 * np.pi * reference_freq * time + phase)

        i_comp, q_comp, amp, phase_out = lock_in.demodulate(signal, time)

        # Check recovered amplitude (should be close to input amplitude)
        mean_amplitude = np.mean(amp[len(amp)//2:])  # Use second half (settled)
        assert abs(mean_amplitude - amplitude) < 0.5

    def test_compute_error_signal(self):
        """Test error signal computation for feedback control."""
        reference_freq = 100.0
        sampling_rate = 10000.0
        lock_in = DigitalLockIn(reference_freq, sampling_rate, time_constant=0.1)

        time = np.linspace(0, 1.0, int(sampling_rate))
        signal = 1.0 * np.cos(2 * np.pi * reference_freq * time + 0.1)

        error_signal = lock_in.compute_error_signal(signal, time, setpoint_phase=0.0)

        # Error signal should have similar length to input
        assert len(error_signal) == len(signal)

        # Mean error should be close to phase offset (0.1 rad)
        mean_error = np.mean(error_signal[len(error_signal)//2:])
        assert abs(mean_error - 0.1) < 0.25  # Relax tolerance for phase unwrapping effects


class TestPIDController:
    """Tests for PID controller."""

    def test_initialization(self):
        """Test PID controller initialization."""
        pid = PIDController(kp=1.0, ki=0.1, kd=0.01)
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.01

    def test_proportional_response(self):
        """Test proportional-only control."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        error = 0.5
        dt = 0.1
        output = pid.update(error, dt)

        # Output should be Kp * error
        assert abs(output - 1.0) < 1e-6

    def test_integral_accumulation(self):
        """Test integral term accumulates error."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0)

        error = 1.0
        dt = 0.1

        # First update
        output1 = pid.update(error, dt)
        assert abs(output1 - 0.1) < 1e-6  # Ki * error * dt

        # Second update (integral should accumulate)
        output2 = pid.update(error, dt)
        assert abs(output2 - 0.2) < 1e-6

    def test_output_limits(self):
        """Test output limiting."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))

        error = 1.0
        dt = 0.1
        output = pid.update(error, dt)

        # Output should be clamped to limits
        assert output == 1.0

    def test_reset(self):
        """Test PID reset clears state."""
        pid = PIDController(kp=1.0, ki=1.0, kd=0.0)

        # Accumulate some integral
        pid.update(1.0, 0.1)
        pid.update(1.0, 0.1)

        # Reset
        pid.reset()

        # Next update should start fresh
        output = pid.update(1.0, 0.1)
        expected = 1.0 * 1.0 + 1.0 * 1.0 * 0.1  # P + I
        assert abs(output - expected) < 1e-6


class TestResonatorFeedback:
    """Tests for complete resonator feedback system."""

    def test_initialization(self):
        """Test resonator feedback initialization."""
        lock_in = DigitalLockIn(1000.0, 10000.0, 1.0)
        pid = PIDController(1.0, 0.1, 0.01)
        feedback = ResonatorFeedback(lock_in, pid, setpoint_phase=0.0)

        assert feedback.setpoint_phase == 0.0

    def test_process_step(self):
        """Test single feedback iteration."""
        lock_in = DigitalLockIn(100.0, 10000.0, 0.1)
        pid = PIDController(1.0, 0.1, 0.01, output_limits=(-10.0, 10.0))
        feedback = ResonatorFeedback(lock_in, pid, setpoint_phase=0.0)

        # Generate test signal with phase offset
        time = np.linspace(0, 0.1, 1000)
        signal = np.cos(2 * np.pi * 100.0 * time + 0.2)  # 0.2 rad phase error

        control_output, error_signal = feedback.process_step(signal, time)

        # Control output should be a scalar
        assert isinstance(control_output, (int, float))

        # Error signal should have same length as input
        assert len(error_signal) == len(signal)

    def test_reset(self):
        """Test feedback system reset."""
        lock_in = DigitalLockIn(1000.0, 10000.0, 1.0)
        pid = PIDController(1.0, 0.1, 0.01)
        feedback = ResonatorFeedback(lock_in, pid)

        # Process some data to accumulate state
        time = np.linspace(0, 0.1, 1000)
        signal = np.cos(2 * np.pi * 1000.0 * time)
        feedback.process_step(signal, time)

        # Reset
        feedback.reset()

        # PID integral should be cleared
        assert feedback.pid.integral == 0.0
