"""FFT-based correlation analysis for MRI signal processing.

Implements frequency-domain correlation for functional connectivity analysis
between brain regions.
"""

from typing import Tuple, Optional, Dict
import numpy as np

__all__ = ['compute_fft_correlation', 'cross_spectrum', 'coherence', 
           'phase_correlation', 'CorrelationResult']


class CorrelationResult:
    """Container for FFT correlation results."""
    
    def __init__(self, correlation: np.ndarray, peak_value: float,
                 peak_lag: int, sampling_rate: float):
        self.correlation = correlation
        self.peak_value = peak_value
        self.peak_lag = peak_lag
        self.sampling_rate = sampling_rate
    
    @property
    def peak_lag_seconds(self) -> float:
        """Peak lag in seconds."""
        return self.peak_lag / self.sampling_rate
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'peak_value': self.peak_value,
            'peak_lag_samples': self.peak_lag,
            'peak_lag_seconds': self.peak_lag_seconds,
            'correlation_length': len(self.correlation)
        }


def compute_fft_correlation(signal_a: np.ndarray, signal_b: np.ndarray,
                            sampling_rate: float = 1000.0,
                            normalize: bool = True) -> CorrelationResult:
    """Compute cross-correlation using FFT.
    
    This method computes correlation in the frequency domain, which is
    efficient for long signals and reveals phase relationships.
    
    Args:
        signal_a: First time-series signal
        signal_b: Second time-series signal
        sampling_rate: Sampling rate in Hz
        normalize: Whether to normalize the correlation
        
    Returns:
        CorrelationResult with correlation values and metrics
    """
    # Ensure same length
    min_len = min(len(signal_a), len(signal_b))
    a = np.array(signal_a[:min_len])
    b = np.array(signal_b[:min_len])
    
    # Remove mean
    a = a - np.mean(a)
    b = b - np.mean(b)
    
    # Compute FFT
    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    
    # Cross-spectrum (a * conj(b))
    cross_spec = fft_a * np.conj(fft_b)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft(cross_spec).real
    
    # Normalize if requested
    if normalize:
        norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if norm > 0:
            correlation = correlation / norm
    
    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_lag = peak_idx if peak_idx < min_len // 2 else peak_idx - min_len
    peak_value = float(np.abs(correlation[peak_idx]))
    
    return CorrelationResult(correlation, peak_value, peak_lag, sampling_rate)


def cross_spectrum(signal_a: np.ndarray, signal_b: np.ndarray,
                   sampling_rate: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-spectrum between two signals.
    
    Args:
        signal_a: First signal
        signal_b: Second signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies, cross_spectrum_magnitude)
    """
    min_len = min(len(signal_a), len(signal_b))
    a = np.array(signal_a[:min_len])
    b = np.array(signal_b[:min_len])
    
    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    cross = fft_a * np.conj(fft_b)
    
    freqs = np.fft.fftfreq(min_len, 1/sampling_rate)
    
    # Return positive frequencies only
    pos_mask = freqs >= 0
    return freqs[pos_mask], np.abs(cross[pos_mask])


def coherence(signal_a: np.ndarray, signal_b: np.ndarray,
              sampling_rate: float = 1000.0,
              segment_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coherence between two signals.
    
    Coherence measures the linear correlation between signals as a function
    of frequency, normalized to be between 0 and 1.
    
    Args:
        signal_a: First signal
        signal_b: Second signal
        sampling_rate: Sampling rate in Hz
        segment_length: Length of segments for averaging (None = no averaging)
        
    Returns:
        Tuple of (frequencies, coherence_values)
    """
    min_len = min(len(signal_a), len(signal_b))
    a = np.array(signal_a[:min_len])
    b = np.array(signal_b[:min_len])
    
    if segment_length is None:
        segment_length = min_len
    
    # Simple implementation without segment averaging
    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    
    # Power spectra
    Paa = np.abs(fft_a)**2
    Pbb = np.abs(fft_b)**2
    Pab = fft_a * np.conj(fft_b)
    
    # Coherence = |Pab|^2 / (Paa * Pbb)
    denom = Paa * Pbb
    coh = np.zeros_like(denom)
    nonzero = denom > 0
    coh[nonzero] = np.abs(Pab[nonzero])**2 / denom[nonzero]
    
    freqs = np.fft.fftfreq(min_len, 1/sampling_rate)
    pos_mask = freqs >= 0
    
    return freqs[pos_mask], coh[pos_mask]


def phase_correlation(signal_a: np.ndarray, signal_b: np.ndarray,
                      sampling_rate: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute phase difference between signals as a function of frequency.
    
    Args:
        signal_a: First signal
        signal_b: Second signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (frequencies, phase_difference_radians)
    """
    min_len = min(len(signal_a), len(signal_b))
    a = np.array(signal_a[:min_len])
    b = np.array(signal_b[:min_len])
    
    fft_a = np.fft.fft(a)
    fft_b = np.fft.fft(b)
    
    # Phase difference
    phase_diff = np.angle(fft_a) - np.angle(fft_b)
    
    # Wrap to [-pi, pi]
    phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
    
    freqs = np.fft.fftfreq(min_len, 1/sampling_rate)
    pos_mask = freqs >= 0
    
    return freqs[pos_mask], phase_diff[pos_mask]
