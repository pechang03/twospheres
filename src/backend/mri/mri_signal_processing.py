"""MRI signal processing for functional connectivity analysis.

Based on MRISpheres/twospheres signal_processing.py and Lu et al. (2019)
"Abnormal intra-network architecture in extra-striate cortices in amblyopia".

Implements:
- Signal alignment via cross-correlation
- FFT-based pairwise correlation
- Distance correlation for multivariate functional connectivity
"""

import asyncio
from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray


async def align_signals(
    signals: List[NDArray],
    peak_index: Optional[int] = None
) -> List[NDArray]:
    """
    Align multiple signals using cross-correlation.

    Computes time shifts relative to the first signal and aligns all
    subsequent signals. Used for MRI time-series alignment across brain regions.

    Args:
        signals: List of 1D arrays (time-series from brain regions)
        peak_index: Optional peak index in first signal (unused, for compatibility)

    Returns:
        List of aligned signals

    Example:
        >>> v1_signal = np.sin(2*np.pi*0.1*np.arange(100))
        >>> v4_signal = np.sin(2*np.pi*0.1*np.arange(100) + 10)  # Shifted
        >>> aligned = await align_signals([v1_signal, v4_signal])
        >>> # aligned[1] is now synchronized with aligned[0]
    """
    def _align():
        # Calculate time shift for each signal relative to first
        shifts = []
        for i in range(1, len(signals)):
            cross_corr = np.correlate(signals[0], signals[i], mode="full")
            shift = np.argmax(cross_corr) - (len(signals[0]) - 1)
            shifts.append(shift)

        # Apply shifts
        aligned_signals = [signals[0]]  # Reference signal
        for i, signal in enumerate(signals[1:]):
            aligned_signal = np.roll(signal, shifts[i])
            aligned_signals.append(aligned_signal)

        return aligned_signals

    return await asyncio.to_thread(_align)


async def compute_pairwise_correlation_fft(
    signals: List[NDArray]
) -> NDArray:
    """
    Compute pairwise correlation between FFTs of multiple signals.

    Frequency-domain correlation measurement used for functional connectivity
    analysis. Equivalent to measuring phase synchronization across frequencies.

    Args:
        signals: List of 1D arrays (MRI time-series)

    Returns:
        NxN correlation matrix (symmetric, values in [-1, 1])

    Example:
        >>> # Two brain regions with synchronized oscillations
        >>> region1 = np.sin(2*np.pi*0.1*np.arange(100))
        >>> region2 = np.sin(2*np.pi*0.1*np.arange(100) + np.pi/4)
        >>> corr_matrix = await compute_pairwise_correlation_fft([region1, region2])
        >>> # corr_matrix[0,1] > 0.8 indicates strong functional connectivity
    """
    def _compute():
        # Compute FFTs for each signal
        fft_signals = [np.fft.fft(signal) for signal in signals]

        # Initialize correlation matrix
        num_signals = len(signals)
        correlations = np.zeros((num_signals, num_signals))

        # Compute pairwise correlation coefficients
        for i in range(num_signals):
            for j in range(i + 1, num_signals):
                fft_i = fft_signals[i]
                fft_j = fft_signals[j]
                # Use real parts for correlation (magnitude correlation)
                corr_coef = np.corrcoef(np.abs(fft_i), np.abs(fft_j))[0, 1]
                correlations[i, j] = corr_coef
                correlations[j, i] = corr_coef

        # Diagonal is self-correlation (always 1.0)
        np.fill_diagonal(correlations, 1.0)

        return correlations

    return await asyncio.to_thread(_compute)


async def compute_stats(
    vectors: List[NDArray]
) -> Tuple[NDArray, NDArray]:
    """
    Compute statistics for multiple vectors.

    Args:
        vectors: List of 1D arrays (e.g., time-series or embeddings)

    Returns:
        Tuple of (mean_vector, std_dev_vector)

    Example:
        >>> signals = [np.random.randn(100) for _ in range(10)]
        >>> mean, std = await compute_stats(signals)
    """
    def _compute():
        # Compute mean and standard deviation
        average_vector = np.mean(vectors, axis=0)
        std_dev = np.std(vectors, axis=0, ddof=1)  # Sample std dev

        return average_vector, std_dev

    return await asyncio.to_thread(_compute)


async def compute_distance_correlation(
    region_a: NDArray,
    region_b: NDArray
) -> float:
    """
    Compute distance correlation between two brain regions.

    Based on Lu et al. (2019) multivariate distance correlation method.
    More robust than Pearson correlation for detecting non-linear dependencies
    and uses multi-voxel patterns instead of averaged BOLD signals.

    Distance correlation formula:
        dCor(A, B) = dCov(A, B) / sqrt(dVar(A) * dVar(B))

    Where:
        dCov = distance covariance
        dVar = distance variance

    Args:
        region_a: Brain region A data, shape (n_voxels_A, n_timepoints)
        region_b: Brain region B data, shape (n_voxels_B, n_timepoints)

    Returns:
        Distance correlation value in [0, 1]

    References:
        Székely et al. (2007) "Measuring and testing dependence by correlation of distances"
        Lu et al. (2019) - GRETNA toolbox implementation

    Example:
        >>> v1_data = np.random.randn(100, 400)  # 100 voxels, 400 timepoints
        >>> v4_data = np.random.randn(80, 400)   # 80 voxels, 400 timepoints
        >>> # Add shared signal (simulate functional connectivity)
        >>> shared = np.random.randn(400)
        >>> v1_data += 0.5 * shared
        >>> v4_data += 0.5 * shared
        >>> dCor = await compute_distance_correlation(v1_data, v4_data)
        >>> # dCor > 0.3 indicates functional connectivity
    """
    def _compute():
        # Ensure correct shape (voxels x timepoints)
        if region_a.ndim == 1:
            region_a_2d = region_a.reshape(1, -1)
        else:
            region_a_2d = region_a

        if region_b.ndim == 1:
            region_b_2d = region_b.reshape(1, -1)
        else:
            region_b_2d = region_b

        n_timepoints = region_a_2d.shape[1]

        # Compute Euclidean distance matrices between timepoints
        # For region A: d_A(t1, t2) = ||voxels_A(t1) - voxels_A(t2)||
        def compute_distance_matrix(data):
            n_time = data.shape[1]
            dist_matrix = np.zeros((n_time, n_time))
            for t1 in range(n_time):
                for t2 in range(t1 + 1, n_time):
                    dist = np.linalg.norm(data[:, t1] - data[:, t2])
                    dist_matrix[t1, t2] = dist
                    dist_matrix[t2, t1] = dist
            return dist_matrix

        dist_a = compute_distance_matrix(region_a_2d)
        dist_b = compute_distance_matrix(region_b_2d)

        # U-centering: set row and column means to zero
        def u_center(dist_matrix):
            n = dist_matrix.shape[0]
            # Row means
            row_means = dist_matrix.mean(axis=1, keepdims=True)
            # Column means
            col_means = dist_matrix.mean(axis=0, keepdims=True)
            # Grand mean
            grand_mean = dist_matrix.mean()

            # U-centered matrix
            centered = dist_matrix - row_means - col_means + grand_mean
            return centered

        centered_a = u_center(dist_a)
        centered_b = u_center(dist_b)

        # Distance covariance
        n = n_timepoints
        d_cov_ab = np.sqrt(np.sum(centered_a * centered_b) / (n * n))

        # Distance variances
        d_var_a = np.sqrt(np.sum(centered_a * centered_a) / (n * n))
        d_var_b = np.sqrt(np.sum(centered_b * centered_b) / (n * n))

        # Distance correlation
        if d_var_a > 0 and d_var_b > 0:
            d_cor = d_cov_ab / np.sqrt(d_var_a * d_var_b)
        else:
            d_cor = 0.0

        return float(np.clip(d_cor, 0.0, 1.0))

    return await asyncio.to_thread(_compute)


async def compute_phase_locking_value(
    signal1: NDArray,
    signal2: NDArray
) -> float:
    """
    Compute Phase-Locking Value (PLV) between two signals.

    PLV measures phase synchronization between brain regions:
        PLV = (1/N) * |Σ exp(i * (φ1(t) - φ2(t)))|

    Args:
        signal1: Time-series from brain region 1
        signal2: Time-series from brain region 2

    Returns:
        PLV in [0, 1], where 1 = perfect phase-locking

    Example:
        >>> # Two signals with strong phase-locking
        >>> t = np.linspace(0, 10, 1000)
        >>> sig1 = np.sin(2*np.pi*5*t)
        >>> sig2 = np.sin(2*np.pi*5*t + np.pi/4)  # Phase offset
        >>> plv = await compute_phase_locking_value(sig1, sig2)
        >>> # plv ≈ 1.0 (perfect synchronization)
    """
    def _compute():
        # Extract phases using Hilbert transform
        from scipy.signal import hilbert

        analytic1 = hilbert(signal1)
        analytic2 = hilbert(signal2)

        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)

        # Phase difference
        phase_diff = phase1 - phase2

        # PLV formula
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        return float(plv)

    return await asyncio.to_thread(_compute)
