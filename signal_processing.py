# signal processing.

import numpy as np


def align_signals(signals, peak_index):
    """
    Align multiple signals by computing time shifts relative to the peak of the first signal.

    Parameters:
        signals (list): List of 1D NumPy arrays representing the signals.
        peak_index (int): Index of the peak in the first signal.

    Returns:
        aligned_signals (list): List of 1D NumPy arrays representing the aligned signals.
    """
    # Calculate the time shift for each signal relative to the peak of the first signal
    shifts = []
    for i in range(1, len(signals)):
        cross_corr = np.correlate(signals[0], signals[i], mode="full")
        shift = np.argmax(cross_corr) - (len(signals[0]) - 1)
        shifts.append(shift)

    # Apply the shifts to align the signals
    aligned_signals = [signals[0]]  # Add the reference signal which is already aligned
    for i, signal in enumerate(signals[1:]):
        aligned_signal = np.roll(signal, shifts[i])
        aligned_signals.append(aligned_signal)

    return aligned_signals


def compute_stats(vectors):
    """
    Compute statistics for a list of vectors.

    Parameters:
        vectors (list): List of 1D NumPy arrays representing the vectors.

    Returns:
        average_vector (ndarray): Average vector.
        std_dev (ndarray): Standard deviation of each element in the vectors.
        average_shift (float or None): Average phase shift (if applicable).
        std_shift (float or None): Standard deviation of phase shifts (if applicable).
    """
    # Compute the average vector
    average_vector = np.mean(vectors, axis=0)

    # Compute standard deviation
    std_dev = np.std(vectors, axis=0, ddof=1)  # ddof=1 for sample standard deviation

    # If phase shifts have been calculated, compute their average and SD
    if "phase_shifts" in globals():
        average_shift = np.mean(phase_shifts)
        std_shift = np.std(phase_shifts, ddof=1)
    else:
        average_shift, std_shift = None, None

    return average_vector, std_dev, average_shift, std_shift


def compute_pairwise_correlation_fft(signals):
    """
    Compute pairwise correlation between FFTs of multiple signals.

    Parameters:
        signals (list): List of 1D NumPy arrays representing the signals.

    Returns:
        correlations (ndarray): Pairwise correlation coefficients between FFTs.
    """
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
            corr_coef = np.corrcoef(fft_i, fft_j)[0, 1]
            correlations[i, j] = corr_coef
            correlations[j, i] = corr_coef
    return correlations
