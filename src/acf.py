import numpy as np


def acf_fft(X, n_timepoints):
    """Estimate the full-lag auto-correlation function (ACF) of a matrix of timeseries using the Fast Fourier Transform.

    Parameters
    ----------
    X : ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_timepoints : int
        Number of timepoints/samples.

    Returns
    -------
    ndarray of shape (n_regions, n_timepoints)
        The full-lag ACF of each timeseries.

    Raises
    ------
    ValueError
        If `X` is not in (n_regions, n_timepoints) form.
    """

    if X.shape[1] != n_timepoints:
        raise ValueError("X should be in (n_regions, n_timepoints) form")

    X -= X.mean(axis=1, keepdims=True)  # mean center timeseries
    X_fft = np.fft.rfft(X, axis=1)  # frequency domain
    X_acov = np.fft.irfft(X_fft * np.conj(X_fft), axis=1)  # time domain
    X_avar = np.sum(X**2, axis=1)  # auto-variances
    X_acf = X_acov / X_avar.reshape(-1, 1)  # auto-covariances > auto-correlations

    return X_acf
