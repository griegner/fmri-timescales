import numpy as np
import scipy as sp
from statsmodels.tsa.arima_process import arma_acf


def acf_fft(X: np.ndarray, n_timepoints: int) -> np.ndarray:
    """Estimate the full-lag auto-correlation function (ACF) of an array of timeseries using the Fast Fourier Transform
    -- O(n_timepoints log n_timepoints) complexity instead of O(n_timepoints^2) for the time-domain method.

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
    n_fft = 2 ** int(np.ceil(np.log2(2 * n_timepoints - 1)))  # zero-pad
    X_fft = np.fft.rfft(X, n=n_fft, axis=1)  # frequency domain
    X_acov = np.fft.irfft(X_fft * np.conj(X_fft), axis=1)[:, :n_timepoints]  # time domain
    X_avar = np.sum(X**2, axis=1)  # auto-variances
    X_acf = X_acov / X_avar.reshape(-1, 1)  # auto-covariances > auto-correlations

    return X_acf


def acf_to_toeplitz(acf: np.ndarray, n_timepoints: int) -> np.ndarray:
    """Converts an auto-correlation function (ACF) to a Toeplitz matrix for one or multiple timeseries.

    Parameters
    ----------
    acf : np.ndarray of shape (n_regions, n_lags) or (n_lags,)
        The auto-correlation function.
        If the shape is (n_lags,), all timeseries will have the same ACF.
        If the shape is (n_regions, n_lags), each timeseries will have a different ACF.
    n_timepoints : int
        Number of timepoints/samples.

    Returns
    -------
    np.ndarray of shape (n_regions, n_timepoints, n_timepoints) or (n_timepoints, n_timepoints)
        The Toeplitz matrix for each timeseries or a single Toeplitz matrix if all timeseries have the same ACF.
        If `n_lags < n_timepoints`, the ACF will be padded with zeros.

    Raises
    ------
    ValueError
        If `acf` is not in (n_timepoints,) or (n_regions, n_timepoints) form.
    """

    if acf.ndim != 1 and acf.ndim != 2:
        raise ValueError("acf should be in ([n_regions], n_timepoints) form")

    if acf.ndim == 1:
        acorr = np.pad(acf, (0, n_timepoints - acf.size))
        return sp.linalg.toeplitz(acorr)

    else:
        acorrs = []
        for region in acf:
            acorr = np.pad(region, (0, n_timepoints - region.size))
            acorrs.append(sp.linalg.toeplitz(acorr))
        return np.array(acorrs)


def ar_to_acf(ar_coeffs: np.ndarray, n_lags: int = 10) -> np.ndarray:
    """Calculates the theoretical autocorrelation function of an AutoRegressive (AR) process.

    Parameters
    ----------
    ar_coeffs : list or ndarray of shape (p,)
        A list of AR(p) coefficients in the form [phi_1, phi_2, ..., phi_p], excluding phi_0.
    n_lags : int, optional
        Number of lags, by default 10

    Returns
    -------
    np.ndarray
        The autocorrelation function for the given AR(p) coefficients up to `n_lags`.
    """
    if isinstance(ar_coeffs, list):
        ar_coeffs = np.array(ar_coeffs)
    return arma_acf(ar=np.r_[1, -ar_coeffs], ma=[1], lags=n_lags)
