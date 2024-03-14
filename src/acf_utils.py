from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg, signal
from statsmodels.tsa.arima_process import arma_acf


class ACF:
    """Empirical Autocorrelation Function (ACF)

    Parameters
    ----------
    n_lags : int, optional
        Number of lags to return autocorrelation for, by default None.
        If None, the full lag ACF will be returned
    copy_X : bool, optional
        If True X will be copied, else it may be overwritten, by default False
    n_jobs : int, optional
        The number of jobs to use for the computation, by default None
    """

    def __init__(
        self,
        n_lags: Optional[int] = None,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_lags = n_lags
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.acorr_ = None

    def fit_transform(self, X: np.ndarray, n_timepoints: int) -> np.ndarray:
        """Fit using the Fast Fourier Transform: O(n log n) time complexity.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_regions)
            An array containing the timeseries of each region
        n_timepoints : int
            The number of timepoints in X

        Returns
        ------
        X_acf_ : np.ndarray of shape (n_lags, n_regions)
            An array containing the empirical ACF for each region in X

        Raises
        ------
        ValueError
            If X is not in (n_timepoints, n_regions) form.
        """

        if X.shape[0] != n_timepoints:
            raise ValueError("X should be in (n_timepoints, n_regions) form")
        X = X.copy() if self.copy_X else X
        X = (X - X.mean(axis=0)) / X.std(axis=0)  # mean zero, variance one

        X_acov_ = Parallel(n_jobs=self.n_jobs)(
            delayed(signal.convolve)(X[:, idx], X[::-1, idx], mode="full", method="fft")
            for idx in range(X.shape[1])
        )
        X_acf_ = np.array(X_acov_).T[n_timepoints - 1 :, : self.n_lags] / n_timepoints
        return X_acf_


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
        return linalg.toeplitz(acorr)

    else:
        acorrs = []
        for region in acf:
            acorr = np.pad(region, (0, n_timepoints - region.size))
            acorrs.append(linalg.toeplitz(acorr))
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
