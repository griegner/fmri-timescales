from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator

from fmri_timescales import acf_utils


def newey_west_omega(u: np.ndarray, n_lags: Optional[int] = None) -> float:
    n_u = len(u)
    if n_lags is None:
        n_lags = int(np.floor(4 * (n_u / 100.0) ** (2 / 9)))
    weights = 1 - np.arange(n_lags + 1) / (n_lags + 1)
    omega = weights[0] * np.sum(u**2)
    for lag in range(1, n_lags + 1):
        omega += weights[lag] * (2 * np.sum(u[lag:] * u[:-lag]))
    return omega


def _phi_to_tau(phis: np.ndarray, se_phis: np.ndarray) -> tuple:
    """phi to tau (timescale), and apply delta method to std err"""
    phis_abs = np.abs(phis)  # tau undefined for negative phi
    taus = -1.0 / np.log(phis_abs)
    se_taus = (1.0 / (phis_abs * np.log(phis_abs) ** 2)) * se_phis
    return taus, se_taus


class LLS(BaseEstimator):
    """Time Domain Linear Model, Fit by Linear Least Squares (LLS).

    Parameters
    ----------
    var_estimator : str, optional
        The variance estimator to use. Options are "newey-west" or "non-robust", by default "newey-west"
    var_n_lags : int, optional
        The lag truncation number for the bartlett kernel, by default None
    X_sfreq : float, optional
        The sampling frequency in Hz, by default 1
    copy_X : bool, optional
        If True X will be copied, else it may be overwritten, by default False
    n_jobs : int, optional
        The number of jobs to use for the computation, by default None

    Attributes
    ----------
    estimates_ : dict
    A dictionary containing four np.ndarray of shape (n_regions, ):
    - "phi": AR(1) coefficient estimates, for each region in X.
    - "se(phi)": Standard errors of AR(1) coefficients.
    - "tau": Timescale estimates, for each region in X.
    - "se(tau)": Standard errors of timescales.

    Examples
    --------
    >>> from fmri_timescales import sim, timescale_utils
    >>> X = sim.sim_ar(ar_coeffs=[0.8], n_timepoints=1000) # x_t = 0.8 x_{t-1} + e_t
    >>> lls = timescale_utils.LLS(var_estimator="newey-west", var_n_lags=10)
    >>> lls.fit(X=X, n_timepoints=1000).estimates_
    {'phi': array([0.79789847]), 'se(phi)': array([0.02045074]), 'tau': array([4.42920958]), 'se(tau)': array([0.50282146])}
    """

    def __init__(
        self,
        var_estimator: str = "newey-west",
        var_n_lags: Optional[int] = None,
        X_sfreq: Optional[float] = 1.0,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.var_estimator = var_estimator
        self.var_n_lags = var_n_lags
        self.X_sfreq = X_sfreq
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    @delayed
    def _fit_lls(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries x in X"""
        T = len(x) - 1
        lag = int(self.X_sfreq)
        # x_t = X[1:], x_{t-1} = x[:-1] (Hz=1)
        phi_ = np.sum(x[lag:] * x[:-lag]) / np.sum(x[:-lag] ** 2)

        # variance estimators
        def non_robust():
            e_ = x[lag:] - phi_ * x[:-lag]
            q_ = np.sum(x[:-lag] ** 2)
            sigma2_ = (1 / T) * np.sum(e_**2)
            return (1 / q_) * sigma2_

        def newey_west():
            e_ = x[lag:] - phi_ * x[:-lag]
            q_ = np.sum(x[:-lag] ** 2)
            u_ = x[:-lag] * e_
            omega_ = newey_west_omega(u_, n_lags=self.var_n_lags)
            return (1 / q_) * omega_ * (1 / q_)

        var_estimators = {"non-robust": non_robust, "newey-west": newey_west}

        if self.var_estimator not in var_estimators:
            raise ValueError("var_estimator must be either 'newey-west' or 'non-robust'")

        var_ = var_estimators[self.var_estimator]()
        return phi_, np.sqrt(var_)

    def fit(self, X: np.ndarray, n_timepoints: int):
        """Fit the LLS model.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_regions)
            An array containing the timeseries of each region.
        n_timepoints : int
            The number of timepoints in X.

        Raises
        ------
        ValueError
            If `X` is not in (n_timepoints, n_regions) form.
        """

        if X.ndim != 2 or X.shape[0] != n_timepoints:
            raise ValueError("X should be in (n_timepoints, n_regions) form")
        X = X.copy() if self.copy_X else X
        X = (X - X.mean(axis=0)) / X.std(axis=0)  # mean zero, variance 1

        with Parallel(n_jobs=self.n_jobs) as parallel:
            lls_fits = parallel(self._fit_lls(X[:, idx]) for idx in range(X.shape[1]))
            phis_, se_phis_ = map(np.array, zip(*lls_fits))
            taus_, se_taus_ = _phi_to_tau(phis_, se_phis_)

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self


class NLS(BaseEstimator):
    """Autocorrelation Domain Nonlinear Model, fit by Nonlinear Least Squares (NLS).

    Parameters
    ----------
    var_estimator : str, optional
        The variance estimator to use. Options are "newey-west" or "non-robust", by default "newey-west"
    var_domain : str, optional
        The domain to fit variance estimator. Options are "time" or "autocorrelation", be default "time"\n
        "autocorrelation" is required if X_domain="autocorrelation"
    var_n_lags : int, optional
        The lag truncation number for the bartlett kernel, by default None
    acf_n_lags : int, optional
        The lag truncation number for the autocorrelation function, by default None
    X_domain : str, optional
        The domain of X. Options are "time" or "autocorrelation", be default "time"
    X_sfreq : float, optional
        The sampling frequency in Hz, by default 1
    copy_X : bool, optional
        If True X will be copied, else it may be overwritten, by default False
    n_jobs : _type_, optional
        The number of jobs to use for the computation, by default None

    Attributes
    ----------
    estimates_ : dict
    A dictionary containing two np.ndarray of shape (n_regions, ):
    - "tau": Timescale estimates, for each region in X.
    - "se(tau)": Standard errors of timescales.

    Examples
    --------
    >>> from fmri_timescales import sim, timescale_utils
    >>> X = sim.sim_ar(ar_coeffs=[0.8], n_timepoints=1000) # x_t = 0.8 x_{t-1} + e_t
    >>> nls = timescale_utils.NLS(var_domain="time", var_estimator="newey-west", var_n_lags=10)
    >>> nls.fit(X=X, n_timepoints=1000).estimates_
    {'phi': array([0.7802222]), 'se(phi)': array([0.03207284]), 'tau': array([4.02938991]), 'se(tau)': array([0.66741761])}
    """

    def __init__(
        self,
        var_estimator: str = "newey-west",
        var_domain: str = "time",
        var_n_lags: Optional[int] = None,
        acf_n_lags: Optional[int] = None,
        X_domain: str = "time",
        X_sfreq: Optional[float] = 1.0,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.var_estimator = var_estimator
        self.var_domain = var_domain
        self.var_n_lags = var_n_lags
        self.acf_n_lags = acf_n_lags
        self.X_domain = X_domain
        self.X_sfreq = X_sfreq
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    @delayed
    def _fit_nls(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries/autocorrelation function x in X"""

        if self.X_domain == "time":
            T = len(x)
            x_acf = acf_utils.ACF(n_lags=self.acf_n_lags).fit_transform(x.reshape(-1, 1), T).squeeze()
            K = len(x_acf)
        elif self.X_domain == "autocorrelation":
            K = len(x)
            x_acf = x
        else:
            raise ValueError("X_domain must be 'time' or 'autocorrelation'")

        # define the regression function (m), and its linearized regressor (dm_dphi)
        ks = np.linspace(0, K - 1, K) * (1.0 / self.X_sfreq)
        m = lambda ks, phi: phi**ks
        dm_dphi = lambda ks, phi: ks * phi ** (ks - 1)
        jac = lambda ks, phi: dm_dphi(ks, phi + 1e-10).reshape(-1, 1)

        # phi estimator
        phi_, _ = curve_fit(f=m, xdata=ks, ydata=x_acf, p0=1e-2, bounds=(-1, +1), ftol=1e-6, jac=jac)
        phi_ = phi_.squeeze()

        lag = int(self.X_sfreq + 1)

        # variance estimators
        def non_robust_time():
            e_ = x[lag:] - phi_ * x[:-lag]
            q_ = np.sum(x[:-lag] ** 2)
            sigma2_ = (1 / T) * np.sum(e_**2)
            return (1 / q_) * sigma2_

        def newey_west_time():
            e_ = x[lag:] - phi_ * x[:-lag]
            q_ = np.sum(x[:-lag] ** 2)
            u_ = x[:-lag] * e_
            omega_ = newey_west_omega(u_, n_lags=self.var_n_lags)
            return (1 / q_) * omega_ * (1 / q_)

        def non_robust_autocorrelation():
            e_ = x_acf - phi_**ks
            q_ = np.sum(dm_dphi(ks, phi_) ** 2)
            sigma2_ = (1 / K) * np.sum(e_**2)
            return (1 / q_) * sigma2_

        def newey_west_autocorrelation():
            e_ = x_acf - phi_**ks
            q_ = np.sum(dm_dphi(ks, phi_) ** 2)
            u_ = dm_dphi(ks, phi_) * e_
            omega_ = newey_west_omega(u_, n_lags=self.var_n_lags)
            return (1 / q_) * omega_ * (1 / q_)

        var_estimators = {
            ("non-robust", "time"): non_robust_time,
            ("newey-west", "time"): newey_west_time,
            ("non-robust", "autocorrelation"): non_robust_autocorrelation,
            ("newey-west", "autocorrelation"): newey_west_autocorrelation,
        }

        if (self.var_estimator, self.var_domain) not in var_estimators:
            raise ValueError(
                "var_estimator must be either 'newey-west' or 'non-robust'\n"
                "var_domain must be either 'time' or 'autocorrelation'"
            )

        if self.X_domain == "autocorrelation" and self.var_domain == "time":
            raise ValueError("var_domain='time' incompatible with X_domain='autocorrelation'.")

        var_ = var_estimators[(self.var_estimator, self.var_domain)]()
        return phi_, np.sqrt(var_)

    def fit(self, X: np.ndarray, n_timepoints: int):
        """Fit the NLS model.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_regions) if X_domain='time'\n
            or shape (n_lags, n_regions) if X_domain='autocorrelation'
        n_features : int
            The number of timepoints/lags in X.

        Raises
        ------
        ValueError
            If `X` is not in (n_timepoints, n_regions) form.
        """
        if X.ndim != 2 or X.shape[0] != n_timepoints:
            raise ValueError("X should be in (n_timepoints, n_regions) form")
        X = X.copy() if self.copy_X else X

        X = (X - X.mean(axis=0)) / X.std(axis=0) if self.X_domain == "time" else X

        with Parallel(n_jobs=self.n_jobs) as parallel:
            nls_fits = parallel(self._fit_nls(X[:, idx]) for idx in range(X.shape[1]))
            phis_, se_phis_ = map(np.array, zip(*nls_fits))
            taus_, se_taus_ = _phi_to_tau(phis_, se_phis_)

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self
