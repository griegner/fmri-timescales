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


def _phi_to_tau(phis: np.ndarray, se_phis: np.ndarray, lag: int = 1) -> tuple:
    """phi to tau (timescale), and apply delta method to std err"""
    phis_abs = np.abs(phis)  # tau undefined for negative phi
    taus = -lag / np.log(phis_abs)
    se_taus = (lag / (phis_abs * np.log(phis_abs) ** 2)) * se_phis
    return taus, se_taus


class TD(BaseEstimator):
    """Time Domain (TD) Linear Model, Fit by Linear Least Squares.

    Parameters
    ----------
    var_estimator : str, optional
        The variance estimator to use. Options are "newey-west" or "non-robust", by default "newey-west"
    var_n_lags : int, optional
        The lag truncation number for the bartlett kernel, by default None
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
    >>> td = timescale_utils.TD(var_estimator="newey-west", var_n_lags=10)
    >>> td.fit(X=X, n_timepoints=1000).estimates_
    {'phi': array([0.79789847]), 'se(phi)': array([0.02045074]), 'tau': array([4.42920958]), 'se(tau)': array([0.50282146])}
    """

    def __init__(
        self,
        var_estimator: str = "newey-west",
        var_n_lags: Optional[int] = None,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.var_estimator = var_estimator
        self.var_n_lags = var_n_lags
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    @delayed
    def _fit_td(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries x in X"""
        T = len(x) - 1
        # x_t = X[1:], x_{t-1} = x[:-1] (Hz=1)
        phi_ = np.sum(x[1:] * x[:-1]) / np.sum(x[:-1] ** 2)

        # variance estimators
        def non_robust():
            e_ = x[1:] - phi_ * x[:-1]
            q_ = np.sum(x[:-1] ** 2)
            sigma2_ = (1 / T) * np.sum(e_**2)
            return (1 / q_) * sigma2_

        def newey_west():
            e_ = x[1:] - phi_ * x[:-1]
            q_ = np.sum(x[:-1] ** 2)
            u_ = x[:-1] * e_
            omega_ = newey_west_omega(u_, n_lags=self.var_n_lags)
            return (1 / q_) * omega_ * (1 / q_)

        var_estimators = {"non-robust": non_robust, "newey-west": newey_west}

        if self.var_estimator not in var_estimators:
            raise ValueError("var_estimator must be either 'newey-west' or 'non-robust'")

        var_ = var_estimators[self.var_estimator]()
        return phi_, np.sqrt(var_)

    def fit(self, X: np.ndarray, n_timepoints: int):
        """Fit the TD model.

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
            td_fits = parallel(self._fit_td(X[:, idx]) for idx in range(X.shape[1]))
            phis_, se_phis_ = map(np.array, zip(*td_fits))
            taus_, se_taus_ = _phi_to_tau(phis_, se_phis_)

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self


class AD(BaseEstimator):
    """Autocorrelation Domain (AD) Nonlinear Model, fit by Nonlinear Least Squares.

    Parameters
    ----------
    var_estimator : str, optional
        The variance estimator to use. Options are "newey-west" or "non-robust", by default "newey-west"
    var_n_lags : int, optional
        The lag truncation number for the bartlett kernel, by default None
    acf_n_lags : int, optional
        The lag truncation number for the autocorrelation function, by default None
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
    >>> ad = timescale_utils.AD(var_estimator="newey-west", var_n_lags=10, acf_n_lags=50)
    >>> ad.fit(X=X, n_timepoints=1000).estimates_
    {'phi': array([0.78021651]), 'se(phi)': array([0.02814532]), 'tau': array([4.02927146]), 'se(tau)': array([0.58565806])}
    """

    def __init__(
        self,
        var_estimator: str = "newey-west",
        var_n_lags: Optional[int] = None,
        acf_n_lags: Optional[int] = None,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.var_estimator = var_estimator
        self.var_n_lags = var_n_lags
        self.acf_n_lags = acf_n_lags
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    @delayed
    def _fit_ad(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries/autocorrelation function x in X"""

        T = len(x)

        # acf estimator
        x_acf = acf_utils.ACF(n_lags=self.acf_n_lags + 1).fit_transform(x.reshape(-1, 1), T).squeeze()[1:]

        # regression function (m), and its linearized regressor (dm_dphi)
        ks = np.arange(1, len(x_acf) + 1)
        m = lambda ks, phi: phi**ks
        jac = lambda ks, phi: (ks * phi ** (ks - 1)).reshape(-1, 1)

        # phi estimator
        eps = 1e-10
        phi_, _ = curve_fit(f=m, xdata=ks, ydata=x_acf, p0=1e-2, bounds=(-1 + eps, +1 - eps), ftol=1e-6, jac=jac)
        phi_ = phi_.squeeze()

        # variance estimators
        def non_robust():
            e_ = x[1:] - phi_ * x[:-1]
            q_ = np.sum(x[:-1] ** 2)
            sigma2_ = (1 / T) * np.sum(e_**2)
            return (1 / q_) * sigma2_

        def newey_west():
            q_ = np.sum((ks * phi_ ** (ks - 1)) ** 2)
            weights = ks * (phi_ ** (ks - 1))
            weights_phi = weights * (phi_**ks)
            conv1 = np.convolve(x, weights, mode="full")
            conv2 = np.convolve(x**2, weights_phi, mode="full")
            u_ = x[len(ks) : T] * conv1[len(ks) - 1 : T - 1] - conv2[len(ks) - 1 : T - 1]
            omega_ = (1 / len(u_) ** 2) * newey_west_omega(u_, n_lags=self.var_n_lags)
            return (1 / q_) * omega_ * (1 / q_)

        var_estimators = {"non-robust": non_robust, "newey-west": newey_west}

        if self.var_estimator not in var_estimators:
            raise ValueError("var_estimator must be either 'newey-west' or 'non-robust'")

        var_ = var_estimators[self.var_estimator]()
        return phi_, np.sqrt(var_)

    def fit(self, X: np.ndarray, n_timepoints: int):
        """Fit the AD model.

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
        if self.acf_n_lags is None:
            self.acf_n_lags = n_timepoints // 100

        with Parallel(n_jobs=self.n_jobs) as parallel:
            ad_fits = parallel(self._fit_ad(X[:, idx]) for idx in range(X.shape[1]))
            phis_, se_phis_ = map(np.array, zip(*ad_fits))
            taus_, se_taus_ = _phi_to_tau(phis_, se_phis_)

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self
