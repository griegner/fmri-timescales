from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator

from fmri_timescales import acf_utils


def newey_west_omega(u, n_lags=None):
    n_u = len(u)
    if n_lags is None:
        n_lags = int(np.floor(4 * (n_u / 100.0) ** (2.0 / 9.0)))

    weights = 1 - np.arange(n_lags + 1) / (n_lags + 1)  # bartlett weights

    S = weights[0] * np.sum(u**2)  # weights[0] is 1

    for lag in range(1, n_lags + 1):
        s = np.sum(u[lag:] * u[:-lag])
        S += weights[lag] * (2 * s)

    return S


class LLS(BaseEstimator):
    """Linear Least Squares Autoregressive Model.

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
    >>> lls = timescale_utils.LLS(var_estimator="newey-west")
    >>> lls.fit(X=X, n_timepoints=1000)
    {'phi': array([0.79789847]), 'se(phi)': array([0.02028763]), 'tau': array([4.42920958]), 'se(tau)': array([0.49881125])}
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
        self.estimates_ = {}

    def _fit_ols(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries x in X"""
        T = x.shape[0] - 1
        # x_t = X[1:], x_{t-1} = x[:-1]
        q_ = np.sum(x[:-1] ** 2)
        phi_ = np.sum((x[1:] * x[:-1])) / q_

        # variance estimators
        if self.var_estimator == "non-robust":
            sigma2_ = (1 / (T - 1)) * np.sum((x[1:] - phi_ * x[:-1]) ** 2)
            var_ = (1 / q_) * sigma2_
            se_phi_ = np.sqrt(var_)
        elif self.var_estimator == "newey-west":
            u_ = x[:-1] * (x[1:] - phi_ * x[:-1])
            omega_ = newey_west_omega(u_, n_lags=self.var_n_lags)
            var_ = (1 / q_) * omega_ * (1 / q_)
            se_phi_ = np.sqrt(var_)
        else:
            raise ValueError("var_estimator must be either 'newey-west' or 'non-robust'")
        return phi_, se_phi_

    def fit(self, X: np.ndarray, n_timepoints: int) -> dict:
        """Fit AR(1) model estimated using OLS.

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

        ols_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_ols)(X[:, idx]) for idx in range(X.shape[1])
        )
        phis_, se_phis_ = map(np.array, zip(*ols_fits))

        # phi to tau (timescale), and apply delta method to std err
        phis_ = np.abs(phis_)  # tau undefined for phi <= 0
        taus_ = -1.0 / np.log(phis_)
        se_taus_ = (1.0 / (phis_ * np.log(phis_) ** 2)) * se_phis_

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self.estimates_


class NLS(BaseEstimator):
    """Non-linear Least Squares.

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
    >>> nls = timescale_utils.NLS(var_estimator="newey-west")
    >>> nls.fit(X=X, n_timepoints=1000)
    {'phi': array([0.78021159]), 'se(phi)': array([0.01071285]), 'tau': array([4.02916908]), 'se(tau)': array([0.22290691])}
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
        self.estimates_ = {}

    def _fit_nls(self, x_acf_: np.ndarray) -> tuple:
        """fit model to a single autocorrelation function x_acf_ in X_acf_"""

        K = len(x_acf_)
        ks = np.linspace(0, K - 1, K)

        m = lambda ks, phi: phi**ks
        dm_dphi = lambda ks, phi: ks * phi ** (ks - 1)
        phi_, _ = curve_fit(f=m, xdata=ks, ydata=x_acf_, p0=0, bounds=(-1, +1), ftol=1e-6)
        phi_ = phi_.squeeze()
        e_ = x_acf_ - phi_**ks
        q_ = np.mean(dm_dphi(ks, phi_))
        if self.var_estimator == "non-robust":
            sigma2_ = np.mean(e_**2)
            var_ = (1 / q_) * sigma2_
            se_phi_ = np.sqrt((1 / K) * var_)
        elif self.var_estimator == "newey-west":
            u_ = dm_dphi(ks, phi_) * e_
            omega_ = (1 / K) * newey_west_omega(u_, n_lags=self.var_n_lags)
            var_ = (1 / q_) * omega_ * (1 / q_)
            se_phi_ = np.sqrt((1 / K) * var_)
        else:
            raise ValueError("var_estimator must be either 'newey-west' or 'non-robust'")
        return phi_, se_phi_

    def fit(self, X: np.ndarray, n_timepoints: int) -> dict:
        """Fit exponential decay function to empirical ACF.

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
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        X_acf_ = acf_utils.ACF(n_lags=self.acf_n_lags).fit_transform(X, X.shape[0])

        nls_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_nls)(X_acf_[:, idx]) for idx in range(X_acf_.shape[1])
        )
        phis_, se_phis_ = map(np.array, zip(*nls_fits))

        # phi to tau (timescale), and apply delta method to std err
        phis_ = np.abs(phis_)  # tau undefined for phi <= 0
        taus_ = -1.0 / np.log(phis_)
        se_taus_ = (1.0 / (phis_ * np.log(phis_) ** 2)) * se_phis_

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}
        return self.estimates_
