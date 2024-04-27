from typing import Optional

import numpy as np
import statsmodels.stats.sandwich_covariance as sw_cov
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator

from src import acf_utils


class OLS(BaseEstimator):
    """Ordinary Least Squares Autoregressive Model.

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
            omega_ = sw_cov.S_hac_simple(
                u_, nlags=self.var_n_lags, weights_func=sw_cov.weights_bartlett
            ).squeeze()
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
        The variance estimator to use. Options are "non-robust", by default "non-robust"
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
    """

    def __init__(
        self,
        var_estimator: str = "non-robust",
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.var_estimator = var_estimator
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.estimates_ = {}

    def _fit_nls(self, x_acf_: np.ndarray) -> tuple:
        """fit model to a single autocorrelation function x_acf_ in X_acf_"""
        K = x_acf_.shape[0]
        ifzero = lambda tau: tau if tau > 0 else 1e-6
        exp_decay = lambda tau, k: np.exp(-k / ifzero(tau))
        exp_decay_dtau = lambda tau, k: (k / ifzero(tau) ** 2) * np.exp(-k / ifzero(tau))
        exp_decay_ddtau = lambda tau, k: (
            (k**2 / ifzero(tau) ** 4) * np.exp(-k / ifzero(tau))
        ) - ((2 * k / ifzero(tau) ** 3) * np.exp(-k / ifzero(tau)))
        loss = lambda tau_, k, y: exp_decay(tau_, k) - y
        ks = np.arange(K)

        nls_fit = least_squares(fun=loss, args=(ks, x_acf_), x0=1.0, bounds=(0, np.inf), ftol=1e-6)
        tau_ = nls_fit.x[0]

        # variance estimators
        q_ = np.mean(exp_decay_ddtau(tau_, ks))
        if self.var_estimator == "non-robust":
            sigma2_ = (1 / (K - 1)) * np.sum(nls_fit.fun**2)
            var_ = (1 / K) * (1 / q_) * sigma2_
            se_tau_ = np.sqrt(var_)
        elif self.var_estimator == "robust":
            omega_ = np.mean((exp_decay_ddtau(tau_, ks) ** 2) * (nls_fit.fun**2))
            var_ = (1 / K) * ((1 / q_) * omega_ * (1 / q_))
            se_tau_ = np.sqrt(var_)
        else:
            raise ValueError("var_estimator must be either 'robust' or 'non-robust'")
        return tau_, se_tau_

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

        X_acf_ = acf_utils.ACF().fit_transform(X, X.shape[0])

        nls_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_nls)(X_acf_[:, idx]) for idx in range(X_acf_.shape[1])
        )
        taus_, se_taus_ = map(np.ravel, zip(*nls_fits))

        self.estimates_ = {"tau": taus_, "se(tau)": se_taus_}
        return self.estimates_
