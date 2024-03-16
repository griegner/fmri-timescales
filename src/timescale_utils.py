from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator

from src import acf_utils


class OLS(BaseEstimator):
    """Ordinary Least Squares Autoregressive Model.

    Parameters
    ----------
    cov_estimator : str, optional
        The covariance estimator to use. Options are "newey-west" or "non-robust", by default "newey-west"
    cov_n_lags : int, optional
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
        cov_estimator: str = "newey-west",
        cov_n_lags: Optional[int] = None,
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.cov_estimator = cov_estimator
        self.cov_n_lags = cov_n_lags
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.estimates_ = {}

    def _fit_ols(self, x: np.ndarray) -> tuple:
        """fit model to a single timeseries x in X"""
        T = x.shape[0] - 1
        # x_t = X[1:], x_{t-1} = x[:-1]
        phi_ = np.sum((x[1:] * x[:-1])) / np.sum(x[:-1] ** 2)
        sigma2_ = np.sum((x[1:] - phi_ * x[:-1]) ** 2) / (T - 1)
        se_phi_ = np.sqrt(sigma2_ / np.sum(x[:-1] ** 2))
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
        copy_X: bool = False,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.estimates_ = {}

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

        exp_decay = lambda k, tau: np.exp(-k / (tau if tau > 0 else 1e-6))
        lags = np.arange(n_timepoints)
        curve_fit_kwargs = dict(bounds=(0, np.inf), ftol=1e-6)

        exp_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(curve_fit)(f=exp_decay, xdata=lags, ydata=X_acf_[:, idx], **curve_fit_kwargs)
            for idx in range(X_acf_.shape[1])
        )
        taus_, var_taus_ = map(np.ravel, zip(*exp_fits))

        self.estimates_ = {"tau": taus_, "se(tau)": np.sqrt(var_taus_)}
        return self.estimates_
