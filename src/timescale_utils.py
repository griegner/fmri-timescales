import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from src import acf_utils


class OLS:
    def __init__(
        self,
        cov_estimator="newey-west",
        cov_n_lags=None,
        copy_X=False,
        standardize_X=True,
        n_jobs=None,
    ):
        self.cov_estimator = cov_estimator
        self.cov_n_lags = cov_n_lags
        self.copy_X = copy_X
        self.standardize_X = standardize_X
        self.n_jobs = n_jobs

    def _fit_ols(self, x):
        T = x.shape[0] - 1
        # x_t = X[1:], x_{t-1} = x[:-1]
        phi_ = np.sum((x[1:] * x[:-1])) / np.sum(x[:-1] ** 2)
        sigma2_ = np.sum((x[1:] - phi_ * x[:-1]) ** 2) / (T - 1)
        se_phi_ = np.sqrt(sigma2_ / np.sum(x[:-1] ** 2))
        return phi_, se_phi_

    def fit(self, X, n_timepoints):

        if X.shape[0] != n_timepoints:
            raise ValueError("X should be in (n_timepoints, n_regions) form")
        X = X.copy() if self.copy_X else X
        X = (X - X.mean(axis=0)) / X.std(axis=0) if self.standardize_X else X

        ols_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_ols)(X[:, idx]) for idx in range(X.shape[1])
        )
        phis_, se_phis_ = map(np.array, zip(*ols_fits))

        # phi to tau (timescale), and apply delta method to std err
        phis_ = np.abs(phis_)  # tau undefined for phi <= 0
        taus_ = -1.0 / np.log(phis_)
        se_taus_ = (1.0 / (phis_ * np.log(phis_) ** 2)) * se_phis_

        self.estimates_ = {"phi": phis_, "se(phi)": se_phis_, "tau": taus_, "se(tau)": se_taus_}


class NLS:
    def __init__(
        self,
        copy_X=False,
        standardize_X=True,
        n_jobs=None,
    ):
        self.copy_X = copy_X
        self.standardize_X = standardize_X
        self.n_jobs = n_jobs

    def fit(self, X, n_timepoints):

        if X.shape[0] != n_timepoints:
            raise ValueError("X should be in (n_timepoints, n_regions) form")
        X = X.copy() if self.copy_X else X
        X = (X - X.mean(axis=0)) / X.std(axis=0) if self.standardize_X else X
        X_acf = acf_utils.acf_fft(X.T, X.shape[0]).T

        exp_decay = lambda k, tau: np.exp(-k / (tau if tau > 0 else 1e-9))
        lags = np.arange(n_timepoints)

        exp_fits = Parallel(n_jobs=self.n_jobs)(
            delayed(curve_fit)(f=exp_decay, xdata=lags, ydata=X_acf[:, idx])
            for idx in range(X_acf.shape[1])
        )
        taus_, se_taus_ = map(np.ravel, zip(*exp_fits))

        self.estimates_ = {"tau": taus_, "se(tau)": se_taus_}
