import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from src import acf_utils


def estimate_timescales_nls(X: np.array, n_regions: int) -> dict:
    """Estimate timescales and standard errors of a stationary AR(p) process using a non-linear least squares (NLS) model.

    Parameters
    ----------
    X : np.array of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_regions : int
        Number of regions/timeseries.

    Returns
    -------
    dict
        A dictionary with two keys, each mapping to an np.ndarray of shape (n_regions,):
        - "tau": Timescales for each region in X.
        - "se(tau)": Standard errors of the timescales.

    Raises
    ------
    ValueError
        If `X` is not in (n_regions, n_timepoints) form.
    """

    if X.shape[0] != n_regions:
        raise ValueError("X should be in (n_regions, n_timepoints) form")

    n_timepoints = X.shape[1]
    timepoints = np.arange(n_timepoints)

    exp_decay = lambda x, tau: np.exp(-x / (tau if tau > 0 else 1e-6))

    df = {"tau": np.zeros(n_regions), "se(tau)": np.zeros(n_regions)}
    for idx, x in enumerate(X):  # loop over regions
        X_acf = acf_utils.acf_fft(x.reshape(1, -1), n_timepoints).squeeze()
        tau, var = curve_fit(f=exp_decay, xdata=timepoints, ydata=X_acf)  # fit NLS
        df["tau"][idx] = tau
        df["se(tau)"][idx] = np.sqrt(var)

    return df


def _fit_ar1(X: np.ndarray, batch: np.ndarray) -> tuple:
    """Fits an AR(1) model estimated using OLS.

    Parameters
    ----------
    X : np.ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    batch : np.ndarray
        The indices of the regions to fit.

    Returns
    -------
    tuple of np.ndarrays with shape (n_voxels,)
        The estimated AR(1) coefficients and their standard errors.
    """
    X = X.T
    T = len(X) - 1
    # X_{t-1} = X[:-1, :], X_t = X[1:, :]
    sxx = np.sum(X[:-1, batch] ** 2, axis=0)
    phi = np.sum((X[:-1, batch] * X[1:, batch]), axis=0) / sxx
    se_phi = np.sqrt(
        np.sum((X[1:, batch] - phi * X[:-1, batch]) ** 2, axis=0)  # rss
        / (T - 1)  # df
        / sxx  # sxx
    )

    return phi, se_phi


def estimate_timescales_ols(X: np.ndarray, n_regions: int, n_jobs=-2, batch_size=100) -> dict:
    """Estimate timescales and standard errors of a stationary AR(p) process using an AR(1) model.

    Parameters
    ----------
    X : np.ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_regions : int
        Number of regions/timeseries.
    n_jobs : int, optional
        The number of jobs to run in parallel. \n
        (n_cpus + 1 + n_jobs) are used, by default -2
    batch_size : int, optional
        The number of voxels to fit in each batch, by default 100

    Returns
    -------
    dict
        A dictionary containing four np.ndarray of shape (n_regions, ):
        - "phi": AR(1) coefficient estimates, for each region in X.
        - "se(phi)": Standard errors of AR(1) coefficients.
        - "tau": Timescale estimates, for each region in X.
        - "se(tau)": Standard errors of timescales.

    Raises
    ------
    ValueError
        If `X` is not in (n_timepoints, n_regions) form.
    """

    if X.shape[0] != n_regions:
        raise ValueError("X should be in (n_regions, n_timepoints) form")

    indices = np.arange(n_regions)
    batches = [indices[idx : idx + batch_size] for idx in range(0, n_regions, batch_size)]
    ar1_fit = Parallel(n_jobs=n_jobs)(delayed(_fit_ar1)(X, batch) for batch in batches)
    phi, se_phi = map(np.concatenate, zip(*ar1_fit))

    # phi to tau (timescale), and apply delta method to std errs
    tau = -1 / np.log(phi)
    se_tau = (1 / (phi * np.log(phi) ** 2)) * se_phi
    return {"phi": phi, "se(phi)": se_phi, "tau": tau, "se(tau)": se_tau}
