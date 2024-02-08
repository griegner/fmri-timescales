import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

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


def estimate_timescales_ols(X: np.ndarray, n_regions: int) -> dict:
    """
    Estimate timescales and standard errors of a stationary AR(p) process using an ordinary least squares model.

    Parameters
    ----------
    X : np.ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_regions : int
        Number of regions/timeseries.

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
        If `X` is not in (n_regions, n_timepoints) form.
    """

    if X.shape[0] != n_regions:
        raise ValueError("X should be in (n_regions, n_timepoints) form")

    # timescale = -1 / ln(slope)
    slope_to_timescale = lambda lm: -1 / np.log(lm.slope if lm.slope > 0 else 1e-9)
    # stderr = (1 / (slope * ln(slope)^2)) * stderr
    delta_method = (
        lambda lm: (1 / (lm.slope * np.log(lm.slope if lm.slope > 0 else 1e-9) ** 2)) * lm.stderr
    )

    df = {
        "phi": np.zeros(n_regions),
        "se(phi)": np.zeros(n_regions),
        "tau": np.zeros(n_regions),
        "se(tau)": np.zeros(n_regions),
    }
    for idx, x in enumerate(X):  # loop over regions
        lm = linregress(x[:-1], x[1:])  # fit OLS
        df["phi"][idx] = lm.slope
        df["se(phi)"][idx] = lm.stderr
        df["tau"][idx] = slope_to_timescale(lm)
        df["se(tau)"][idx] = delta_method(lm)

    return df
