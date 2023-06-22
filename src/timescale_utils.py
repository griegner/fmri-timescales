import numpy as np
from scipy.stats import linregress


def estimate_timescales(X: np.ndarray, n_regions: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate timescales and standard errors of a stationary AR(p) process using an AR(1) model.

    Parameters
    ----------
    X : np.ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_regions : int
        Number of regions/timeseries.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or shape (n_regions,), (n_regions,)
        A tuple containing two arrays:
        - An array of timescales, for each region in X.
        - An array of standard errors, for each timescale.

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

    timescales, stderrs = np.zeros(n_regions), np.zeros(n_regions)
    for idx, region in enumerate(X):  # loop over regions
        lm = linregress(region[:-1], region[1:])
        timescales[idx] = slope_to_timescale(lm)
        stderrs[idx] = delta_method(lm)

    return timescales, stderrs
