import numpy as np


def sim_fmri(
    xcov: np.ndarray, acov: np.ndarray, n_regions: int, n_timepoints: int, random_seed: int = 0
) -> np.ndarray:
    """Sample cross-correlated (`xcov`) and auto-correlated (`acov`) timeseries from the multivariate normal distribution
    -- multiply a standard normal random variable `z` by the Cholesky decomposition of the `xcov` and `acov` matrices.

    Parameters
    ----------
    xcov : ndarray of shape (n_regions, n_regions)
        The cross-covariance matrix, i.e. spatial correlation.
    acov : ndarray of shape either (n_timepoints, n_timepoints) or (n_regions, n_timepoints, n_timepoints)
        The auto-covariance matrix, i.e. temporal correlation.
        If the shape is (n_timepoints, n_timepoints), all timeseries will have the same ACF.
        If the shape is (n_regions, n_timepoints, n_timepoints), each timeseries will have a different ACF.
    n_regions : int
        Number of regions/timeseries.
    n_timepoints : int
        Number of timepoints/samples.
    random_seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    ndarray of shape (n_regions, n_timepoints)
        Simulated timeseries with the specified `xcov` and `acov`.

    Raises
    ------
    ValueError
        If `acov` is not in (n_timepoints, n_timepoints) or (n_regions, n_timepoints, n_timepoints) form.
    """

    if acov.ndim != 2 and acov.ndim != 3:
        raise ValueError("acov should be in ([n_regions], n_timepoints, n_timepoints) form")

    random_state = np.random.default_rng(seed=random_seed)
    rv = random_state.standard_normal(size=(n_regions, n_timepoints))

    xcov_rv = np.linalg.cholesky(xcov) @ rv  # cross-correlation

    # all timeseries have the same ACF
    if acov.ndim == 2:
        X = xcov_rv @ np.linalg.cholesky(acov)  # auto-correlation
        return X

    # each timeseries has a different ACF
    else:
        if acov.shape[0] != n_regions:
            raise ValueError("acov should index n_regions")

        X = np.zeros((n_regions, n_timepoints))
        for idx, region in enumerate(acov):
            X[idx] = xcov_rv[idx, :] @ np.linalg.cholesky(region)  # auto-correlation
        return X
