import numpy as np
from scipy.signal import lfilter

from src import acf_utils


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


def calc_xcorr(X: np.ndarray, n_timepoints: int, corrected: bool = True) -> np.ndarray:
    """
    Compute pairwise cross-correlations from simulated cross- and auto-correlated timeseries.


    Parameters
    ----------
    X : np.ndarray of shape (n_regions, n_timepoints)
        An array containing the timeseries of each region.
    n_timepoints : int
        Number of timepoints/samples.
    corrected : bool, optional
        When each timeseries is generated with a different auto-correlation function, spurious cross-correlations can arise.
        This can be corrected by `Eq. S9 (Afyouni et al, 2019)`. By default True.

    Returns
    -------
    np.ndarray of shape (n_regions, n_regions)
        The (corrected) cross-correlation matrix of `X`.

    Raises
    ------
    ValueError
        If `X` is not in (n_regions, n_timepoints) form.

    References
    ----------
    Afyouni, Soroosh, Stephen M. Smith, and Thomas E. Nichols.
    “Effective Degrees of Freedom of the Pearson's Correlation Coefficient under Autocorrelation.”
    NeuroImage 199 (October 2019): 609-25. https://doi.org/10.1016/j.neuroimage.2019.05.011.
    """

    if X.shape[1] != n_timepoints:
        raise ValueError("X should be in (n_regions, n_timepoints) form")

    acf = acf_utils.acf_fft(X, n_timepoints)
    acorr = acf_utils.acf_to_toeplitz(acf, n_timepoints)
    xcorr = np.corrcoef(X)

    if not corrected:
        return xcorr

    else:  # Eq S9 in Section S3.1 (Afyouni et al., 2019)
        xcorr_corrected = np.ones_like(xcorr)
        for i, region_i in enumerate(acorr):
            cholesky_i = np.linalg.cholesky(region_i)
            for j, region_j in enumerate(acorr):
                cholesky_j = np.linalg.cholesky(region_j)
                xcorr_corrected[i, j] = (
                    xcorr[i, j] / np.trace(cholesky_i @ cholesky_j.T) * n_timepoints
                )
        return xcorr_corrected


def gen_ar2_coeffs(oscillatory=False, random_seed=4):
    """Generate coefficients for an stationary AR(2) process.\n
    An AR(2) process is stable when the coefficients are in the triangle -1 < phi2 < 1 - |phi1|;\n
    and oscillatory if -1 < phi2 < -0.25 * phi1^2.

    Parameters
    ----------
    oscillatory : bool, optional
        Whether the coefficients should generate an oscillating AR(2) process, by default False
    random_seed : int, optional
        Seed for the random number generator, by default 4

    Returns
    -------
    list
        A list of two coefficients for the AR(2) process
    """
    rng = np.random.default_rng(seed=random_seed)
    phi1 = rng.uniform(-2, 2)
    if oscillatory:
        phi2 = rng.uniform(-1, -0.25 * phi1**2)
    else:
        phi2 = rng.uniform(np.max([-1, -0.25 * phi1**2]), np.min([1 + phi1, 1 - phi1]))
    return [phi1, phi2]


def sim_ar(
    ar_coeffs: np.ndarray, n_timepoints: int, scale: float = 1.0, random_seed: int = 0
) -> np.ndarray:
    """Simulate a univariate AR(p) timeseries.

    Parameters
    ----------
    ar_coeffs : list or ndarray of shape (p,)
        A list of AR(p) coefficients in the form [phi_1, phi_2, ..., phi_p], excluding phi_0.
    n_timepoints : int
        Number of timepoints/samples.
    scale : float, optional
        Standard deviation of the noise, by default 1.0.
    random_seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Simulated timeseries with the specified AR(p) coefficients.
    """

    if isinstance(ar_coeffs, list):
        ar_coeffs = np.array(ar_coeffs)

    random_state = np.random.default_rng(seed=random_seed)
    rv = scale * random_state.standard_normal(size=n_timepoints)

    return lfilter([1], np.r_[1, -ar_coeffs], rv)
