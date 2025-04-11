from typing import Union

import numpy as np
from scipy.signal import lfilter

from fmri_timescales import acf_utils


def sim_fmri(xcm: np.ndarray, acm: np.ndarray, n_regions: int, n_timepoints: int, random_seed: int = 0) -> np.ndarray:
    """Sample cross-correlated and auto-correlated timeseries from the multivariate normal distribution.

    Parameters
    ----------
    xcm : np.ndarray of shape (n_regions, n_regions)
        The cross-correlation matrix, i.e. spatial correlation.
    acm : np.ndarray of shape either (n_timepoints, n_timepoints) or (n_timepoints, n_timepoints, n_regions)
        The auto-correlation toeplitz matrix, i.e. temporal correlation.
        If the shape is (n_timepoints, n_timepoints), all timeseries will have the same acf.
        If the shape is (n_timepoints, n_timepoints, n_regions), each timeseries will have a different acf.
    n_regions : int
        Number of regions/timeseries.
    n_timepoints : int
        Number of timepoints/samples.
    random_seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    np.ndarray of shape (n_timepoints, n_regions)
        Simulated timeseries with the specified {cross, auto}-correlations.

    Raises
    ------
    ValueError
        If `xcm` is not in (n_regions, n_regions) form.
        If `acm` is not in (n_timepoints, n_timepoints) or (n_timepoints, n_timepoints, n_regions) form.
    """

    if xcm.ndim != 2 or xcm.shape != (n_regions, n_regions):
        raise ValueError("xcm should be in (n_regions, n_regions) form")

    rng = np.random.default_rng(seed=random_seed)
    Z = rng.standard_normal(size=(n_regions, n_timepoints))  # ~ WN(0,1)

    xcf_Z = np.linalg.cholesky(xcm) @ Z  # cross-correlation

    # all timeseries have the same acf
    if acm.ndim == 2 and acm.shape == (n_timepoints, n_timepoints):
        X = (xcf_Z @ np.linalg.cholesky(acm)).T  # auto-correlation
        return X

    # each timeseries has a different acf
    elif acm.ndim == 3 and acm.shape == (n_timepoints, n_timepoints, n_regions):
        X = np.zeros((n_timepoints, n_regions))
        for idx in range(acm.shape[-1]):
            X[:, idx] = xcf_Z[idx, :] @ np.linalg.cholesky(acm[..., idx])  # auto-correlation
        return X

    else:
        raise ValueError(
            "acm should be in (n_timepoints, n_timepoints) or (n_timepoints, n_timepoints, n_regions) form"
        )


def calc_xcm(X: np.ndarray, n_timepoints: int, corrected: bool = True) -> np.ndarray:
    """
    Compute the cross-correlation matrix of X generated from `sim_fmri()`.


    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints, n_regions)
        An array containing the timeseries of each region.
    n_timepoints : int
        Number of timepoints/samples.
    corrected : bool, optional
        If each timeseries is generated with a different auto-correlation function, spurious cross-correlations can arise.
        This can be corrected by `Eq. S9 (Afyouni et al, 2019)`. By default True.

    Returns
    -------
    np.ndarray of shape (n_regions, n_regions)
        The (corrected) cross-correlation matrix of `X`.

    Raises
    ------
    ValueError
        If `X` is not in (n_timepoints, n_regions) form.

    References
    ----------
    Afyouni, Soroosh, Stephen M. Smith, and Thomas E. Nichols.
    “Effective Degrees of Freedom of the Pearson's Correlation Coefficient under Autocorrelation.”
    NeuroImage 199 (October 2019): 609-25. https://doi.org/10.1016/j.neuroimage.2019.05.011.
    """

    if X.ndim != 2 or X.shape[0] != n_timepoints:
        raise ValueError("X should be in (n_timepoints, n_regions) form")

    X_acf_ = acf_utils.ACF().fit_transform(X, X.shape[0])
    acm = acf_utils.acf_to_toeplitz(X_acf_, X.shape[0])
    xcm = np.corrcoef(X.T)

    if not corrected:
        return xcm

    else:  # Eq S9 in Section S3.1 (Afyouni et al., 2019)
        xcm_corrected = np.ones_like(xcm)
        for i in range(acm.shape[-1]):
            cholesky_i = np.linalg.cholesky(acm[..., i])
            for j in range(acm.shape[-1]):
                cholesky_j = np.linalg.cholesky(acm[..., j])
                xcm_corrected[i, j] = xcm[i, j] / np.trace(cholesky_i @ cholesky_j.T) * n_timepoints
        return xcm_corrected


def gen_ar2_coeffs(oscillatory: bool = False, random_seed: int = 4) -> np.ndarray:
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
    np.ndarray of shape (2, )
        The two coefficients of a stationary AR(2) process
    """
    rng = np.random.default_rng(seed=random_seed)
    phi1 = rng.uniform(-2, 2)
    if oscillatory:
        phi2 = rng.uniform(-1, -0.25 * phi1**2)
    else:
        phi2 = rng.uniform(np.max([-1, -0.25 * phi1**2]), np.min([1 + phi1, 1 - phi1]))
    return np.array([phi1, phi2])


def sim_ar(
    ar_coeffs: Union[list, np.ndarray],
    n_timepoints: int,
    n_repeats: int = 1,
    scale: float = 1.0,
    random_seed: int = 0,
) -> np.ndarray:
    """Simulate multiple univariate AR(p) timeseries.

    Parameters
    ----------
    ar_coeffs : list or ndarray of shape (p,)
        A list of AR(p) coefficients in the form [phi_1, phi_2, ..., phi_p], excluding phi_0.
    n_timepoints : int
        Number of timepoints/samples.
    n_repeats : int, optional
        Number of timeseries to generate, be default 1.
    scale : float, optional
        Standard deviation of the noise, by default 1.0.
    random_seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    np.ndarray of shape (n_repeats, n_timepoints)
        Simulated timeseries with the specified AR(p) coefficients.
    """

    if isinstance(ar_coeffs, list):
        ar_coeffs = np.array(ar_coeffs)

    random_state = np.random.default_rng(seed=random_seed)
    Z = scale * random_state.standard_normal(size=(n_timepoints, n_repeats))  # ~ WN(0,1)

    return lfilter([1], np.r_[1, -ar_coeffs], Z, axis=0)  # type: ignore
