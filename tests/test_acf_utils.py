import numpy as np
import pytest

from fmri_timescales import acf_utils, sim

# set parameters
n_timepoints, n_regions = 10000, 3
phis = [[0.9], [1.77, -0.89]]  # AR(1), AR(2)


@pytest.mark.parametrize("phi", phis)
def test_ACF(phi):
    """Test against np.correlate function, which is slower for large inputs"""
    n_lags = 50
    acf = acf_utils.ar_to_acf(phi, n_lags)

    X = sim.sim_ar(phi, n_timepoints, random_seed=3)
    X_acf_ = acf_utils.ACF(n_lags, n_jobs=-2).fit_transform(X, X.shape[0])

    assert np.allclose(acf, X_acf_.flatten(), atol=0.06)


def test_ACF_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_regions, n_timepoints))
    with pytest.raises(ValueError):
        acf_utils.ACF().fit_transform(X, n_timepoints)


def test_acf_to_toeplitz():
    """Test if the shape of the toeplitx matrix matches what is expected"""
    n_lags = 1200

    # all timeseries have the same ACF
    acf = np.zeros(shape=n_lags)
    toeplitz = acf_utils.acf_to_toeplitz(acf, n_lags)
    assert toeplitz.shape == (n_lags, n_lags)

    # each timeseries has a different ACF
    acf = np.zeros(shape=(n_lags, n_regions))
    toeplitz = acf_utils.acf_to_toeplitz(acf, n_lags)
    assert toeplitz.shape == (n_lags, n_lags, n_regions)

    # check fail
    with pytest.raises(ValueError):
        acf = np.zeros(shape=(n_regions, n_lags))
        acf_utils.acf_to_toeplitz(acf, n_lags)


def test_ar_to_acf():
    """Test theoretical calculations vs returned values"""

    # AR(1)
    n_lags = 10
    ar_coeffs = np.array([0.8])
    acf = np.power(ar_coeffs, np.arange(0, n_lags))
    assert np.allclose(acf_utils.ar_to_acf(ar_coeffs, n_lags=n_lags), acf)

    # AR(2)
    n_lags = 3
    ar_coeffs = np.array([0.75, -0.25])
    acf = [1, 0.6, 0.2]  # solved by Yule-Walker
    assert np.allclose(acf_utils.ar_to_acf(ar_coeffs, n_lags=n_lags), acf)
