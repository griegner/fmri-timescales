import numpy as np
import pytest

from src import acf_utils, sim

# set parameters
n_regions, n_timepoints = 3, 1200
xcorr = np.eye(n_regions)
acorrs = [
    np.eye(n_timepoints),
    acf_utils.acf_to_toeplitz(np.power(0.99, np.arange(n_timepoints)), n_timepoints),
    acf_utils.acf_to_toeplitz(np.load("tests/data/fmri-acf.npy"), n_timepoints),
]


@pytest.mark.parametrize("acorr", acorrs)
def test_acf_fft(acorr):
    "Test against np.correlate function, which is slower for large inputs"
    X = sim.sim_fmri(xcorr, acorr, n_regions, n_timepoints, random_seed=10)

    # compute ACF in the frequency domain
    X_acf = acf_utils.acf_fft(X, n_timepoints)

    # compute ACF in the time domain
    np_acf = np.zeros_like(X_acf)
    for region in range(n_regions):
        acov = np.correlate(X[region, :], X[region, :], mode="full")[n_timepoints - 1 :]
        np_acf[region, :] = acov / np.var(X[region, :]) / n_timepoints

    assert np.allclose(X_acf, np_acf)


def test_acf_fft_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_timepoints, n_regions))
    with pytest.raises(ValueError):
        acf_utils.acf_fft(X, n_timepoints)


def test_acf_to_toeplitz():
    "Test if the shape of the toeplitx matrix matches what is expected"
    n_timepoints = 1200

    # all timeseries have the same ACF
    acf = np.array([1.0, 0.9, 0.5, 0.3, 0.1])
    toeplitz = acf_utils.acf_to_toeplitz(acf, n_timepoints)
    assert toeplitz.shape == (n_timepoints, n_timepoints)

    # each timeseries has a different ACF
    acf = np.array([[1.0, 0.9, 0.0], [1.0, 0.8, 0.0], [1.0, 0.7, 0.0]])
    toeplitz = acf_utils.acf_to_toeplitz(acf, n_timepoints)
    assert toeplitz.shape == (n_regions, n_timepoints, n_timepoints)
