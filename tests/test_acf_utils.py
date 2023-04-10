import numpy as np

from src import acf_utils, sim


def test_acf_fft():
    n_regions, n_timepoints = 3, 1200
    xcorr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]])
    acorr = np.eye(n_timepoints)
    X = sim.sample_mv_normal(xcorr, acorr, n_regions, n_timepoints, random_seed=0)

    X_acf = acf_utils.acf_fft(X, n_timepoints)

    assert X_acf is not None  # smoke test


def test_acf_to_toeplitz():
    n_timepoints = 1200
    acf = np.array([1.0, 0.9, 0.5, 0.3, 0.1])
    toeplitz = acf_utils.acf_to_toeplitz(acf, n_timepoints)
    assert toeplitz.shape == (n_timepoints, n_timepoints)
