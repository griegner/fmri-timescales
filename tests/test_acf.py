import numpy as np

from src import sim, acf


def test_acf_fft():
    n_regions, n_timepoints = 3, 1200
    xcorr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]])
    acorr = np.eye(n_timepoints)
    X = sim.sample_mv_normal(xcorr, acorr, n_regions, n_timepoints, random_seed=0)

    X_acf = acf.acf_fft(X, n_timepoints)

    assert X_acf is not None  # smoke test
