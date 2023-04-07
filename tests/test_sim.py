import numpy as np

from src import sim


def test_sample_mv_normal():
    random_state = np.random.default_rng(seed=1)

    n_regions, n_timepoints = 3, 1200
    xcorr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]])
    acorr = np.eye(n_timepoints)

    X = sim.sample_mv_normal(xcorr, acorr, n_regions, n_timepoints, random_seed=0)

    assert np.allclose(xcorr, np.corrcoef(X), atol=0.1)  # cross-correlation
