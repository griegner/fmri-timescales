import numpy as np
import pytest

from src import sim, timescale_utils

n_regions, n_timepoints = 1, 4800


def _calc_ar_bias(phi, n_repeats=1000, random_seed=0):
    """repeate simulations of AR(p) process with fixed coefficients phi_1, phi_2, ..., phi_p"""
    random_state = np.random.default_rng(random_seed)
    random_seeds = random_state.integers(0, 100_000, size=n_repeats)

    timescales, stderrs = np.zeros(n_repeats), np.zeros(n_repeats)
    for idx, rs in enumerate(random_seeds):  # n_repeats
        # simulate AR(p) process
        X = sim.sim_ar(phi, n_timepoints, random_seed=rs)
        # fit AR(1) model
        timescale, stderr = timescale_utils.estimate_timescales(X.reshape(1, -1), n_regions)
        timescales[idx] = timescale
        stderrs[idx] = stderr

    return timescales, stderrs


def test_estimate_timescales():
    """Test if the AR(1) model estimates the expected coefficients and standard errors of an AR(p) process"""

    # AR(1) process
    ar1_phis = [0.9, 0.7, 0.5, 0.3, 0.1]  # AR coefficients

    for ar1_phi in ar1_phis:
        timescales, stderrs = _calc_ar_bias(ar1_phi)
        ar1_tau = -1 / np.log(ar1_phi)  # true timescale
        # timescales: true vs estimate (mean of timescales distribution)
        assert np.isclose(ar1_tau, timescales.mean(), atol=0.081)
        # stderrs: true (std of timescales) vs estimate (mean of stderr distribution)
        assert np.isclose(stderrs.mean(), timescales.std(), atol=0.012)

    # AR(2) process, w/in stationarity triangle
    ar2_phis = [[phi, (0.5 - 0.5 * phi)] for phi in ar1_phis]
    # AR(1) coeffs calculated empirically using an AR(1) model w/ 100,000 timepoints
    ar1_phis = [0.948, 0.823, 0.664, 0.458, 0.176]

    for ar1_phi, ar2_phi in zip(ar1_phis, ar2_phis):
        timescales, stderrs = _calc_ar_bias(ar2_phi)
        ar1_tau = -1 / np.log(ar1_phi)  # true timescale
        # timescales: true vs estimate (mean of timescales distribution)
        assert np.isclose(ar1_tau, timescales.mean(), atol=0.5)
        # stderrs: true (std of timescales) vs estimate (mean of stderr distribution)
        assert np.isclose(stderrs.mean(), timescales.std(), atol=0.05)


def test_estimate_timescales_checkfail():
    """Test if the function raises the expected ValueError"""
    n_regions, n_timepoints = 5, 4800
    X = np.random.random(size=(n_timepoints, n_regions))

    with pytest.raises(ValueError):
        timescale_utils.estimate_timescales(X, n_regions)
