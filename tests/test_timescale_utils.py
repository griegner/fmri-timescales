import numpy as np
import pytest

from src import sim, timescale_utils

n_timepoints, n_repeats = 4800, 1000


def test_OLS():
    """Test if the OLS estimator returns the expected estimates of an AR(1) process"""

    phi = np.array(0.75)
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats)

    ols = timescale_utils.OLS(n_jobs=-2)
    ols.fit(X.T, n_timepoints)

    # test difference btw true and estimated paramaters
    assert np.isclose(phi, ols.estimates_["phi"].mean(), atol=0.001)
    assert np.isclose(ols.estimates_["phi"].std(), ols.estimates_["se(phi)"].mean(), atol=0.001)
    assert np.isclose(tau, ols.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(ols.estimates_["tau"].std(), ols.estimates_["se(tau)"].mean(), atol=0.01)


def test_NLS():
    """Test if the NLS estimator returns the expected estimates of an AR(1) process"""

    phi = np.array(0.75)
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats)

    nls = timescale_utils.NLS(n_jobs=-2)
    nls.fit(X.T, n_timepoints)

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.26)
