import numpy as np
import pytest
from statsmodels.regression.linear_model import OLS as SMOLS

from src import sim, timescale_utils

n_timepoints, n_repeats = 4800, 1000


def test_OLS():
    """Test if the OLS estimator returns the expected estimates of an AR(1) process"""

    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    ols = timescale_utils.OLS(n_jobs=-2)
    ols.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(phi, ols.estimates_["phi"].mean(), atol=0.001)
    assert np.isclose(ols.estimates_["phi"].std(), ols.estimates_["se(phi)"].mean(), atol=0.001)
    assert np.isclose(tau, ols.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(ols.estimates_["tau"].std(), ols.estimates_["se(tau)"].mean(), atol=0.01)


def test_OLS_vs_statsmodels():
    """Test against statsmodels newey-west, which is slower for multiple tests"""
    ar2_phi = [0.1, 0.45]
    X = sim.sim_ar(ar2_phi, n_timepoints, n_repeats=1, random_seed=0)

    # non-robust std errors
    ols = timescale_utils.OLS(var_estimator="non-robust", n_jobs=1)
    ols.fit(X, n_timepoints)
    sm_ols = SMOLS(X[:-1], X[1:]).fit()
    assert np.isclose(ols.estimates_["phi"], sm_ols.params, atol=1e-4)
    assert np.isclose(ols.estimates_["se(phi)"], sm_ols.bse, atol=1e-4)

    # newey-west std errors
    ols.set_params(**dict(var_estimator="newey-west", var_n_lags=100))
    ols.fit(X, n_timepoints)
    sm_ols = SMOLS(X[:-1], X[1:]).fit(cov_type="HAC", cov_kwds=dict(maxlags=100))
    assert np.isclose(ols.estimates_["phi"], sm_ols.params, atol=1e-4)
    assert np.isclose(ols.estimates_["se(phi)"], sm_ols.bse, atol=1e-4)


def test_OLS_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    ols = timescale_utils.OLS()
    with pytest.raises(ValueError):  # X shape error
        ols.fit(X, n_timepoints)
    with pytest.raises(ValueError):  # var_estimator not ["non-robust" or "newey-west"]
        ols.set_params(**{"var_estimator": "nw"})
        ols.fit(X.T, n_timepoints)


def test_NLS():
    """Test if the NLS estimator returns the expected estimates of an AR(1) process"""

    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    # non-robust std errors
    nls = timescale_utils.NLS(var_estimator="non-robust", n_jobs=-2)
    nls.fit(X, X.shape[0])
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.2)

    # newey-west std errors
    nls.set_params(**dict(var_estimator="newey-west", var_n_lags=100))
    nls.fit(X, n_timepoints)
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.2)


def test_NLS_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    nls = timescale_utils.NLS()
    with pytest.raises(ValueError):
        nls.fit(X, n_timepoints)
