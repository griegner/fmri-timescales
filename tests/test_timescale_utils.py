import numpy as np
import pytest
from scipy.optimize import curve_fit
from statsmodels.regression.linear_model import OLS as SMOLS

from fmri_timescales import acf_utils, sim, timescale_utils

n_timepoints, n_lags, n_repeats = 4800, 100, 1000


def test_TD():
    """Test if the TD estimator returns the expected estimates of an AR(1) process"""

    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    td = timescale_utils.TD(n_jobs=-2, var_estimator="non-robust")
    td.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, td.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(td.estimates_["tau"].std(), td.estimates_["se(tau)"].mean(), atol=0.01)

    td.set_params(var_estimator="newey-west")
    td.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, td.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(td.estimates_["tau"].std(), td.estimates_["se(tau)"].mean(), atol=0.01)


def test_TD_vs_statsmodels():
    """Test against statsmodels non-robust and newey-west"""
    ar2_phi = [0.1, 0.45]
    X = sim.sim_ar(ar2_phi, n_timepoints, n_repeats=1, random_seed=0)

    # non-robust std errors
    td = timescale_utils.TD(var_estimator="non-robust")
    td.fit(X, n_timepoints)
    sm_td = SMOLS(X[:-1], X[1:]).fit()
    assert np.isclose(td.estimates_["phi"], sm_td.params, atol=1e-4)
    assert np.isclose(td.estimates_["se(phi)"], sm_td.bse, atol=1e-4)

    # newey-west std errors
    td.set_params(var_estimator="newey-west", var_n_lags=100)
    td.fit(X, n_timepoints)
    sm_td = SMOLS(X[:-1], X[1:]).fit(cov_type="HAC", cov_kwds=dict(maxlags=100))
    assert np.isclose(td.estimates_["phi"], sm_td.params, atol=1e-4)
    assert np.isclose(td.estimates_["se(phi)"], sm_td.bse, atol=1e-4)


def test_TD_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    td = timescale_utils.TD()
    with pytest.raises(ValueError):  # X shape error
        td.fit(X, n_timepoints)
    with pytest.raises(ValueError):  # var_estimator not ["non-robust" or "newey-west"]
        td.set_params(var_estimator="nw")
        td.fit(X.T, n_timepoints)


def test_AD_time():
    """Test if the AD estimator returns the expected estimates of an AR(1) process"""

    # X_domain='time'
    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    # var_domain='time'
    ad = timescale_utils.AD(n_jobs=-2, var_estimator="non-robust")
    ad.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, ad.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(ad.estimates_["tau"].std(), ad.estimates_["se(tau)"].mean(), atol=0.1)

    ad.set_params(var_estimator="newey-west")
    ad.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, ad.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(ad.estimates_["tau"].std(), ad.estimates_["se(tau)"].mean(), atol=0.1)


def test_AD_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    ad = timescale_utils.AD()
    with pytest.raises(ValueError):  # X shape error
        ad.fit(X, n_timepoints)
    with pytest.raises(ValueError):  # var_estimator not ["non-robust" or "newey-west"]
        ad.set_params(var_estimator="nw")
        ad.fit(X.T, n_timepoints)
