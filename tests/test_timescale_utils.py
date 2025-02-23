import numpy as np
import pytest
from scipy.optimize import curve_fit
from statsmodels.regression.linear_model import OLS as SMOLS

from fmri_timescales import acf_utils, sim, timescale_utils

n_timepoints, n_lags, n_repeats = 4800, 100, 1000


def test_LLS():
    """Test if the OLS estimator returns the expected estimates of an AR(1) process"""

    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    lls = timescale_utils.LLS(n_jobs=-2, var_estimator="non-robust")
    lls.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, lls.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(lls.estimates_["tau"].std(), lls.estimates_["se(tau)"].mean(), atol=0.01)

    lls.set_params(var_estimator="newey-west")
    lls.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, lls.estimates_["tau"].mean(), atol=0.01)
    assert np.isclose(lls.estimates_["tau"].std(), lls.estimates_["se(tau)"].mean(), atol=0.01)


def test_LLS_vs_statsmodels():
    """Test against statsmodels non-robust and newey-west"""
    ar2_phi = [0.1, 0.45]
    X = sim.sim_ar(ar2_phi, n_timepoints, n_repeats=1, random_seed=0)

    # non-robust std errors
    lls = timescale_utils.LLS(var_estimator="non-robust")
    lls.fit(X, n_timepoints)
    sm_lls = SMOLS(X[:-1], X[1:]).fit()
    assert np.isclose(lls.estimates_["phi"], sm_lls.params, atol=1e-4)
    assert np.isclose(lls.estimates_["se(phi)"], sm_lls.bse, atol=1e-4)

    # newey-west std errors
    lls.set_params(var_estimator="newey-west", var_n_lags=100)
    lls.fit(X, n_timepoints)
    sm_lls = SMOLS(X[:-1], X[1:]).fit(cov_type="HAC", cov_kwds=dict(maxlags=100))
    assert np.isclose(lls.estimates_["phi"], sm_lls.params, atol=1e-4)
    assert np.isclose(lls.estimates_["se(phi)"], sm_lls.bse, atol=1e-4)


def test_LLS_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    lls = timescale_utils.LLS()
    with pytest.raises(ValueError):  # X shape error
        lls.fit(X, n_timepoints)
    with pytest.raises(ValueError):  # var_estimator not ["non-robust" or "newey-west"]
        lls.set_params(var_estimator="nw")
        lls.fit(X.T, n_timepoints)


def test_NLS_time():
    """Test if the NLS estimator returns the expected estimates of an AR(1) process"""

    # X_domain='time'
    phi = [0.75]
    tau = -1.0 / np.log(phi)
    X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=0)

    # var_domain='time'
    nls = timescale_utils.NLS(n_jobs=-2, var_estimator="non-robust", var_domain="time")
    nls.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.1)

    nls.set_params(var_estimator="newey-west")
    nls.fit(X, X.shape[0])

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.1)


def test_NLS_autocorrelation():
    """Test when X_domain='autocorrelation'"""

    rng = np.random.default_rng(seed=0)

    phi = [0.9]
    tau = -1.0 / np.log(phi)
    X = acf_utils.ar_to_acf(phi, n_lags=n_lags)
    X = np.tile(acf_utils.ar_to_acf(phi, n_lags=n_lags), (n_repeats, 1)).T
    X += rng.normal(loc=0, scale=0.05, size=X.shape)

    nls = timescale_utils.NLS(X_domain="autocorrelation", var_estimator="non-robust", var_domain="autocorrelation")
    nls.fit(X, n_lags)

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.005)

    nls.set_params(var_estimator="newey-west")
    nls.fit(X, n_lags)

    # test difference btw true and estimated paramaters
    assert np.isclose(tau, nls.estimates_["tau"].mean(), atol=0.015)
    assert np.isclose(nls.estimates_["tau"].std(), nls.estimates_["se(tau)"].mean(), atol=0.05)


def test_NLS_vs_scipy():
    """Test against scipy non-robust (no scipy newey-west implementation :/)"""

    ar2_phi = [0.1, 0.45]
    X = sim.sim_ar(ar2_phi, n_timepoints, n_repeats=1, random_seed=0)

    # variance estimation in autocorrelation domain only
    # non-robust std errors
    nls = timescale_utils.NLS(var_estimator="non-robust", var_domain="autocorrelation")
    nls.fit(X, n_timepoints)
    x_acf_ = acf_utils.ACF().fit_transform(X.reshape(-1, 1), n_timepoints).squeeze()
    ks = np.linspace(0, n_timepoints - 1, n_timepoints)
    m = lambda ks, phi: phi**ks
    phi_, var_phi_ = curve_fit(f=m, xdata=ks, ydata=x_acf_, p0=0, bounds=(-1, +1), ftol=1e-6)
    assert np.isclose(nls.estimates_["phi"], phi_, atol=1e-4)
    assert np.isclose(nls.estimates_["se(phi)"], np.sqrt(var_phi_), atol=1e-4)

    # newey-west std errors (test close to non-robust)
    nls.set_params(var_estimator="newey-west")
    nls.fit(X, n_timepoints)
    assert np.isclose(nls.estimates_["se(phi)"], np.sqrt(var_phi_), atol=0.05)


def test_NLS_checkfail():
    """Test if the function raises the expected ValueError"""
    X = np.zeros((n_repeats, n_timepoints))
    nls = timescale_utils.NLS()
    with pytest.raises(ValueError):  # X shape error
        nls.fit(X, n_timepoints)
    with pytest.raises(ValueError):  # var_estimator not ["non-robust" or "newey-west"]
        nls.set_params(var_estimator="nw")
        nls.fit(X.T, n_timepoints)
    with pytest.raises(ValueError):  # var_domain not ["time" or "autocorrelation"]
        nls.set_params(var_domain="ad")
        nls.fit(X.T, n_timepoints)
