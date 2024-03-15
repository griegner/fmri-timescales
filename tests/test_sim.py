import numpy as np
import pytest
from statsmodels.tsa.arima.model import ARIMA

from src import acf_utils, sim

# set parameters
n_regions, n_timepoints = 3, 1200
xcms = [
    np.eye(n_regions),
    np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]]),
]
acms = [
    np.eye(n_timepoints),
    acf_utils.acf_to_toeplitz(np.load("tests/data/fmri-acf.npy"), n_timepoints),
]


@pytest.mark.parametrize("xcm", xcms)
@pytest.mark.parametrize("acm", acms)
def test_sim_fmri(xcm, acm):
    """Test if the generated data returns the expected {auto,cross}-correlation parameters"""
    xcm_corrected = True if acm.ndim == 3 else False
    acf = np.tile(acm[0, ...], (3, 1)).T if acm.ndim == 2 else acm[0, ...]
    X = sim.sim_fmri(xcm, acm, n_regions, n_timepoints, random_seed=0)

    # cross-correlation
    assert np.allclose(xcm, sim.calc_xcm(X, n_timepoints, xcm_corrected), atol=0.1)
    # auto-correlation
    assert np.allclose(acf, acf_utils.ACF().fit_transform(X, n_timepoints), atol=0.3)


def test_sim_fmri_checkfail():
    """Test if the function raises the expected ValueError"""
    xcm = np.eye(n_regions)
    acm = np.zeros((n_regions, n_timepoints, n_timepoints))

    with pytest.raises(ValueError):
        sim.sim_fmri(xcm, acm, n_regions, n_timepoints)


def test_calc_xcorr():
    "Test if simulation artifacts are removed by the correction"
    xcm = xcms[1]
    acm = acms[1]
    X = sim.sim_fmri(xcm, acm, n_regions, n_timepoints, random_seed=0)

    # not corrected
    assert not np.allclose(xcm, sim.calc_xcm(X, n_timepoints, corrected=False), atol=0.1)
    # corrected
    assert np.allclose(xcm, sim.calc_xcm(X, n_timepoints, corrected=True), atol=0.1)


def test_gen_ar2_coeffs():
    """Test if generated AR(2) coefficients are within the expected ranges"""

    # non-oscillatory AR(2) coefficients
    coeffs = sim.gen_ar2_coeffs(oscillatory=False, random_seed=0)
    assert len(coeffs) == 2
    assert -2 < coeffs[0] < 2  # phi1
    assert max(-1, -0.25 * coeffs[0] ** 2) <= coeffs[1] <= min(1 + coeffs[0], 1 - coeffs[0])  # phi2

    # oscillatory AR(2) coefficients
    coeffs = sim.gen_ar2_coeffs(oscillatory=True, random_seed=0)
    assert len(coeffs) == 2
    assert -2 < coeffs[0] < 2  # phi1
    assert -1 < coeffs[1] < -0.25 * coeffs[0] ** 2  # phi2


def test_sim_ar():
    """Test if generated AR process returns the expected coefficients"""
    ar_coeffs = [[0.6], [0.6, -0.4], [0.6, -0.4, 0.2]]  # AR(1,2,3)

    for ar_coeff in ar_coeffs:
        p = len(ar_coeff)
        X = sim.sim_ar(ar_coeff, n_timepoints).squeeze()
        ar_coeff_hat = ARIMA(X, order=(p, 0, 0)).fit().params[1:-1]

        assert np.allclose(ar_coeff, ar_coeff_hat, atol=0.1)


def test_sim_ar_shape():
    """Test function returns the correct shape"""
    assert sim.sim_ar([1], n_timepoints, n_repeats=n_regions).shape == (n_timepoints, n_regions)
