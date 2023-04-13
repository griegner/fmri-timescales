import numpy as np
import pytest

from src import acf_utils, sim

# set parameters
n_regions, n_timepoints = 3, 1200
xcorrs = [
    np.eye(n_regions),
    np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]]),
]
acorrs = [
    np.eye(n_timepoints),
    acf_utils.acf_to_toeplitz(np.load("tests/data/fmri-acf.npy"), n_timepoints),
]


@pytest.mark.parametrize("xcorr", xcorrs)
@pytest.mark.parametrize("acorr", acorrs)
def test_sim_fmri(xcorr, acorr):
    """Test if the generated data returns the expected {auto,cross}-correlation parameters"""
    xcorr_corrected = True if acorr.ndim == 3 else False
    acf = acorr[:, 0] if acorr.ndim == 2 else acorr[:, :, 0]
    X = sim.sim_fmri(xcorr, acorr, n_regions, n_timepoints, random_seed=0)

    # cross-correlation
    assert np.allclose(xcorr, acf_utils.calc_xcorr(X, n_timepoints, xcorr_corrected), atol=0.1)
    # auto-correlation
    assert np.allclose(acf, acf_utils.acf_fft(X, n_timepoints), atol=0.3)


def test_sim_fmri_checkfail():
    """Test if the function raises the expected ValueError"""
    xcorr = np.eye(n_regions)
    acorr = np.zeros((n_timepoints, n_timepoints, n_regions))

    with pytest.raises(ValueError):
        sim.sim_fmri(xcorr, acorr, n_regions, n_timepoints)
