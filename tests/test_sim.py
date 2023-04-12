import numpy as np
import pytest

from src import acf_utils, sim

n_regions, n_timepoints = 3, 1200
xcorrs = [
    np.eye(n_regions),
    np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0], [0.3, 0, 1.0]]),
]
acfs = [
    np.hstack([1, np.zeros(n_timepoints - 1)]),
    np.power(0.9, np.arange(n_timepoints)),
]
acfs = [np.tile(acf, (n_regions, 1)) for acf in acfs]


@pytest.mark.parametrize("xcorr", xcorrs)
@pytest.mark.parametrize("acf", acfs)
def test_sim_fmri(xcorr, acf):
    """Test if the generated data returns the expected {auto,cross}-correlation parameters"""

    acorr = acf_utils.acf_to_toeplitz(acf, n_timepoints)
    X = sim.sim_fmri(xcorr, acorr, n_regions, n_timepoints, random_seed=0)

    assert np.allclose(xcorr, np.corrcoef(X), atol=0.1)  # cross-correlation
    assert np.allclose(acf, acf_utils.acf_fft(X, n_timepoints), atol=0.3)  # auto-correlation


def test_sim_fmri_checkfail():
    xcorr = np.eye(n_regions)
    acorr = np.zeros((n_timepoints, n_timepoints, n_regions))
    with pytest.raises(ValueError):
        sim.sim_fmri(xcorr, acorr, n_regions, n_timepoints)
