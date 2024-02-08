import numpy as np
import pytest

from src import sim, timescale_utils

n_timepoints, n_repeats = 4800, 1000


@pytest.mark.parametrize(
    "model", [timescale_utils.estimate_timescales_ols, timescale_utils.estimate_timescales_nls]
)
def test_estimate_timescales(model):
    """Test if the models estimate the expected coefficients and standard errors of an AR(1) process"""

    # AR(1) process
    phi = 0.75
    tau = -1 / np.log(phi)  # true timescale

    X = sim.sim_ar(phi, n_timepoints, n_repeats)
    df = model(X, n_repeats)
    # timescales: true vs estimate (mean of timescales distribution)
    assert np.isclose(tau, df["tau"].mean(), atol=0.01)
    # stderrs: true (std of timescales) vs estimate (mean of stderr distribution)
    assert np.isclose(df["tau"].std(), df["se(tau)"].mean(), atol=0.19)


@pytest.mark.parametrize(
    "model", [timescale_utils.estimate_timescales_ols, timescale_utils.estimate_timescales_nls]
)
def test_estimate_timescales_checkfail(model):
    """Test if the functions raise the expected ValueError"""

    n_regions, n_timepoints = 5, 4800
    X = np.random.random(size=(n_timepoints, n_regions))

    with pytest.raises(ValueError):
        model(X, n_regions)
