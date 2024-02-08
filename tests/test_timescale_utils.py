import numpy as np
import pytest

from src import sim, timescale_utils

n_regions, n_timepoints = 1, 4800


def _calc_ar_bias(phi, n_repeats=1000, random_seed=0, model="ols"):
    """repeate simulations of AR(p) process with fixed coefficients phi_1, phi_2, ..., phi_p"""

    models = dict(
        ols=timescale_utils.estimate_timescales_ols, nls=timescale_utils.estimate_timescales_nls
    )

    X = np.zeros((n_repeats, n_timepoints))
    for idx in range(n_repeats):
        X[idx, :] = sim.sim_ar(phi, n_timepoints, random_seed=idx)

    return models[model](X, n_repeats)


@pytest.mark.parametrize("model", ["ols", "nls"])
def test_estimate_timescales(model):
    """Test if the models estimate the expected coefficients and standard errors of an AR(1) process"""
    # AR(1) process
    phi = 0.75
    df = _calc_ar_bias(phi, model=model)
    tau = -1 / np.log(phi)  # true timescale
    # timescales: true vs estimate (mean of timescales distribution)
    assert np.isclose(tau, df["tau"].mean(), atol=0.022)
    # stderrs: true (std of timescales) vs estimate (mean of stderr distribution)
    assert np.isclose(df["tau"].std(), df["se(tau)"].mean(), atol=0.18)


@pytest.mark.parametrize("model", ["ols", "nls"])
def test_estimate_timescales_checkfail(model):
    """Test if the functions raise the expected ValueError"""

    models = dict(
        ols=timescale_utils.estimate_timescales_ols, nls=timescale_utils.estimate_timescales_nls
    )

    n_regions, n_timepoints = 5, 4800
    X = np.random.random(size=(n_timepoints, n_regions))

    with pytest.raises(ValueError):
        models[model](X, n_regions)
