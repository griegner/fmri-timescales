import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src import acf_utils, sim, timescale_utils


def ols_simulation(phis, n_timepoints, ols, acm=None, n_repeats=1000, random_seed=10):
    ols_ = {}
    for idx, phi in enumerate(phis):
        if acm is None:  # simulate autoregression
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
        else:  # simulate from autocovariance matrix
            X = sim.sim_fmri(
                np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed
            )
        ols_nr_ = ols["ols_nr"].fit(X, n_timepoints)
        ols_nw_ = ols["ols_nw"].fit(X, n_timepoints)
        ols_[str(phi)] = (ols_nr_, ols_nw_)
    return ols_


def nls_simulation(phis, n_lags, n_interp, acfs=None, n_repeats=1000, random_seed=10):
    nls_ = {}
    rng = np.random.default_rng(seed=random_seed)
    ks = np.linspace(0, n_lags - 1, n_lags)
    ks_interp = np.linspace(0, n_lags - 1, (n_lags * n_interp))
    for idx, phi in enumerate(phis):
        if acfs is None:
            acf = acf_utils.ar_to_acf(phi, n_lags)
        else:
            acf = acfs[:, idx]

        acf_interp = np.interp(ks_interp, ks, acf)

        m = lambda ks, phi: phi**ks

        phis_, nr_se_, nw_se_ = (
            np.zeros(n_repeats),
            np.zeros(n_repeats),
            np.zeros(n_repeats),
        )
        for rep in range(n_repeats):
            acf_e = acf_interp + rng.normal(0, 0.2, size=(n_lags * n_interp))
            phi_, _ = curve_fit(m, ks_interp, acf_e, p0=0, bounds=(-1, +1))
            phis_[rep] = phi_.squeeze()

            # standard errors
            dm_dphi = ks_interp * phi_ ** (ks_interp - 1)
            q_ = np.mean(dm_dphi**2)
            e_ = acf_e - phi_**ks_interp

            # NR
            sigma2_ = np.mean(e_**2)
            var_ = (1 / q_) * sigma2_
            nr_se_[rep] = np.sqrt((1 / (n_lags * n_interp)) * var_)

            # NW
            u_ = dm_dphi * e_
            omega_ = (1 / (n_lags * n_interp)) * timescale_utils.newey_west_omega(u_, n_lags=3)
            var_ = (1 / q_) * omega_ * (1 / q_)
            nw_se_[rep] = np.sqrt((1 / (n_lags * n_interp)) * var_)

        nls_nr_ = {
            "phi": phis_,
            "se(phi)": nr_se_,
            "tau": -1 / np.log(phis_),
            "se(tau)": (1.0 / (phis_ * np.log(phis_) ** 2)) * nr_se_,
        }
        nls_nw_ = {
            "phi": phis_,
            "se(phi)": nw_se_,
            "tau": -1 / np.log(phis_),
            "se(tau)": (1.0 / (phis_ * np.log(phis_) ** 2)) * nw_se_,
        }

        nls_[str(phi)] = (nls_nr_, nls_nw_)

    return nls_


def plot_simulation(ols_, nls_, param, ols_params, nls_params, fig_title=None):
    assert param in ["phi", "tau"], "param must be either 'phi' or 'tau'"
    colors = ["#000000", "#B66B7C", "#8D9FCB", "#66C2A6", "#7D7D7D"]
    hist_kwargs = dict(bins=25, histtype="step", lw=1)
    vline_kwargs = dict(lw=2)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 5))
    fig.suptitle(fig_title if fig_title is not None else "", weight="bold")

    # share x-axis pairs
    axs[0, 0].sharex(axs[1, 0])

    axs[0, 1].sharex(axs[0, 2])
    axs[0, 2].sharex(axs[1, 1])
    axs[1, 1].sharex(axs[1, 2])

    for idx, phi in enumerate(ols_.keys()):
        ols_nr_, ols_nw_ = ols_[phi]
        nls_nr_, nls_nw_ = nls_[phi]

        # --- row 0 --- #

        # ols: param
        axs[0, 0].set_ylabel("OLS", weight="bold")
        axs[0, 0].set_title(r"$\hat{\tau}$" if param == "tau" else r"$\hat{\phi}$")
        axs[0, 0].hist(ols_nr_[param], color=colors[idx], **hist_kwargs)
        axs[0, 0].axvline(ols_params[idx], color=colors[idx], **vline_kwargs)

        ## ols: non-robust se(param)
        axs[0, 1].set_title(r"$se_{NR}(\hat\tau)$" if param == "tau" else r"$se_{NR}(\hat\phi)$")
        axs[0, 1].hist(ols_nr_[f"se({param})"], color=colors[idx], **hist_kwargs)
        axs[0, 1].axvline(ols_nr_[param].std(), color=colors[idx], **vline_kwargs)

        ## ols: newey-west se(param)
        axs[0, 2].set_title(r"$se_{NW}(\hat\tau)$" if param == "tau" else r"$se_{NW}(\hat\phi)$")
        axs[0, 2].hist(ols_nw_[f"se({param})"], color=colors[idx], **hist_kwargs)
        axs[0, 2].axvline(ols_nw_["tau"].std(), color=colors[idx], **vline_kwargs)

        # --- row 1 --- #

        # nls: param
        axs[1, 0].set_ylabel("NLS", weight="bold")
        axs[1, 0].set_title(r"$\hat{\tau}$" if param == "tau" else r"$\hat{\phi}$")
        axs[1, 0].hist(nls_nr_[param], color=colors[idx], **hist_kwargs)
        axs[1, 0].axvline(nls_params[idx], color=colors[idx], **vline_kwargs)

        # nls: non-robust se(param)
        axs[1, 1].set_title(r"$se_{NR}(\hat\tau)$" if param == "tau" else r"$se_{NR}(\hat\phi)$")
        axs[1, 1].hist(nls_nr_[f"se({param})"], color=colors[idx], **hist_kwargs)
        axs[1, 1].axvline(nls_nr_[param].std(), color=colors[idx], **vline_kwargs)

        # nls: newey-west se(param)
        axs[1, 2].set_title(r"$se_{NW}(\hat\tau)$" if param == "tau" else r"$se_{NW}(\hat\phi)$")
        axs[1, 2].hist(nls_nw_[f"se({param})"], color=colors[idx], **hist_kwargs)
        axs[1, 2].axvline(nls_nw_[param].std(), color=colors[idx], **vline_kwargs)

    fig.tight_layout(pad=1)
