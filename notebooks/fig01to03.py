import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import curve_fit

from fmri_timescales import acf_utils, sim, timescale_utils


def lls_simulation(phis, n_timepoints, lls, acm=None, n_repeats=1000, random_seed=10):
    lls_ = {}
    for idx, phi in enumerate(phis):
        if acm is None:  # simulate autoregression
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
        else:  # simulate from autocovariance matrix
            X = sim.sim_fmri(
                np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed
            )
        lls_nr_ = lls["lls_nr"].fit(X, n_timepoints)
        lls_nw_ = lls["lls_nw"].fit(X, n_timepoints)
        lls_[str(phi)] = (lls_nr_, lls_nw_)
    return lls_


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


def plot_simulation(lls_, nls_, lls_params, nls_params):
    colors = ["#313695", "#72ABD0", "#FEDE8E", "#F57245", "#A70226"]
    hist_kwargs = dict(bins=25, histtype="step", lw=3)
    vline_kwargs = dict(lw=5)

    layout = """
    .b
    .B
    ..
    cd
    CD
    ..
    ef
    EF
    """

    # subplot setup
    fig, axs = plt.subplot_mosaic(layout, figsize=(24, 18), layout="constrained")

    # tick formatting
    for ax in axs.values():
        ax.yaxis.set_major_formatter(mt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 1))
        ax.yaxis.get_offset_text().set_fontsize(15)

    # share x-axis pairs
    axs["b"].sharex(axs["B"])

    axs["c"].sharex(axs["d"])
    axs["d"].sharex(axs["C"])
    axs["C"].sharex(axs["D"])

    axs["e"].sharex(axs["E"])
    axs["f"].sharex(axs["F"])

    for idx, phi in enumerate(lls_.keys()):
        lls_nr_, lls_nw_ = lls_[phi]
        nls_nr_, nls_nw_ = nls_[phi]

        # --- row 0 --- #

        # lls: tau
        axs["b"].set_ylabel("LLS")
        axs["b"].set_xlabel(r"$\hat\tau_\text{LLS}$")
        axs["b"].hist(lls_nr_["tau"], color=colors[idx], **hist_kwargs)
        axs["b"].axvline(lls_params[idx], color=colors[idx], **vline_kwargs)

        # --- row 1 --- #

        # nls: tau
        axs["B"].set_ylabel("NLS")
        axs["B"].set_xlabel(r"$\hat\tau_\text{NLS}$")
        axs["B"].hist(nls_nr_["tau"], color=colors[idx], **hist_kwargs)
        axs["B"].axvline(nls_params[idx], color=colors[idx], **vline_kwargs)

        # --- row 2 --- #

        ## lls: naive se(tau)
        axs["c"].set_ylabel("LLS")
        axs["c"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau_\text{LLS})$")
        axs["c"].hist(lls_nr_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["c"].axvline(lls_nr_["tau"].std(), color=colors[idx], **vline_kwargs)

        ## lls: newey-west se(tau)
        axs["d"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau_\text{LLS})$")
        axs["d"].hist(lls_nw_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["d"].axvline(lls_nw_["tau"].std(), color=colors[idx], **vline_kwargs)

        # --- row 3 --- #

        # nls: naive se(tau)
        axs["C"].set_ylabel("NLS")
        axs["C"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau_\text{NLS})$")
        axs["C"].hist(nls_nr_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["C"].axvline(nls_nr_["tau"].std(), color=colors[idx], **vline_kwargs)

        # nls: newey-west se(tau)
        axs["D"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau_\text{NLS})$")
        axs["D"].hist(nls_nw_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["D"].axvline(nls_nw_["tau"].std(), color=colors[idx], **vline_kwargs)

        # --- row 4 --- #

        ## lls:  tau / newey-west se(tau)
        axs["e"].set_ylabel("LLS")
        axs["e"].set_xlabel(
            r"$\hat\tau_\text{LLS} \;/\; \widehat{se}_\text{NW}(\hat\tau_\text{LLS})$"
        )
        axs["e"].hist(lls_nw_["tau"] / lls_nw_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["e"].axvline(lls_params[idx] / lls_nw_["tau"].std(), color=colors[idx], **vline_kwargs)

        ## lls: newey-west se(tau) / tau
        axs["f"].set_xlabel(
            r"$\widehat{se}_\text{NW}(\hat\tau_\text{LLS}) \;/\; \hat\tau_\text{LLS}$"
        )
        axs["f"].hist(lls_nw_["se(tau)"] / lls_nw_["tau"], color=colors[idx], **hist_kwargs)
        axs["f"].axvline(lls_nw_["tau"].std() / lls_params[idx], color=colors[idx], **vline_kwargs)

        # --- row 5 --- #

        ## nls:  tau / newey-west se(tau)
        axs["E"].set_ylabel("NLS")
        axs["E"].set_xlabel(
            r"$\hat\tau_\text{NLS} \;/\; \widehat{se}_\text{NW}(\hat\tau_\text{NLS})$"
        )
        axs["E"].hist(nls_nw_["tau"] / nls_nw_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs["E"].axvline(nls_params[idx] / nls_nw_["tau"].std(), color=colors[idx], **vline_kwargs)

        ## nls: newey-west se / tau
        axs["F"].set_xlabel(
            r"$\widehat{se}_\text{NW}(\hat\tau_\text{NLS}) \;/\; \hat\tau_\text{NLS}$"
        )
        axs["F"].hist(nls_nw_["se(tau)"] / nls_nw_["tau"], color=colors[idx], **hist_kwargs)
        axs["F"].axvline(nls_nw_["tau"].std() / nls_params[idx], color=colors[idx], **vline_kwargs)

    return fig
