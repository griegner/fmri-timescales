import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import curve_fit

from fmri_timescales import acf_utils, sim, timescale_utils


def run_simulation(phis, n_timepoints, models, acm=None, n_repeats=1000, random_seed=10):
    results = {}
    for idx, phi in enumerate(phis):
        if acm is None:  # simulate autoregression
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
        else:  # simulate from autocovariance matrix
            X = sim.sim_fmri(
                np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed
            )

        results[str(phi)] = {m_name: m.fit(X, n_timepoints) for m_name, m in models.items()}

    return results


def plot_simulation(results, lls_params, nls_params):
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

    for idx, phi in enumerate(results.keys()):
        lls_nr_ = results[phi]["lls_nr"]
        lls_nw_ = results[phi]["lls_nw"]
        nls_nr_ = results[phi]["nls_nr"]
        nls_nw_ = results[phi]["nls_nw"]

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
