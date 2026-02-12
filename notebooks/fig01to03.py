import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.base import clone

from fmri_timescales import acf_utils, sim

rrmse = lambda true, estimates_: np.sqrt(np.mean((estimates_ - true) ** 2)) / np.abs(true)


def get_ad_params(coeffs, coeff_type, n_timepoints):
    """theoretical ad phi parameters"""

    if coeff_type == "ar_coeffs":
        coeffs = acf_utils.ar_to_acf(coeffs, n_lags=n_timepoints)
    elif coeff_type != "acf":
        raise ValueError("coeff_type in 'ar_coeffs' or 'acf'")

    ks = np.arange(len(coeffs))
    m = lambda ks, phi: phi**ks
    phi, _ = curve_fit(f=m, xdata=ks, ydata=coeffs, p0=1e-2, bounds=(-1, +1), ftol=1e-6)
    return phi.squeeze()


def gridsearch_var_n_lags(estimator, X, n_rows, search_space=np.arange(0, 10)):
    """grid search var_n_lags to minimize rRMSE(std(tau), se(tau))"""
    best_lags, best_score = None, np.inf
    for n_lags in search_space:
        est = estimator.set_params(var_n_lags=n_lags)
        est.fit(X, n_rows)
        score = rrmse(est.estimates_["tau"].std(), est.estimates_["se(tau)"])
        if score < best_score:
            best_score, best_lags = score, n_lags
    return best_lags


def run_simulation(phis, n_timepoints, estimators, acm=None, n_repeats=1000, random_seed=0):
    """simulate realizations of AR1, AR2, HCP and fit estimators"""

    results = {}
    for idx, phi in enumerate(phis):
        X = (
            sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
            if acm is None
            else sim.sim_fmri(np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed)
        )

        results[idx] = {}
        for name, estimator in estimators.items():
            est = clone(estimator)
            results[idx][name] = {}

            if "nw" in name:
                var_n_lags = gridsearch_var_n_lags(est, X[:, :100], n_rows=n_timepoints)
                results[idx][name]["var_n_lags"] = var_n_lags
                est = est.set_params(var_n_lags=var_n_lags)

            results[idx][name].update(est.fit(X, n_timepoints).estimates_)

    return results


def plot_simulation(results, td_taus, ad_taus):
    """compare true and estimated timescales + standard errors"""

    hist_range = lambda estimates_: (
        np.min(estimates_),
        np.mean(estimates_) + 3 * np.std(estimates_),
    )

    colors = ["#313695", "#72ABD0", "#FEDE8E", "#F57245", "#A70226"]
    hist_kwargs = dict(bins=25, histtype="step", lw=3)
    vline_kwargs = dict(lw=5)
    scatter_kwargs = dict(s=100, lw=2)

    layout = """
    .b
    .B
    ..
    CD
    ef
    """

    # subplot setup
    fig, axs = plt.subplot_mosaic(
        layout,
        figsize=(24, 11.4),
        layout="constrained",
        gridspec_kw={"wspace": 0.1, "height_ratios": [1, 1, 0.5, 1, 1]},
    )
    axs_inset = {k: v.inset_axes([1.1, 0, 0.3, 0.99]) for k, v in axs.items()}

    # tick formatting
    for ax in axs.values():
        ax.yaxis.set_major_formatter(mt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 1))
        ax.yaxis.get_offset_text().set_fontsize(15)

    for k, ax in axs_inset.items():
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.2f"))
        ax.set_xticklabels("")
        if k in ["b", "B", "d"]:
            ax.axhline(y=0.1, c="k", ls="--")
        else:
            ax.axhline(y=0.2, c="k", ls="--")

    # share axis pairs
    share_pairs = [("b", "B"), ("C", "D"), ("D", "e"), ("e", "f")]
    for pair in share_pairs:
        axs[pair[0]].sharex(axs[pair[1]])
        axs_inset[pair[0]].sharey(axs_inset[pair[1]])

    for idx, result in results.items():
        # --- row 0 --- #

        # td: tau
        true, estimates_ = td_taus[idx], result["td_nr"]["tau"]
        axs["b"].set_ylabel("TD")
        axs["b"].set_xlabel(r"$\hat\tau^*_\text{TD}$")
        axs["b"].hist(estimates_, color=colors[idx], **hist_kwargs)
        axs["b"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["b"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["b"].set_xlabel(r"$\text{rRMSE}(\hat\tau^*_\text{TD})$")

        # --- row 1 --- #

        # ad: tau
        true, estimates_ = ad_taus[idx], result["ad_nr"]["tau"]
        axs["B"].set_ylabel("AD")
        axs["B"].set_xlabel(r"$\hat\tau^*_\text{AD}$")
        axs["B"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["B"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["B"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["B"].set_xlabel(r"$\text{rRMSE}(\hat\tau^*_\text{AD})$")

        # --- row 2 --- #

        ## td: naive se(tau)
        true, estimates_ = result["td_nr"]["tau"].std(), result["td_nr"]["se(tau)"]
        axs["C"].set_ylabel("TD")
        axs["C"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{TD})$")
        axs["C"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["C"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["C"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["C"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        ## td: newey-west se(tau)
        true, estimates_ = result["td_nw"]["tau"].std(), result["td_nw"]["se(tau)"]
        axs["D"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{TD})$")
        axs["D"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["D"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["D"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["D"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

        # --- row 3 --- #

        # ad: naive se(tau)
        true, estimates_ = result["ad_nr"]["tau"].std(), result["ad_nr"]["se(tau)"]
        axs["e"].set_ylabel("AD")
        axs["e"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["e"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["e"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["e"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["e"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        # ad: newey-west se(tau)
        true, estimates_ = result["ad_nw"]["tau"].std(), result["ad_nw"]["se(tau)"]
        axs["f"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["f"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["f"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["f"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["f"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig
