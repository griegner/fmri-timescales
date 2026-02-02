import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import curve_fit

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


def gridsearch_n_lags(estimator, X, n_rows, var_n_lags=np.arange(1, 31)):
    """grid search var_n_lags to minimize rrmse"""
    best_n_lags = None
    best_rrmse = np.inf
    for n_lags in var_n_lags:
        nw = estimator.set_params(var_estimator="newey-west", var_n_lags=n_lags)
        nw.fit(X, n_rows)
        se_true = nw.estimates_["tau"].std()
        se_estimates = nw.estimates_["se(tau)"]
        rrmse_ = rrmse(se_true, se_estimates)
        if rrmse_ < best_rrmse:
            best_rrmse = rrmse_
            best_n_lags = n_lags
    return best_n_lags


def run_simulation(phis, n_timepoints, n_lags, estimators, acm=None, n_repeats=1000, random_seed=0):
    """simulate realizations of AR1, AR2, HCP and fit estimators"""

    # simulation settings
    rng = np.random.default_rng(seed=random_seed)
    sfreq = 10
    ks, ks_interp = np.arange(n_lags), np.linspace(0, n_lags - 1, (n_lags * sfreq))
    ar1_phis = np.linspace(0.1, 0.8, 5)
    scales = np.sqrt(1 / (1 - ar1_phis**2)) * (1 / sfreq)  # match var of AR1 w sigma^2=1

    # generate X in time domain, X_acf in autocorrelation domain
    results = {}
    for idx, phi in enumerate(phis):
        results[idx] = {}

        if acm is None:
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
            X_acf = (
                acf_utils.ar_to_acf(phi, n_lags=n_lags, sfreq=sfreq)
                .repeat(n_repeats)
                .reshape(n_lags * sfreq, n_repeats)
            )
            X_acf += rng.normal(0, scales[idx], size=X_acf.shape)

        else:
            X = sim.sim_fmri(np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed)
            X_acf = np.interp(ks_interp, ks, acm[:n_lags, 0, idx]).repeat(n_repeats).reshape(n_lags * sfreq, n_repeats)
            X_acf += rng.normal(0, scales[idx], size=X_acf.shape)

        for name, estimator in estimators.items():
            if name not in results[idx]:
                results[idx][name] = {}

            if "aa" in name:
                estimator.set_params(lag_interval=sfreq)
                if "nw" in name:
                    var_n_lags = gridsearch_n_lags(estimator, X_acf[:, :100], n_rows=n_lags * sfreq)
                    results[idx][name]["var_n_lags"] = var_n_lags
                    estimator.set_params(var_n_lags=var_n_lags)
                results[idx][name].update(estimator.fit(X_acf, n_lags * sfreq).estimates_)
            else:
                if "nw" in name:
                    var_n_lags = gridsearch_n_lags(estimator, X[:, :100], n_rows=n_timepoints)
                    results[idx][name]["var_n_lags"] = var_n_lags
                    estimator.set_params(var_n_lags=var_n_lags)
                results[idx][name].update(estimator.fit(X, n_timepoints).estimates_)

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

        # ad TT/TA: tau
        true, estimates_ = ad_taus[idx], result["ad_tt_nr"]["tau"]
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

        # ad var_domain="time": naive se(tau)
        true, estimates_ = result["ad_tt_nr"]["tau"].std(), result["ad_tt_nr"]["se(tau)"]
        axs["e"].set_ylabel("AD/TD")
        axs["e"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["e"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["e"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["e"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["e"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        # ad var_domain="time": newey-west se(tau)
        true, estimates_ = result["ad_tt_nw"]["tau"].std(), result["ad_tt_nw"]["se(tau)"]
        axs["f"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["f"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["f"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["f"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["f"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig


def run_simulation(phis, n_timepoints, n_lags, estimators, acm=None, n_repeats=1000, random_seed=0):
    """simulate realizations of AR1, AR2, HCP and fit estimators"""

    # simulation settings
    rng = np.random.default_rng(seed=random_seed)
    sfreq = 10
    ks, ks_interp = np.arange(n_lags), np.linspace(0, n_lags - 1, (n_lags * sfreq))
    ar1_phis = np.linspace(0.1, 0.8, 5)
    scales = np.sqrt(1 / (1 - ar1_phis**2)) * (1 / sfreq)  # match var of AR1 w sigma^2=1

    # generate X in time domain, X_acf in autocorrelation domain
    results = {}
    for idx, phi in enumerate(phis):
        results[idx] = {}

        if acm is None:
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
            X_acf = (
                acf_utils.ar_to_acf(phi, n_lags=n_lags, sfreq=sfreq)
                .repeat(n_repeats)
                .reshape(n_lags * sfreq, n_repeats)
            )
            X_acf += rng.normal(0, scales[idx], size=X_acf.shape)

        else:
            X = sim.sim_fmri(np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed)
            X_acf = np.interp(ks_interp, ks, acm[:n_lags, 0, idx]).repeat(n_repeats).reshape(n_lags * sfreq, n_repeats)
            X_acf += rng.normal(0, scales[idx], size=X_acf.shape)

        for name, estimator in estimators.items():
            if name not in results[idx]:
                results[idx][name] = {}

            if "aa" in name:
                estimator.set_params(lag_interval=sfreq)
                if "nw" in name:
                    var_n_lags = gridsearch_n_lags(estimator, X_acf[:, :100], n_rows=n_lags * sfreq)
                    results[idx][name]["var_n_lags"] = var_n_lags
                    estimator.set_params(var_n_lags=var_n_lags)
                results[idx][name].update(estimator.fit(X_acf, n_lags * sfreq).estimates_)
            else:
                if "nw" in name:
                    var_n_lags = gridsearch_n_lags(estimator, X[:, :100], n_rows=n_timepoints)
                    results[idx][name]["var_n_lags"] = var_n_lags
                    estimator.set_params(var_n_lags=var_n_lags)
                results[idx][name].update(estimator.fit(X, n_timepoints).estimates_)

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

        # ad TT/TA: tau
        true, estimates_ = ad_taus[idx], result["ad_tt_nr"]["tau"]
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

        # ad var_domain="time": naive se(tau)
        true, estimates_ = result["ad_tt_nr"]["tau"].std(), result["ad_tt_nr"]["se(tau)"]
        axs["e"].set_ylabel("AD/TD")
        axs["e"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["e"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["e"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["e"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["e"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        # ad var_domain="time": newey-west se(tau)
        true, estimates_ = result["ad_tt_nw"]["tau"].std(), result["ad_tt_nw"]["se(tau)"]
        axs["f"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["f"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["f"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["f"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["f"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig


def plot_simulation_sup(results, td_taus, ad_taus):
    """compare true and estimated timescales + standard errors"""

    hist_range = lambda estimates_: (
        np.min(estimates_),
        np.mean(estimates_) + 3 * np.std(estimates_),
    )

    colors = ["#313695", "#72ABD0", "#FEDE8E", "#F57245", "#A70226"]
    hist_kwargs = dict(bins=25, histtype="step", lw=3)
    vline_kwargs = dict(lw=5, ls="--")
    scatter_kwargs = dict(s=100, lw=2)

    layout = """
    aA
    bB
    cC
    """

    # subplot setup
    fig, axs = plt.subplot_mosaic(
        layout, figsize=(23.8, 9), layout="constrained", gridspec_kw={"wspace": 0.1, "hspace": 0.2}
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
        ax.axhline(y=0.2, c="k", ls="--")

    # share axis pairs
    share_pairs = [("a", "A"), ("A", "b"), ("b", "B"), ("B", "c"), ("c", "C")]
    for pair in share_pairs:
        axs[pair[0]].sharex(axs[pair[1]])
        axs_inset[pair[0]].sharey(axs_inset[pair[1]])

    for idx, result in results.items():
        # --- row 0 --- #
        axs["a"].set_ylabel("AD")
        true, estimates_ = result["ar1"]["ad_aa_nr"]["tau"].std(), result["ar1"]["ad_aa_nr"]["se(tau)"]
        axs["a"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["a"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["a"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["a"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["a"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        true, estimates_ = result["ar1"]["ad_aa_nw"]["tau"].std(), result["ar1"]["ad_aa_nw"]["se(tau)"]
        axs["A"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["A"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["A"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["A"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["A"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

        # --- row 1 --- #
        axs["b"].set_ylabel("AD")
        true, estimates_ = result["ar2"]["ad_aa_nr"]["tau"].std(), result["ar2"]["ad_aa_nr"]["se(tau)"]
        axs["b"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["b"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["b"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["b"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["b"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        true, estimates_ = result["ar2"]["ad_aa_nw"]["tau"].std(), result["ar2"]["ad_aa_nw"]["se(tau)"]
        axs["B"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["B"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["B"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["B"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["B"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

        # --- row 2 --- #
        axs["c"].set_ylabel("AD")
        true, estimates_ = result["hcp"]["ad_aa_nr"]["tau"].std(), result["hcp"]["ad_aa_nr"]["se(tau)"]
        axs["c"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau^*_\text{AD})$")
        axs["c"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["c"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["c"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["c"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        true, estimates_ = result["hcp"]["ad_aa_nw"]["tau"].std(), result["hcp"]["ad_aa_nw"]["se(tau)"]
        axs["C"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{AD})$")
        axs["C"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), ls="--", **hist_kwargs)
        axs["C"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["C"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )
        axs_inset["C"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig
