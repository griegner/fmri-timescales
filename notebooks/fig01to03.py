import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from scipy.optimize import curve_fit

from fmri_timescales import acf_utils, sim

rrmse = lambda true, estimates_: np.sqrt(np.mean((estimates_ - true) ** 2)) / np.abs(true)


def get_nls_params(coeffs, coeff_type, n_timepoints):
    """theoretical nls phi parameters"""

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
        lls_nw = estimator.set_params(var_estimator="newey-west", var_n_lags=n_lags)
        lls_nw.fit(X, n_rows)
        se_true = lls_nw.estimates_["tau"].std()
        se_estimates = lls_nw.estimates_["se(tau)"]
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
                estimator.set_params(X_sfreq=sfreq)
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


def plot_simulation(results, lls_taus, nls_taus):
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
    .d
    ..
    CD
    ef
    EF
    """

    # subplot setup
    fig, axs = plt.subplot_mosaic(layout, figsize=(24, 15), layout="constrained")
    axs_inset = {k: v.inset_axes([1.1, 0, 0.3, 0.99]) for k, v in axs.items()}

    # tick formatting
    for ax in axs.values():
        ax.yaxis.set_major_formatter(mt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 1))
        ax.yaxis.get_offset_text().set_fontsize(15)

    for k, ax in axs_inset.items():
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.1f"))
        ax.set_xticklabels("")
        if k in ["b", "B", "d"]:
            ax.axhline(y=0.1, c="k", ls="--")
        else:
            ax.axhline(y=0.2, c="k", ls="--")

    # share axis pairs
    share_pairs = [("b", "B"), ("B", "d"), ("C", "D"), ("D", "e"), ("e", "f"), ("f", "E"), ("E", "F")]
    for pair in share_pairs:
        axs[pair[0]].sharex(axs[pair[1]])
        axs_inset[pair[0]].sharey(axs_inset[pair[1]])

    for idx, result in results.items():
        # --- row 0 --- #

        # lls: tau
        true, estimates_ = lls_taus[idx], result["lls_nr"]["tau"]
        axs["b"].set_ylabel("LLS")
        axs["b"].set_xlabel(r"$\hat\tau_\text{LLS}$")
        axs["b"].hist(estimates_, color=colors[idx], **hist_kwargs)
        axs["b"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["b"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["b"].set_xlabel(r"$\text{rRMSE}(\hat\tau_\text{LLS})$")

        # --- row 1 --- #

        # nls TT/TA: tau
        true, estimates_ = nls_taus[idx], result["nls_tt_nr"]["tau"]
        axs["B"].set_ylabel("NLS")
        axs["B"].set_xlabel(r"$\hat\tau_\text{NLS}$")
        axs["B"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["B"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["B"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["B"].set_xlabel(r"$\text{rRMSE}(\hat\tau_\text{NLS})$")

        # --- row 2 --- #

        # nls AA: tau
        true, estimates_ = nls_taus[idx], result["nls_aa_nr"]["tau"]
        axs["d"].set_ylabel("NLS")
        axs["d"].set_xlabel(r"$\hat\tau_\text{NLS}$")
        axs["d"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["d"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["d"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["d"].set_xlabel(r"$\text{rRMSE}(\hat\tau_\text{NLS})$")

        # --- row 3 --- #

        ## lls: naive se(tau)
        true, estimates_ = result["lls_nr"]["tau"].std(), result["lls_nr"]["se(tau)"]
        axs["C"].set_ylabel("LLS")
        axs["C"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau_\text{LLS})$")
        axs["C"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["C"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["C"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["C"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        ## lls: newey-west se(tau)
        true, estimates_ = result["lls_nw"]["tau"].std(), result["lls_nw"]["se(tau)"]
        axs["D"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau_\text{LLS})$")
        axs["D"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["D"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["D"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["D"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

        # --- row 4 --- #

        # nls var_domain="autocorrelation": naive se(tau)
        true, estimates_ = result["nls_tt_nr"]["tau"].std(), result["nls_tt_nr"]["se(tau)"]
        axs["e"].set_ylabel("NLS")
        axs["e"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau_\text{NLS})$")
        axs["e"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["e"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["e"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["e"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        true, estimates_ = result["nls_ta_nr"]["tau"].std(), result["nls_ta_nr"]["se(tau)"]
        axs["e"].hist(estimates_, color=colors[idx], alpha=0.75, ls="--", **hist_kwargs)
        axs_inset["e"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )

        # nls var_domain="autocorrelation": newey-west se(tau)
        true, estimates_ = result["nls_tt_nw"]["tau"].std(), result["nls_tt_nw"]["se(tau)"]
        axs["f"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau_\text{NLS})$")
        axs["f"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["f"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["f"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["f"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

        true, estimates_ = result["nls_ta_nw"]["tau"].std(), result["nls_ta_nw"]["se(tau)"]
        axs["f"].hist(estimates_, color=colors[idx], alpha=0.75, ls="--", range=hist_range(estimates_), **hist_kwargs)
        axs_inset["f"].scatter(
            idx, rrmse(true, estimates_), color=colors[idx], ls="--", facecolors="none", **scatter_kwargs
        )

        # --- row 5 --- #

        # nls var_domain="time": naive se(tau)
        true, estimates_ = result["nls_aa_nr"]["tau"].std(), result["nls_aa_nr"]["se(tau)"]
        axs["E"].set_ylabel("NLS")
        axs["E"].set_xlabel(r"$\widehat{se}_\text{Naive}(\hat\tau_\text{NLS})$")
        axs["E"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["E"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["E"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["E"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{Naive})$")

        # nls var_domain="time": newey-west se(tau)
        true, estimates_ = result["nls_aa_nw"]["tau"].std(), result["nls_aa_nw"]["se(tau)"]
        axs["F"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau_\text{NLS})$")
        axs["F"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["F"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["F"].scatter(idx, rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["F"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig
