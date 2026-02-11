import fig01to03
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np

from fmri_timescales import sim


def run_simulation(phis, n_timepoints, estimator, n_repeats=1000, random_seed=0):
    """simulate realizations of AR1, AR2, HCP and fit estimators"""

    # simulation settings
    rng = np.random.default_rng(seed=random_seed)

    if isinstance(phis, float):  # measurement noise
        # generate X with innovations, then add IID measurement noise
        stds = np.sqrt([2, 1.5, 1, 0.5, 0])
        X = sim.sim_ar(phis, n_timepoints=n_timepoints, n_repeats=n_repeats, random_seed=rng)
        results = {}
        for idx, std in enumerate(stds):
            X_mn = X + rng.normal(loc=0, scale=std, size=(n_timepoints, n_repeats))
            var_n_lags = fig01to03.gridsearch_n_lags(
                estimator, X_mn[:, :200], n_rows=n_timepoints, var_n_lags=np.arange(1, 50, 5)
            )
            print(var_n_lags)
            estimator.set_params(var_n_lags=var_n_lags)
            results[idx] = estimator.fit(X_mn, n_timepoints).estimates_

    else:  # oscillations
        results = {}
        for idx, phi in enumerate(phis):
            results[idx] = {}
            X = sim.sim_ar(phi, n_timepoints=n_timepoints, n_repeats=n_repeats)
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            var_n_lags = fig01to03.gridsearch_n_lags(
                estimator, X[:, :200], n_rows=n_timepoints, var_n_lags=np.arange(1, 200, 5)
            )
            print(var_n_lags)
            estimator.set_params(var_n_lags=var_n_lags)
            results[idx] = estimator.fit(X, n_timepoints).estimates_

    return results


def plot_simulation(results, tau):
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
    a
    b
    """

    fig, axs = plt.subplot_mosaic(layout, figsize=(10, 5), layout="constrained")

    axs_inset = {k: v.inset_axes([1.1, 0, 0.3, 0.99]) for k, v in axs.items()}
    for k, ax in axs_inset.items():
        ax.yaxis.set_major_formatter(mt.FormatStrFormatter("%.2f"))
        ax.set_xticklabels("")
        if k == "a":
            ax.axhline(y=0.1, c="k", ls="--")
        else:
            ax.axhline(y=0.2, c="k", ls="--")

    for ax in axs.values():
        ax.yaxis.set_major_formatter(mt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 1))
        ax.yaxis.get_offset_text().set_fontsize(15)

    for idx, result in results.items():

        # tau
        true, estimates_ = tau, result["tau"]
        axs["a"].set_ylabel("TD")
        axs["a"].set_xlabel(r"$\hat\tau^*_\text{TD}$")
        axs["a"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["a"].axvline(tau, color="k", **vline_kwargs)
        axs_inset["a"].scatter(idx, fig01to03.rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["a"].set_xlabel(r"$\text{rRMSE}(\hat\tau^*_\text{TD})$")

        # robust se(tau)
        true, estimates_ = result["tau"].std(), result["se(tau)"]
        axs["b"].set_ylabel("TD")
        axs["b"].set_xlabel(r"$\widehat{se}_\text{NW}(\hat\tau^*_\text{TD})$")
        axs["b"].hist(estimates_, color=colors[idx], range=hist_range(estimates_), **hist_kwargs)
        axs["b"].axvline(true, color=colors[idx], **vline_kwargs)
        axs_inset["b"].scatter(idx, fig01to03.rrmse(true, estimates_), color=colors[idx], **scatter_kwargs)
        axs_inset["b"].set_xlabel(r"$\text{rRMSE}(\widehat{se}_\text{NW})$")

    return fig
