import matplotlib.pyplot as plt
import numpy as np

from src import acf_utils, sim, timescale_utils


def bias_std_rmse(true, estimates):
    bias = np.mean(estimates) - true
    std = np.std(estimates)
    rmse = np.sqrt(np.mean((estimates - true) ** 2))
    return bias, std, rmse


def plot_nls_vs_ols(X, n_timepoints, nls_, ols_):
    acf = acf_utils.ACF(n_jobs=-2)
    X_acf_ = acf.fit_transform(X, n_timepoints).squeeze()
    timepoints = np.arange(n_timepoints)

    # nls
    exp_decay = lambda x, tau: np.exp(-x / (tau if tau > 0 else 1e-6))
    nls_fit = exp_decay(timepoints, nls_["tau"])
    nls_residuals = X_acf_ - nls_fit

    fig, axs = plt.subplots(2, 1, figsize=(15, 4), sharex=True)

    axs[0].set_title("non-linear least squares")
    axs[0].plot(timepoints, X_acf_, lw=1, alpha=0.5, label="ACF")
    axs[0].plot(timepoints, nls_fit, c="k", lw=1, label="fit")
    axs[0].set_ylabel(r"$\hat \rho_k$")
    axs[0].annotate(f"<-- timescale = {nls_['tau']}", (nls_["tau"], 1 / np.e))
    axs[0].legend()

    axs[1].scatter(timepoints, nls_residuals, s=1, alpha=0.5, label="residuals")
    axs[1].set_ylabel(r"$\hat e_k$")
    axs[1].set_xlabel(r"lag $k$")
    axs[1].legend()
    fig.tight_layout()

    # ols
    ols_fit = ols_["phi"] * X[:-1]
    ols_residuals = X[1:] - ols_fit

    fig, axs = plt.subplots(1, 2, figsize=(15, 3))

    axs[0].set_title("ordinary least squares")

    axs[0].scatter(X[:-1], X[1:], s=4, alpha=0.5, label="AR1")

    axs[0].plot(X[:-1], ols_fit, c="k", lw=1, label="fit")
    axs[0].annotate(
        f"slope = {ols_['phi']}, timescale = {ols_['tau']}\nslope se = {ols_['se(phi)']}, timescale se = {ols_['se(tau)']}",
        (0, -5),
    )
    axs[0].set_xlabel(r"$X_{t-1}$")
    axs[0].set_ylabel(r"$X_t$")
    axs[0].legend()

    axs[1].scatter(X[:-1], ols_residuals, s=4, alpha=0.5, label="residuals")
    axs[1].set_xlabel(r"$X_t$")
    axs[1].set_ylabel(r"$\hat e_t$")
    axs[1].legend()
    fig.tight_layout()


def mc_simulation(phis, n_timepoints, estimators, acm=None, n_repeats=1000, random_seed=10):
    estimates_ = {}
    for idx, phi in enumerate(phis):
        if acm is None:  # simulate autoregression
            X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
        else:  # simulate from autocovariance matrix
            X = sim.sim_fmri(
                np.eye(n_repeats), acm[..., idx], n_repeats, n_timepoints, random_seed=random_seed
            )
        ols_nr_ = estimators["ols_nr"].fit(X, n_timepoints)
        ols_nw_ = estimators["ols_nw"].fit(X, n_timepoints)
        nls_nr_ = estimators["nls_nr"].fit(X, n_timepoints)
        nls_nw_ = estimators["nls_nw"].fit(X, n_timepoints)
        estimates_[str(phi)] = (ols_nr_, ols_nw_, nls_nr_, nls_nw_)
    return estimates_


def plot_simulation(estimates_, param, ols_params, nls_params, fig_title=None):
    assert param in ["phi", "tau"], "param must be either 'phi' or 'tau'"
    colors = ["#000000", "#B66B7C", "#8D9FCB", "#66C2A6", "#7D7D7D"]
    hist_kwargs = dict(bins=25, histtype="step", lw=1)
    vline_kwargs = dict(lw=2)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 4))
    fig.suptitle(fig_title if fig_title is not None else "", weight="bold")

    # share x-axis pairs
    axs[0, 0].sharex(axs[1, 0])

    axs[0, 1].sharex(axs[0, 2])
    axs[1, 1].sharex(axs[1, 2])

    for idx, phi in enumerate(estimates_.keys()):
        ols_nr_, ols_nw_, nls_nr_, nls_nw_ = estimates_[phi]

        # --- row 0 --- #

        # ols: param
        axs[0, 0].set_ylabel("ols", fontsize=14, weight="bold")
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
        axs[1, 0].set_ylabel("nls", fontsize=14, weight="bold")
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

    fig.tight_layout(pad=2)
