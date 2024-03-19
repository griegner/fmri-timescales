import matplotlib.pyplot as plt
import numpy as np

from src import acf_utils, sim, timescale_utils


def mc_simulation(phis, n_timepoints, n_repeats=1000, random_seed=10):
    ols = timescale_utils.OLS(cov_estimator="newey-west", cov_n_lags=100, n_jobs=-2)
    nls = timescale_utils.NLS(n_jobs=-2)
    estimates_ = {}
    for phi in phis:
        X = sim.sim_ar(phi, n_timepoints, n_repeats, random_seed=random_seed)
        ols_ = ols.fit(X, n_timepoints)
        nls_ = nls.fit(X, n_timepoints)
        estimates_[str(phi)] = (ols_, nls_)
    return estimates_


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


def plot_simulation(estimates_, ar1_phis, ar1_taus, tau_xlim=None, se_xlim=None, fig_title=None):
    colors = ["#000000", "#B66B7C", "#8D9FCB", "#66C2A6", "#7D7D7D"]
    hist_kwargs = dict(bins=50, histtype="step", lw=1)
    vline_kwargs = dict(lw=2)

    fig, axs = plt.subplots(3, 2, figsize=(16, 6))
    fig.suptitle(fig_title if fig_title is not None else "")

    for idx, phi in enumerate(estimates_.keys()):
        ols_, nls_ = estimates_[phi]

        # ols-phi
        axs[0, 0].hist(ols_["phi"], color=colors[idx], **hist_kwargs)
        axs[0, 0].axvline(ar1_phis[idx], color=colors[idx], **vline_kwargs)

        axs[0, 1].hist(ols_["se(phi)"], color=colors[idx], **hist_kwargs)
        axs[0, 1].axvline(ols_["phi"].std(), color=colors[idx], **vline_kwargs)

        # ols-tau
        axs[1, 0].hist(ols_["tau"], color=colors[idx], **hist_kwargs)
        axs[1, 0].axvline(ar1_taus[idx], color=colors[idx], **vline_kwargs)
        axs[1, 0].set_xlim(*tau_xlim) if tau_xlim else None

        axs[1, 1].hist(ols_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs[1, 1].axvline(ols_["tau"].std(), color=colors[idx], **vline_kwargs)
        axs[1, 1].set_xlim(*se_xlim) if se_xlim else None

        # nls-tau
        axs[2, 0].hist(nls_["tau"], color=colors[idx], **hist_kwargs)
        axs[2, 0].axvline(ar1_taus[idx], color=colors[idx], **vline_kwargs)
        axs[2, 0].set_xlim(*tau_xlim) if tau_xlim else None

        # se(tau)
        axs[2, 1].hist(nls_["se(tau)"], color=colors[idx], **hist_kwargs)
        axs[2, 1].axvline(nls_["tau"].std(), color=colors[idx], **vline_kwargs)
        axs[2, 1].set_xlim(*se_xlim) if se_xlim else None

    axs[0, 0].set_title(r"$\hat{\phi}$")
    axs[0, 0].set_ylabel("ols", fontsize=14)
    axs[0, 1].set_title(r"$se(\hat{\phi})$")
    axs[1, 0].set_title(r"$\hat{\tau}$")
    axs[1, 0].set_ylabel("ols", fontsize=14)
    axs[1, 1].set_title(r"$se(\hat{\tau})$")
    axs[2, 0].set_title(r"$\hat{\tau}$")
    axs[2, 0].set_ylabel("nls", fontsize=14)
    axs[2, 1].set_title(r"$se(\hat{\tau})$")

    fig.tight_layout(pad=2)
