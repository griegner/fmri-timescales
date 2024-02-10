import matplotlib.pyplot as plt
import numpy as np

from src import acf_utils, sim, timescale_utils


def mc_simulation(phis, n_timepoints, n_repeats=1000):
    df = {}
    for phi in phis:
        X = sim.sim_ar(phi, n_timepoints, n_repeats)
        ols = timescale_utils.estimate_timescales_ols(X, n_repeats)
        nls = timescale_utils.estimate_timescales_nls(X, n_repeats)
        df[str(phi)] = (ols, nls)
    return df


def bias_std_rmse(true, estimates):
    bias = np.mean(estimates) - true
    std = np.std(estimates)
    rmse = np.sqrt(np.mean((estimates - true) ** 2))
    return bias, std, rmse


def plot_nls_vs_ols(X, n_timepoints, nls_df, ols_df):

    X_acf = acf_utils.acf_fft(X.reshape(1, -1), n_timepoints).squeeze()
    timepoints = np.arange(n_timepoints)

    # nls
    exp_decay = lambda x, tau: np.exp(-x / (tau if tau > 0 else 1e-6))
    nls_fit = exp_decay(timepoints, nls_df["tau"])
    nls_residuals = X_acf - nls_fit

    fig, axs = plt.subplots(2, 1, figsize=(15, 4), sharex=True)

    axs[0].set_title("non-linear least squares")
    axs[0].plot(timepoints, X_acf, lw=1, alpha=0.5, label="ACF")
    axs[0].plot(timepoints, nls_fit, c="k", lw=1, label="fit")
    axs[0].set_ylabel(r"$\hat \rho_k$")
    axs[0].annotate(f"<-- timescale = {nls_df['tau']}", (nls_df["tau"], 1 / np.e))
    axs[0].legend()

    axs[1].scatter(timepoints, nls_residuals, s=1, alpha=0.5, label="residuals")
    axs[1].set_ylabel(r"$\hat e_k$")
    axs[1].set_xlabel(r"lag $k$")
    axs[1].legend()
    fig.tight_layout()

    # ols
    ols_fit = ols_df["phi"] * X[:-1]
    ols_residuals = X[1:] - ols_fit

    fig, axs = plt.subplots(1, 2, figsize=(15, 3))

    axs[0].set_title("ordinary least squares")

    axs[0].scatter(X[:-1], X[1:], s=4, alpha=0.5, label="AR1")

    axs[0].plot(X[:-1], ols_fit, c="k", lw=1, label="fit")
    axs[0].annotate(
        f"slope = {ols_df['phi']}, timescale = {ols_df['tau']}\nslope se = {ols_df['se(phi)']}, timescale se = {ols_df['se(tau)']}",
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


def plot_simulation(df, ar1_phis, ar1_taus, tau_xlim=None, se_xlim=None, fig_title=None):
    colors = ["#000000", "#B66B7C", "#8D9FCB", "#66C2A6", "#7D7D7D"]
    hist_kwargs = dict(bins=50, histtype="step", lw=1)
    vline_kwargs = dict(lw=2)

    fig, axs = plt.subplots(3, 2, figsize=(16, 6))
    fig.suptitle(fig_title)

    for idx, phi in enumerate(df.keys()):
        df_ols, df_nls = df[phi]

        # ols-phi
        axs[0, 0].hist(df_ols["phi"], color=colors[idx], **hist_kwargs)
        axs[0, 0].axvline(ar1_phis[idx], color=colors[idx], **vline_kwargs)

        axs[0, 1].hist(df_ols["se(phi)"], color=colors[idx], **hist_kwargs)
        axs[0, 1].axvline(df_ols["phi"].std(), color=colors[idx], **vline_kwargs)

        # ols-tau
        axs[1, 0].hist(df_ols["tau"], color=colors[idx], **hist_kwargs)
        axs[1, 0].axvline(ar1_taus[idx], color=colors[idx], **vline_kwargs)
        axs[1, 0].set_xlim(*tau_xlim) if tau_xlim else None

        axs[1, 1].hist(df_ols["se(tau)"], color=colors[idx], **hist_kwargs)
        axs[1, 1].axvline(df_ols["tau"].std(), color=colors[idx], **vline_kwargs)
        axs[1, 1].set_xlim(*se_xlim) if se_xlim else None

        # nls-tau
        axs[2, 0].hist(df_nls["tau"], color=colors[idx], **hist_kwargs)
        axs[2, 0].axvline(ar1_taus[idx], color=colors[idx], **vline_kwargs)
        axs[2, 0].set_xlim(*tau_xlim) if tau_xlim else None

        # se(tau)
        axs[2, 1].hist(df_nls["se(tau)"], color=colors[idx], **hist_kwargs)
        axs[2, 1].axvline(df_nls["tau"].std(), color=colors[idx], **vline_kwargs)
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
