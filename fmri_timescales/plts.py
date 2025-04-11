import numpy as np


def plot_stationarity_triangle(ax, fill_alpha=0.25, annotate=True):
    """Plot stationarity triangle for an AR(2) process.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes used to display the plot.
    fill_alpha : float, optional
        The opacity (btw 0-1) of the fill color, by default 0.25.
    annotate : bool, optional
        If True, annotate triangle bounds, by default True.
    """
    ax.set_xlabel(r"$\phi_1$")
    ax.set_ylabel(r"$\phi_2$")

    # set x, y limits
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlim([-2.1, 2.1])
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim([-1.1, 1.1])

    # plot AR(2) stationarity triangle
    phi1 = np.linspace(-2, 2, 100)
    lwr, upr1, upr2 = -1, 1 + phi1, 1 - phi1
    ax.fill_between(phi1, np.maximum(lwr, np.minimum(upr1, upr2)), lwr, color="gray", alpha=fill_alpha)
    ax.plot(phi1, (-0.25 * phi1**2), c="k", ls="--", lw=0.5)

    # add annotations
    if annotate:
        text_kwargs = dict(horizontalalignment="center", verticalalignment="center")
        ax.text(-1.5, 0.5, r"$\phi_2 > 1 + \phi_1$", **text_kwargs)
        ax.text(1.5, 0.5, r"$\phi_2 > 1 - \phi_1$", **text_kwargs)
        ax.text(0, -0.5, r"$\phi_2 < - \phi_1^2 / 4$", **text_kwargs)
