import matplotlib.pyplot as plt
import pytest

from fmri_timescales import plts


@pytest.mark.parametrize("annotate", [True, False])
def test_plot_stationarity_triangle(annotate):
    fig, ax = plt.subplots()
    plts.plot_stationarity_triangle(ax, annotate=annotate)
