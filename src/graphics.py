from typing import Dict
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator, LogLocator, MultipleLocator
import os
import numpy as np

os.environ['PATH'] = f"/Library/TeX/texbin:{os.environ['PATH']}"
rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rcParams.update({'font.size': 15})


def plot_roc_curve(
        signal_eff: Dict[str, np.ndarray], background_eff: Dict[str, np.ndarray], labels: Dict[str, str],
        colors: Dict[str, str], file_path=None
):
    """Plots the roc curve for a list of classifiers"""

    for classifier in signal_eff:
        plt.plot(background_eff[classifier], signal_eff[classifier], label=labels[classifier],
                 color=colors[classifier])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel(r"Background Efficiency ($\epsilon_b$)")
    plt.ylabel(r"Signal Efficiency ($\epsilon_s$)")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=2, direction="in")
    plt.tick_params(axis="both", which="major", top=True, right=True, length=5, direction="in")
    plt.legend(loc="best", frameon=False, framealpha=1, fontsize="12", fancybox=False)
    plt.minorticks_on()
    if file_path is not None:
        plt.savefig(file_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()
