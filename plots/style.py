"""Shared matplotlib style for all figures."""
import matplotlib.pyplot as plt
import config


def apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": config.INK_SECONDARY,
        "axes.labelcolor": config.INK,
        "xtick.color": config.INK_SECONDARY,
        "ytick.color": config.INK_SECONDARY,
    })
