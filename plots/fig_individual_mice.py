"""
FIGURE 2 - Individual mice: Glia index and NeuN index vs memory.

Each dot is one stained mouse, coloured by group; dashed line = simple fit.
Title gives raw Spearman rho (not group-adjusted).
Output: results/figures/fig2_individual_mice.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config
from features.dataset import get_dataset
from plots.style import apply_style
from utils.io import header, save_figure
from utils.stats_helpers import spearman


def run():
    header("FIGURE 2  Individual mice")
    apply_style()
    df = get_dataset()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, x in zip(axes, ["Glia_idx", "NeuN_idx"]):
        for g in config.GROUPS:
            d = df[df.group == g]
            ax.scatter(d[x], d.Mem12, s=40, color=config.GROUP_COLOURS[g], label=g, edgecolor="white", linewidth=.8)
        d = df[[x, "Mem12"]].dropna()
        b = np.polyfit(d[x], d.Mem12, 1)
        xs = np.linspace(d[x].min(), d[x].max(), 2)
        ax.plot(xs, np.polyval(b, xs), color=config.INK_SECONDARY, lw=1.5, ls="--")
        rho, p, n = spearman(df[x], df.Mem12)
        ax.set_title(f"rho = {rho:.2f}, p = {p:.3f}, n = {n}", loc="left", fontsize=10)
        ax.set_xlabel(config.LABELS[x] + " (z)")
    axes[0].set_ylabel(config.LABELS["Mem12"] + " (z)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig2_individual_mice")
    plt.close(fig)


if __name__ == "__main__":
    run()
