"""
FIGURE 1 - Group dissociation: NeuN vs IBA1 vs GFAP vs memory.

One panel per measure; dots = mice, bar = group mean. Shows that N EV has
low NeuN (like PM2.5) but Saline-level glia and Saline-level memory.
Output: results/figures/fig1_group_dissociation.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import config
from features.dataset import get_dataset
from plots.style import apply_style
from utils.io import header, save_figure

PANELS = [("NeuN_pct", "NeuN (neurons)", "% of Saline, mean of 5 regions", 100),
          ("IBA1_pct", "IBA1 (microglia)", "% of Saline, mean of 5 regions", 100),
          ("GFAP_pct", "GFAP (astrocytes)", "% of Saline, mean of 5 regions", 100),
          ("Mem12", "Memory, week 12", "composite z (T-maze 15s+60s, NOR)", 0)]


def run():
    header("FIGURE 1  Group dissociation")
    apply_style()
    df = get_dataset()
    gen = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (col, title, ylab, ref) in zip(axes, PANELS):
        for i, g in enumerate(config.GROUPS):
            v = df.loc[df.group == g, col].dropna()
            ax.scatter(i + gen.uniform(-.15, .15, len(v)), v, s=18, color=config.GROUP_COLOURS[g],
                       alpha=.6, edgecolor="white", linewidth=.5)
            ax.plot([i - .28, i + .28], [v.mean()] * 2, color=config.INK, lw=2)
        ax.axhline(ref, color="#bbbbbb", lw=1, ls="--", zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels([g.replace(" EV", "\nEV") if g == "PM2.5 EV" else g for g in config.GROUPS])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylab, fontsize=8.5, color=config.INK_SECONDARY)
    fig.suptitle("N EV: NeuN low, glia and memory at Saline level", x=.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "fig1_group_dissociation")
    plt.close(fig)


if __name__ == "__main__":
    run()
