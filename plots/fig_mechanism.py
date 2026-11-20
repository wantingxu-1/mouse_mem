"""
FIGURE 4 - Which stains go with behaviour, and is memory above chance?

A  Canonical PLS (IHC <-> behaviour, component 1): weight of each of the
   15 stains with bootstrap 95% CI (from advanced/pls_integration.py).
   Filled = glia stain (IBA1, GFAP), open = NeuN.
B  T-maze week 12, proportion correct per group with GEE 95% CI, 15 s and
   60 s delay; dashed line = chance (0.5) (from robustness/tmaze_chance_level.py).
Reads the saved tables, so run those modules first (run_all.py does).
Output: results/figures/fig4_mechanism.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from plots.style import apply_style
from utils.io import header, load_table, save_figure


def run():
    header("FIGURE 4  PLS stain weights and T-maze vs chance")
    apply_style()
    w = load_table("adv_pls_canonical_weights")
    w = w[w.block == "IHC"].copy()
    w["marker"] = w.variable.str.split(" ").str[0]
    order = ["IBA1", "GFAP", "NeuN"]
    w["k"] = w.marker.map({m: i for i, m in enumerate(order)})
    w = w.sort_values(["k", "variable"]).reset_index(drop=True)
    ch = load_table("robust_tmaze_chance")
    ch = ch[(ch.week == "wk12") & (ch.group.isin(config.GROUPS))]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, r in w.iterrows():
        glia = r.marker != "NeuN"
        colr = config.INK if glia else config.INK_SECONDARY
        a.plot([r["95% low"], r["95% high"]], [i, i], color=colr, lw=1.5)
        a.plot(r.weight, i, "o" if glia else "s", color=colr, mfc=colr if glia else "white", ms=6, mew=1.5)
    a.axvline(0, color=config.INK_SECONDARY, lw=0.8, ls="--")
    a.set_yticks(range(len(w)))
    a.set_yticklabels(w.variable)
    a.invert_yaxis()
    a.set_xlabel("Weight on shared stain-behaviour axis (95% CI)")
    a.set_title("A  Stains that covary with behaviour\n(right of 0 = more stain, worse T-maze wk12 15 s)", loc="left", fontsize=10)

    for j, delay in enumerate(["15s", "60s"]):
        for k, g in enumerate(config.GROUPS):
            r = ch[(ch.delay == delay) & (ch.group == g)].iloc[0]
            x = j * 5 + k
            b.plot([x, x], [r["GEE 95% low"], r["GEE 95% high"]], color=config.GROUP_COLOURS[g], lw=2)
            b.plot(x, r["proportion correct"], "o", color=config.GROUP_COLOURS[g], ms=8,
                   label=g if j == 0 else None, mec="white", mew=1)
    b.axhline(0.5, color=config.INK_SECONDARY, lw=0.8, ls="--")
    b.text(4.0, 0.505, "chance", fontsize=8, color=config.INK_SECONDARY, ha="center", va="bottom")
    b.set_xticks([1.5, 6.5])
    b.set_xticklabels(["15 s delay", "60 s delay"])
    b.set_ylim(0.25, 1.0)
    b.set_ylabel("Proportion correct, week 12 (GEE 95% CI)")
    b.set_title("B  T-maze vs chance", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig4_mechanism")
    plt.close(fig)


if __name__ == "__main__":
    run()
