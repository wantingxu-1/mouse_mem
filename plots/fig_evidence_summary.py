"""
FIGURE 3 - Evidence summary: glia vs NeuN effect across outcomes and checks.

Each row is one joint model  y ~ glia_high + neun_low (+ covariates),
outcome z-scored, so effects are in SD units with 95% CI.
Filled circle = glia pattern, open square = NeuN pattern.
Negative = worse (memory) / less open-arm time (EPM).
Output: results/figures/fig3_evidence_summary.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from features.dataset import get_dataset
from plots.style import apply_style
from utils.io import header, save_figure
from utils.stats_helpers import ols

ROWS = [  # (label, outcome, covariates, groups kept)
    ("Memory composite wk12 (main)", "Mem12", [], config.GROUPS),
    ("  + EPM activity covariates", "Mem12", ["EPMentries12", "EPMcentre12", "EPMopen12"], config.GROUPS),
    ("  without PM2.5 EV", "Mem12", [], ["Saline", "PM2.5", "N EV"]),
    ("T-maze wk12 15 s", "T12_15", [], config.GROUPS),
    ("T-maze wk12 60 s", "T12_60", [], config.GROUPS),
    ("NOR wk12", "NOR12", [], config.GROUPS),
    ("Memory composite wk4", "Mem4", [], config.GROUPS),
    ("EPM open time wk12 (anxiety)", "EPMopen12", [], config.GROUPS),
]


def run():
    header("FIGURE 3  Evidence summary (forest plot)")
    apply_style()
    df = get_dataset()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, y, covs, groups) in enumerate(ROWS):
        d = df[df.group.astype(str).isin(groups)][[y, "glia_high", "neun_low"] + covs].dropna().copy()
        d[y] = (d[y] - d[y].mean()) / d[y].std()
        for c in covs:
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        m = ols(f"{y} ~ glia_high + neun_low" + "".join(f" + {c}" for c in covs), d)
        ci = m.conf_int()
        for term, dy, kw in [("glia_high", -0.13, dict(marker="o", color=config.INK, mfc=config.INK)),
                             ("neun_low", 0.13, dict(marker="s", color=config.INK_SECONDARY, mfc="white"))]:
            ax.plot([ci.loc[term, 0], ci.loc[term, 1]], [i + dy] * 2, color=kw["color"], lw=1.5)
            ax.plot(m.params[term], i + dy, ms=7, mew=1.5, ls="none", **kw)
        ax.text(1.62, i, f"n={len(d)}", va="center", fontsize=8, color=config.INK_SECONDARY)
    ax.axvline(0, color=config.INK_SECONDARY, lw=0.8, ls="--")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlim(-1.8, 1.8)
    ax.set_xlabel("Effect in SD of the outcome (95% CI); negative = worse")
    ax.plot([], [], "o", color=config.INK, label="glia pattern (PM2.5, PM2.5 EV)")
    ax.plot([], [], "s", color=config.INK_SECONDARY, mfc="white", label="NeuN pattern (all treated)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig3_evidence_summary")
    plt.close(fig)


if __name__ == "__main__":
    run()
