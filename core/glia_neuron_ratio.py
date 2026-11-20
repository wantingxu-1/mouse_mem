"""
CORE 8 - Glia burden per neuron: does a glia/NeuN ratio explain memory better?

Idea     : Glia activation and neuron loss may matter together: the same
           glial load could be worse when fewer neurons remain. A per-mouse
           ratio (glia relative to NeuN) captures "glia burden per neuron".
           If the ratio predicts memory better than glia alone, both cell
           types contribute; if not, glia alone is the better account.
Ratios   : from % of Saline values (mean over 5 regions)
             log2(GFAP% / NeuN%), log2(IBA1% / NeuN%), log2(Glia% / NeuN%)
             with Glia% = mean(GFAP%, IBA1%). 0 = Saline-like balance.
Steps
  1 Ratio by group (back-transformed fold vs Saline) and Kruskal-Wallis.
  2 Group level: memory group means vs predictor group means (4 points,
    descriptive only).
  3 Individual level, same stained mice: Mem12 / T12_15 / NOR12 ~ one
    predictor (glia, NeuN, each ratio), glia + NeuN, glia x NeuN.
    R^2, AIC, LOOCV R^2.
  4 Within-group association (group-centred ranks, permutation p).
Output   : results/tables/core_ratio_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.permutation import within_group_corr_perm
from utils.io import header, log, save_table
from utils.stats_helpers import loocv_r2, ols, rng

OUTCOMES = ["Mem12", "T12_15", "NOR12"]
RATIOS = ["GFAP/NeuN", "IBA1/NeuN", "Glia/NeuN"]


def add_ratios(df):
    d = df.copy()
    d["Glia_pct"] = d[["GFAP_pct", "IBA1_pct"]].mean(axis=1, skipna=False)
    for m in ["GFAP", "IBA1", "Glia"]:
        d[f"{m}/NeuN"] = np.log2(d[f"{m}_pct"] / d["NeuN_pct"])
    return d


def loo_ols_r2(d, y, xs):
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    yv = d[y].to_numpy(float)
    pred = np.empty(len(d))
    for i in range(len(d)):
        tr = np.arange(len(d)) != i
        b, *_ = np.linalg.lstsq(X[tr], yv[tr], rcond=None)
        pred[i] = X[i] @ b
    return loocv_r2(yv, pred)


def q(name):
    return f"Q('{name}')" if "/" in name else name


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("CORE 8  Glia/NeuN ratio vs memory")
    df = add_ratios(get_dataset())
    gen = rng()

    # 1 ratio by group
    rows = []
    for r in RATIOS:
        means = df.groupby("group", observed=True)[r].mean()
        rows.append({"ratio": r, "n": int(df[r].notna().sum()),
                     **{f"{g} fold vs Saline": 2 ** means[g] for g in config.GROUPS},
                     "KW p": stats.kruskal(*[df.loc[df.group == g, r].dropna() for g in config.GROUPS]).pvalue})
    by = pd.DataFrame(rows)
    log("Glia/NeuN ratio by group (fold of Saline balance):\n" + by.round(3).to_string(index=False))
    save_table(by, "core_ratio_by_group")

    # 2 group level (4 points)
    gm = df.groupby("group", observed=True)[OUTCOMES + RATIOS + ["Glia_idx", "NeuN_idx"]].mean()
    rows = []
    for y in OUTCOMES:
        for x in ["Glia_idx", "NeuN_idx"] + RATIOS:
            rows.append({"outcome": config.LABELS[y], "predictor": x, "group-mean r": np.corrcoef(gm[x], gm[y])[0, 1]})
    glev = pd.DataFrame(rows)
    log("Group-mean correlations (4 groups, descriptive):\n" +
        glev.pivot(index="predictor", columns="outcome", values="group-mean r").round(2).to_string())
    save_table(glev, "core_ratio_group_level")

    # 3 individual-level model comparison on the same mice
    models = {"Glia index": ["Glia_idx"], "NeuN index": ["NeuN_idx"],
              **{f"log2 {r}": [r] for r in RATIOS},
              "Glia + NeuN": ["Glia_idx", "NeuN_idx"], "Glia x NeuN": ["Glia_idx", "NeuN_idx", "GxN"]}
    rows = []
    for y in OUTCOMES:
        d = df[[y, "Glia_idx", "NeuN_idx"] + RATIOS].dropna().copy()
        d["GxN"] = d.Glia_idx * d.NeuN_idx
        for name, xs in models.items():
            m = ols(f"{y} ~ " + " + ".join(q(x) for x in xs), d)
            first = q(xs[0]) if len(xs) == 1 else None
            rows.append({"outcome": config.LABELS[y], "n": len(d), "model": name, "R2": m.rsquared, "AIC": m.aic,
                         "LOOCV R2": loo_ols_r2(d, y, xs),
                         "p (single predictor)": m.pvalues[first] if first else np.nan,
                         "p interaction": m.pvalues["GxN"] if "GxN" in xs else np.nan})
    mc = pd.DataFrame(rows)
    mc["dAIC vs best"] = mc.AIC - mc.groupby("outcome").AIC.transform("min")
    log("Individual-level models (stained mice):\n" + mc.round(3).to_string(index=False))
    save_table(mc, "core_ratio_model_comparison")

    # 4 within-group association
    rows = []
    for r in RATIOS:
        for y in OUTCOMES:
            rho, p, _ = within_group_corr_perm(df, r, y, min(n_perm, 4999), gen)
            rows.append({"ratio": r, "outcome": config.LABELS[y], "within-group rho": rho, "perm p": p})
    wg = pd.DataFrame(rows)
    log("Within-group association of ratio with memory:\n" + wg.round(3).to_string(index=False))
    save_table(wg, "core_ratio_within_group")
    return by, glev, mc, wg


if __name__ == "__main__":
    run()
