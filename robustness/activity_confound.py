"""
ROBUSTNESS 10 - Is the memory effect an activity / anxiety artefact?

Why      : T-maze and NOR need the mouse to move and explore. If a group is
           hypoactive or more anxious, it can score lower on memory without
           any memory deficit. N EV is the group that decides "glia vs NeuN",
           and N EV is also the most active group in the EPM (most crossings,
           most centre time). So the question is whether the glia effect on
           memory survives when activity / anxiety is held constant.
Activity proxies (the only ones in the data)
  EPMentries12 : EPM crossing frequency wk12 (locomotion)
  EPMcentre12  : EPM centre time wk12 (exploration / decision time)
  EPMopen12    : EPM open-arm time wk12 (anxiety; low = anxious)
  Total EPM time is ~600 s for every mouse, so it carries no information.
Steps
  1 Do activity proxies differ by group / follow either pattern?
  2 Do activity proxies predict memory (all mice; within groups)?
  3 y ~ glia_high + neun_low + covariate(s) for Mem12, T12_15, NOR12.
    Glia coefficient with and without covariates on the SAME mice;
    p from OLS and from the conditional permutation (labels shuffled
    within the neun_low stratum; outcome and covariate stay with the mouse).
  4 T-maze trial-level GEE (wk12, both delays) with EPM crossings as a
    mouse-level covariate: glia and NeuN odds ratios at 15 s.
Output   : results/tables/robust_activity_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import config
from core.pattern_contrast import contrast_table
from features.dataset import get_dataset
from features.tmaze_trials import long_trials
from utils.permutation import conditional_pattern_perm
from utils.io import header, log, save_table
from utils.stats_helpers import group_centred_ranks, ols, rng

ACTIVITY = ["EPMentries12", "EPMcentre12", "EPMopen12"]
OUTCOMES = ["Mem12", "T12_15", "NOR12"]
COVARIATE_SETS = {"crossings": ["EPMentries12"], "centre time": ["EPMcentre12"],
                  "open time": ["EPMopen12"], "all three": ACTIVITY}


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("ROBUSTNESS 10  Activity / anxiety confound of the memory effect")
    df = get_dataset()
    gen = rng()
    z = df.copy()
    for c in ACTIVITY:
        z[c] = (z[c] - z[c].mean()) / z[c].std()

    # 1 activity by group
    act = contrast_table(df, ACTIVITY)
    act["KW p"] = [stats.kruskal(*[df.loc[df.group == g, c].dropna() for g in config.GROUPS]).pvalue for c in ACTIVITY]
    act = pd.concat([act, df.groupby("group", observed=True)[ACTIVITY].mean().T.reset_index(drop=True)], axis=1)
    log("Activity proxies by group and pattern:\n" +
        act[["outcome", "n", "KW p", "glia p", "NeuN p", "joint glia p", "joint NeuN p"] + config.GROUPS].round(3).to_string(index=False))
    save_table(act, "robust_activity_by_group")

    # 2 activity vs memory
    rows = []
    for a in ACTIVITY:
        for y in OUTCOMES:
            d = df[[a, y, "group"]].dropna()
            r_all, p_all = stats.spearmanr(d[a], d[y])
            c = group_centred_ranks(d, [a, y])
            r_w, p_w = stats.pearsonr(c[a], c[y])
            rows.append({"activity": config.LABELS[a], "memory": config.LABELS[y], "n": len(d),
                         "rho all mice": r_all, "p all": p_all, "rho within group": r_w, "p within": p_w})
    cor = pd.DataFrame(rows)
    log("Activity vs memory:\n" + cor.round(3).to_string(index=False))
    save_table(cor, "robust_activity_memory_correlation")

    # 3 adjusted pattern contrast
    rows = []
    for y in OUTCOMES:
        for name, covs in COVARIATE_SETS.items():
            d = z[[y, "glia_high", "neun_low", "group"] + covs].dropna()
            base = ols(f"{y} ~ glia_high + neun_low", d)
            adj = ols(f"{y} ~ glia_high + neun_low + " + " + ".join(covs), d)
            rows.append({"outcome": config.LABELS[y], "covariates": name, "n": len(d),
                         "glia coef unadjusted": base.params["glia_high"], "glia p unadjusted": base.pvalues["glia_high"],
                         "glia coef adjusted": adj.params["glia_high"], "glia p adjusted": adj.pvalues["glia_high"],
                         "glia perm p adjusted": conditional_pattern_perm(d, y, "glia_high", n_perm, gen, covs)[1],
                         "% change in glia coef": 100 * (adj.params["glia_high"] / base.params["glia_high"] - 1),
                         "NeuN coef adjusted": adj.params["neun_low"], "NeuN p adjusted": adj.pvalues["neun_low"],
                         "covariate p (min)": adj.pvalues[covs].min(),
                         "R2 gain from covariates": adj.rsquared - base.rsquared})
    adj = pd.DataFrame(rows)
    log("Glia/NeuN pattern contrast adjusted for activity (covariates z-scored):\n" +
        adj[["outcome", "covariates", "n", "glia coef unadjusted", "glia coef adjusted", "glia p adjusted",
             "glia perm p adjusted", "% change in glia coef", "NeuN p adjusted", "covariate p (min)"]].round(3).to_string(index=False))
    save_table(adj, "robust_activity_adjusted_contrast")

    # 4 T-maze GEE with activity covariate (week 12 trials)
    t = long_trials()
    t = t[t.week == "wk12"].copy()
    t["act"] = t.mouse.map(z["EPMentries12"])
    t = t.dropna(subset=["act"])
    rows = []
    for label, f in [("unadjusted", "correct ~ (glia_high + neun_low) * d60"),
                     ("+ EPM crossings", "correct ~ (glia_high + neun_low) * d60 + act")]:
        res = smf.gee(f, groups="mouse", data=t, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = res.conf_int()
        for term in ["glia_high", "neun_low"]:
            rows.append({"model": label, "effect at 15 s": "glia" if term == "glia_high" else "NeuN",
                         "mice": t.mouse.nunique(), "odds ratio": np.exp(res.params[term]),
                         "OR 95% low": np.exp(ci.loc[term, 0]), "OR 95% high": np.exp(ci.loc[term, 1]),
                         "p": res.pvalues[term],
                         "activity OR per SD": np.exp(res.params["act"]) if "act" in res.params else np.nan,
                         "activity p": res.pvalues["act"] if "act" in res.params else np.nan})
    gee = pd.DataFrame(rows)
    log("T-maze wk12 GEE, 15 s delay:\n" + gee.round(3).to_string(index=False))
    save_table(gee, "robust_activity_tmaze_gee")
    return act, cor, adj, gee


if __name__ == "__main__":
    run()
