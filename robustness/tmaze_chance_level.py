"""
ROBUSTNESS 14 - Is T-maze performance above chance (50%)?

Why      : At week 12 with the 60 s delay all groups score 0.50-0.67, so the
           condition may be too hard to show group differences. That was a
           guess; here it is tested. Chance = 0.5 assumes a two-arm choice
           (spontaneous alternation or rewarded arm).
Tests per group x week x delay
  Trial level : exact binomial test of pooled correct trials vs 0.5
                (ignores clustering of 3 trials within a mouse).
  Mouse level : one-sample Wilcoxon of the mouse scores vs 0.5
                (one value per mouse; respects clustering, low power).
  GEE         : intercept-only logistic GEE clustered by mouse, robust SE
                (the main test).
Also Saline only (the reference group) and all groups pooled per condition.
Output   : results/tables/robust_tmaze_chance.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import config
from features.tmaze_trials import long_trials
from utils.io import header, log, save_table


def gee_vs_half(d):
    res = sm.GEE(d.correct.to_numpy(), np.ones((len(d), 1)), groups=d.mouse.to_numpy(),
                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, se = res.params[0], res.bse[0]
    lo, hi = b - 1.96 * se, b + 1.96 * se
    return 1 / (1 + np.exp(-b)), 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi)), 2 * stats.norm.sf(abs(b / se))


def run():
    header("ROBUSTNESS 14  T-maze: above chance (0.5)?")
    t = long_trials()
    rows = []
    for (week, delay), cell in t.groupby(["week", "delay"]):
        for gname in config.GROUPS + ["all groups"]:
            d = cell if gname == "all groups" else cell[cell.group == gname]
            k, n = int(d.correct.sum()), len(d)
            scores = d.groupby("mouse").correct.mean()
            w = stats.wilcoxon(scores - 0.5) if (scores != 0.5).any() else None
            p_hat, lo, hi, p_gee = gee_vs_half(d)
            rows.append({"week": week, "delay": delay, "group": gname, "mice": d.mouse.nunique(), "trials": n,
                         "proportion correct": k / n, "GEE 95% low": lo, "GEE 95% high": hi, "GEE p vs 0.5": p_gee,
                         "binomial p (trials)": stats.binomtest(k, n, 0.5).pvalue,
                         "Wilcoxon p (mice)": w.pvalue if w else np.nan})
    r = pd.DataFrame(rows)
    log(r.round(3).to_string(index=False))
    save_table(r, "robust_tmaze_chance")
    return r


if __name__ == "__main__":
    run()
