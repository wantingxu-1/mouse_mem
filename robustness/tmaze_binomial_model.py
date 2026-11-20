"""
ROBUSTNESS 8 - T-maze as trials: binomial models with all delays and weeks.

Why      : Each T-maze score is k correct out of 3 trials (0, 1/3, 2/3, 1).
           Averaging or OLS treats it as continuous. Here every trial is a
           0/1 outcome, and all four conditions per mouse (15 s / 60 s delay
           x week 4 / week 12) enter one model with mouse as a cluster.
           This replaces the noisy "T60 - T15" difference: the delay
           question becomes a proper interaction test.
Assumption: score = proportion correct of
           3 trials; 15/60 s = delay between trials.

Models (logistic, 48 mice x 4 conditions x 3 trials = 576 trials)
  M1  correct ~ group * delay * week
      -> Wald tests: does the group effect depend on delay? on week?
  M2  correct ~ (glia_high + neun_low) * delay * week
      -> glia vs NeuN odds ratios in each delay x week cell.
Estimation
  Main  : GEE, binomial, exchangeable correlation within mouse, robust SEs
          (population-averaged effects; reliable with 48 clusters).
  Check : Bayesian binomial mixed model with a random intercept per mouse
          (statsmodels BinomialBayesMixedGLM, variational Bayes), M2 only.
          Use it for the DIRECTION of effects only: variational Bayes is
          known to give too-narrow posterior SDs, so its intervals are
          not reliable. Inference is based on the GEE.
Output   : results/tables/robust_tmaze_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from features.tmaze_trials import long_trials  # noqa: F401 (re-exported)
from utils.io import header, log, save_table


def gee(formula, d):
    return smf.gee(formula, groups="mouse", data=d, family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def wald_terms(res):
    out = res.wald_test_terms(scalar=True).table
    out = out.reset_index().rename(columns={"index": "term"})
    return out


def cell_odds_ratios(res, d):
    """Glia and NeuN odds ratios in each delay x week cell from M2 (GEE)."""
    p, cov = res.params, res.cov_params()
    rows = []
    for d60 in (0, 1):
        for w12 in (0, 1):
            for term in ("glia_high", "neun_low"):
                names = [term, f"{term}:d60", f"{term}:w12", f"{term}:d60:w12"]
                w = np.array([1, d60, w12, d60 * w12], float)
                est = w @ p[names].values
                se = np.sqrt(w @ cov.loc[names, names].values @ w)
                z = est / se
                rows.append({"delay": "60s" if d60 else "15s", "week": "wk12" if w12 else "wk4",
                             "effect": "glia" if term == "glia_high" else "NeuN",
                             "odds ratio": np.exp(est), "OR 95% low": np.exp(est - 1.96 * se),
                             "OR 95% high": np.exp(est + 1.96 * se),
                             "p": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))})
    return pd.DataFrame(rows)


def bayes_mixed_check(d):
    """M2 with random intercept per mouse (variational Bayes); returns posterior means/SDs."""
    model = BinomialBayesMixedGLM.from_formula("correct ~ (glia_high + neun_low) * d60 * w12",
                                               {"mouse": "0 + C(mouse)"}, d)
    res = model.fit_vb()
    names = model.exog_names
    fe = pd.DataFrame({"term": names, "posterior mean": res.fe_mean, "posterior SD": res.fe_sd})
    fe["95% CrI low"] = fe["posterior mean"] - 1.96 * fe["posterior SD"]
    fe["95% CrI high"] = fe["posterior mean"] + 1.96 * fe["posterior SD"]
    fe["mouse random-effect SD"] = np.exp(res.vcp_mean[0])
    return fe


def run():
    header("ROBUSTNESS 8  T-maze trial-level binomial models")
    d = long_trials()
    log(f"Trials: {len(d)} from {d.mouse.nunique()} mice")
    obs = d.groupby(["group", "week", "delay"], observed=True).correct.mean().unstack(["week", "delay"])
    log("Observed proportion correct:\n" + obs.round(3).to_string())

    m1 = gee("correct ~ group * d60 * w12", d)
    w1 = wald_terms(m1)
    log("M1 (group x delay x week), GEE Wald tests:\n" + w1.round(4).to_string(index=False))
    save_table(w1, "robust_tmaze_group_wald")

    m2 = gee("correct ~ (glia_high + neun_low) * d60 * w12", d)
    w2 = wald_terms(m2)
    log("M2 (glia/NeuN pattern x delay x week), GEE Wald tests:\n" + w2.round(4).to_string(index=False))
    save_table(w2, "robust_tmaze_pattern_wald")
    cells = cell_odds_ratios(m2, d)
    log("M2 odds ratios per cell (OR < 1 = fewer correct choices):\n" + cells.round(3).to_string(index=False))
    save_table(cells, "robust_tmaze_pattern_odds_ratios")

    fe = bayes_mixed_check(d)
    log("M2 Bayesian mixed model (random intercept per mouse):\n" + fe.round(3).to_string(index=False))
    save_table(fe, "robust_tmaze_bayes_mixed")
    return w1, w2, cells, fe


if __name__ == "__main__":
    run()
