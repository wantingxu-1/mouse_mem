"""
ROBUSTNESS 7 - Evidence FOR "no NeuN effect" (Bayes factors + equivalence tests).

Why      : p = 0.91 for NeuN only says "no evidence of an effect". It does not
           say "evidence of no effect". The conclusion "behaviour follows glia,
           not NeuN" needs the second statement too.

Methods
  (a) BIC Bayes factor (Wagenmakers 2007, unit-information prior):
        BF01 = exp((BIC_with_term - BIC_without_term) / 2)
      in the joint model y ~ glia_high + neun_low. BF01 > 1 favours "no effect
      of the term"; 3-10 = moderate, >10 = strong. BF10 = 1 / BF01.
  (b) JZS Bayes factor (Rouder et al. 2009, Cauchy prior r = 0.707) for the
      two-sample comparison Saline vs N EV - same glia status, different NeuN.
      If NeuN mattered, these two groups should differ.
  (c) Equivalence test (TOST, alpha = 0.05 -> 90% CI). Smallest effect of
      interest fixed in advance: half an outcome SD (Cohen's d = 0.5).
      "Equivalent" = the whole 90% CI lies inside +/- that bound.
      Also reported: the narrowest bound that would still pass.
Output   : results/tables/robust_bayes_factors.csv, robust_equivalence.csv
"""
import numpy as np
import pandas as pd
from scipy import integrate, stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import ols

SESOI_D = 0.5        # smallest effect size of interest, in SD of the outcome
JZS_R = np.sqrt(2) / 2


def bic_bf01(d, y, term):
    other = [t for t in ["glia_high", "neun_low"] if t != term]
    full = ols(f"{y} ~ glia_high + neun_low", d)
    reduced = ols(f"{y} ~ {' + '.join(other)}", d)
    return np.exp((full.bic - reduced.bic) / 2)


def jzs_bf10(t, n1, n2, r=JZS_R):
    """JZS Bayes factor for a two-sample t statistic (Rouder et al. 2009, eq. 1 with scale r)."""
    neff = n1 * n2 / (n1 + n2)
    v = n1 + n2 - 2

    def integrand(g):
        return ((1 + neff * g * r ** 2) ** -0.5
                * (1 + t ** 2 / ((1 + neff * g * r ** 2) * v)) ** (-(v + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    num = integrate.quad(integrand, 0, np.inf)[0]
    den = (1 + t ** 2 / v) ** (-(v + 1) / 2)
    return num / den


def label(bf01):
    if bf01 >= 10:
        return "strong for no effect"
    if bf01 >= 3:
        return "moderate for no effect"
    if bf01 > 1 / 3:
        return "inconclusive"
    if bf01 > 1 / 10:
        return "moderate for an effect"
    return "strong for an effect"


def run():
    header("ROBUSTNESS 7  Bayes factors and equivalence tests")
    df = get_dataset()
    bf_rows, eq_rows = [], []
    for y in config.PRIMARY_OUTCOMES + ["Mem4"]:
        d = df[[y, "group", "glia_high", "neun_low"]].dropna()
        sd = d[y].std()
        bound = SESOI_D * sd
        joint = ols(f"{y} ~ glia_high + neun_low", d)
        ci90 = joint.conf_int(alpha=0.10)
        for term, name in [("glia_high", "glia effect"), ("neun_low", "NeuN effect")]:
            bf01 = bic_bf01(d, y, term)
            bf_rows.append({"outcome": config.LABELS[y], "test": f"{name} (joint model, BIC)", "n": len(d),
                            "BF01 (for no effect)": bf01, "BF10 (for effect)": 1 / bf01, "reading": label(bf01)})
            lo, hi = ci90.loc[term]
            eq_rows.append({"outcome": config.LABELS[y], "effect": name, "estimate": joint.params[term],
                            "90% CI low": lo, "90% CI high": hi, "bound (+/- 0.5 SD)": bound,
                            "equivalent (inside bound)": bool(lo > -bound and hi < bound),
                            "narrowest passing bound (SD units)": max(abs(lo), abs(hi)) / sd})
        a, b = d.loc[d.group == "Saline", y], d.loc[d.group == "N EV", y]
        t = stats.ttest_ind(a, b).statistic
        bf10 = jzs_bf10(t, len(a), len(b))
        bf_rows.append({"outcome": config.LABELS[y], "test": "Saline vs N EV (JZS t-test)", "n": len(a) + len(b),
                        "BF01 (for no effect)": 1 / bf10, "BF10 (for effect)": bf10, "reading": label(1 / bf10)})
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        dof = len(a) + len(b) - 2
        q = stats.t.ppf(0.95, dof)
        lo, hi = diff - q * se, diff + q * se
        eq_rows.append({"outcome": config.LABELS[y], "effect": "N EV minus Saline", "estimate": diff,
                        "90% CI low": lo, "90% CI high": hi, "bound (+/- 0.5 SD)": bound,
                        "equivalent (inside bound)": bool(lo > -bound and hi < bound),
                        "narrowest passing bound (SD units)": max(abs(lo), abs(hi)) / sd})
    bf, eq = pd.DataFrame(bf_rows), pd.DataFrame(eq_rows)
    log(bf.round(3).to_string(index=False))
    log(eq.round(3).to_string(index=False))
    save_table(bf, "robust_bayes_factors")
    save_table(eq, "robust_equivalence")
    return bf, eq


if __name__ == "__main__":
    run()
