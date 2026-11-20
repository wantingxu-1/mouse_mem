"""
CORE 3 - Neurons or glia? Group-pattern contrast (all 48 mice).  KEY ANALYSIS.

Question : Does behaviour follow the NeuN pattern (all treated groups low) or
           the glia pattern (only PM2.5 and PM2.5 EV high)?
Logic    : The two patterns differ only in N EV (NeuN low, glia normal).
           If memory in N EV is normal, behaviour follows glia, not NeuN.
Method   : OLS on all mice with the outcome:
             y ~ glia_high          (glia pattern alone)
             y ~ neun_low           (NeuN pattern alone)
             y ~ glia_high + neun_low   (which one survives when both are in)
           Compare by R^2, AIC and the joint-model coefficients.
           Also: Saline vs N EV Mann-Whitney (same glia, different NeuN).
Output   : results/tables/core_pattern_contrast.csv
The function `contrast_table` is reused by core.exclusion_sensitivity.
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import ols


def contrast_table(df, outcomes):
    separable = not (df.glia_high == df.neun_low).all()
    rows = []
    for y in outcomes:
        d = df[[y, "glia_high", "neun_low", "group"]].dropna()
        mg, mn = ols(f"{y} ~ glia_high", d), ols(f"{y} ~ neun_low", d)
        r = {"outcome": config.LABELS.get(y, y), "n": len(d),
             "glia R2": mg.rsquared, "glia p": mg.pvalues["glia_high"], "glia AIC": mg.aic,
             "NeuN R2": mn.rsquared, "NeuN p": mn.pvalues["neun_low"], "NeuN AIC": mn.aic,
             "joint glia coef": np.nan, "joint glia p": np.nan, "joint NeuN coef": np.nan, "joint NeuN p": np.nan,
             "separable": separable}
        if separable:
            mb = ols(f"{y} ~ glia_high + neun_low", d)
            r.update({"joint glia coef": mb.params["glia_high"], "joint glia p": mb.pvalues["glia_high"],
                      "joint NeuN coef": mb.params["neun_low"], "joint NeuN p": mb.pvalues["neun_low"]})
        if {"Saline", "N EV"} <= set(d.group.astype(str)):
            r["Saline vs N EV p"] = stats.mannwhitneyu(d.loc[d.group == "Saline", y], d.loc[d.group == "N EV", y]).pvalue
        rows.append(r)
    return pd.DataFrame(rows)


def run():
    header("CORE 3  Glia pattern vs NeuN pattern (all mice)")
    df = get_dataset()
    t = contrast_table(df, config.MEMORY_OUTCOMES + config.ANXIETY_OUTCOMES)
    log(t[["outcome", "n", "glia R2", "glia p", "NeuN R2", "NeuN p", "joint glia p", "joint NeuN p"]].round(3).to_string(index=False))
    save_table(t, "core_pattern_contrast")
    return t


if __name__ == "__main__":
    run()
