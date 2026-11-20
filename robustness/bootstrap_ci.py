"""
ROBUSTNESS 2 - Bootstrap confidence intervals (resampling whole mice).

Why      : Gives a range of plausible values for the key effects, not just p.
Method   : Stratified bootstrap - within each group, draw mice with
           replacement (group sizes kept). A mouse is drawn with all its
           measurements. 95% percentile intervals.
Statistics
  - glia-pattern R^2, joint glia coefficient, joint NeuN coefficient
    (y ~ glia_high + neun_low) for Mem12 and T12_15, all 48 mice
  - Spearman rho of Glia index / NeuN index with Mem12 (stained mice)
Note     : The bootstrap shows sensitivity to *these* mice; it is not a new
           experiment.
Output   : results/tables/robust_bootstrap.csv
"""
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import design, fast_ols, rng


def stratified_sample(d, gen):
    idx = []
    for _, g in d.groupby("group", observed=True):
        idx.extend(gen.choice(g.index.to_numpy(), size=len(g), replace=True))
    return d.loc[idx]


def statistics(d_all, d_ihc):
    out = {}
    for y in ["Mem12", "T12_15"]:
        d = d_all[[y, "glia_high", "neun_low"]].dropna()
        out[f"{y}: glia-pattern R2"] = fast_ols(design(d, ["glia_high"]), d[y])[2]
        b = fast_ols(design(d, ["glia_high", "neun_low"]), d[y])[0]
        out[f"{y}: joint glia coef"], out[f"{y}: joint NeuN coef"] = b[1], b[2]
    for x in ["Glia_idx", "NeuN_idx"]:
        d = d_ihc[[x, "Mem12"]].dropna()
        out[f"rho {x} vs Mem12"] = stats.spearmanr(d[x], d["Mem12"])[0]
    return out


def run(n_boot=None):
    n_boot = n_boot or config.N_BOOTSTRAP
    header(f"ROBUSTNESS 2  Stratified bootstrap ({n_boot} resamples)")
    df = get_dataset()
    ihc = df[df[["Glia_idx", "NeuN_idx"]].notna().any(axis=1)]
    gen = rng()
    obs = statistics(df, ihc)
    boots = pd.DataFrame([statistics(stratified_sample(df, gen), stratified_sample(ihc, gen)) for _ in range(n_boot)])
    t = pd.DataFrame({"statistic": list(obs), "estimate": list(obs.values()),
                      "CI 2.5%": boots.quantile(.025).values, "CI 97.5%": boots.quantile(.975).values})
    t["CI excludes 0"] = (t["CI 2.5%"] > 0) | (t["CI 97.5%"] < 0)
    log(t.round(3).to_string(index=False))
    save_table(t, "robust_bootstrap")
    return t


if __name__ == "__main__":
    run()
