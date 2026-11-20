"""
ROBUSTNESS 1 - Permutation tests (no distributional assumptions).

Why      : n is small and T-maze has only 4 possible values, so p-values
           from t/F distributions may be inaccurate. A permutation p asks:
           "if labels were exchangeable, how often is the statistic this big?"
Tests
  (a) Group effect: Kruskal-Wallis H with group labels shuffled across mice.
  (b) Glia effect given NeuN: shuffle group labels only among treated mice
      (neun_low = 1: PM2.5, PM2.5 EV, N EV). NeuN status is unchanged, so this
      tests glia_high while holding the NeuN pattern fixed.
      Statistic: t of glia_high in y ~ glia_high + neun_low.
  (c) NeuN effect given glia: shuffle labels only among glia-normal mice
      (Saline, N EV). Statistic: t of neun_low in the same model.
  (d) Individual correlations within groups: shuffle behaviour within each
      group (keeps group differences), statistic = group-adjusted rho.
All p-values are two-sided: (1 + #|perm| >= |obs|) / (1 + N).
Output   : results/tables/robust_permutation.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.permutation import conditional_pattern_perm, within_group_corr_perm
from utils.stats_helpers import rng


def kw_perm(d, y, n_perm, gen):
    vals, labels = d[y].to_numpy(), d.group.astype(str).to_numpy()
    H = lambda lab: stats.kruskal(*[vals[lab == g] for g in np.unique(lab)]).statistic
    obs = H(labels)
    null = np.array([H(gen.permutation(labels)) for _ in range(n_perm)])
    return obs, (1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header(f"ROBUSTNESS 1  Permutation tests ({n_perm} permutations)")
    df = get_dataset()
    gen = rng()
    rows = []
    for y in config.PRIMARY_OUTCOMES + ["Mem4"]:
        d = df[[y, "group", "glia_high", "neun_low"]].dropna()
        H, p = kw_perm(d, y, n_perm, gen)
        rows.append({"test": "group effect (Kruskal-Wallis H)", "outcome": config.LABELS[y], "n": len(d), "statistic": H, "perm p": p})
        for term, name in [("glia_high", "glia effect | NeuN fixed (t)"), ("neun_low", "NeuN effect | glia fixed (t)")]:
            t, p = conditional_pattern_perm(d, y, term, n_perm, gen)
            rows.append({"test": name, "outcome": config.LABELS[y], "n": len(d), "statistic": t, "perm p": p})
    for x in ["Glia_idx", "NeuN_idx"]:
        for y in ["Mem12", "T12_15"]:
            r, p, n = within_group_corr_perm(df, x, y, min(n_perm, 4999), gen)
            rows.append({"test": f"within-group rho {x}", "outcome": config.LABELS[y], "n": n, "statistic": r, "perm p": p})
    t = pd.DataFrame(rows)
    log(t.round(4).to_string(index=False))
    save_table(t, "robust_permutation")
    return t


if __name__ == "__main__":
    run()
