"""
ROBUSTNESS 4 - Holm correction for the pre-specified primary family.

Family   : For each primary week-12 memory outcome (config.PRIMARY_OUTCOMES:
           Mem12, T12_15, T12_60, NOR12), two tests from the joint model
           y ~ glia_high + neun_low:  glia effect and NeuN effect.
           = 8 tests. Holm controls the chance of ANY false positive in
           this family. Applied to both the OLS p and the permutation p.
Caveat   : Holm cannot remove bias from earlier rounds of looking at the
           same mice (the glia/NeuN patterns were chosen from the IHC, not
           from behaviour, which limits but does not remove this concern).
Output   : results/tables/robust_holm_primary.csv
"""
import pandas as pd
import config
from core.pattern_contrast import contrast_table
from features.dataset import get_dataset
from utils.permutation import conditional_pattern_perm
from utils.io import header, log, save_table
from utils.stats_helpers import holm, rng


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("ROBUSTNESS 4  Holm-corrected primary family")
    df = get_dataset()
    ct = contrast_table(df, config.PRIMARY_OUTCOMES)
    gen = rng()
    rows = []
    for y, (_, r) in zip(config.PRIMARY_OUTCOMES, ct.iterrows()):
        d = df[[y, "group", "glia_high", "neun_low"]].dropna()
        for term, label, pcol, ccol in [("glia_high", "glia effect", "joint glia p", "joint glia coef"),
                                        ("neun_low", "NeuN effect", "joint NeuN p", "joint NeuN coef")]:
            _, pp = conditional_pattern_perm(d, y, term, n_perm, gen)
            rows.append({"outcome": config.LABELS[y], "effect": label, "coef": r[ccol], "OLS p": r[pcol], "perm p": pp})
    t = pd.DataFrame(rows)
    t["Holm OLS p"] = holm(t["OLS p"])
    t["Holm perm p"] = holm(t["perm p"])
    log(t.round(4).to_string(index=False))
    save_table(t, "robust_holm_primary")
    return t


if __name__ == "__main__":
    run()
