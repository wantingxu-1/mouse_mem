"""
CORE 5 - Individual mice: do brain markers and behaviour vary together?

Question : Do mice with more glia / less NeuN behave worse?
Method   : For every (brain measure, behaviour) pair:
             raw Spearman rho over all mice that have both values;
             group-adjusted rho (rank, subtract group mean rank, correlate) -
             asks whether a mouse that is high *for its group* is also
             high/low *for its group* in behaviour.
           BH correction across all pairs, separately for raw and adjusted.
Note     : Only 5-6 stained mice per group, so the group-adjusted test has
           very low power.
Output   : results/tables/core_individual_correlation.csv
"""
import pandas as pd
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import bh, spearman, within_group_spearman

PREDICTORS = ["NeuN_idx", "GFAP_idx", "IBA1_idx", "Glia_idx", "NeuN CA1", "IBA1 DG", "IBA1 CA2", "GFAP CA2"]
OUTCOMES = ["Mem12", "T12_15", "T12_60", "NOR12", "EPMopen12", "EPMentries12", "Mem4", "T4_15", "EPMopen4"]


def run():
    header("CORE 5  Individual-mouse correlations")
    df = get_dataset()
    rows = []
    for x in PREDICTORS:
        for y in OUTCOMES:
            rho, p, n = spearman(df[x], df[y])
            rw, pw, _ = within_group_spearman(df, x, y)
            rows.append({"brain": x, "behaviour": config.LABELS.get(y, y), "n": n,
                         "rho raw": rho, "p raw": p, "rho group-adjusted": rw, "p group-adjusted": pw})
    t = pd.DataFrame(rows)
    t["q raw"] = bh(t["p raw"])
    t["q group-adjusted"] = bh(t["p group-adjusted"])
    log(f"Pairs tested: {len(t)}; raw p<0.05: {(t['p raw'] < .05).sum()}; "
        f"group-adjusted p<0.05: {(t['p group-adjusted'] < .05).sum()}; "
        f"min q raw {t['q raw'].min():.3f}; min q adjusted {t['q group-adjusted'].min():.3f}")
    key = t[t.brain.isin(["Glia_idx", "NeuN_idx"]) & t.behaviour.isin([config.LABELS["Mem12"], config.LABELS["T12_15"]])]
    log(key.round(3).to_string(index=False))
    save_table(t, "core_individual_correlation")
    return t


if __name__ == "__main__":
    run()
