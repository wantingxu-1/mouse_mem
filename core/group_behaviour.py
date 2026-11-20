"""
CORE 1 - Group differences in behaviour (all 48 mice).

Question : Which behaviours differ between the four groups?
Method   : Group mean +/- SEM; Kruskal-Wallis across 4 groups (rank-based,
           suits the 4-level T-maze scores); Mann-Whitney for fixed pairs
           (each group vs Saline; PM2.5 vs each EV group).
           BH correction across all behaviour outcomes for the KW test.
Output   : results/tables/core_group_behaviour.csv
"""
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import bh, kruskal_by_group

PAIRS = [("Saline", "PM2.5"), ("Saline", "PM2.5 EV"), ("Saline", "N EV"),
         ("PM2.5", "PM2.5 EV"), ("PM2.5", "N EV")]


def run():
    header("CORE 1  Group differences in behaviour")
    df = get_dataset()
    outcomes = list(config.BEHAVIOUR_COLUMNS.values()) + ["Mem4", "Mem12"]
    rows = []
    for y in outcomes:
        r = {"outcome": config.LABELS.get(y, y)}
        for g in config.GROUPS:
            v = df.loc[df.group == g, y].dropna()
            r[f"{g} mean"], r[f"{g} sem"], r[f"{g} n"] = v.mean(), v.sem(), len(v)
        r["KW p"] = kruskal_by_group(df, y)
        for a, b in PAIRS:
            r[f"MW p {a} vs {b}"] = stats.mannwhitneyu(df.loc[df.group == a, y].dropna(),
                                                      df.loc[df.group == b, y].dropna()).pvalue
        rows.append(r)
    t = pd.DataFrame(rows)
    t.insert(t.columns.get_loc("KW p") + 1, "KW q (BH)", bh(t["KW p"]))
    log(t[["outcome"] + [f"{g} mean" for g in config.GROUPS] + ["KW p", "KW q (BH)"]].round(3).to_string(index=False))
    save_table(t, "core_group_behaviour")
    return t


if __name__ == "__main__":
    run()
