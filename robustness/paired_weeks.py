"""
ROBUSTNESS 5 - Week 4 -> week 12 change within the same mice.

Question : Does behaviour change between week 4 and week 12, and does the
           change differ between groups?
Method   : Per group, Wilcoxon signed-rank test (paired, same mouse) on
           week 12 minus week 4; Kruskal-Wallis across groups on the change.
Caveat   : Week 4 is already 4 weeks into treatment - NOT a pre-treatment
           baseline. Descriptive only.
Output   : results/tables/robust_paired_weeks.csv
"""
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import kruskal_by_group

PAIRS = {"T-maze 15s": ("T4_15", "T12_15"), "T-maze 60s": ("T4_60", "T12_60"),
         "NOR": ("NOR4", "NOR12"), "EPM open time": ("EPMopen4", "EPMopen12")}


def run():
    header("ROBUSTNESS 5  Paired week 4 -> week 12 change")
    df = get_dataset()
    rows = []
    for name, (w4, w12) in PAIRS.items():
        d = df[[w4, w12, "group"]].dropna().assign(change=lambda x: x[w12] - x[w4])
        r = {"measure": name, "KW p (change differs by group)": kruskal_by_group(d, "change")}
        for g in config.GROUPS:
            c = d.loc[d.group == g, "change"]
            r[f"{g} mean change"] = c.mean()
            r[f"{g} Wilcoxon p"] = stats.wilcoxon(c).pvalue if (c != 0).sum() >= 3 else float("nan")
        rows.append(r)
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "robust_paired_weeks")
    return t


if __name__ == "__main__":
    run()
