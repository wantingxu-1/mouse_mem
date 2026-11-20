"""
CORE 2 - Group differences in brain markers (IHC, 5-6 mice per group).

Question : Which markers (NeuN = neurons, IBA1 = microglia, GFAP = astrocytes)
           change, in which groups and regions?
Method   : Group mean +/- SEM and % of Saline; one-way ANOVA per region;
           Welch t-tests (unequal variances) for fixed pairs; BH across all
           pairwise tests (15 regions x 5 pairs).
Output   : results/tables/core_group_ihc.csv, core_group_ihc_percent.csv
"""
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset, ihc_columns
from utils.io import header, log, save_table
from utils.stats_helpers import bh

PAIRS = [("Saline", "PM2.5"), ("Saline", "PM2.5 EV"), ("Saline", "N EV"),
         ("PM2.5", "N EV"), ("PM2.5 EV", "N EV")]


def run():
    header("CORE 2  Group differences in IHC markers")
    df = get_dataset()
    cols = ihc_columns() + ["NeuN_idx", "GFAP_idx", "IBA1_idx"]
    rows = []
    for c in cols:
        v = {g: df.loc[df.group == g, c].dropna() for g in config.GROUPS}
        r = {"measure": c}
        for g in config.GROUPS:
            r[f"{g} mean"], r[f"{g} sem"], r[f"{g} n"] = v[g].mean(), v[g].sem(), len(v[g])
        r["ANOVA p"] = stats.f_oneway(*v.values()).pvalue
        for a, b in PAIRS:
            r[f"Welch p {a} vs {b}"] = stats.ttest_ind(v[a], v[b], equal_var=False).pvalue
        rows.append(r)
    t = pd.DataFrame(rows)
    pcols = [c for c in t.columns if c.startswith("Welch p")]
    region = t.measure.isin(ihc_columns())
    flat = t.loc[region, pcols].values.ravel()
    q = bh(flat).reshape(region.sum(), len(pcols))
    for i, c in enumerate(pcols):
        t.loc[region, c.replace("Welch p", "Welch q")] = q[:, i]
    save_table(t, "core_group_ihc")

    pct = pd.DataFrame({g: t[f"{g} mean"] / t["Saline mean"] * 100 for g in config.GROUPS})
    pct.insert(0, "measure", t.measure)
    pct = pct[region.values]
    log("% of Saline (group means):\n" + pct.round(0).to_string(index=False))
    log(f"Pairwise region tests with BH q<0.05: {(q < 0.05).sum()} of {q.size}")
    save_table(pct, "core_group_ihc_percent")
    return t


if __name__ == "__main__":
    run()
