"""
CORE 4 - Does the conclusion depend on one group? (exclusion sensitivity)

Question : What happens if N EV, or PM2.5 EV, is removed?
Method   : Re-run the pattern contrast (CORE 3) and the individual-mouse
           Spearman correlations of Glia/NeuN index with behaviour on each
           subset. Without N EV the two patterns are identical, so the
           question cannot be answered - this shows N EV is the key group.
Output   : results/tables/core_exclusion_sensitivity.csv
"""
import pandas as pd
from core.pattern_contrast import contrast_table
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import spearman

SCENARIOS = {"All 4 groups": [], "Exclude N EV": ["N EV"], "Exclude PM2.5 EV": ["PM2.5 EV"]}
OUTCOMES = ["Mem12", "T12_15", "T12_60", "NOR12", "Mem4", "T4_15", "T4_60"]


def run():
    header("CORE 4  Exclusion sensitivity")
    df = get_dataset()
    parts = []
    for name, excluded in SCENARIOS.items():
        d = df[~df.group.isin(excluded)]
        t = contrast_table(d, OUTCOMES)
        for i, y in enumerate(OUTCOMES):
            t.loc[i, "indiv Glia rho"], t.loc[i, "indiv Glia p"], _ = spearman(d.Glia_idx, d[y])
            t.loc[i, "indiv NeuN rho"], t.loc[i, "indiv NeuN p"], _ = spearman(d.NeuN_idx, d[y])
        t.insert(0, "scenario", name)
        parts.append(t)
        log(f"{name}: glia and NeuN patterns separable = {t.separable.iloc[0]}")
    t = pd.concat(parts, ignore_index=True)
    log(t[["scenario", "outcome", "n", "glia p", "NeuN p", "joint glia p", "joint NeuN p", "indiv Glia rho", "indiv NeuN rho"]].round(3).to_string(index=False))
    save_table(t, "core_exclusion_sensitivity")
    return t


if __name__ == "__main__":
    run()
