"""
CORE 10 - Do the composite scores depend on how missing items are handled?

Memory composite: NOR wk12 is missing for 6 mice, so their composite is the
  mean of 2 items instead of 3. Re-run the primary contrast
  (y ~ glia_high + neun_low) with
    available items (main) | complete cases, 3 items | T-maze items only.
Glia index: 4 mice have only one of GFAP / IBA1. Re-run the individual
  association with memory (raw and group-adjusted Spearman) with
    available markers (main) | both markers only | GFAP index | IBA1 index.
Output   : results/tables/core_composition_sensitivity.csv
"""
import pandas as pd
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import ols, spearman, within_group_spearman, zscore


def run():
    header("CORE 10  Composite sensitivity: memory composite items, glia index markers")
    df = get_dataset()
    z = {c: zscore(df[c]) for c in config.MEMORY_ITEMS["Mem12"]}
    df["Mem12_complete3"] = pd.concat(z, axis=1).mean(axis=1, skipna=False)
    df["Mem12_tmaze_only"] = pd.concat([z["T12_15"], z["T12_60"]], axis=1).mean(axis=1)
    both = df[["GFAP_idx", "IBA1_idx"]].notna().all(axis=1)
    df["Glia_both_markers"] = df.Glia_idx.where(both)

    rows = []
    for y, label in [("Mem12", "available items (main)"), ("Mem12_complete3", "complete cases, 3 items"),
                     ("Mem12_tmaze_only", "T-maze items only")]:
        d = df[[y, "glia_high", "neun_low"]].dropna()
        m = ols(f"{y} ~ glia_high + neun_low", d)
        rows.append({"analysis": "memory composite wk12: primary contrast", "version": label, "n": len(d),
                     "outcome SD": d[y].std(), "glia-pattern coef": m.params["glia_high"],
                     "glia-pattern p": m.pvalues["glia_high"], "NeuN-pattern p": m.pvalues["neun_low"]})
    for x, label in [("Glia_idx", "available markers (main)"), ("Glia_both_markers", "both markers only"),
                     ("GFAP_idx", "GFAP index"), ("IBA1_idx", "IBA1 index")]:
        for y in ["Mem12", "T12_15"]:
            rho, p, n = spearman(df[x], df[y])
            rw, pw, _ = within_group_spearman(df, x, y)
            rows.append({"analysis": f"glia index vs {config.LABELS[y]}", "version": label, "n": n,
                         "rho": rho, "p": p, "within-group rho": rw, "within-group p": pw})
    t = pd.DataFrame(rows)
    log(t.round(4).to_string(index=False))
    save_table(t, "core_composition_sensitivity")
    return t


if __name__ == "__main__":
    run()
