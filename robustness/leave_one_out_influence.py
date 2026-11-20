"""
ROBUSTNESS 3 - Leave-one-mouse-out influence.

Question : Is a key result carried by one or two mice?
Method   : Remove each mouse in turn and recompute
             - joint glia and NeuN coefficients/p (y ~ glia_high + neun_low)
               for Mem12 and T12_15;
             - Spearman rho of Glia index with Mem12.
           Report the range over all drops and the most influential mouse.
Output   : results/tables/robust_loo_influence_summary.csv,
           robust_loo_influence_by_mouse.csv
"""
import pandas as pd
from scipy import stats
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import ols


def stats_without(df, drop):
    d = df.drop(index=drop) if drop is not None else df
    out = {}
    for y in ["Mem12", "T12_15"]:
        m = ols(f"{y} ~ glia_high + neun_low", d[[y, "glia_high", "neun_low"]].dropna())
        out[f"{y} glia coef"], out[f"{y} glia p"] = m.params["glia_high"], m.pvalues["glia_high"]
        out[f"{y} NeuN coef"], out[f"{y} NeuN p"] = m.params["neun_low"], m.pvalues["neun_low"]
    dd = d[["Glia_idx", "Mem12"]].dropna()
    out["Glia rho"], out["Glia rho p"] = stats.spearmanr(dd.Glia_idx, dd.Mem12)
    return out


def run():
    header("ROBUSTNESS 3  Leave-one-mouse-out influence")
    df = get_dataset()
    full = stats_without(df, None)
    by = pd.DataFrame({m: stats_without(df, m) for m in df.index}).T
    by.index.name = "dropped mouse"
    rows = []
    for k, v in full.items():
        dev = (by[k] - v).abs()
        rows.append({"statistic": k, "all mice": v, "min after drop": by[k].min(), "max after drop": by[k].max(),
                     "most influential mouse": int(dev.idxmax()), "group of that mouse": str(df.group[dev.idxmax()])})
    t = pd.DataFrame(rows)
    log(t.round(4).to_string(index=False))
    save_table(t, "robust_loo_influence_summary")
    save_table(by.reset_index(), "robust_loo_influence_by_mouse")
    return t


if __name__ == "__main__":
    run()
