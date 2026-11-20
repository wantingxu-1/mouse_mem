"""
CORE 9 - Inside N EV: do all mice recover, or only some (responders)?

Why      : N EV is the group that separates "glia" from "NeuN". Its group
           mean is Saline-like for glia and memory, but a mean can hide a
           mix: some mice fully recovered, others still damaged. If the
           mice whose glia did NOT recover are also the ones with poor
           memory, that is individual-level support for the glia account.
Method   (descriptive; n = 12 behaviour, 5-6 IHC in N EV)
  Reference distributions
    "Saline-like" = Saline group; "damaged" = PM2.5 + PM2.5 EV pooled.
  1 Recovery score per mouse and measure:
      s = (x - damaged mean) / (Saline mean - damaged mean)
      s ~ 1 Saline-like, s ~ 0 like the damaged groups.
  2 Classification: normal likelihood under Saline vs damaged (pooled SD);
    P(Saline-like) per mouse. Calibrated by classifying Saline and
    damaged mice themselves leave-one-out (how often each is called
    Saline-like). Fraction of N EV called Saline-like per measure.
  3 Spread: is N EV more variable than Saline? (Levene, Brown-Forsythe)
  4 Stained N EV mice: glia recovery vs memory side by side, and
    Spearman within N EV (n <= 6, descriptive only).
Output   : results/tables/core_nev_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table

MEASURES = ["Mem12", "T12_15", "NOR12", "GFAP_pct", "IBA1_pct", "NeuN_pct"]
DAMAGED = ["PM2.5", "PM2.5 EV"]


def p_saline(x, sal, dam):
    """P(Saline-like) under two normals with a pooled SD (equal priors)."""
    sd = np.sqrt(((len(sal) - 1) * sal.var() + (len(dam) - 1) * dam.var()) / (len(sal) + len(dam) - 2))
    l_s, l_d = stats.norm.logpdf(x, sal.mean(), sd), stats.norm.logpdf(x, dam.mean(), sd)
    return 1 / (1 + np.exp(l_d - l_s))


def run():
    header("CORE 9  Heterogeneity within N EV (responders?)")
    df = get_dataset()
    g = df.group.astype(str)
    nev = df[g == "N EV"]

    rec, cls, cal, spread = {}, {}, [], []
    for m in MEASURES:
        sal, dam = df.loc[g == "Saline", m].dropna(), df.loc[g.isin(DAMAGED), m].dropna()
        x = nev[m]
        rec[m] = (x - dam.mean()) / (sal.mean() - dam.mean())
        cls[m] = x.apply(lambda v: p_saline(v, sal, dam) if pd.notna(v) else np.nan)
        # leave-one-out calibration
        sal_loo = [p_saline(sal[i], sal.drop(i), dam) > .5 for i in sal.index]
        dam_loo = [p_saline(dam[i], sal, dam.drop(i)) > .5 for i in dam.index]
        k, n = int((cls[m].dropna() > .5).sum()), int(x.notna().sum())
        cal.append({"measure": config.LABELS.get(m, m), "N EV called Saline-like": f"{k}/{n}",
                    "N EV fraction": k / n, "Saline called Saline-like (LOO)": np.mean(sal_loo),
                    "damaged called Saline-like (LOO)": np.mean(dam_loo),
                    "separation d (Saline vs damaged)": (sal.mean() - dam.mean()) / np.sqrt((sal.var() + dam.var()) / 2),
                    "N EV mean recovery score": rec[m].mean()})
        spread.append({"measure": config.LABELS.get(m, m), "SD Saline": sal.std(), "SD N EV": x.std(),
                       "SD damaged": dam.std(), "Levene N EV vs Saline p": stats.levene(x.dropna(), sal).pvalue})
    cal, spread = pd.DataFrame(cal), pd.DataFrame(spread)
    log("Fraction of N EV mice classified Saline-like (with leave-one-out calibration):\n" + cal.round(2).to_string(index=False))
    log("Spread:\n" + spread.round(3).to_string(index=False))
    save_table(cal, "core_nev_classification")
    save_table(spread, "core_nev_spread")

    per = pd.concat({**{f"{m} recovery": rec[m] for m in MEASURES},
                     **{f"{m} P(Saline-like)": cls[m] for m in MEASURES}}, axis=1)
    per.insert(0, "Glia recovery", per[["GFAP_pct recovery", "IBA1_pct recovery"]].mean(axis=1, skipna=False))
    per.insert(1, "Memory recovery", per[["T12_15 recovery", "NOR12 recovery"]].mean(axis=1))
    log("Per N EV mouse (recovery 1 = Saline-like, 0 = damaged-like):\n" +
        per[["Glia recovery", "GFAP_pct recovery", "IBA1_pct recovery", "NeuN_pct recovery",
             "Memory recovery", "Mem12 recovery", "T12_15 recovery", "NOR12 recovery"]].round(2).to_string())
    save_table(per.reset_index(), "core_nev_per_mouse")

    rows = []
    for x in ["Glia recovery", "GFAP_pct recovery", "IBA1_pct recovery", "NeuN_pct recovery"]:
        for y in ["Mem12 recovery", "T12_15 recovery", "NOR12 recovery"]:
            d = per[[x, y]].dropna()
            rho, p = stats.spearmanr(d[x], d[y]) if len(d) >= 4 else (np.nan, np.nan)
            rows.append({"IHC": x, "memory": y, "n": len(d), "rho (within N EV)": rho, "p": p})
    wc = pd.DataFrame(rows)
    log("Within N EV, IHC recovery vs memory recovery (descriptive):\n" + wc.round(3).to_string(index=False))
    save_table(wc, "core_nev_within_correlation")
    return cal, spread, per, wc


if __name__ == "__main__":
    run()
