"""
ROBUSTNESS 11 - Mediation: does treatment act on memory THROUGH glia or NeuN?

Idea     : The pattern contrast asks which group pattern memory follows.
           Mediation asks it per mouse: treatment -> mediator -> memory.
           Parallel mediation with both mediators at once:
             M_glia = a1 X + e,  M_neun = a2 X + e
             Y      = c' X + b1 M_glia + b2 M_neun + e
           indirect via glia = a1*b1, via NeuN = a2*b2, direct = c'.
Contrasts (stained mice with both indices and memory)
  PM2.5 groups vs Saline : PM2.5 + PM2.5 EV (X=1) vs Saline (X=0)
  N EV vs PM2.5 groups   : N EV (X=1) vs PM2.5 + PM2.5 EV (X=0)
  any treatment : PM2.5, PM2.5 EV, N EV (X=1) vs Saline (X=0)
Inference: 5,000 bootstrap resamples within each X level; percentile 95% CI.
Caveats  : n is small (12-24 mice per contrast); mediators were not
           randomised, so a causal reading needs "no unmeasured confounding
           of mediator and outcome". Read as a structured description.
Output   : results/tables/robust_mediation.csv
"""
import numpy as np
import pandas as pd
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import rng

CONTRASTS = {"PM2.5 groups vs Saline": (["PM2.5", "PM2.5 EV"], ["Saline"]),
             "N EV vs PM2.5 groups": (["N EV"], ["PM2.5", "PM2.5 EV"]),
             "any treatment vs Saline": (["PM2.5", "PM2.5 EV", "N EV"], ["Saline"])}
OUTCOMES = ["Mem12", "T12_15"]


def effects(x, mg, mn, y):
    X1 = np.column_stack([np.ones_like(x), x])
    a1 = np.linalg.lstsq(X1, mg, rcond=None)[0][1]
    a2 = np.linalg.lstsq(X1, mn, rcond=None)[0][1]
    b = np.linalg.lstsq(np.column_stack([np.ones_like(x), x, mg, mn]), y, rcond=None)[0]
    c = np.linalg.lstsq(X1, y, rcond=None)[0][1]
    return {"total effect": c, "direct effect": b[1], "a glia": a1, "b glia": b[2], "indirect via glia": a1 * b[2],
            "a NeuN": a2, "b NeuN": b[3], "indirect via NeuN": a2 * b[3]}


def run():
    header("ROBUSTNESS 11  Parallel mediation: treatment -> glia / NeuN -> memory")
    df = get_dataset()
    g = df.group.astype(str)
    gen = rng()
    rows = []
    for cname, (treat, ref) in CONTRASTS.items():
        for y in OUTCOMES:
            d = df[g.isin(treat + ref)][["Glia_idx", "NeuN_idx", y]].dropna()
            x = g.loc[d.index].isin(treat).astype(float).to_numpy()
            mg, mn, yy = d.Glia_idx.to_numpy(), d.NeuN_idx.to_numpy(), d[y].to_numpy(float)
            est = effects(x, mg, mn, yy)
            i1, i0 = np.where(x == 1)[0], np.where(x == 0)[0]
            boot = []
            for _ in range(config.N_BOOT_MEDIATION):
                i = np.r_[gen.choice(i1, len(i1)), gen.choice(i0, len(i0))]
                boot.append(effects(x[i], mg[i], mn[i], yy[i]))
            boot = pd.DataFrame(boot)
            r = {"contrast": cname, "outcome": config.LABELS[y], "n treated": len(i1), "n reference": len(i0)}
            for k in ["total effect", "direct effect", "indirect via glia", "indirect via NeuN", "a glia", "a NeuN"]:
                lo, hi = np.percentile(boot[k], [2.5, 97.5])
                r[k] = est[k]
                r[f"{k} 95% CI"] = f"[{lo:.2f}, {hi:.2f}]"
                r[f"{k} CI excludes 0"] = (lo > 0) or (hi < 0)
            rows.append(r)
    t = pd.DataFrame(rows)
    show = ["contrast", "outcome", "n treated", "n reference", "total effect", "indirect via glia",
            "indirect via glia 95% CI", "indirect via NeuN", "indirect via NeuN 95% CI", "direct effect", "direct effect 95% CI"]
    log(t[show].round(3).to_string(index=False))
    save_table(t, "robust_mediation")
    return t


if __name__ == "__main__":
    run()
