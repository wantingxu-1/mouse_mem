"""
ROBUSTNESS 13 - Are the stained (and proteomics) mice representative?

Why      : Only 5-6 of 12 mice per group were stained, and 6 per group had
           proteomics. If those mice were chosen for extreme or typical
           behaviour, every individual-level IHC/proteomics result (and the
           IHC group means) could be biased. How they were chosen is not
           recorded, so check it in the behaviour, which all 48 mice have.
Checks, for each subset (any IHC stain; all three stains; proteomics)
  1 Which mice are in the subset, per group (IDs).
  2 Behaviour of subset vs non-subset mice:
      per group: Mann-Whitney;
      overall: y ~ group + in_subset (OLS) and a stratified permutation
      test (subset labels shuffled within group).
  3 Does the main contrast (glia vs NeuN pattern) look the same inside the
    subset as in all 48 mice? Coefficients side by side.
Output   : results/tables/robust_selection_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset, ihc_columns
from utils.io import header, log, save_table
from utils.permutation import perm_p, within_group_permutation
from utils.stats_helpers import ols, rng

OUTCOMES = ["Mem12", "T12_15", "T12_60", "NOR12", "Mem4", "EPMopen12", "EPMentries12"]


def subsets(df):
    ihc = ihc_columns()
    return {"any IHC stain": df[ihc].notna().any(axis=1),
            "all three stains (indices)": df[["GFAP_idx", "IBA1_idx", "NeuN_idx"]].notna().all(axis=1),
            "proteomics": df.has_proteomics.astype(bool)}


def strat_perm(d, y, n_perm, gen):
    g = d.group.astype(str).to_numpy()
    s = d.inset.to_numpy().astype(int)
    yv = d[y].to_numpy(float)
    def stat(lab):
        return np.mean([yv[(g == k) & (lab == 1)].mean() - yv[(g == k) & (lab == 0)].mean()
                        for k in np.unique(g) if 0 < lab[g == k].sum() < (g == k).sum()])
    obs = stat(s)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stat(s[within_group_permutation(g, gen)])
    return obs, perm_p(obs, null)


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("ROBUSTNESS 13  Selection check: are stained / proteomics mice representative?")
    df = get_dataset()
    gen = rng()
    ids, rows, crows = [], [], []
    for sname, mask in subsets(df).items():
        d0 = df.assign(inset=mask.astype(int))
        for gname in config.GROUPS:
            ids.append({"subset": sname, "group": gname,
                        "mice in subset": ", ".join(map(str, d0.index[(d0.group == gname) & (d0.inset == 1)])),
                        "n": int(((d0.group == gname) & (d0.inset == 1)).sum())})
        for y in OUTCOMES:
            d = d0[[y, "group", "inset", "glia_high", "neun_low"]].dropna()
            r = {"subset": sname, "outcome": config.LABELS[y], "n in": int(d.inset.sum()), "n out": int((1 - d.inset).sum())}
            for gname in config.GROUPS:
                a = d.loc[(d.group == gname) & (d.inset == 1), y]
                b = d.loc[(d.group == gname) & (d.inset == 0), y]
                r[f"{gname} in - out"] = a.mean() - b.mean()
                r[f"{gname} MW p"] = stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan
            m = ols(f"{y} ~ C(group) + inset", d)
            r["in-subset coef (OLS, adj. group)"] = m.params["inset"]
            r["OLS p"] = m.pvalues["inset"]
            r["mean within-group difference"], r["stratified perm p"] = strat_perm(d, y, n_perm, gen)
            rows.append(r)
            if y in ("Mem12", "T12_15"):
                full = ols(f"{y} ~ glia_high + neun_low", d)
                sub = ols(f"{y} ~ glia_high + neun_low", d[d.inset == 1])
                crows.append({"subset": sname, "outcome": config.LABELS[y],
                              "glia coef, all mice": full.params["glia_high"], "glia coef, subset": sub.params["glia_high"],
                              "glia p, subset": sub.pvalues["glia_high"],
                              "NeuN coef, all mice": full.params["neun_low"], "NeuN coef, subset": sub.params["neun_low"],
                              "NeuN p, subset": sub.pvalues["neun_low"], "n subset": int(d.inset.sum())})
    idt, t, ct = pd.DataFrame(ids), pd.DataFrame(rows), pd.DataFrame(crows)
    log("Mice in each subset:\n" + idt.to_string(index=False))
    show = ["subset", "outcome", "n in", "n out"] + [f"{g} in - out" for g in config.GROUPS] + \
           ["in-subset coef (OLS, adj. group)", "OLS p", "stratified perm p"]
    log("Subset vs other mice (behaviour):\n" + t[show].round(3).to_string(index=False))
    log("Main contrast inside each subset vs all mice:\n" + ct.round(3).to_string(index=False))
    save_table(idt, "robust_selection_ids")
    save_table(t, "robust_selection_behaviour")
    save_table(ct, "robust_selection_contrast")
    return idt, t, ct


if __name__ == "__main__":
    run()
