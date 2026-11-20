"""
ROBUSTNESS 12 - Possible sex confound: sensitivity analysis with Ddx3y.

Background: In the proteomics, the Y-linked protein Ddx3y is detected in
  Saline 0/6, PM2.5 2/6, PM2.5 EV 5/6, N EV 2/6 mice (qc/proteomics_integrity.py).
  If detection means "male", group sex ratios differ and sex could explain
  part of the group differences. No animal records are available, so the
  question is answered from the data only.
Checks (24 proteomics mice)
  1 Is Ddx3y a real sex marker? X-linked genes that escape X inactivation
    (Ddx3x, Kdm5c, Kdm6a, Eif2s3x) are expressed more in females. If
    Ddx3y-positive mice have LOWER levels of these proteins, Ddx3y tracks
    sex; if not, detection is more likely peptide mis-assignment / noise.
    Compared with a null of 1,000 random protein sets of the same size.
  2 Does Ddx3y relate to memory or to the IHC indices (all mice and within
    group)?
  3 Main contrast on the same 24 mice, without and with Ddx3y as a
    covariate: Mem12 / T12_15 ~ glia_high + neun_low (+ ddx3y); OLS and
    conditional permutation p for glia.
  4 Within each Ddx3y stratum: glia-high vs glia-normal memory means
    (descriptive; is the direction the same in both "sexes"?).
Output   : results/tables/robust_sex_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from features.proteomics_prep import get_proteomics, protein_column
from utils.permutation import conditional_pattern_perm
from utils.io import header, log, save_table
from utils.stats_helpers import group_centred_ranks, ols, rng

X_ESCAPEES = ["Ddx3x", "Kdm5c", "Kdm6a", "Eif2s3x", "Utp14a", "Kdm5d"]
OUTCOMES = ["Mem12", "T12_15", "NOR12"]


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("ROBUSTNESS 12  Sex sensitivity (Ddx3y as a sex proxy)")
    L, grp, keep = get_proteomics()
    df = get_dataset().loc[L.index].copy()
    df["ddx3y"] = L[protein_column(L, "Ddx3y")].notna().astype(int)
    gen = rng()

    # 1 validation with X-escapee proteins
    esc = [protein_column(L, p) for p in X_ESCAPEES[:-1] if protein_column(L, p) in keep]
    rows = []
    for c in esc:
        a, b = L.loc[df.ddx3y == 1, c].dropna(), L.loc[df.ddx3y == 0, c].dropna()
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        rows.append({"X-escapee protein": c.split(";")[0], "n Ddx3y+": len(a), "n Ddx3y-": len(b),
                     "log2 difference (Ddx3y+ minus -)": a.mean() - b.mean(), "effect size d": d,
                     "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue})
    xv = pd.DataFrame(rows)
    full = [c for c in keep if L[c].notna().sum() >= 20]
    Z = (L[full] - L[full].mean()) / L[full].std()
    diff = Z[df.ddx3y == 1].mean() - Z[df.ddx3y == 0].mean()
    obs = diff[esc].mean()
    null = np.array([diff[gen.choice(full, len(esc), replace=False)].mean() for _ in range(config.N_RANDOM_SETS)])
    xv_summary = (f"Mean z-difference of {len(esc)} X-escapee proteins (Ddx3y+ minus -) = {obs:.3f}; "
                  f"random sets: mean {null.mean():.3f}, percentile of observed {(null < obs).mean():.3f} "
                  f"(low percentile = escapees lower in Ddx3y+ mice, as expected if Ddx3y marks males)")
    log("X-escapee check:\n" + xv.round(3).to_string(index=False))
    log(xv_summary)
    xv["summary"] = [xv_summary] + [""] * (len(xv) - 1)
    save_table(xv, "robust_sex_xescapee_check")

    # 2 Ddx3y vs memory / IHC
    rows = []
    for y in OUTCOMES + ["Glia_idx", "NeuN_idx"]:
        d = df[[y, "ddx3y", "group"]].dropna()
        a, b = d.loc[d.ddx3y == 1, y], d.loc[d.ddx3y == 0, y]
        c = group_centred_ranks(d, [y, "ddx3y"])
        rw, pw = stats.pearsonr(c[y], c["ddx3y"])
        rows.append({"variable": config.LABELS.get(y, y), "n": len(d), "mean Ddx3y+": a.mean(), "mean Ddx3y-": b.mean(),
                     "Mann-Whitney p (all mice)": stats.mannwhitneyu(a, b).pvalue,
                     "within-group r": rw, "within-group p": pw})
    assoc = pd.DataFrame(rows)
    log("Ddx3y vs memory and IHC:\n" + assoc.round(3).to_string(index=False))
    save_table(assoc, "robust_sex_association")

    # 3 main contrast with / without Ddx3y
    rows = []
    for y in OUTCOMES:
        d = df[[y, "glia_high", "neun_low", "ddx3y", "group"]].dropna()
        m0 = ols(f"{y} ~ glia_high + neun_low", d)
        m1 = ols(f"{y} ~ glia_high + neun_low + ddx3y", d)
        rows.append({"outcome": config.LABELS[y], "n": len(d),
                     "glia coef (no sex term)": m0.params["glia_high"], "glia p (no sex term)": m0.pvalues["glia_high"],
                     "glia coef (+ Ddx3y)": m1.params["glia_high"], "glia p (+ Ddx3y)": m1.pvalues["glia_high"],
                     "glia perm p (+ Ddx3y)": conditional_pattern_perm(d, y, "glia_high", n_perm, gen, ["ddx3y"])[1],
                     "% change in glia coef": 100 * (m1.params["glia_high"] / m0.params["glia_high"] - 1),
                     "NeuN p (+ Ddx3y)": m1.pvalues["neun_low"], "Ddx3y coef": m1.params["ddx3y"],
                     "Ddx3y p": m1.pvalues["ddx3y"]})
    adj = pd.DataFrame(rows)
    log("Glia/NeuN contrast on the 24 proteomics mice, with and without Ddx3y:\n" + adj.round(3).to_string(index=False))
    save_table(adj, "robust_sex_adjusted_contrast")

    # 4 within strata
    rows = []
    for s in (0, 1):
        for y in OUTCOMES:
            d = df[(df.ddx3y == s)][[y, "glia_high", "group"]].dropna()
            rows.append({"Ddx3y": "detected" if s else "not detected", "outcome": config.LABELS[y],
                         "groups (n)": ", ".join(f"{k} {v}" for k, v in d.group.astype(str).value_counts().sort_index().items()),
                         "glia-normal mean": d.loc[d.glia_high == 0, y].mean(), "glia-high mean": d.loc[d.glia_high == 1, y].mean(),
                         "difference (high - normal)": d.loc[d.glia_high == 1, y].mean() - d.loc[d.glia_high == 0, y].mean()})
    strata = pd.DataFrame(rows)
    log("Within Ddx3y strata (descriptive):\n" + strata.round(3).to_string(index=False))
    save_table(strata, "robust_sex_strata")
    return xv, assoc, adj, strata


if __name__ == "__main__":
    run()
