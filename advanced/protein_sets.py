"""
ADVANCED 1 - Knowledge-guided protein sets (idea from MetaGeno / PRS).

Literature
  MetaGeno (Yang et al., Brief Bioinform 2025) groups thousands of SNPs by
  chromosome and summarises each group, and uses SNPs pre-selected from
  external PGS scores instead of selecting them on its own outcome.
  PRS / transfer learning likewise import weights from outside the cohort.
  The shared principle for small n, large p: reduce dimension with PRIOR
  KNOWLEDGE, not with the outcome.

Here
  ~3,500 proteins -> 14 literature-defined cell-type / pathway sets
  (config.PROTEIN_SETS). Set score = mean z-score of detected members
  (>= 20 mice), equal weights, no fitting on outcomes.

Tests per set (24 proteomics mice)
  - group differences: ANOVA; Welch N EV vs PM2.5 and PM2.5 vs Saline;
  - association with IHC indices (Glia, IBA1, GFAP, NeuN) and memory
    (Mem12, T12_15): raw Spearman and group-adjusted rho;
  - BH across all set x target tests;
  - competitive test: is the set's |rho| larger than for 1,000 random sets
    of the same size? (guards against "everything moves together").
Output   : results/tables/adv_protein_set_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from features.protein_sets import resolve_members, set_scores, zmatrix  # noqa: F401 (re-exported)
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from utils.stats_helpers import bh, rng, spearman, within_group_spearman

TARGETS = ["Glia_idx", "IBA1_idx", "GFAP_idx", "NeuN_idx", "Mem12", "T12_15"]



def run():
    header("ADVANCED 1  Knowledge-guided protein sets")
    L, grp, _ = get_proteomics()
    df = get_dataset().loc[L.index]
    members = resolve_members(L)
    Z = zmatrix(L)
    S = set_scores(Z, members)
    log("Sets used: " + "; ".join(f"{k} ({len(v)})" for k, v in members.items()))
    save_table(pd.DataFrame([{"set": k, "n proteins": len(v), "proteins": " ".join(c.split(";")[0] for c in v)}
                             for k, v in members.items()]), "adv_protein_set_members")

    # group differences
    g = grp.astype(str)
    grow = []
    for k in S:
        vals = {x: S.loc[g == x, k] for x in config.GROUPS}
        grow.append({"set": k, **{f"{x} mean z": vals[x].mean() for x in config.GROUPS},
                     "ANOVA p": stats.f_oneway(*vals.values()).pvalue,
                     "PM2.5 vs Saline p": stats.ttest_ind(vals["PM2.5"], vals["Saline"], equal_var=False).pvalue,
                     "N EV vs PM2.5 p": stats.ttest_ind(vals["N EV"], vals["PM2.5"], equal_var=False).pvalue})
    gt = pd.DataFrame(grow)
    gt["ANOVA q"] = bh(gt["ANOVA p"])
    log(gt.round(3).to_string(index=False))
    save_table(gt, "adv_protein_set_groups")

    # associations + competitive null
    gen = rng()
    pool = [c for c in L.columns if L[c].notna().sum() >= config.MIN_MICE_FOR_MARKER_SCORE]
    d = pd.concat([S, df[TARGETS + ["group"]]], axis=1)
    rows = []
    for k, cols in members.items():
        size = len(cols)
        rand = np.array([Z[gen.choice(pool, size, replace=False)].mean(axis=1).to_numpy() for _ in range(config.N_RANDOM_SETS)])
        for t in TARGETS:
            rho, p, n = spearman(d[k], d[t])
            rw, pw, _ = within_group_spearman(d, k, t)
            y = d[t]
            ok = y.notna().to_numpy()
            null = np.array([stats.spearmanr(r[ok], y[ok])[0] for r in rand])
            rows.append({"set": k, "target": t, "n": n, "rho raw": rho, "p raw": p,
                         "rho group-adjusted": rw, "p group-adjusted": pw,
                         "competitive p (vs random sets)": (1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null))})
    at = pd.DataFrame(rows)
    at["q raw"] = bh(at["p raw"])
    at["q group-adjusted"] = bh(at["p group-adjusted"])
    log(f"Set x target tests: {len(at)}; raw p<0.05: {(at['p raw'] < .05).sum()} (chance {0.05 * len(at):.1f}); "
        f"min q raw {at['q raw'].min():.3f}; min q group-adjusted {at['q group-adjusted'].min():.3f}")
    top = at.sort_values("p raw").head(12)
    log(top.round(3).to_string(index=False))
    save_table(at, "adv_protein_set_associations")
    S.to_csv(config.TABLE_DIR / "adv_protein_set_scores.csv")
    return gt, at


if __name__ == "__main__":
    run()
