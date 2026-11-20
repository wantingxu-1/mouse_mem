"""
ROBUSTNESS 6 - Whole 5-region profile per marker (instead of region by region).

Questions
  (a) Does the 5-region profile of a marker differ between groups at all?
  (b) Is the N EV profile closer to Saline than the PM2.5 profile is?
Method
  Each region z-scored; mice with all 5 regions only.
  (a) PERMANOVA: pseudo-F on Euclidean distances, group labels permuted.
  (b) Distance from each group's centroid to the Saline centroid.
      Test: D = dist(PM2.5, Saline) - dist(N EV, Saline); permute labels
      between PM2.5 and N EV mice. Positive D = N EV closer to Saline.
Caveat   : "Closer to Saline" in a stain profile is not the same as
           functional recovery.
Output   : results/tables/robust_multiregion_profile.csv
"""
import numpy as np
import pandas as pd
import config
from features.dataset import get_dataset, ihc_columns
from features.ihc_indices import marker_columns
from utils.io import header, log, save_table
from utils.stats_helpers import rng


def pseudo_F(X, labels):
    grand = X.mean(0)
    ss_total = ((X - grand) ** 2).sum()
    ss_within = sum(((X[labels == g] - X[labels == g].mean(0)) ** 2).sum() for g in np.unique(labels))
    k, n = len(np.unique(labels)), len(X)
    return ((ss_total - ss_within) / (k - 1)) / (ss_within / (n - k))


def centroid_dist(X, labels, g, ref):
    return np.linalg.norm(X[labels == g].mean(0) - X[labels == ref].mean(0))


def run(n_perm=None):
    n_perm = n_perm or config.N_PERMUTATIONS
    header("ROBUSTNESS 6  Multi-region profile (PERMANOVA, distance to Saline)")
    df = get_dataset()
    gen = rng()
    rows = []
    for m in config.IHC_MARKERS:
        cols = marker_columns(ihc_columns(), m)
        d = df[cols + ["group"]].dropna()
        X = d[cols].apply(lambda s: (s - s.mean()) / s.std()).to_numpy()
        lab = d.group.astype(str).to_numpy()
        F = pseudo_F(X, lab)
        pF = (1 + sum(pseudo_F(X, gen.permutation(lab)) >= F for _ in range(n_perm))) / (1 + n_perm)
        dist = {g: centroid_dist(X, lab, g, "Saline") for g in config.GROUPS[1:]}
        D = dist["PM2.5"] - dist["N EV"]
        sub = np.isin(lab, ["PM2.5", "N EV"])
        null = []
        for _ in range(n_perm):
            lp = lab.copy()
            lp[sub] = gen.permutation(lp[sub])
            null.append(centroid_dist(X, lp, "PM2.5", "Saline") - centroid_dist(X, lp, "N EV", "Saline"))
        pD = (1 + np.sum(np.array(null) >= D)) / (1 + n_perm)   # one-sided: N EV closer
        rows.append({"marker": m, "n": len(d), "PERMANOVA pseudo-F": F, "PERMANOVA p": pF,
                     **{f"dist {g} to Saline": v for g, v in dist.items()},
                     "N EV closer than PM2.5 (D)": D, "one-sided p": pD})
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "robust_multiregion_profile")
    return t


if __name__ == "__main__":
    run()
