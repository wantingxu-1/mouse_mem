"""
ADVANCED 2 - Stability selection with decoy features (Stabl-like).

Literature
  Meinshausen & Buhlmann (2010) stability selection; Hedou et al., Nature
  Biotechnology 2024 ("Stabl"): add artificial decoy features that cannot
  relate to the outcome, run a sparse model on many subsamples, and keep
  only real features selected more often than decoys. The decoys give a
  data-driven false-discovery estimate, which is exactly what small-n,
  large-p omics needs.

Here (per target in config.STABILITY_TARGETS)
  X = filtered proteins (log2, median-normalised, per-protein median
      imputation, z-scored) + one decoy per protein (its values shuffled
      across mice, so any link to the target is destroyed).
  Repeat STABILITY_SUBSAMPLES times: draw half of the mice (stratified by
  group), fit the Lasso path (30 penalties), record for each feature the
  highest selection frequency over the path.
  Threshold t chosen to minimise FDP+(t) = (1 + #decoys >= t) / #real >= t.
  Report proteins with frequency >= t and the estimated FDP.
  If the best FDP+ is still high (> 0.5), the honest result is "nothing
  reliable".
Positive control
  Target = number of proteins detected per mouse (measurement depth), which
  is known to drive the largest proteomic variation. The method SHOULD find
  stable proteins here; if it does, a null result for biology targets means
  "no reliable signal", not "method cannot find anything".
Output   : results/tables/adv_stability_selection.csv (+ top features per target)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path
import config
from features.dataset import get_dataset
from features.proteomics_prep import detected_per_mouse, get_proteomics
from utils.io import header, log, save_table
from utils.stats_helpers import rng

N_ALPHAS = 30


def prepare_X(L, keep):
    X = L[keep].apply(lambda s: s.fillna(s.median()))
    return (X - X.mean()) / X.std()


def selection_frequencies(X, y, groups, gen):
    n, p = X.shape
    counts = np.zeros((N_ALPHAS, p))
    yc = (y - y.mean()) / y.std()
    alpha_max = np.abs(X.T @ yc).max() / n
    alphas = alpha_max * np.logspace(0, -1.5, N_ALPHAS)
    idx_by_group = [np.where(groups == g)[0] for g in np.unique(groups)]
    for _ in range(config.STABILITY_SUBSAMPLES):
        sub = np.concatenate([gen.choice(ix, max(1, len(ix) // 2), replace=False) for ix in idx_by_group])
        Xs = X[sub]
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
        ys = yc[sub] - yc[sub].mean()
        _, coefs, _ = lasso_path(Xs, ys, alphas=alphas, max_iter=2000)
        counts += (np.abs(coefs.T) > 1e-10)
    return (counts / config.STABILITY_SUBSAMPLES).max(axis=0)


def choose_threshold(freq_real, freq_decoy):
    best = (np.inf, None)
    for t in np.arange(0.3, 1.0, 0.01):
        n_real = (freq_real >= t).sum()
        if n_real == 0:
            break
        fdp = (1 + (freq_decoy >= t).sum()) / n_real
        if fdp < best[0]:
            best = (fdp, t)
    return best


def run():
    header(f"ADVANCED 2  Stability selection with decoys ({config.STABILITY_SUBSAMPLES} subsamples)")
    L, grp, keep = get_proteomics()
    df = get_dataset().loc[L.index]
    df["POSITIVE CONTROL: detection depth"] = detected_per_mouse().loc[L.index]
    gen = rng()
    X_all = prepare_X(L, keep)
    summary, selected = [], []
    for target in config.STABILITY_TARGETS + ["POSITIVE CONTROL: detection depth"]:
        ok = df[target].notna()
        X = X_all.loc[ok].to_numpy()
        decoy = np.column_stack([gen.permutation(X[:, j]) for j in range(X.shape[1])])
        Xf = np.hstack([X, decoy])
        y = df.loc[ok, target].to_numpy(float)
        freq = selection_frequencies(Xf, y, grp[ok].astype(str).to_numpy(), gen)
        fr, fd = freq[:X.shape[1]], freq[X.shape[1]:]
        fdp, t = choose_threshold(fr, fd)
        top = np.argsort(-fr)[:10]
        summary.append({"target": target, "n mice": int(ok.sum()), "n proteins": X.shape[1],
                        "max freq real": fr.max(), "max freq decoy": fd.max(),
                        "real >= 0.6": int((fr >= .6).sum()), "decoy >= 0.6": int((fd >= .6).sum()),
                        "best threshold": t, "estimated FDP at best": fdp,
                        "n selected (FDP<=0.5)": int((fr >= t).sum()) if t is not None and fdp <= 0.5 else 0,
                        "top 5 proteins (freq)": ", ".join(f"{keep[i].split(';')[0]} ({fr[i]:.2f})" for i in top[:5])})
        for rank, i in enumerate(top, 1):
            selected.append({"target": target, "rank": rank, "protein": keep[i], "selection frequency": fr[i],
                             "passes threshold": bool(t is not None and fdp <= 0.5 and fr[i] >= t)})
    s = pd.DataFrame(summary)
    log(s.round(3).to_string(index=False))
    save_table(s, "adv_stability_selection")
    save_table(pd.DataFrame(selected), "adv_stability_top_features")
    return s


if __name__ == "__main__":
    run()
