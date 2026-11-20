"""
ADVANCED 4 - Multi-model weighting / stacking (idea from the seizure-type paper).

Literature
  "Multi-model transfer learning and genotypic analysis for seizure type
  classification" (Yang et al., 2023) combines several models with a
  learned weighting system instead of trusting one model. With small data,
  a weighted ensemble can be more stable than any single model.

Here (mice with all IHC + proteomics; nested leave-one-mouse-out)
  Base models
    1 group means            (what the group label alone predicts)
    2 Ridge on 15 IHC values
    3 Ridge on Glia + NeuN index (2 features, rebuilt from the 15 IHC
      values inside each training fold)
    4 Ridge on 14 protein-set scores (members and z-scaling learnt inside
      each training fold, so the left-out mouse never shapes its features)
  Stacking: inside each outer training fold, inner LOO predictions of the
  base models are combined with non-negative least squares weights
  (summing to 1). The weighted ensemble then predicts the left-out mouse.
  Average weights tell which information source the ensemble relies on.
Output  : results/tables/adv_ensemble_scores.csv, adv_ensemble_weights.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
import config
from features.ihc_indices import FoldIHCIndices
from features.protein_sets import FoldSetScores
from ml.nested_cv import mae, train_mean_predict, tuned
from features.dataset import get_dataset, ihc_columns
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from utils.stats_helpers import loocv_r2

ALPHAS = np.logspace(-2, 4, 7)
OUTCOMES = ["Mem12", "T12_15"]


def base_predict(name, cols, d, y, tr, te, L):
    if name == "group means":
        g = d.group.astype(str).to_numpy()
        return np.array([y[tr][g[tr] == g[i]].mean() for i in te])
    # every fitted step (set scores, indices, scaling) sits inside the Ridge alpha search
    if cols == "SETS":
        X, pre = L.loc[d.index], [FoldSetScores()]
    elif cols == "INDICES":
        X, pre = d[ihc_columns()], [FoldIHCIndices()]
    else:
        X, pre = d[cols], []
    m = tuned(Ridge(), {"alpha": ALPHAS}, *pre, impute=False).fit(X.iloc[tr], y[tr])
    return m.predict(X.iloc[te])


def run():
    header("ADVANCED 4  Multi-model weighting (stacking, nested LOOCV)")
    df = get_dataset()
    L, _, _ = get_proteomics()
    bases = {"group means": None, "Ridge 15 IHC": ihc_columns(), "Ridge Glia+NeuN index": "INDICES",
             "Ridge protein sets": "SETS"}
    score_rows, weight_rows = [], []
    for out in OUTCOMES:
        d = df.loc[L.index, ihc_columns() + ["group", out]].dropna()
        y = d[out].to_numpy(float)
        n = len(d)
        outer = {b: np.zeros(n) for b in bases}
        ens = np.zeros(n)
        W = []
        for tr, te in LeaveOneOut().split(d):
            # inner out-of-fold predictions for weighting
            inner = np.zeros((len(tr), len(bases)))
            for k, (itr, ite) in enumerate(LeaveOneOut().split(tr)):
                for j, (b, cols) in enumerate(bases.items()):
                    inner[ite, j] = base_predict(b, cols, d, y, tr[itr], tr[ite], L)
            w, _ = nnls(inner, y[tr])
            w = w / w.sum() if w.sum() > 0 else np.full(len(bases), 1 / len(bases))
            W.append(w)
            preds = np.array([base_predict(b, cols, d, y, tr, te, L)[0] for b, cols in bases.items()])
            for j, b in enumerate(bases):
                outer[b][te] = preds[j]
            ens[te] = preds @ w
        tm = train_mean_predict(y)
        score_rows.append({"outcome": config.LABELS[out], "n": n, "model": "training mean (baseline)",
                           "LOOCV R2": loocv_r2(y, tm), "MAE": mae(y, tm)})
        for b in bases:
            score_rows.append({"outcome": config.LABELS[out], "n": n, "model": b, "LOOCV R2": loocv_r2(y, outer[b]),
                               "MAE": mae(y, outer[b])})
        score_rows.append({"outcome": config.LABELS[out], "n": n, "model": "weighted ensemble", "LOOCV R2": loocv_r2(y, ens),
                           "MAE": mae(y, ens)})
        weight_rows.append({"outcome": config.LABELS[out], **dict(zip(bases, np.mean(W, axis=0)))})
    s, w = pd.DataFrame(score_rows), pd.DataFrame(weight_rows)
    log(s.round(3).to_string(index=False))
    log("Average ensemble weights:\n" + w.round(2).to_string(index=False))
    save_table(s, "adv_ensemble_scores")
    save_table(w, "adv_ensemble_weights")
    return s, w


if __name__ == "__main__":
    run()
