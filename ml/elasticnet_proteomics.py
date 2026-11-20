"""
ML 2 - Can the whole proteome predict an unseen mouse's behaviour? (Elastic Net)

Role     : Negative control / completeness check. With 24 mice and ~3500
           proteins, a model can easily fit noise; nested CV tells whether
           anything generalises.
Model    : Elastic Net (sparse selection + shrinkage). Inside each training
           fold (outer and inner): detection filter (FoldDetectionFilter),
           median imputation, scaling; penalty strength and L1 ratio chosen
           by an inner grid search -> nested LOOCV.
Inputs   : log2, median-normalised proteins (features.proteomics_prep); the
           ">=4/6 detected in every group" filter is refitted per fold, so a
           left-out mouse never decides which proteins are kept.
Baseline : group training means on the same 24 mice.
Note     : Non-zero weights are NOT mechanisms; they are reported only as
           "most often selected" for transparency.
Output   : results/tables/ml_elasticnet_proteomics.csv
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import config
from features.dataset import get_dataset
from features.proteomics_prep import FoldDetectionFilter, get_proteomics
from ml.nested_cv import group_mean_baseline, regression_score, train_mean_predict
from utils.stats_helpers import loocv_r2
from utils.io import header, log, save_table

OUTCOMES = ["Mem12", "T12_15"]


GRID = {"alpha": np.logspace(-2, 1, 8), "l1_ratio": [0.1, 0.5, 0.9]}


def estimator():
    return ElasticNet(max_iter=5000, random_state=config.SEED)


def run(n_perm=0):
    header("ML 2  Elastic Net: proteome -> behaviour (nested LOOCV)")
    L, grp, keep = get_proteomics()
    df = get_dataset().loc[L.index]
    rows = []
    for y in OUTCOMES:
        ok = df[y].notna()
        X, yy, g = L.loc[ok], df.loc[ok, y], grp[ok].astype(str)
        r2, p = regression_score(estimator(), X, yy, FoldDetectionFilter(grp), n_perm=n_perm, grid=GRID)
        rows.append({"outcome": config.LABELS[y], "n mice": int(ok.sum()), "n proteins (all-mice filter)": len(keep),
                     "Elastic Net LOOCV R2": r2, "perm p": p, "training-mean LOOCV R2": loocv_r2(yy, train_mean_predict(yy)),
                     "group-means LOOCV R2": group_mean_baseline(g, yy)})
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "ml_elasticnet_proteomics")
    return t


if __name__ == "__main__":
    run()
