"""
ML 1 - Can brain stains predict an unseen mouse's behaviour? (Ridge)

Question : Trained on the other mice, how well do IHC values predict a
           left-out mouse's memory? Do glia stains predict better than
           NeuN stains?
Model    : Ridge regression (shrinks all weights; suits correlated regions).
           Penalty chosen inside each training fold (RidgeCV, efficient
           leave-one-out on the training mice) -> nested LOOCV.
Inputs   : all 15 IHC values / NeuN only (5) / glia only (GFAP+IBA1, 10).
           Same mice for every feature set (mice with all 15 values).
Baseline : predict each mouse by its group's training mean.
Output   : results/tables/ml_ridge_ihc_behaviour.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import config
from features.dataset import get_dataset, ihc_columns
from ml.nested_cv import group_mean_baseline, regression_score, train_mean_predict
from utils.stats_helpers import loocv_r2
from utils.io import header, log, save_table

OUTCOMES = ["Mem12", "T12_15", "NOR12", "Mem4"]
ALPHAS = np.logspace(-2, 4, 7)


def feature_sets():
    cols = ihc_columns()
    return {"all 15 IHC": cols,
            "NeuN only (5)": [c for c in cols if c.startswith("NeuN")],
            "glia only (10)": [c for c in cols if c.startswith(("GFAP", "IBA1"))]}


def run(n_perm=None):
    n_perm = config.N_ML_PERMUTATIONS if n_perm is None else n_perm
    header(f"ML 1  Ridge: IHC -> behaviour (nested LOOCV, {n_perm} permutations)")
    df = get_dataset()
    rows = []
    for y in OUTCOMES:
        d = df[ihc_columns() + [y, "group"]].dropna()
        rows.append({"outcome": config.LABELS[y], "features": "training mean (baseline)", "n": len(d),
                     "LOOCV R2": loocv_r2(d[y], train_mean_predict(d[y])), "perm p": np.nan})
        rows.append({"outcome": config.LABELS[y], "features": "group means (baseline)", "n": len(d),
                     "LOOCV R2": group_mean_baseline(d.group.astype(str), d[y]), "perm p": np.nan})
        for name, cols in feature_sets().items():
            r2, p = regression_score(Ridge(), d[cols], d[y], n_perm=n_perm, grid={"alpha": ALPHAS})
            rows.append({"outcome": config.LABELS[y], "features": name, "n": len(d), "LOOCV R2": r2, "perm p": p})
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "ml_ridge_ihc_behaviour")
    return t


if __name__ == "__main__":
    run()
