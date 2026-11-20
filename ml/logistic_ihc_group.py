"""
ML 3 - Can brain stains tell which group a mouse came from? (Logistic regression)

Question : Do the IHC profiles separate the groups for unseen mice?
           (This answers "do groups differ in the brain", not "what causes
           the behaviour".)
Model    : L2-penalised logistic regression; penalty chosen inside each
           training fold (LogisticRegressionCV, stratified 3-fold) ->
           nested LOOCV. Multinomial for 4 groups; binary tasks below.
Tasks    : 4 groups; Saline vs rest; N EV vs PM2.5 + PM2.5 EV.
Features : all 15 IHC; NeuN only; glia only.
Chance   : balanced accuracy 0.25 (4 groups) / 0.5 (binary); permutation p.
Output   : results/tables/ml_logistic_ihc_group.csv, ml_confusion_4groups.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import config
from features.dataset import get_dataset, ihc_columns
from ml.nested_cv import classification_score
from ml.ridge_ihc_behaviour import feature_sets
from utils.io import header, log, save_table


GRID = {"C": np.logspace(-3, 2, 8)}


def estimator():
    return LogisticRegression(max_iter=5000)


def tasks(d):
    g = d.group.astype(str)
    return {"4 groups": (g, d.index),
            "Saline vs rest": (np.where(g == "Saline", "Saline", "rest"), d.index),
            "N EV vs PM2.5 groups": (g[g != "Saline"].where(g[g != "Saline"] == "N EV", "PM2.5 groups"), d.index[(g != "Saline").to_numpy()])}


def run(n_perm=None):
    n_perm = config.N_ML_LOGISTIC_PERMUTATIONS if n_perm is None else n_perm
    header(f"ML 3  Logistic regression: IHC -> group (nested LOOCV, {n_perm} permutations)")
    d = get_dataset()
    d = d[ihc_columns() + ["group"]].dropna()
    rows = []
    for task, (labels, idx) in tasks(d).items():
        labels = np.asarray(labels)
        for fname, cols in feature_sets().items():
            acc, bacc, p, pred = classification_score(estimator(), d.loc[idx, cols], labels, n_perm, grid=GRID,
                                                       cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
                                                       scoring="accuracy")
            rows.append({"task": task, "features": fname, "n": len(idx), "chance": 1 / len(np.unique(labels)),
                         "accuracy": acc, "balanced accuracy": bacc, "perm p": p})
            if task == "4 groups" and fname == "all 15 IHC":
                cm = pd.DataFrame(confusion_matrix(labels, pred, labels=config.GROUPS),
                                  index=[f"true {g}" for g in config.GROUPS], columns=[f"pred {g}" for g in config.GROUPS])
                log("Confusion matrix (4 groups, all 15 IHC):\n" + cm.to_string())
                save_table(cm.reset_index(), "ml_confusion_4groups")
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "ml_logistic_ihc_group")
    return t


if __name__ == "__main__":
    run()
