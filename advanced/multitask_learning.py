"""
ADVANCED 3 - Multi-task learning (idea from MetaGeno).

Literature
  MetaGeno predicts ischaemic stroke together with its risk factors
  (AF, hypertension, cholesterol, diabetes, CAD) in one shared model, so
  related tasks share information. With few samples, sharing strength
  across related outcomes can stabilise the weights.

Here (nested leave-one-mouse-out; alpha / l1_ratio search refits feature
      construction and X / y scaling in every inner training fold)
  A. Behaviour tasks from IHC (mice with all 15 IHC values):
     tasks = T-maze 15 s / 60 s at week 4 and week 12 (no missing values).
     Single-task Elastic Net per task  vs  Multi-task Elastic Net (shared
     sparsity pattern across tasks)  vs  group-mean baseline.
  B. MetaGeno-style auxiliary tasks (mice with proteomics + all IHC):
     main task Mem12 from the 14 protein-set scores; auxiliary tasks =
     Glia index and NeuN index. Does adding the auxiliary IHC tasks help
     the proteome predict memory? Set scores are rebuilt inside each
     training fold (fold_set_scores).
Score   : prediction R^2 on left-out mice, per task.
Output  : results/tables/adv_multitask.csv
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config
from features.protein_sets import FoldSetScores
from features.dataset import get_dataset, ihc_columns
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from ml.nested_cv import train_mean_predict
from utils.stats_helpers import loocv_r2

L1 = [0.1, 0.5, 0.9]
ALPHAS = np.logspace(-3, 1, 9)


def inner_cv():
    return KFold(5, shuffle=True, random_state=config.SEED)


def search(estimator, pre):
    """Alpha / l1_ratio search; feature steps, X scaling and y scaling are refitted in every inner fold."""
    pipe = make_pipeline(*pre, StandardScaler(), estimator)
    name = pipe.steps[-1][0]
    model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
    grid = {f"regressor__{name}__alpha": ALPHAS, f"regressor__{name}__l1_ratio": L1}
    return GridSearchCV(model, grid, cv=inner_cv(), scoring="neg_mean_squared_error")


def nested_multitask(X, Y, groups, pre=(), which=("single", "multi")):
    """LOO predictions (n x k) for single-task / multi-task Elastic Net and group means.

    X: array or DataFrame; pre: feature steps fitted inside every fold
    (e.g. FoldSetScores on the protein matrix).
    """
    n, k = Y.shape
    pred = {"group means": np.zeros((n, k))}
    pred.update({m: np.zeros((n, k)) for m, w in [("single-task Elastic Net", "single"), ("multi-task Elastic Net", "multi")]
                 if w in which})
    rows = (lambda idx: X.iloc[idx]) if hasattr(X, "iloc") else (lambda idx: X[idx])
    for tr, te in LeaveOneOut().split(Y):
        Xtr, Xte = rows(tr), rows(te)
        if "multi" in which:
            mt = search(MultiTaskElasticNet(max_iter=100000, tol=1e-4), pre).fit(Xtr, Y[tr])
            pred["multi-task Elastic Net"][te] = mt.predict(Xte)
        if "single" in which:
            pred["single-task Elastic Net"][te] = np.column_stack(
                [search(ElasticNet(max_iter=100000, tol=1e-4), pre).fit(Xtr, Y[tr][:, j]).predict(Xte) for j in range(k)])
        same = groups[tr] == groups[te][0]
        pred["group means"][te] = Y[tr][same].mean(0)
    return pred


def score(pred, Y, task_names, setting, n):
    pred = {"training mean (baseline)": np.column_stack([train_mean_predict(Y[:, j]) for j in range(Y.shape[1])]), **pred}
    rows = []
    for m, P in pred.items():
        for j, t in enumerate(task_names):
            rows.append({"setting": setting, "model": m, "task": t, "n": n, "LOOCV R2": loocv_r2(Y[:, j], P[:, j])})
    return rows


def run():
    header("ADVANCED 3  Multi-task learning (nested LOOCV)")
    df = get_dataset()
    rows = []

    # A: behaviour tasks from IHC
    tasks = ["T4_15", "T4_60", "T12_15", "T12_60"]
    d = df[ihc_columns() + tasks + ["group"]].dropna()
    pred = nested_multitask(d[ihc_columns()].to_numpy(), d[tasks].to_numpy(float), d.group.astype(str).to_numpy())
    rows += score(pred, d[tasks].to_numpy(float), [config.LABELS[t] for t in tasks], "A: IHC -> 4 T-maze tasks", len(d))

    # B: protein sets -> memory, with / without auxiliary IHC tasks
    L, grp, _ = get_proteomics()
    main, aux = ["Mem12"], ["Glia_idx", "NeuN_idx"]
    d = df.loc[L.index, main + aux + ["group"]].dropna()
    g = d.group.astype(str).to_numpy()
    X, pre = L.loc[d.index], (FoldSetScores(),)
    p_multi = nested_multitask(X, d[main + aux].to_numpy(float), g, pre, which=("multi",))
    p_single = nested_multitask(X, d[main].to_numpy(float), g, pre, which=("single",))
    y = d[main].to_numpy(float)
    for label, P in [("protein sets, Mem12 only", p_single["single-task Elastic Net"][:, [0]]),
                     ("protein sets, Mem12 + Glia/NeuN auxiliary tasks", p_multi["multi-task Elastic Net"][:, [0]]),
                     ("group means", p_multi["group means"][:, [0]]),
                     ("training mean (baseline)", train_mean_predict(y[:, 0])[:, None])]:
        rows.append({"setting": "B: proteome -> memory, auxiliary IHC tasks", "model": label, "task": config.LABELS["Mem12"],
                     "n": len(d), "LOOCV R2": loocv_r2(y[:, 0], P[:, 0])})
    t = pd.DataFrame(rows)
    summary = t.groupby(["setting", "model"], sort=False)["LOOCV R2"].mean().rename("mean R2 over tasks").reset_index()
    log(t.round(3).to_string(index=False))
    log(summary.round(3).to_string(index=False))
    save_table(t, "adv_multitask")
    return t


if __name__ == "__main__":
    run()
