"""
ML 5 - Do measured stains add information beyond the treatment group?

Question : The pattern contrast shows that memory follows a group pattern.
           Once a mouse's group is known, does its own glia or NeuN value
           improve the prediction of its memory?
Models   (same 21 fully stained mice, same leave-one-mouse-out folds)
  M0  training mean
  M1  group means                         (group only)
  M2  group + NeuN index
  M3  group + Glia index
  M4  group + Glia + NeuN indices
  Indices are rebuilt from the 15 IHC values on the training mice of each
  fold (same definition as FoldIHCIndices); models are OLS with one
  intercept per group.
Scores   : LOOCV R^2, MAE; gain over M1 = mean drop in squared error per
           mouse, with a paired bootstrap 95% CI over mice; permutation p
           for the gain with the stain values shuffled WITHIN group (keeps
           all group differences, breaks only the individual link).
Output   : results/tables/ml_incremental_value.csv
"""
import numpy as np
import pandas as pd
import config
from features.dataset import get_dataset, ihc_columns
from ml.nested_cv import group_mean_predict, mae, train_mean_predict
from utils.io import header, log, save_table
from utils.permutation import within_group_permutation
from utils.stats_helpers import loocv_r2, rng

OUTCOMES = ["Mem12", "T12_15"]
MODELS = {"M2 group + NeuN": [1], "M3 group + Glia": [0], "M4 group + Glia + NeuN": [0, 1]}


def fold_indices(X, cols, tr, te):
    """[Glia, NeuN] for training and test rows; same definition as FoldIHCIndices (numpy, for speed)."""
    mu, sd = X[tr].mean(0), X[tr].std(0, ddof=1)
    out = []
    for rows in (tr, te):
        Z = (X[rows] - mu) / sd
        m = {k: Z[:, [j for j, c in enumerate(cols) if c.split(" ", 1)[0] == k]].mean(1) for k in config.IHC_MARKERS}
        out.append(np.column_stack([(m["GFAP"] + m["IBA1"]) / 2, m["NeuN"]]))
    return out


def loo_group_plus(X, cols, groups, y, which):
    """LOO predictions of y ~ group intercepts + chosen fold-built indices."""
    G = pd.get_dummies(pd.Series(groups)).to_numpy(float)
    pred = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        Itr, Ite = fold_indices(X, cols, tr, [i])
        D = np.column_stack([G[tr], Itr[:, which]])
        b, *_ = np.linalg.lstsq(D, y[tr], rcond=None)
        pred[i] = np.r_[G[i], Ite[0, which]] @ b
    return pred


def run(n_perm=None):
    n_perm = n_perm or config.N_PERM_BLOCK
    header(f"ML 5  Incremental value of measured stains beyond group ({n_perm} within-group permutations)")
    df = get_dataset()
    gen = rng()
    rows = []
    for y in OUTCOMES:
        d = df[ihc_columns() + [y, "group"]].dropna()
        cols = ihc_columns()
        X, g, yv = d[cols].to_numpy(float), d.group.astype(str).to_numpy(), d[y].to_numpy(float)
        p0, p1 = train_mean_predict(yv), group_mean_predict(g, yv)
        e1 = (yv - p1) ** 2
        for name, pred in [("M0 training mean", p0), ("M1 group means", p1)]:
            rows.append({"outcome": config.LABELS[y], "model": name, "n": len(d), "LOOCV R2": loocv_r2(yv, pred),
                         "MAE": mae(yv, pred)})
        for name, which in MODELS.items():
            pred = loo_group_plus(X, cols, g, yv, which)
            gain = e1 - (yv - pred) ** 2
            boot = [gain[gen.integers(0, len(gain), len(gain))].mean() for _ in range(config.N_BOOTSTRAP)]
            null = np.empty(n_perm)
            for k in range(n_perm):
                Xp = X[within_group_permutation(g, gen)]
                null[k] = (e1 - (yv - loo_group_plus(Xp, cols, g, yv, which)) ** 2).mean()
            rows.append({"outcome": config.LABELS[y], "model": name, "n": len(d), "LOOCV R2": loocv_r2(yv, pred),
                         "MAE": mae(yv, pred), "gain vs M1 (mean SE drop)": gain.mean(),
                         "gain 95% low": np.percentile(boot, 2.5), "gain 95% high": np.percentile(boot, 97.5),
                         "within-group perm p (gain)": (1 + np.sum(null >= gain.mean())) / (1 + n_perm)})
    t = pd.DataFrame(rows)
    log(t.round(4).to_string(index=False))
    save_table(t, "ml_incremental_value")
    return t


if __name__ == "__main__":
    run()
