"""
ML 4 - Model zoo with honest error bars: linear, Bayesian, kernel, tree models.

Literature
  Vabalas et al. 2019 (PLoS One): with small n, only nested CV (all tuning
  and feature handling inside the training fold) gives unbiased estimates.
  Varoquaux 2018 (NeuroImage, "Cross-validation failure"): with small n the
  error bars of CV scores are huge, and fold-to-fold SE underestimates them;
  report CIs and compare against a baseline.
Models (all nested leave-one-mouse-out; inner tuning on training mice only)
  Ridge (reference), PLS regression (1-3 components), ARD (sparse Bayesian
  linear, "automatic relevance determination"), Gaussian process (RBF +
  noise kernel), linear SVR, k-nearest neighbours, random forest,
  gradient boosting (shallow trees).
Inputs   : 15 IHC values, or the 2 indices (Glia, NeuN) rebuilt from the
           15 values inside every training fold (FoldIHCIndices).
Scores   : LOOCV R^2 and MAE; 95% CI of R^2 by bootstrapping mice over the
           out-of-fold predictions (does not refit models, so it shows
           sample-composition uncertainty only); permutation p (outcome
           shuffled, whole procedure repeated); training-mean and
           group-mean baselines on the same mice.
Output   : results/tables/ml_model_zoo.csv
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ARDRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import config
from features.dataset import get_dataset, ihc_columns
from features.ihc_indices import FoldIHCIndices
from ml.nested_cv import baseline_rows, mae, nested_predict, parallel_map
from utils.io import header, log, save_table
from utils.stats_helpers import loocv_r2, rng

OUTCOMES = ["Mem12", "T12_15"]


class PLS1(PLSRegression):
    """PLSRegression returning a 1-D prediction (so it behaves like other regressors)."""
    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def models(n_features):
    """name -> (estimator, grid or None). Grids are searched with the whole pipeline refitted per inner fold."""
    return {
        "Ridge": (Ridge(), {"alpha": np.logspace(-2, 4, 7)}),
        "PLS": (PLS1(scale=False), {"n_components": list(range(1, min(3, n_features) + 1))}),
        "ARD (Bayesian sparse)": (ARDRegression(), None),
        "Gaussian process": (GaussianProcessRegressor(ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(), normalize_y=True,
                                                      random_state=config.SEED), None),
        "Linear SVR": (SVR(kernel="linear"), {"C": [0.01, 0.1, 1], "epsilon": [0.1, 0.3]}),
        "kNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
        "Random forest": (RandomForestRegressor(n_estimators=150, max_depth=3, min_samples_leaf=3, random_state=config.SEED), None),
        "Gradient boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                       random_state=config.SEED), None),
    }


def boot_ci(y, pred, gen, n=config.N_BOOTSTRAP):
    idx = [gen.integers(0, len(y), len(y)) for _ in range(n)]
    vals = [loocv_r2(y[i], pred[i]) for i in idx if np.var(y[i]) > 0]
    return np.percentile(vals, [2.5, 97.5])


def run(n_perm=None):
    n_perm = config.N_ML_ZOO_PERMUTATIONS if n_perm is None else n_perm
    header(f"ML 4  Model zoo, nested LOOCV with bootstrap CI and {n_perm} permutations")
    df = get_dataset()
    gen = rng()
    rows = []
    for y in OUTCOMES:
        d = df[ihc_columns() + [y, "group"]].dropna()
        yv = d[y].to_numpy(float)
        rows += baseline_rows(d.group.astype(str), yv, outcome=config.LABELS[y], features="-")
        X = d[ihc_columns()]
        for fname, pre, k in [("15 IHC", (), 15), ("Glia + NeuN index", (FoldIHCIndices(),), 2)]:
            for mname, (est, grid) in models(k).items():
                pred = nested_predict(est, X, yv, *pre, grid=grid)
                r2 = loocv_r2(yv, pred)
                lo, hi = boot_ci(yv, pred, gen)
                perms = [gen.permutation(yv) for _ in range(n_perm)]
                null = parallel_map(lambda yp: loocv_r2(yp, nested_predict(est, X, yp, *pre, grid=grid)), perms)
                rows.append({"outcome": config.LABELS[y], "features": fname, "model": mname, "n": len(d), "LOOCV R2": r2,
                             "MAE": mae(yv, pred), "R2 95% low": lo, "R2 95% high": hi,
                             "perm p": (1 + np.sum(np.array(null) >= r2)) / (1 + n_perm)})
                log(f"  {y:7s} {fname:18s} {mname:22s} R2={r2:6.3f} [{lo:6.2f},{hi:5.2f}]")
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "ml_model_zoo")
    return t


if __name__ == "__main__":
    run()
