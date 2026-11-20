"""
Shared machine-learning machinery: nested leave-one-mouse-out CV.

Why nested?
  Outer loop: leave one mouse out, train on the rest, predict the left-out
  mouse. Inner loop: hyper-parameters are chosen by a search whose every
  inner training fold refits the WHOLE pipeline (feature construction,
  imputation, scaling, model). So neither the outer test mouse nor an
  inner validation mouse shapes any fitted step it is scored on.

Scores
  Regression     : prediction R^2 on left-out mice and MAE. R^2 = 0 is the
                   test-set mean; the honest floor is the training-mean
                   baseline (train_mean_predict), whose R^2 is 1-(n/(n-1))^2.
  Classification : accuracy and balanced accuracy on left-out mice.
Permutation p (optional): repeat the whole nested procedure with the outcome
  shuffled; p = how often a shuffled outcome is predicted as well.
"""
import numpy as np
from joblib import Parallel, delayed
import config
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils.stats_helpers import loocv_r2, rng


def make_model(estimator, *pre):
    """[pre steps ->] median imputation -> standardisation -> estimator, all fitted per fold."""
    return make_pipeline(*pre, SimpleImputer(strategy="median"), StandardScaler(), estimator)


def inner_cv():
    return KFold(3, shuffle=True, random_state=config.SEED)


def tuned(estimator, grid, *pre, cv=None, scoring="neg_mean_squared_error", impute=True):
    """Grid search over `grid` (estimator parameters) with the whole pipeline refitted per inner fold."""
    pipe = make_model(estimator, *pre) if impute else make_pipeline(*pre, StandardScaler(), estimator)
    name = pipe.steps[-1][0]
    return GridSearchCV(pipe, {f"{name}__{k}": v for k, v in grid.items()}, cv=cv or inner_cv(), scoring=scoring)


def build(estimator, *pre, grid=None, **kw):
    """Untuned pipeline, or a tuned one when a grid is given."""
    return tuned(estimator, grid, *pre, **kw) if grid else make_model(estimator, *pre)


def nested_predict(estimator, X, y, *pre, grid=None, **kw):
    return cross_val_predict(build(estimator, *pre, grid=grid, **kw), X, y, cv=LeaveOneOut())


def parallel_map(fn, items):
    """fn over items on all cores (used for permutation nulls; each item runs serially inside)."""
    return Parallel(n_jobs=-1)(delayed(fn)(it) for it in items)


def mae(y, pred):
    """Mean absolute prediction error, in the outcome's own units."""
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def train_mean_predict(y):
    """LOO baseline: each left-out mouse is predicted by the mean of the other mice.

    Its R^2 is exactly 1 - (n / (n - 1))^2 (e.g. -0.1025 for n = 21), so a
    model with that R^2 has shrunk to a constant prediction.
    """
    y = np.asarray(y, float)
    return (y.sum() - y) / (len(y) - 1)


def group_mean_predict(groups, y):
    """LOO baseline: each left-out mouse is predicted by its group's training mean."""
    groups = np.asarray(groups)
    y = np.asarray(y, float)
    return np.array([y[(groups == groups[i]) & (np.arange(len(y)) != i)].mean() for i in range(len(y))])


def regression_score(estimator, X, y, *pre, n_perm=0, grid=None):
    y = np.asarray(y, float)
    r2 = loocv_r2(y, nested_predict(estimator, X, y, *pre, grid=grid))
    p = np.nan
    if n_perm:
        gen = rng()
        perms = [gen.permutation(y) for _ in range(n_perm)]
        null = parallel_map(lambda yp: loocv_r2(yp, nested_predict(estimator, X, yp, *pre, grid=grid)), perms)
        p = (1 + np.sum(np.array(null) >= r2)) / (1 + n_perm)
    return r2, p


def classification_score(estimator, X, y, n_perm=0, grid=None, **kw):
    y = np.asarray(y)
    pred = nested_predict(estimator, X, y, grid=grid, **kw)
    acc, bacc = accuracy_score(y, pred), balanced_accuracy_score(y, pred)
    p = np.nan
    if n_perm:
        gen = rng()
        perms = [gen.permutation(y) for _ in range(n_perm)]
        null = parallel_map(lambda yp: balanced_accuracy_score(yp, nested_predict(estimator, X, yp, grid=grid, **kw)), perms)
        p = (1 + np.sum(np.array(null) >= bacc)) / (1 + n_perm)
    return acc, bacc, p, pred


def group_mean_baseline(groups, y):
    """LOOCV R^2 when each left-out mouse is predicted by its group's training mean."""
    return loocv_r2(y, group_mean_predict(groups, y))


def baseline_rows(groups, y, **labels):
    """Two table rows (training mean, group means) with LOOCV R^2 and MAE, for the same mice."""
    out = []
    for name, pred in [("training mean (baseline)", train_mean_predict(y)),
                       ("group means (baseline)", group_mean_predict(groups, y))]:
        out.append({**labels, "model": name, "n": len(y), "LOOCV R2": loocv_r2(y, pred), "MAE": mae(y, pred)})
    return out
