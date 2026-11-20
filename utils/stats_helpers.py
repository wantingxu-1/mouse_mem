"""
Small statistical helpers shared by several modules.

Everything here is a thin, documented wrapper so that each analysis module
reads like a description of the method rather than of the plumbing.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


def zscore(s):
    """z-score a Series across all mice that have a value (sample SD)."""
    return (s - s.mean()) / s.std()


def bh(pvalues):
    """Benjamini-Hochberg q-values (NaNs are kept as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def holm(pvalues):
    """Holm-adjusted p-values for a small, pre-specified family."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def group_centred_ranks(df, cols, group_col="group"):
    """Rank each column across mice, then subtract the group mean rank.

    The result describes how high a mouse is *relative to its own group*.
    """
    r = df[cols].rank()
    return r - r.groupby(df[group_col], observed=True).transform("mean")


def spearman(x, y):
    """Spearman rho, p and n on the mice that have both values."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 4:
        return np.nan, np.nan, len(d)
    rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return rho, p, len(d)


def within_group_spearman(df, x, y, group_col="group"):
    """Group-adjusted rank correlation.

    Steps: keep mice with both values -> rank x and y -> subtract each group's
    mean rank -> Pearson correlation of the centred ranks.
    """
    d = df[[x, y, group_col]].dropna()
    if len(d) < 4:
        return np.nan, np.nan, len(d)
    c = group_centred_ranks(d, [x, y], group_col)
    r, p = stats.pearsonr(c[x], c[y])
    return r, p, len(d)


def ols(formula, data):
    """Ordinary least squares via statsmodels formula interface."""
    return smf.ols(formula, data).fit()


def kruskal_by_group(df, col, group_col="group"):
    groups = [g[col].dropna() for _, g in df.groupby(group_col, observed=True)]
    groups = [g for g in groups if len(g) > 0]
    return stats.kruskal(*groups).pvalue


def mean_sem_table(df, cols, group_col="group"):
    g = df.groupby(group_col, observed=True)[cols]
    return g.mean().T, g.sem().T, g.count().T


def loocv_r2(y, pred):
    """Prediction R^2 for left-out mice (0 = as good as predicting the mean)."""
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def rng(seed=None):
    import config
    return np.random.default_rng(config.SEED if seed is None else seed)


def fast_ols(X, y):
    """Least squares with intercept already in X. Returns (coef, t, r2).

    Used inside permutation / bootstrap loops where statsmodels is too slow.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    t = beta / np.sqrt(np.diag(cov))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, t, r2


def design(df, cols):
    """Design matrix [1, cols...] as a numpy array."""
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])


def standardise(M):
    """Column-wise z-score of a 2-D array (sample SD)."""
    M = np.asarray(M, float)
    return (M - M.mean(0)) / M.std(0, ddof=1)


def residualise_columns(L, covars):
    """Remove covariates from each column by least squares, keeping the column mean.

    NaN-aware: each column uses only rows with a value. Columns with too few
    values (<= n covariates + 4) are returned unchanged.
    """
    out = L.copy()
    C = np.column_stack([np.ones(len(covars)), covars.to_numpy(float)])
    for c in L.columns:
        y = L[c].to_numpy()
        ok = ~np.isnan(y)
        if ok.sum() > C.shape[1] + 3:
            b, *_ = np.linalg.lstsq(C[ok], y[ok], rcond=None)
            out.loc[ok, c] = y[ok] - C[ok] @ b + y[ok].mean()
    return out


def residualise_matrix(M, z):
    """Residuals of every column of a complete 2-D array on covariate(s) z (with intercept)."""
    Z = np.column_stack([np.ones(len(z)), z])
    b, *_ = np.linalg.lstsq(Z, M, rcond=None)
    return M - Z @ b


def group_centred_rank_matrix(M, groups):
    """Rank every column of a DataFrame (NaN kept), subtract the group mean rank."""
    R = M.rank()
    return R - R.groupby(groups).transform("mean")
