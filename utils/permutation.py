"""
Shared permutation machinery.

Every permutation test in the project is built from these pieces, so the
shuffling scheme is defined once:

  perm_p                    two-sided p = (1 + #|null| >= |obs|) / (1 + N)
  within_group_permutation  index array that shuffles mice only inside their
                            own group (keeps group differences intact)
  conditional_pattern_perm  t of glia_high or neun_low in
                            y ~ glia_high + neun_low (+ covariates), with group
                            labels shuffled only inside the stratum that holds
                            the other pattern fixed
  within_group_corr_perm    group-adjusted rank correlation, behaviour shuffled
                            within group

Random-number use is part of the contract: each helper draws exactly as the
original per-module loops did, so results are reproducible with config.SEED.
"""
import numpy as np
from scipy import stats
import config
from utils.stats_helpers import design, fast_ols, group_centred_ranks


def perm_p(obs, null):
    """Two-sided permutation p-value with the +1 correction."""
    return (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null))


def within_group_permutation(groups, gen, order=None):
    """Return an index array that permutes positions only within each group.

    groups : 1-D array of group labels (one per mouse/row)
    order  : group order in which random draws are made (default: sorted
             unique labels). Changing it changes the random stream, not the
             distribution.
    """
    groups = np.asarray(groups)
    idx = np.arange(len(groups))
    for g in (np.unique(groups) if order is None else order):
        ix = np.where(groups == g)[0]
        idx[ix] = gen.permutation(ix)
    return idx


def conditional_pattern_perm(d, y, term, n_perm, gen, covariates=()):
    """Permutation p for one pattern term, holding the other pattern fixed.

    glia_high: labels shuffled among treated mice (neun_low == 1), so NeuN
               status is unchanged.
    neun_low : labels shuffled among glia-normal mice (glia_high == 0).
    Outcome and covariates stay with their mouse. Statistic: OLS t of term.
    """
    cols = ["glia_high", "neun_low"] + list(covariates)
    k = cols.index(term) + 1
    obs = fast_ols(design(d, cols), d[y])[1][k]
    stratum = (d.neun_low == 1) if term == "glia_high" else (d.glia_high == 0)
    idx = np.where(stratum.to_numpy())[0]
    groups = d.group.astype(str).to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        g = groups.copy()
        g[idx] = gen.permutation(g[idx])
        dp = d.assign(glia_high=np.isin(g, config.GLIA_HIGH_GROUPS).astype(int),
                      neun_low=np.isin(g, config.NEUN_LOW_GROUPS).astype(int))
        null[i] = fast_ols(design(dp, cols), d[y])[1][k]
    return obs, perm_p(obs, null)


def within_group_corr_perm(d, x, y, n_perm, gen):
    """Group-adjusted rank correlation of x and y; y shuffled within group.

    Returns (rho, permutation p, n mice).
    """
    d = d[[x, y, "group"]].dropna()
    c = group_centred_ranks(d, [x, y])
    obs = stats.pearsonr(c[x], c[y])[0]
    yv = d[y].to_numpy().copy()
    groups = d.group.astype(str).to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        dp = d.assign(**{y: yv[within_group_permutation(groups, gen)]})
        cp = group_centred_ranks(dp, [x, y])
        null[i] = stats.pearsonr(cp[x], cp[y])[0]
    return obs, perm_p(obs, null), len(d)
