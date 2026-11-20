"""Permutation helpers: correct shuffling scheme and the same random stream as the original loops."""
import numpy as np
from utils.permutation import conditional_pattern_perm, perm_p, within_group_permutation
from utils.stats_helpers import design, fast_ols
from features.dataset import get_dataset
import config


def test_within_group_permutation_stays_inside_groups():
    groups = np.array(list("aabbbcc"))
    idx = within_group_permutation(groups, np.random.default_rng(3))
    assert sorted(idx) == list(range(len(groups)))
    assert (groups[idx] == groups).all()


def test_within_group_permutation_matches_value_shuffle_loop():
    """Index-based helper == the per-module value loops it replaced (same seed, same result)."""
    groups = np.array(list("babacabc"))
    values = np.arange(8) * 1.5
    old = values.copy()
    g1 = np.random.default_rng(42)
    for k in np.unique(groups):
        ix = np.where(groups == k)[0]
        old[ix] = g1.permutation(old[ix])
    new = values[within_group_permutation(groups, np.random.default_rng(42))]
    np.testing.assert_array_equal(old, new)


def test_perm_p_bounds():
    assert perm_p(0.0, np.array([1.0, -1.0])) == 1.0
    assert perm_p(5.0, np.zeros(99)) == 0.01


def test_conditional_perm_with_covariate_matches_original_glia_perm():
    """conditional_pattern_perm(..., covariates) reproduces the former activity_confound.glia_perm."""
    d = get_dataset()[["Mem12", "glia_high", "neun_low", "group", "EPMentries12"]].dropna()
    covs = ["EPMentries12"]
    cols = ["glia_high", "neun_low"] + covs
    obs = fast_ols(design(d, cols), d.Mem12)[1][1]
    idx = np.where((d.neun_low == 1).to_numpy())[0]
    groups = d.group.astype(str).to_numpy()
    gen = np.random.default_rng(7)
    null = np.empty(50)
    for i in range(50):
        g = groups.copy()
        g[idx] = gen.permutation(g[idx])
        dp = d.assign(glia_high=np.isin(g, config.GLIA_HIGH_GROUPS).astype(int))
        null[i] = fast_ols(design(dp, cols), d.Mem12)[1][1]
    old = perm_p(obs, null)
    t, new = conditional_pattern_perm(d, "Mem12", "glia_high", 50, np.random.default_rng(7), covs)
    assert t == obs and new == old
