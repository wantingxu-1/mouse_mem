"""Derived features follow their documented definitions."""
import numpy as np
import pytest
import config
from features.dataset import get_dataset, ihc_columns
from features.ihc_indices import FoldIHCIndices
from features.proteomics_prep import get_proteomics, technical_factors, TECHNICAL_FACTORS
from features.protein_sets import all_set_scores
from features.tmaze_trials import long_trials
from utils.stats_helpers import zscore


@pytest.fixture(scope="module")
def df():
    return get_dataset()


def test_group_patterns(df):
    for g in config.GROUPS:
        sub = df[df.group == g]
        assert (sub.glia_high == int(g in config.GLIA_HIGH_GROUPS)).all()
        assert (sub.neun_low == int(g in config.NEUN_LOW_GROUPS)).all()
    # the two patterns differ only in N EV
    differs = df.loc[df.glia_high != df.neun_low, "group"].astype(str).unique()
    assert list(differs) == ["N EV"]


def test_memory_composite_is_mean_of_z(df):
    items = config.MEMORY_ITEMS["Mem12"]
    expected = np.nanmean(np.column_stack([zscore(df[c]) for c in items]), axis=1)
    np.testing.assert_allclose(df.Mem12.to_numpy(), expected, atol=1e-10)


def test_glia_index_is_mean_of_glia_markers(df):
    expected = df[["GFAP_idx", "IBA1_idx"]].mean(axis=1)
    np.testing.assert_allclose(df.Glia_idx, expected, atol=1e-12, equal_nan=True)


def test_percent_of_saline_reference(df):
    for m in config.IHC_MARKERS:
        assert df.loc[df.group == "Saline", f"{m}_pct"].mean() == pytest.approx(100, abs=1e-9)


def test_proteomics_prep():
    L, grp, keep = get_proteomics()
    assert len(L) == 24
    assert (grp.astype(str).value_counts() == 6).all()
    assert len(keep) == 3492
    tech = technical_factors(L)
    assert list(tech.columns) == TECHNICAL_FACTORS
    assert tech.notna().all().all()
    S = all_set_scores(L)
    assert S.shape == (24, 14)


def test_tmaze_trials_reproduce_scores(df):
    t = long_trials()
    assert len(t) == 48 * 4 * 3
    per_mouse = t[(t.week == "wk12") & (t.delay == "15s")].groupby("mouse").correct.mean()
    np.testing.assert_allclose(per_mouse.sort_index(), df.T12_15.dropna().sort_index(), atol=1e-9)


def test_fold_ihc_indices_use_training_mice_only(df):
    """Changing a test mouse's stains must not change the fitted scaling; definition matches add_ihc_indices."""
    d = df[ihc_columns()].dropna()
    t = FoldIHCIndices().fit(d)
    full = d.apply(zscore)
    glia = full[[c for c in d.columns if c.split()[0] in ("GFAP", "IBA1")]]
    expected_neun = full[[c for c in d.columns if c.startswith("NeuN")]].mean(axis=1)
    np.testing.assert_allclose(t.transform(d)[:, 1], expected_neun, atol=1e-12)
    assert glia.shape[1] == 10
    tr, te = d.iloc[1:], d.iloc[[0]]
    a = FoldIHCIndices().fit(tr)
    te2 = te * 10
    b = FoldIHCIndices().fit(tr)
    np.testing.assert_array_equal(a.mean_, b.mean_)
    assert not np.allclose(a.transform(te), b.transform(te2))
