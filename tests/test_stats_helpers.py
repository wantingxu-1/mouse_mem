"""Small statistical helpers behave as documented."""
import numpy as np
import pandas as pd
import pytest
from utils.stats_helpers import (bh, group_centred_rank_matrix, group_centred_ranks, holm, loocv_r2,
                                 residualise_columns, residualise_matrix, standardise, zscore)


def test_bh_and_holm_known_values():
    p = [0.01, 0.04, 0.03, np.nan]
    np.testing.assert_allclose(bh(p)[:3], [0.03, 0.04, 0.04])
    np.testing.assert_allclose(holm(p)[:3], [0.03, 0.06, 0.06])
    assert np.isnan(bh(p)[3]) and np.isnan(holm(p)[3])


def test_zscore_and_standardise():
    s = pd.Series([1.0, 2.0, 3.0, np.nan])
    z = zscore(s)
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    M = np.random.default_rng(0).normal(size=(20, 3)) * [1, 5, 10] + 7
    Z = standardise(M)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(0, ddof=1), 1)


def test_group_centred_ranks_have_zero_group_mean():
    d = pd.DataFrame({"x": [5, 1, 3, 9, 7, 2], "group": list("aaabbb")})
    c = group_centred_ranks(d, ["x"])
    np.testing.assert_allclose(c.groupby(d.group).x.mean(), 0, atol=1e-12)
    M = pd.DataFrame({"x": [5.0, 1, 3, 9, 7, 2], "y": [1.0, np.nan, 2, 3, 4, 5]})
    cm = group_centred_rank_matrix(M, d.group)
    np.testing.assert_allclose(cm.x, c.x)


def test_residualise_removes_covariate():
    rng = np.random.default_rng(1)
    z = rng.normal(size=30)
    M = np.column_stack([2 * z + rng.normal(size=30), -z + rng.normal(size=30)])
    R = residualise_matrix(M, z)
    np.testing.assert_allclose(R.T @ (z - z.mean()), 0, atol=1e-9)
    L = pd.DataFrame(M + 5, columns=["a", "b"])
    L.iloc[0, 0] = np.nan
    out = residualise_columns(L, pd.DataFrame({"z": z}))
    assert out.a.mean() == pytest.approx(L.a.mean())
    ok = L.a.notna()
    assert abs(np.corrcoef(out.a[ok], z[ok])[0, 1]) < 1e-9


def test_loocv_r2():
    y = np.array([1.0, 2, 3, 4])
    assert loocv_r2(y, y) == 1
    assert loocv_r2(y, np.full(4, y.mean())) == 0
