"""Regression guard: headline numbers of the reported analysis must not drift."""
import pytest
from core.pattern_contrast import contrast_table
from features.dataset import get_dataset


@pytest.fixture(scope="module")
def table():
    return contrast_table(get_dataset(), ["Mem12", "T12_15"]).set_index("outcome")


def test_glia_pattern_explains_memory(table):
    r = table.loc["Memory composite wk12"]
    assert r["glia R2"] == pytest.approx(0.222, abs=5e-4)
    assert r["glia p"] == pytest.approx(0.0007, abs=5e-4)
    assert r["joint NeuN p"] == pytest.approx(0.914, abs=5e-4)
    assert r["joint glia coef"] == pytest.approx(-0.512, abs=5e-4)


def test_tmaze_15s(table):
    r = table.loc["T-maze wk12 15s"]
    assert r["glia p"] == pytest.approx(0.004, abs=1e-3)
    assert r["joint NeuN p"] == pytest.approx(0.636, abs=5e-4)


def test_ihc_group_means():
    df = get_dataset()
    pct = df.groupby("group", observed=True)[["NeuN_pct", "IBA1_pct", "GFAP_pct"]].mean().round(1)
    assert pct.loc["N EV", "NeuN_pct"] == pytest.approx(79.4, abs=0.05)
    assert pct.loc["PM2.5", "IBA1_pct"] == pytest.approx(171.2, abs=0.1)
    assert pct.loc["N EV", "IBA1_pct"] == pytest.approx(103.1, abs=0.1)
