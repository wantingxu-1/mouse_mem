"""
Proteomics preprocessing.

Steps
  1. Keep mice with proteomics (>100 detected proteins; 6 per group).
  2. log2 transform the raw intensities (blank = not detected = NaN, never 0).
  3. Median-normalise: subtract each mouse's median log2 intensity, so mice
     measured at different overall depth/loading become comparable.
  4. Filter: keep proteins detected in >=4/6 mice in every group.
Missing values are NOT imputed here; ML pipelines impute inside training folds.
"""
import math
from functools import lru_cache
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config
from data.load_data import load_raw


@lru_cache(maxsize=1)
def get_proteomics():
    """Return (log2 normalised matrix [mice x proteins], group Series, filtered protein list)."""
    raw, blocks = load_raw()
    prot = raw[blocks["protein"]].apply(lambda s: s.astype(float))
    has = prot.notna().sum(axis=1) > config.MIN_PROTEINS_PER_MOUSE
    P = prot.loc[has]
    L = np.log2(P)
    L = L.sub(L.median(axis=1), axis=0)
    grp = raw.loc[L.index, "group"]
    counts = L.notna().groupby(grp, observed=True).sum()
    keep = counts.columns[(counts >= config.MIN_DETECTED_PER_GROUP).all()].tolist()
    return L, grp, keep


def detected_per_mouse():
    raw, blocks = load_raw()
    n = raw[blocks["protein"]].notna().sum(axis=1)
    return n[n > config.MIN_PROTEINS_PER_MOUSE]


def protein_column(L, gene):
    """Column of L whose first gene name (before ';') is `gene`, or None."""
    base = {c.split(";")[0]: c for c in L.columns}
    return base.get(gene)


HAEMOGLOBIN = ["Hba", "Hbb-bs", "Hbb-b1", "Hbb-b2"]
TECHNICAL_FACTORS = ["depth", "blood (Hb log2)", "Omp detected", "Ddx3y detected"]


def technical_factors(L):
    """Per-mouse technical covariates of the proteomics (see qc/proteomics_integrity.py).

    depth           number of detected proteins
    blood (Hb log2) mean log2 haemoglobin (perfusion / blood content)
    Omp detected    olfactory-bulb marker present (dissection)
    Ddx3y detected  Y-linked protein present (possible sex marker)
    """
    import pandas as pd
    hb = [protein_column(L, x) for x in HAEMOGLOBIN if protein_column(L, x)]
    return pd.DataFrame({"depth": detected_per_mouse().loc[L.index],
                         "blood (Hb log2)": L[hb].mean(axis=1),
                         "Omp detected": L[protein_column(L, "Omp")].notna().astype(int),
                         "Ddx3y detected": L[protein_column(L, "Ddx3y")].notna().astype(int)})


class FoldDetectionFilter(BaseEstimator, TransformerMixin):
    """Step 4 (detection filter) refitted on the training mice of each fold.

    Keeps proteins detected in >= MIN_DETECTED_PER_GROUP/6 of the training
    mice of every group (threshold scaled to the group's training size, e.g.
    4 of 5 when one mouse is left out). Group labels are looked up by mouse
    ID, so X must be a DataFrame indexed by mouse. Used by ML pipelines so a
    left-out mouse never influences which proteins are kept.
    """
    def __init__(self, groups=None):
        self.groups = groups

    def fit(self, X, y=None):
        g = self.groups.loc[X.index]
        det = X.notna().groupby(g.to_numpy(), observed=True)
        need = det.size().map(lambda n: math.ceil(config.MIN_DETECTED_PER_GROUP / 6 * n))
        counts = det.sum()
        self.columns_ = counts.columns[(counts.ge(need, axis=0)).all()].tolist()
        return self

    def transform(self, X):
        return X[self.columns_]
