"""
Knowledge-guided protein-set scores (feature construction, no outcome used).

resolve_members(L) -> {set name: [protein columns]} for config.PROTEIN_SETS,
                      keeping proteins detected in >= MIN_MICE_FOR_MARKER_SCORE
                      mice and sets with >= MIN_SET_MEMBERS proteins.
zmatrix(L)         -> each protein z-scored across mice.
set_scores(Z, m)   -> per mouse, mean z of the set's members (equal weights).
FoldSetScores      -> scikit-learn transformer: protein matrix -> set scores, with members and
                      z-scaling learnt in fit() (refitted in every outer AND inner CV fold).
fold_set_scores    -> one-off helper built on FoldSetScores (train ids, test ids).
Used by qc, advanced and ML modules; the analysis of the sets themselves is
advanced/protein_sets.py.
"""
import math
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import config


def resolve_members(L, min_mice=None):
    """Map each set to the protein columns present with >= min_mice values (default MIN_MICE_FOR_MARKER_SCORE)."""
    min_mice = config.MIN_MICE_FOR_MARKER_SCORE if min_mice is None else min_mice
    base = {c.split(";")[0]: c for c in L.columns}
    ok = lambda c: L[c].notna().sum() >= min_mice
    out = {}
    for name, spec in config.PROTEIN_SETS.items():
        if isinstance(spec, str) and spec.startswith("re:"):
            pat = re.compile(spec[3:])
            cols = [c for g, c in base.items() if pat.match(g)]
        else:
            cols = [base[g] for g in dict.fromkeys(spec) if g in base]
        cols = [c for c in cols if ok(c)]
        if len(cols) >= config.MIN_SET_MEMBERS:
            out[name] = cols
    return out


def zmatrix(L):
    return L.apply(lambda s: (s - s.mean()) / s.std())


def set_scores(Z, members):
    return pd.DataFrame({k: Z[v].mean(axis=1) for k, v in members.items()})


def all_set_scores(L):
    """Convenience: set scores for the proteomics matrix L."""
    return set_scores(zmatrix(L), resolve_members(L))


class FoldSetScores(BaseEstimator, TransformerMixin):
    """Protein matrix (DataFrame, proteomics mice only) -> protein-set scores.

    fit() learns set members (detection threshold scaled to the training size,
    same share as MIN_MICE_FOR_MARKER_SCORE of 24 mice) and each protein's
    mean / SD from the training mice only.
    """
    N_REFERENCE = 24

    def fit(self, X, y=None):
        min_mice = math.ceil(config.MIN_MICE_FOR_MARKER_SCORE / self.N_REFERENCE * len(X))
        self.members_ = resolve_members(X, min_mice)
        self.cols_ = sorted({c for v in self.members_.values() for c in v})
        self.mu_, self.sd_ = X[self.cols_].mean(), X[self.cols_].std()
        return self

    def transform(self, X):
        return set_scores((X[self.cols_] - self.mu_) / self.sd_, self.members_)


def fold_set_scores(L, train_ids, test_ids):
    """Set scores for one CV fold (members and z-scaling from the training mice only)."""
    t = FoldSetScores().fit(L.loc[train_ids])
    return t.transform(L.loc[train_ids]), t.transform(L.loc[test_ids])
