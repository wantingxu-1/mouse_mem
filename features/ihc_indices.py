"""
IHC summary features.

Index per marker
    Each of the 5 regions is z-scored across all stained mice, then the 5
    z-scores are averaged. A mouse gets an index only if all 5 regions exist.
    0 = average mouse, +1 = one SD above average.
Glia index
    Mean of GFAP index and IBA1 index (uses whichever is available).
% of Saline
    Each region divided by the Saline group mean, averaged over regions.
    Used for plots and plain-language tables.
FoldIHCIndices
    The same Glia / NeuN indices for prediction: region means and SDs are
    learnt from the training mice only (a scikit-learn transformer, so it
    is refitted inside every cross-validation fold).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config
from utils.stats_helpers import zscore


def marker_columns(df_cols, marker):
    return [c for c in df_cols if c.split(" ", 1)[0] == marker]


def add_ihc_indices(df, ihc_cols):
    out = df.copy()
    for m in config.IHC_MARKERS:
        cols = marker_columns(ihc_cols, m)
        out[f"{m}_idx"] = out[cols].apply(zscore).mean(axis=1, skipna=False)
    out["Glia_idx"] = out[["GFAP_idx", "IBA1_idx"]].mean(axis=1)
    return out


def percent_of_reference(df, ihc_cols, marker):
    """Per mouse: mean over 5 regions of (value / reference-group mean * 100)."""
    cols = marker_columns(ihc_cols, marker)
    ref = df.loc[df.group == config.REFERENCE_GROUP, cols].mean()
    return (df[cols] / ref * 100).mean(axis=1, skipna=False)


class FoldIHCIndices(BaseEstimator, TransformerMixin):
    """15 IHC columns (DataFrame) -> [Glia_idx, NeuN_idx], z-scaling learnt in fit().

    Same definition as add_ihc_indices: each region z-scored, averaged per
    marker (all 5 regions needed), Glia = mean of the GFAP and IBA1 indices.
    """

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.mean_ = X.mean().to_numpy(float)
        self.sd_ = X.std().to_numpy(float)
        return self

    def transform(self, X):
        Z = (X[self.columns_].to_numpy(float) - self.mean_) / self.sd_
        idx = {}
        for m in config.IHC_MARKERS:
            j = [i for i, c in enumerate(self.columns_) if c.split(" ", 1)[0] == m]
            idx[m] = Z[:, j].mean(axis=1)
        glia = np.nanmean(np.column_stack([idx["GFAP"], idx["IBA1"]]), axis=1)
        return np.column_stack([glia, idx["NeuN"]])

    def get_feature_names_out(self, input_features=None):
        return np.array(["Glia_idx", "NeuN_idx"])
