"""
Build the analysis-ready table used by every module.

get_dataset() returns one row per mouse with:
  group, short-name behaviour columns, memory composites,
  the 15 IHC columns (original names), NeuN/GFAP/IBA1/Glia indices,
  % of Saline per marker, glia_high/neun_low indicators, has_proteomics.
Proteomics stays separate (features.proteomics_prep) because it is wide.
"""
from functools import lru_cache
import config
from data.load_data import load_raw
from features.behaviour_scores import add_behaviour_scores
from features.ihc_indices import add_ihc_indices, percent_of_reference
from features.group_patterns import add_pattern_indicators


@lru_cache(maxsize=1)
def _build():
    raw, blocks = load_raw()
    df = raw[["group"] + blocks["behaviour"] + blocks["ihc"]].copy()
    df = add_behaviour_scores(df)
    df = add_ihc_indices(df, blocks["ihc"])
    for m in config.IHC_MARKERS:
        df[f"{m}_pct"] = percent_of_reference(df, blocks["ihc"], m)
    df = add_pattern_indicators(df)
    df["has_proteomics"] = raw[blocks["protein"]].notna().sum(axis=1) > config.MIN_PROTEINS_PER_MOUSE
    return df


def get_dataset():
    return _build().copy()


def ihc_columns():
    return load_raw()[1]["ihc"]
