"""
Behaviour features.

Short names
    Columns C-O are renamed to short names (see config.BEHAVIOUR_COLUMNS).
Memory composite (Mem4, Mem12)
    Mean of the z-scores of T-maze 15 s, T-maze 60 s and NOR for that week,
    using whichever items the mouse has. Averaging reduces noise: each T-maze
    score comes from only 3 trials.
T12_diff (kept only for transparency, NOT recommended)
    T-maze wk12 60 s minus 15 s. A difference of two 3-trial scores is very
    noisy and was constructed post hoc in earlier work.
"""
import config
from utils.stats_helpers import zscore


def add_behaviour_scores(df):
    out = df.rename(columns=config.BEHAVIOUR_COLUMNS)
    for name, items in config.MEMORY_ITEMS.items():
        out[name] = out[items].apply(zscore).mean(axis=1)
    out["T12_diff"] = out["T12_60"] - out["T12_15"]
    return out
