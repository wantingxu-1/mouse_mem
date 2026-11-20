"""
T-maze scores expanded to single trials.

Each T-maze score is k correct out of 3 trials (0, 1/3, 2/3, 1).
long_trials() returns one row per trial: mouse, group, pattern indicators,
delay (15s/60s), week (wk4/wk12), correct (0/1), and 0/1 dummies d60, w12.
48 mice x 4 conditions x 3 trials = 576 rows.
"""
import pandas as pd
import config
from features.dataset import get_dataset

CONDITIONS = {("15s", "wk4"): "T4_15", ("60s", "wk4"): "T4_60", ("15s", "wk12"): "T12_15", ("60s", "wk12"): "T12_60"}
N_TRIALS = 3


def long_trials():
    """One row per trial: mouse, group, pattern indicators, delay, week, correct (0/1)."""
    df = get_dataset()
    rows = []
    for (delay, week), col in CONDITIONS.items():
        for m, r in df.iterrows():
            if pd.isna(r[col]):
                continue
            k = int(round(r[col] * N_TRIALS))
            for t in range(N_TRIALS):
                rows.append({"mouse": m, "group": str(r.group), "glia_high": r.glia_high, "neun_low": r.neun_low,
                             "delay": delay, "week": week, "correct": int(t < k)})
    d = pd.DataFrame(rows)
    d["group"] = pd.Categorical(d.group, config.GROUPS)
    d["d60"] = (d.delay == "60s").astype(int)
    d["w12"] = (d.week == "wk12").astype(int)
    return d
