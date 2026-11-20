"""
Group-pattern indicators used to test "neurons or glia?".

glia_high = 1 for groups whose IBA1/GFAP are raised (PM2.5, PM2.5 EV)
neun_low  = 1 for groups whose NeuN is lowered (all except Saline)

The two indicators differ only in the N EV group (NeuN low, glia normal),
which is what makes the two explanations separable.
"""
import config


def add_pattern_indicators(df):
    out = df.copy()
    out["glia_high"] = out.group.isin(config.GLIA_HIGH_GROUPS).astype(int)
    out["neun_low"] = out.group.isin(config.NEUN_LOW_GROUPS).astype(int)
    return out
