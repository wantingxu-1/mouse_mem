"""
Read the raw Excel sheet and return one tidy table: one row per mouse.

Output
------
load_raw() -> (df, blocks)
    df      : DataFrame indexed by mouse ID (1-48), original column names,
              plus a 'group' column (ordered categorical).
    blocks  : dict with lists of column names: 'behaviour', 'ihc', 'protein'.
"""
from functools import lru_cache
import pandas as pd
import config


@lru_cache(maxsize=1)
def _read_excel():
    df = pd.read_excel(config.RAW_EXCEL, header=config.HEADER_ROW, nrows=config.N_MICE)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_raw():
    df = _read_excel().copy()
    cols = list(df.columns)
    id_col = cols[1]                       # column B holds the mouse number
    df["mouse"] = df[id_col].astype(int)
    df = df.set_index("mouse")

    blocks = {
        "behaviour": cols[config.FIRST_BEHAVIOUR_COL:config.FIRST_IHC_COL],
        "ihc": cols[config.FIRST_IHC_COL:config.FIRST_PROTEIN_COL],
        "protein": cols[config.FIRST_PROTEIN_COL:],
    }
    missing = set(config.BEHAVIOUR_COLUMNS) - set(blocks["behaviour"])
    if missing:
        raise ValueError(f"Behaviour columns not found in sheet: {missing}")

    df["group"] = pd.Categorical(df.index.map(config.GROUP_OF_MOUSE), categories=config.GROUPS, ordered=True)
    sheet = groups_from_sheet()
    wrong = [m for m in df.index if sheet[m] != str(df.loc[m, "group"])]
    if wrong:
        raise ValueError(f"config.GROUP_OF_MOUSE disagrees with column A of the sheet for mice {wrong}")
    return df, blocks


def groups_from_sheet():
    """Group of each mouse read from column A of the sheet (independent of config).

    Column A holds a group label on the first row of each block (Saline's
    label sits on mouse 2 because mouse 1's cell holds the 'Sample ID' header);
    every following mouse belongs to the last label above it.
    """
    df = _read_excel()
    labels = df.iloc[:, 0].astype("string").str.strip().replace("Sample ID", pd.NA)
    labels = labels.ffill().bfill()
    return pd.Series(labels.to_numpy(), index=df.iloc[:, 1].astype(int).to_numpy())


def ihc_column(marker, region):
    """Return the sheet's column name for a marker/region (case-insensitive region)."""
    _, blocks = load_raw()
    for c in blocks["ihc"]:
        m, r = c.split(" ", 1)
        if m == marker and r.lower() == region.lower():
            return c
    raise KeyError(f"{marker} {region}")
