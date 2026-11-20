"""
QC 1 - Data quality checks (run first).

Question: which mice have which measurements, and are there obvious
recording problems?

Checks
  1. Coverage: which mouse IDs have each behaviour, each IHC marker, proteomics.
  2. EPM zone times should add up to ~600 s (10 min test).
  3. T-maze takes only 0, 1/3, 2/3, 1 -> 3 trials per condition (noisy).
  4. Number of detected proteins per mouse (proteomics depth).
"""
import pandas as pd
import config
from data.load_data import load_raw
from features.dataset import get_dataset
from utils.io import header, log, save_table


def run():
    header("QC: coverage and recording checks")
    raw, blocks = load_raw()
    df = get_dataset()

    # 1 coverage
    rows = []
    for col in list(config.BEHAVIOUR_COLUMNS.values()) + [f"{m}_idx" for m in config.IHC_MARKERS] + ["has_proteomics"]:
        s = df[col] if col != "has_proteomics" else df["has_proteomics"].where(df["has_proteomics"])
        present = s.notna()
        rows.append({"measure": col, **{g: int(present[df.group == g].sum()) for g in config.GROUPS},
                     "mouse_ids": " ".join(map(str, df.index[present])) if present.sum() < 48 else "all"})
    cov = pd.DataFrame(rows)
    log(cov.drop(columns="mouse_ids").to_string(index=False))
    save_table(cov, "qc_coverage")

    all_ihc = df[[f"{m}_idx" for m in config.IHC_MARKERS]].notna().all(axis=1)
    log(f"Mice with all 3 IHC markers: {int(all_ihc.sum())}; with IHC and proteomics: {int((all_ihc & df.has_proteomics).sum())}")

    # 2 EPM sums
    issues = []
    for wk in ("4", "12"):
        s = df[[f"EPMclosed{wk}", f"EPMcentre{wk}", f"EPMopen{wk}"]].sum(axis=1, min_count=3)
        for m in s.index[(s - 600).abs() > 2]:
            issues.append({"mouse": m, "check": f"EPM week {wk} zone sum", "value": s[m]})
        for m in df.index[df[f"EPMclosed{wk}"].notna() & df[f"EPMopen{wk}"].isna()]:
            issues.append({"mouse": m, "check": f"EPM week {wk} only closed-arm time recorded", "value": df.at[m, f"EPMclosed{wk}"]})
    # 3 T-maze resolution
    for c in ("T4_15", "T4_60", "T12_15", "T12_60"):
        vals = sorted(df[c].dropna().round(2).unique())
        log(f"{c}: distinct values {vals}")
    # 4 proteomics depth
    prot = raw.loc[df.has_proteomics, blocks["protein"]]
    depth = prot.notna().sum(axis=1)
    log("Proteins detected per mouse, group means:", depth.groupby(df.group[df.has_proteomics], observed=True).mean().round(0).to_dict())
    iss = pd.DataFrame(issues)
    log(iss.to_string(index=False))
    save_table(iss, "qc_issues")


if __name__ == "__main__":
    run()
