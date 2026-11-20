"""
CORE 6 - Which single brain summary explains memory best? (stained mice)

Question : Among mice with all 3 IHC markers, does a glia index explain
           memory as well as knowing the group?
Method   : OLS models on the same mice: intercept only, group (3 df),
           NeuN index, IBA1 index, GFAP index, Glia index, NeuN+IBA1,
           group+IBA1, group+NeuN. Compare AIC (lower = better) and R^2.
           Leave-one-mouse-out prediction R^2 for the simple models
           (no tuning, so plain LOOCV is valid).
Output   : results/tables/core_model_comparison_aic.csv,
           core_model_comparison_loocv.csv
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import config
from features.dataset import get_dataset
from utils.io import header, log, save_table
from utils.stats_helpers import ols, loocv_r2

FORMULAS = ["1", "C(group)", "NeuN_idx", "IBA1_idx", "GFAP_idx", "Glia_idx",
            "NeuN_idx + IBA1_idx", "C(group) + IBA1_idx", "C(group) + NeuN_idx"]
OUTCOMES = ["Mem12", "T12_15", "NOR12", "Mem4", "EPMopen12"]


def run():
    header("CORE 6  Model comparison on stained mice")
    df = get_dataset()
    df["group"] = df.group.astype(str)
    rows, cv_rows = [], []
    for y in OUTCOMES:
        d = df[[y, "group", "NeuN_idx", "GFAP_idx", "IBA1_idx", "Glia_idx"]].dropna()
        for f in FORMULAS:
            m = ols(f"{y} ~ {f}", d)
            rows.append({"outcome": config.LABELS.get(y, y), "n": len(d), "model": f, "AIC": m.aic, "R2": m.rsquared})
        X = {"group means": pd.get_dummies(d.group).astype(float), "Glia index": d[["Glia_idx"]],
             "IBA1 index": d[["IBA1_idx"]], "GFAP index": d[["GFAP_idx"]], "NeuN index": d[["NeuN_idx"]]}
        for name, x in X.items():
            pred = cross_val_predict(LinearRegression(), x, d[y], cv=LeaveOneOut())
            cv_rows.append({"outcome": config.LABELS.get(y, y), "n": len(d), "predictor": name, "LOOCV R2": loocv_r2(d[y], pred)})
    t, cv = pd.DataFrame(rows), pd.DataFrame(cv_rows)
    log(t.pivot(index="model", columns="outcome", values="AIC").round(1).to_string())
    log(cv.pivot(index="predictor", columns="outcome", values="LOOCV R2").round(2).to_string())
    save_table(t, "core_model_comparison_aic")
    save_table(cv, "core_model_comparison_loocv")
    return t, cv


if __name__ == "__main__":
    run()
