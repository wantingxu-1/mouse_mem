"""
ROBUSTNESS 9 - IHC with region as a repeated measure (linear mixed model).

Why      : Averaging 5 regions loses information; testing regions one by one
           multiplies tests. A mixed model uses all regions of a mouse at once
           and asks whether the group effect differs between regions.
Scale    : log(value / Saline mean of that region). A coefficient of
           log(1.5) = +50 % vs Saline, comparable across regions.
Models (per marker, random intercept per mouse, ML fit)
  A  y ~ group + region                (same group effect in every region)
  B  y ~ group * region                (region-specific group effects)
  Likelihood-ratio test A vs B: does the group effect depend on region?
  From B (cell means): per region, % vs Saline for each group, and
  N EV vs Saline / N EV vs PM2.5 contrasts (Wald z).
  Recovery fraction = (PM2.5 - N EV) / (PM2.5 - Saline) on the log scale:
  1 = N EV fully back to Saline, 0 = same as PM2.5.
Note     : On this log-ratio scale the between-mouse variance is estimated
           at ~0 (statsmodels warns "covariance is singular / boundary").
           The model then behaves like ordinary least squares; estimates are
           still valid, and the warnings are expected.
Output   : results/tables/robust_ihc_mixed_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import config
from features.dataset import get_dataset, ihc_columns
from features.ihc_indices import marker_columns
from utils.io import header, log, save_table


def long_marker(df, marker):
    cols = marker_columns(ihc_columns(), marker)
    d = df[cols + ["group"]].reset_index().melt(id_vars=["mouse", "group"], var_name="column", value_name="value").dropna()
    d["region"] = d.column.str.split(" ", n=1).str[1].str.upper().str.replace("CORTEX", "Cortex")
    ref = d[d.group == config.REFERENCE_GROUP].groupby("region").value.mean()
    d["y"] = np.log(d.value / d.region.map(ref))
    d["group"] = pd.Categorical(d.group.astype(str), config.GROUPS)
    return d


def fit(formula, d):
    return smf.mixedlm(formula, d, groups=d["mouse"]).fit(reml=False, method="lbfgs")


def contrast(res, a, b):
    """Wald z for difference of two cell-mean coefficients a - b."""
    p, cov = res.fe_params, res.cov_params()
    est = p[a] - p[b]
    se = np.sqrt(cov.loc[a, a] + cov.loc[b, b] - 2 * cov.loc[a, b])
    return est, 2 * (1 - stats.norm.cdf(abs(est / se)))


def run():
    header("ROBUSTNESS 9  IHC linear mixed models (region as repeated measure)")
    df = get_dataset()
    lrt_rows, cell_rows, overall_rows = [], [], []
    for marker in config.IHC_MARKERS:
        d = long_marker(df, marker)
        mA = fit("y ~ C(group) + C(region)", d)
        mB = fit("y ~ C(group) * C(region)", d)
        lr = 2 * (mB.llf - mA.llf)
        dfd = len(mB.fe_params) - len(mA.fe_params)
        lrt_rows.append({"marker": marker, "n mice": d.mouse.nunique(), "n values": len(d),
                         "LR statistic": lr, "df": dfd, "p (group effect differs by region)": stats.chi2.sf(lr, dfd)})
        # overall (model A) group effects vs Saline
        for g in config.GROUPS[1:]:
            k = f"C(group)[T.{g}]"
            ci = mA.conf_int().loc[k]
            overall_rows.append({"marker": marker, "group": g, "% vs Saline": (np.exp(mA.fe_params[k]) - 1) * 100,
                                 "95% CI low %": (np.exp(ci[0]) - 1) * 100, "95% CI high %": (np.exp(ci[1]) - 1) * 100,
                                 "p": mA.pvalues[k]})
        nev_pm, pnev_pm = contrast(mA, "C(group)[T.N EV]", "C(group)[T.PM2.5]")
        overall_rows.append({"marker": marker, "group": "N EV vs PM2.5", "% vs Saline": (np.exp(nev_pm) - 1) * 100,
                             "p": pnev_pm, "recovery fraction": 1 - mA.fe_params["C(group)[T.N EV]"] / mA.fe_params["C(group)[T.PM2.5]"]})
        # cell means model for per-region contrasts
        mC = fit("y ~ 0 + C(region):C(group)", d)
        for reg in sorted(d.region.unique()):
            name = lambda g: f"C(region)[{reg}]:C(group)[{g}]"
            r = {"marker": marker, "region": reg}
            for g in config.GROUPS[1:]:
                est, p = contrast(mC, name(g), name("Saline"))
                r[f"{g} % vs Saline"], r[f"{g} p"] = (np.exp(est) - 1) * 100, p
            est, p = contrast(mC, name("N EV"), name("PM2.5"))
            r["N EV vs PM2.5 p"] = p
            pm = mC.fe_params[name("PM2.5")] - mC.fe_params[name("Saline")]
            r["recovery fraction"] = 1 - (mC.fe_params[name("N EV")] - mC.fe_params[name("Saline")]) / pm
            cell_rows.append(r)
    lrt, overall, cells = pd.DataFrame(lrt_rows), pd.DataFrame(overall_rows), pd.DataFrame(cell_rows)
    log("Does the group effect differ by region? (LR test)\n" + lrt.round(4).to_string(index=False))
    log("Overall group effect (model A):\n" + overall.round(3).to_string(index=False))
    log("Per region (cell-means model):\n" + cells.round(3).to_string(index=False))
    save_table(lrt, "robust_ihc_mixed_region_interaction")
    save_table(overall, "robust_ihc_mixed_overall")
    save_table(cells, "robust_ihc_mixed_by_region")
    return lrt, overall, cells


if __name__ == "__main__":
    run()
