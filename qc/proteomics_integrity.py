"""
QC 2 - Why does the proteomics carry no signal? Integrity checks.

Every proteomics analysis was negative, and proteomic Gfap does not track
IHC GFAP in the same mice. This module looks for technical explanations.

Checks
  1 Tissue composition: are marker proteins of other brain regions present
    (striatum, olfactory bulb, hypothalamus, cerebellum)? If yes, the
    sample is a large brain piece / hemisphere, not hippocampus, and the
    hippocampal IHC would not be expected to match.
  2 Blood content: haemoglobin/albumin abundance per mouse (perfusion?).
  3 Possible sex marker: Ddx3y (Y-linked) detection per mouse and group.
    Caveat: Ddx3y and Ddx3x share most of their sequence, so detection
    can be peptide mis-assignment; the other Y proteins (Kdm5d, Eif2s3y,
    Uty) are not detected. Must be checked against animal records.
  4 Dissection variability: olfactory-bulb marker Omp per mouse.
  5 What drives the main proteomic variation? PCA (complete proteins),
    R^2 of PC1-PC3 explained by depth, blood, Omp, Ddx3y, group.
  6 Sample-to-mouse alignment: concordance between proteomic glia/neuron
    markers and IHC in the same mice (mean Spearman of 5 pairs), compared
    with (a) proteomics IDs shuffled within group, (b) shuffled across all
    mice, (c) cyclic shifts of IDs within each group (off-by-k errors).
  7 Re-test after removing technical factors (depth, blood, Omp, Ddx3y)
    from every protein: group ANOVA count and Gfap-IHC correlation.
  8 Is the number of group-ANOVA p<0.05 unusual? Compare with the same
    count under shuffled group labels (as measured and after step 7).
Output   : results/tables/qc_proteomics_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from data.load_data import load_raw
from features.dataset import get_dataset
from features.protein_sets import resolve_members, set_scores, zmatrix
from features.proteomics_prep import get_proteomics, protein_column, technical_factors
from utils.io import header, log, save_table
from utils.permutation import within_group_permutation
from utils.stats_helpers import residualise_columns, rng

REGION_MARKERS = {
    "Hippocampus": ["Prox1", "C1ql2", "Pcp4", "Wfs1", "Necab2", "Calb2", "Zbtb20", "Neurod6"],
    "Cortex": ["Satb2", "Tbr1", "Bcl11b", "Neurod2", "Rorb"],
    "Striatum": ["Ppp1r1b", "Penk", "Pde10a", "Drd1", "Tac1", "Rgs9", "Gpr88", "Adora2a"],
    "Olfactory bulb": ["Omp"],
    "Hypothalamus": ["Pmch", "Hcrt", "Oxt", "Avp", "Pomc"],
    "Cerebellum": ["Pcp2", "Gabra6", "Cbln1", "Car8", "Grid2"],
    "Blood": ["Hba", "Hbb-bs", "Hbb-b1", "Hbb-b2", "Alb", "Apoa1"],
    "Y chromosome": ["Ddx3y", "Kdm5d", "Eif2s3y", "Uty"],
}
CONCORDANCE_PAIRS = [("Gfap", "GFAP_idx"), ("set:Reactive astrocyte", "GFAP_idx"),
                     ("Hexb", "IBA1_idx"), ("set:Microglia (Hexb, Tmem119, C1q)", "IBA1_idx"),
                     ("set:Neuronal structure", "NeuN_idx")]


def concordance(P, ihc, mapping):
    """Mean Spearman over CONCORDANCE_PAIRS when proteomics row i is assigned to mouse mapping[i]."""
    rs = []
    for p, i in CONCORDANCE_PAIRS:
        x = pd.Series(P[p].to_numpy(), index=mapping)
        y = ihc.loc[mapping, i]
        ok = x.notna().to_numpy() & y.notna().to_numpy()
        if ok.sum() >= 6:
            rs.append(stats.spearmanr(x.to_numpy()[ok], y.to_numpy()[ok])[0])
    return np.mean(rs)


def run():
    header("QC 2  Proteomics integrity checks")
    raw, blocks = load_raw()
    L, grp, keep = get_proteomics()
    df = get_dataset()
    g = grp.astype(str)
    rawP = raw.loc[L.index, blocks["protein"]].astype(float)
    pct = rawP.median().rank(pct=True)

    # 1-3 marker presence
    rows = []
    for region, genes in REGION_MARKERS.items():
        for gene in genes:
            c = protein_column(L, gene)
            rows.append({"region / type": region, "protein": gene, "mice detected (of 24)": int(L[c].notna().sum()) if c else 0,
                         "abundance percentile": pct[c] if c else np.nan})
    comp = pd.DataFrame(rows)
    log(comp.round(2).to_string(index=False))
    save_table(comp, "qc_proteomics_tissue_markers")

    # per-mouse technical factors
    tech = technical_factors(L)
    tech.insert(0, "group", g)
    log(tech.to_string())
    save_table(tech.reset_index(), "qc_proteomics_per_mouse")
    by = tech.groupby("group")[["depth", "blood (Hb log2)", "Omp detected", "Ddx3y detected"]].agg(["mean"]).round(2)
    log("By group:\n" + by.to_string())
    ft = pd.crosstab(tech.group, tech["Ddx3y detected"])
    chi = stats.chi2_contingency(ft)
    p_sal_pmev = stats.fisher_exact(pd.crosstab(tech.group, tech["Ddx3y detected"]).loc[["Saline", "PM2.5 EV"]])[1]
    log(f"Ddx3y detection vs group: chi2 p = {chi.pvalue:.3f}; Saline vs PM2.5 EV Fisher p = {p_sal_pmev:.3f}")
    log(f"Blood (Hb) vs group ANOVA p = {stats.f_oneway(*[tech.loc[g == k, 'blood (Hb log2)'] for k in config.GROUPS]).pvalue:.3f}")

    # 5 PCA drivers
    comp_cols = [c for c in keep if L[c].notna().all()]
    Z = L[comp_cols].apply(lambda s: (s - s.mean()) / s.std())
    U, s, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    pcs = pd.DataFrame(U[:, :3] * s[:3], index=L.index, columns=["PC1", "PC2", "PC3"])
    var = s ** 2 / (s ** 2).sum()
    drows = []
    for k, pc in enumerate(pcs):
        r = {"PC": pc, "% variance": var[k] * 100}
        for f in ["depth", "blood (Hb log2)", "Omp detected", "Ddx3y detected"]:
            r[f"R2 {f}"] = stats.pearsonr(pcs[pc], tech[f])[0] ** 2
        y = pcs[pc]
        r["R2 group"] = 1 - ((y - y.groupby(g).transform("mean")) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        drows.append(r)
    drivers = pd.DataFrame(drows)
    log(f"PCA on {len(comp_cols)} proteins complete in all 24 mice:\n" + drivers.round(2).to_string(index=False))
    save_table(drivers, "qc_proteomics_pca_drivers")

    # 6 alignment
    S = set_scores(zmatrix(L), resolve_members(L))
    P = pd.DataFrame({p: (S[p[4:]] if p.startswith("set:") else L[protein_column(L, p)]) for p, _ in CONCORDANCE_PAIRS})
    ihc = df[["GFAP_idx", "IBA1_idx", "NeuN_idx"]]
    ids = L.index.to_numpy()
    obs = concordance(P, ihc, ids)
    gen = rng()
    within = []
    for _ in range(config.N_SHUFFLE_ALIGNMENT):
        m = ids[within_group_permutation(g.to_numpy(), gen, order=config.GROUPS)]
        within.append(concordance(P, ihc, m))
    across = [concordance(P, ihc, gen.permutation(ids)) for _ in range(config.N_SHUFFLE_ALIGNMENT)]
    within, across = np.array(within), np.array(across)
    shifts = []
    for k in range(6):
        m = ids.copy()
        for grp_name in config.GROUPS:
            ix = np.where(g.to_numpy() == grp_name)[0]
            m[ix] = np.roll(ids[ix], k)
        shifts.append({"within-group shift": k, "concordance (mean rho)": concordance(P, ihc, m)})
    al = pd.DataFrame([{"mapping": "as recorded", "concordance (mean rho)": obs,
                        "percentile vs within-group shuffles": (within < obs).mean(),
                        "percentile vs all shuffles": (across < obs).mean(),
                        "shuffle within mean": within.mean(), "shuffle all mean": across.mean()}])
    log(f"Alignment: recorded concordance {obs:.3f}; within-group shuffles mean {within.mean():.3f} "
        f"(recorded is at percentile {(within < obs).mean():.2f}); all shuffles mean {across.mean():.3f} "
        f"(percentile {(across < obs).mean():.2f})")
    sh = pd.DataFrame(shifts)
    log(sh.round(3).to_string(index=False))
    save_table(al, "qc_proteomics_alignment")
    save_table(sh, "qc_proteomics_alignment_shifts")

    # 7 re-test after removing technical factors
    cov = tech[["depth", "blood (Hb log2)", "Omp detected", "Ddx3y detected"]]
    R = residualise_columns(L[keep], cov)
    pa_raw = np.array([stats.f_oneway(*[L.loc[g == k, c].dropna() for k in config.GROUPS]).pvalue for c in keep])
    pa_res = np.array([stats.f_oneway(*[R.loc[g == k, c].dropna() for k in config.GROUPS]).pvalue for c in keep])
    gf = protein_column(L, "Gfap")
    r_raw = stats.spearmanr(L[gf], df.loc[L.index, "GFAP_idx"], nan_policy="omit")[0]
    r_res = stats.spearmanr(R[gf], df.loc[L.index, "GFAP_idx"], nan_policy="omit")[0]
    # 8 null distribution of the p<0.05 count
    # Null: shuffle group labels (proteins are correlated, so the count's null spread is wide)
    def count_p05(M, labels):
        return sum(stats.f_oneway(*[M[c][labels == k].dropna() for k in config.GROUPS]).pvalue < .05 for c in keep)
    rows = []
    for name, M, pa, r in [("as measured", L, pa_raw, r_raw), ("technical factors removed", R, pa_res, r_res)]:
        null = np.array([count_p05(M, gen.permutation(g.to_numpy())) for _ in range(config.N_NULL_COUNT_SHUFFLES)])
        obs_n = int((pa < .05).sum())
        rows.append({"data": name, f"ANOVA p<0.05 (of {len(keep)})": obs_n, "Gfap vs IHC GFAP rho": r,
                     "shuffled-label count mean": null.mean(), "shuffled 2.5%": np.percentile(null, 2.5),
                     "shuffled 97.5%": np.percentile(null, 97.5), "percentile of observed": (null < obs_n).mean()})
    re = pd.DataFrame(rows)
    log(re.round(3).to_string(index=False))
    save_table(re, "qc_proteomics_after_adjustment")


if __name__ == "__main__":
    run()
