"""
ADVANCED 8 - Re-check of the earlier protein claims against proper nulls.

Earlier reports (Integrated_Research_Report.docx and others) kept a few
protein findings after screening ~4,800 proteins:
  LRRC7-SHANK2 co-abundance (group-adjusted rho ~0.94)
  DDC-SLC18A2 pair  <-> T60-vs-T15 contrast (rho ~0.67)
  RBM8A <-> NOR wk12 (group-adjusted r ~0.61)
  MRPS30 <-> EPM open time (r ~ -0.87), BCLAF1 <-> T15 (r ~0.65)
A finding picked as the best of thousands needs a null for "the best of
thousands", not the usual single-test p.

Checks
  1 Co-abundance: where does the LRRC7-SHANK2 correlation sit among ALL
    protein pairs (complete proteins), as measured, group-adjusted and
    after removing technical factors? And among postsynaptic pairs?
  2 Where do DDC / SLC18A2 come from? Correlation with dopamine-terminal
    (Th, Slc6a3) and striatal (Ppp1r1b, Penk, Pde10a) proteins: if they
    track striatal content, the pair measures how much striatum was in
    the sample (tissue composition), not hippocampal biology.
  3 Proteome-wide max-statistic permutation (Westfall-Young): behaviour
    shuffled within group 2,000 times; each time the largest
    |group-adjusted rho| over all proteins is kept. Family-wise adjusted
    p for each earlier candidate = share of shuffles whose best protein
    beats the candidate. Also: how many proteins reach the candidate's
    |rho| in the real data.
Output   : results/tables/adv_legacy_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from features.proteomics_prep import get_proteomics, protein_column, technical_factors
from utils.io import header, log, save_table
from utils.permutation import within_group_permutation
from utils.stats_helpers import group_centred_rank_matrix, residualise_columns, rng

CANDIDATES = [("Rbm8a", "NOR12"), ("Mrps30", "EPMopen12"), ("Bclaf1", "T12_15"),
              ("Ddc", "T12_15"), ("Ddc", "T12_60"), ("Slc18a2", "T12_15"), ("Slc18a2", "T12_60")]
POSTSYNAPTIC = ["Dlg4", "Dlg3", "Dlg2", "Shank1", "Shank2", "Shank3", "Homer1", "Lrrc7", "Dlgap1", "Dlgap2",
                "Dlgap3", "Dlgap4", "Camk2a", "Camk2b", "Grin1", "Grin2a", "Grin2b", "Gria1", "Gria2", "Syngap1"]
STRIATAL = ["Th", "Slc6a3", "Ppp1r1b", "Penk", "Pde10a", "Drd1", "Rgs9", "Gpr88"]


def corr_cols(A, b):
    """Pearson of every column of A with vector b, using complete pairs per column."""
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        ok = ~np.isnan(A[:, j]) & ~np.isnan(b)
        if ok.sum() >= 10:
            out[j] = np.corrcoef(A[ok, j], b[ok])[0, 1]
    return out


def run(n_perm=None):
    n_perm = n_perm or config.N_PERM_PROTEOME
    header("ADVANCED 8  Re-check of earlier protein claims (proteome-wide nulls)")
    L, grp, keep = get_proteomics()
    df = get_dataset().loc[L.index]
    g = grp.astype(str)
    tech = technical_factors(L)
    comp = [c for c in keep if L[c].notna().all()]
    gen = rng()

    # 1 co-abundance percentile
    versions = {"as measured": L[comp],
                "group-adjusted ranks": group_centred_rank_matrix(L[comp], g),
                "technical factors removed + group-adjusted": group_centred_rank_matrix(residualise_columns(L[comp], tech), g)}
    a, b = protein_column(L, "Lrrc7"), protein_column(L, "Shank2")
    psd = [protein_column(L, p) for p in POSTSYNAPTIC if protein_column(L, p) in comp]
    rows = []
    for name, M in versions.items():
        C = np.corrcoef(M.to_numpy(float), rowvar=False)
        iu = np.triu_indices_from(C, 1)
        allr = C[iu]
        i, j = comp.index(a), comp.index(b)
        r = C[i, j]
        pi = [comp.index(p) for p in psd]
        psd_r = np.array([C[x, y] for k, x in enumerate(pi) for y in pi[k + 1:]])
        rows.append({"version": name, "LRRC7-SHANK2 r": r, "percentile among all pairs": (allr < r).mean(),
                     "pairs with r >= this": int((allr >= r).sum()), "all pairs": len(allr),
                     "median r all pairs": np.median(allr), "median r postsynaptic pairs": np.median(psd_r),
                     "share of all pairs with |r| > 0.8": (np.abs(allr) > 0.8).mean()})
    co = pd.DataFrame(rows)
    log("LRRC7-SHANK2 among all protein pairs:\n" + co.round(4).to_string(index=False))
    save_table(co, "adv_legacy_coabundance")

    # 2 origin of DDC / SLC18A2
    rows = []
    adj = group_centred_rank_matrix(L[keep], g)
    for p in ["Ddc", "Slc18a2"]:
        for s in STRIATAL:
            c1, c2 = protein_column(L, p), protein_column(L, s)
            if c2 is None or c2 not in keep:
                continue
            for name, M in [("as measured", L), ("group-adjusted ranks", adj)]:
                ok = M[c1].notna() & M[c2].notna()
                rr, pp = stats.spearmanr(M.loc[ok, c1], M.loc[ok, c2])
                rows.append({"protein": p, "partner": s, "version": name, "n": int(ok.sum()), "rho": rr, "p": pp})
    ori = pd.DataFrame(rows)
    log("DDC / SLC18A2 vs dopamine-terminal and striatal proteins:\n" +
        ori.pivot_table(index=["protein", "partner"], columns="version", values="rho").round(2).to_string())
    save_table(ori, "adv_legacy_ddc_origin")

    # 3 proteome-wide max-statistic permutation
    A = group_centred_rank_matrix(L[keep], g).to_numpy(float)
    rows = []
    for prot, beh in CANDIDATES:
        y = df[beh]
        ok = y.notna().to_numpy()
        Ay = A[ok]
        gy = g[ok].to_numpy()
        yv = y[ok]
        yr = (yv.rank() - yv.rank().groupby(gy).transform("mean")).to_numpy(float)
        obs_all = corr_cols(Ay, yr)
        j = keep.index(protein_column(L, prot))
        obs = obs_all[j]
        maxnull = np.empty(n_perm)
        for i in range(n_perm):
            yp = yv.to_numpy()[within_group_permutation(gy, gen)]
            ypr = pd.Series(yp).rank()
            ypr = (ypr - ypr.groupby(gy).transform("mean")).to_numpy(float)
            maxnull[i] = np.nanmax(np.abs(corr_cols(Ay, ypr)))
        rows.append({"protein": prot, "behaviour": config.LABELS[beh], "mice with behaviour": int(ok.sum()),
                     "group-adjusted rho": obs, "single-test p": stats.t.sf(abs(obs) * np.sqrt((ok.sum() - 2) / (1 - obs ** 2)), ok.sum() - 2) * 2,
                     "rank among proteins": int((np.abs(obs_all) > abs(obs)).sum()) + 1, "proteins tested": int(np.isfinite(obs_all).sum()),
                     "proteins with |rho| >= this": int((np.abs(obs_all) >= abs(obs)).sum()),
                     "best |rho| by chance (median)": np.median(maxnull),
                     "best |rho| by chance (95th pct)": np.percentile(maxnull, 95),
                     "family-wise p (max-statistic)": (1 + np.sum(maxnull >= abs(obs))) / (1 + n_perm)})
        log(f"  {prot} ~ {beh}: rho={obs:.3f}, FWER p={rows[-1]['family-wise p (max-statistic)']:.3f}")
    fw = pd.DataFrame(rows)
    log("Earlier candidates vs proteome-wide max-statistic null:\n" + fw.round(3).to_string(index=False))
    save_table(fw, "adv_legacy_maxstat")
    return co, ori, fw


if __name__ == "__main__":
    run()
