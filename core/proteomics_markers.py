"""
CORE 7 - Does the proteomics support the IHC picture? (24 mice)

Question : Do cell-type marker proteins show the same group pattern as IHC,
           and does any protein differ between groups or track memory?
Method   : (a) log2 group differences vs Saline for known marker proteins,
               one-way ANOVA per marker;
           (b) cell-type scores = mean z-score of marker proteins
               (markers detected in >=20 mice), ANOVA across groups;
           (c) agreement: Spearman of protein vs matching IHC index in the
               same mice;
           (d) global screen: ANOVA per protein and Spearman vs Mem12,
               BH across proteins; compare #p<0.05 with the chance level.
Output   : results/tables/core_proteomics_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config
from features.dataset import get_dataset
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from utils.stats_helpers import bh, spearman


def run():
    header("CORE 7  Proteomics: cell-type markers and global screen")
    L, grp, keep = get_proteomics()
    df = get_dataset().loc[L.index]

    # (a) marker proteins
    rows = []
    for ctype, genes in config.CELL_TYPE_MARKERS.items():
        for g in genes:
            if g not in L:
                continue
            means = L[g].groupby(grp, observed=True).mean()
            vals = [L.loc[grp == k, g].dropna() for k in config.GROUPS]
            p = stats.f_oneway(*vals).pvalue if min(map(len, vals)) >= 3 else np.nan
            rows.append({"cell type": ctype, "protein": g, "n mice": int(L[g].notna().sum()),
                         **{f"log2 diff {k} vs Saline": means[k] - means["Saline"] for k in config.GROUPS[1:]},
                         "ANOVA p": p})
    markers = pd.DataFrame(rows)
    save_table(markers, "core_proteomics_marker_proteins")

    # (b) cell-type scores
    Z = L.apply(lambda s: (s - s.mean()) / s.std())
    scores = pd.DataFrame({k: Z[[g for g in v if g in L and L[g].notna().sum() >= config.MIN_MICE_FOR_MARKER_SCORE]].mean(axis=1)
                           for k, v in config.CELL_TYPE_MARKERS.items()})
    srows = []
    for k in scores:
        srows.append({"score": k, **scores[k].groupby(grp, observed=True).mean().to_dict(),
                      "ANOVA p": stats.f_oneway(*[scores.loc[grp == g, k] for g in config.GROUPS]).pvalue,
                      **{f"rho vs {b}": spearman(scores[k], df[b])[0] for b in ["GFAP_idx", "IBA1_idx", "NeuN_idx", "Mem12"]}})
    srows = pd.DataFrame(srows)
    log(srows.round(2).to_string(index=False))
    save_table(srows, "core_proteomics_celltype_scores")

    # (c) agreement with IHC
    arows = []
    for prot, ihc in config.PROTEIN_IHC_PAIRS:
        rho, p, n = spearman(L[prot], df[ihc])
        arows.append({"protein": prot, "IHC": ihc, "n": n, "rho": rho, "p": p})
    agree = pd.DataFrame(arows)
    log(agree.round(3).to_string(index=False))
    save_table(agree, "core_proteomics_ihc_agreement")

    # (d) global screen
    pa = np.array([stats.f_oneway(*[L.loc[grp == k, c].dropna() for k in config.GROUPS]).pvalue for c in keep])
    pm = np.array([spearman(L[c], df.Mem12)[1] for c in keep])
    screen = pd.DataFrame({"protein": keep, "ANOVA p": pa, "ANOVA q": bh(pa), "rho vs Mem12 p": pm, "rho vs Mem12 q": bh(pm)})
    log(f"Proteins tested: {len(keep)}; expected p<0.05 by chance: {0.05 * len(keep):.0f}")
    log(f"  group ANOVA p<0.05: {(pa < .05).sum()}, min q {np.nanmin(screen['ANOVA q']):.3f}")
    log(f"  vs Mem12   p<0.05: {(pm < .05).sum()}, min q {np.nanmin(screen['rho vs Mem12 q']):.3f}")
    save_table(screen.sort_values("ANOVA p"), "core_proteomics_global_screen")


if __name__ == "__main__":
    run()
