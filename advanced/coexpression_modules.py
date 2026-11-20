"""
ADVANCED 7 - Protein co-expression modules (WGCNA-style) with stability check.

Literature
  WGCNA (Langfelder & Horvath 2008, BMC Bioinformatics): cluster features
  into co-expression modules, summarise each by its eigengene (PC1), and
  relate eigengenes to traits; turns thousands of tests into a handful.
  Small-n guidance: below ~15-20 samples modules are unstable, so module
  stability must be checked by bootstrap (e.g. SABRE-type resampling).
Here (24 proteomics mice)
  1 Complete proteins; technical factors (depth, blood, Omp, Ddx3y) removed
    first (see qc/proteomics_integrity.py), then the 1,000 most variable.
  2 Signed adjacency ((1 + r) / 2)^beta, beta = smallest power giving a
    scale-free fit R^2 >= 0.8 (max 14); topological overlap (TOM);
    average-linkage clustering on 1 - TOM; tree cut into modules of
    >= 20 proteins (smaller ones -> "grey", unassigned).
  3 Stability: 100 bootstrap resamples of mice; adjusted Rand index of
    module assignments vs the full-data modules, and per-module best
    Jaccard match.
  4 Eigengene per module: group ANOVA; Spearman with Glia/NeuN index and
    memory (all mice and within group); BH over all tests. Hub proteins
    and overlap with the literature protein sets (Fisher).
Output   : results/tables/adv_wgcna_*.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
import config
from features.protein_sets import resolve_members
from features.dataset import get_dataset
from features.proteomics_prep import get_proteomics, technical_factors
from utils.io import header, log, save_table
from utils.stats_helpers import bh, residualise_columns, rng, spearman, within_group_spearman

N_TOP = 1000
MIN_MODULE = 20
N_CLUSTERS = 25   # tree cut (maxclust); clusters < MIN_MODULE go to grey
TRAITS = ["Glia_idx", "NeuN_idx", "Mem12", "T12_15"]


def pick_beta(R):
    k_all = []
    for beta in range(1, 15):
        A = ((1 + R) / 2) ** beta
        k = A.sum(0) - 1
        h, e = np.histogram(k, bins=10)
        mid = (e[:-1] + e[1:]) / 2
        ok = h > 0
        r = stats.linregress(np.log10(mid[ok]), np.log10(h[ok] / h.sum()))
        fit = -np.sign(r.slope) * r.rvalue ** 2
        k_all.append((beta, fit))
        if fit >= 0.8:
            return beta, k_all
    return max(k_all, key=lambda x: x[1])[0], k_all


def modules(X, beta, n_clusters=None):
    R = np.corrcoef(X, rowvar=False)
    A = ((1 + R) / 2) ** beta
    np.fill_diagonal(A, 0)
    k = A.sum(0)
    TOM = (A @ A + A) / (np.minimum.outer(k, k) + 1 - A)
    np.fill_diagonal(TOM, 1)
    D = 1 - TOM
    Z = linkage(squareform(D, checks=False), "average")
    lab = fcluster(Z, t=n_clusters or N_CLUSTERS, criterion="maxclust")
    sizes = pd.Series(lab).value_counts()
    lab = np.array([l if sizes[l] >= MIN_MODULE else 0 for l in lab])
    return lab


def eigengene(X):
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    e = U[:, 0] * s[0]
    return e * np.sign(np.corrcoef(e, Xs.mean(1))[0, 1]), s[0] ** 2 / (s ** 2).sum()


def run():
    header("ADVANCED 7  Protein co-expression modules (WGCNA-style)")
    L, grp, keep = get_proteomics()
    df = get_dataset()
    g = grp.astype(str)
    tech = technical_factors(L)
    comp = [c for c in keep if L[c].notna().all()]
    R = residualise_columns(L[comp], tech)
    top = R.var().sort_values(ascending=False).index[:N_TOP]
    X = R[top].to_numpy(float)
    beta, fits = pick_beta(np.corrcoef(X, rowvar=False))
    lab = modules(X, beta)
    names = {m: f"M{i + 1}" for i, m in enumerate(sorted(set(lab) - {0}))}
    if dict(fits)[beta] < 0.8:
        log(f"NOTE: scale-free fit never reached 0.8 (best {max(f for _, f in fits):.2f}); using beta = {beta}")
    log(f"Proteins: {len(top)} most variable of {len(comp)} complete (technical factors removed); "
        f"soft power beta = {beta}; modules: {len(names)}; grey (unassigned): {(lab == 0).sum()}")

    # stability
    gen = rng()
    ari, jac = [], {m: [] for m in names}
    for _ in range(config.N_BOOT_MODULES):
        i = gen.integers(0, X.shape[0], X.shape[0])
        if np.any(X[i].std(0) == 0):
            continue
        lb = modules(X[i], beta)
        ari.append(adjusted_rand_score(lab, lb))
        for m in names:
            a = set(np.where(lab == m)[0])
            best = max((len(a & set(np.where(lb == k)[0])) / len(a | set(np.where(lb == k)[0]))
                        for k in set(lb) - {0}), default=0)
            jac[m].append(best)
    log(f"Bootstrap stability: adjusted Rand index median {np.median(ari):.2f} (IQR {np.percentile(ari, 25):.2f}-"
        f"{np.percentile(ari, 75):.2f})")

    # eigengenes and traits
    members = resolve_members(L)
    rows, ME = [], {}
    for m, nm in names.items():
        idx = np.where(lab == m)[0]
        e, ve = eigengene(X[:, idx])
        ME[nm] = pd.Series(e, index=R.index)
        kME = np.array([np.corrcoef(e, X[:, j])[0, 1] for j in idx])
        hubs = [top[idx[j]].split(";")[0] for j in np.argsort(-kME)[:5]]
        mod_prot = set(top[idx])
        best_set, best_p = "", 1.0
        for sname, sm in members.items():
            a = len(mod_prot & set(sm))
            b = len(set(sm) & set(top)) - a
            c = len(mod_prot) - a
            dd = len(top) - a - b - c
            p = stats.fisher_exact([[a, b], [c, dd]], alternative="greater")[1] if a else 1.0
            if p < best_p:
                best_set, best_p = f"{sname} ({a})", p
        r = {"module": nm, "proteins": len(idx), "variance explained by eigengene": ve,
             "bootstrap Jaccard median": np.median(jac[m]), "hub proteins": ", ".join(hubs),
             "best protein-set overlap": best_set, "overlap p": best_p,
             "group ANOVA p": stats.f_oneway(*[e[(g == k).to_numpy()] for k in config.GROUPS]).pvalue}
        tr = df.loc[R.index].assign(ME=e)
        for t in TRAITS:
            r[f"rho {t}"], r[f"p {t}"], _ = spearman(tr.ME, tr[t])
            r[f"within rho {t}"], r[f"within p {t}"], _ = within_group_spearman(tr, "ME", t)
        rows.append(r)
    t = pd.DataFrame(rows)
    pcols = ["group ANOVA p"] + [f"p {x}" for x in TRAITS] + [f"within p {x}" for x in TRAITS]
    allp = t[pcols].to_numpy().ravel()
    t["min BH q (all tests)"] = bh(allp).reshape(t[pcols].shape).min(1)
    log(t.drop(columns=[c for c in t.columns if c.startswith("within rho")]).round(3).to_string(index=False))
    save_table(t, "adv_wgcna_modules")
    save_table(pd.DataFrame({"protein": [p.split(";")[0] for p in top],
                             "module": [names.get(l, "grey") for l in lab]}), "adv_wgcna_membership")
    save_table(pd.DataFrame(fits, columns=["beta", "signed scale-free fit"]), "adv_wgcna_soft_power")
    return t


if __name__ == "__main__":
    run()
