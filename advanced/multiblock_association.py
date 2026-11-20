"""
ADVANCED 5 - Block-to-block association: RV coefficient and distance correlation.

Literature
  Josse & Holmes 2016 (Statistics Surveys, "Measuring multivariate
  association and beyond") review the RV coefficient and distance
  covariance (dCov) for testing association between two blocks of
  variables; Smilde et al. 2009 (Bioinformatics) give the modified RV for
  high-dimensional blocks (removes the bias that inflates RV when p >> n).
  Tests are by permutation. dCor also detects non-linear dependence.
Why here : Instead of testing 15 stains x 3 memory scores one by one
  (many tests), ask one question per pair of blocks:
     glia stains (10) <-> memory (3),  NeuN stains (5) <-> memory (3),
     glia / NeuN <-> anxiety (EPM, 3), IHC (15) <-> proteomics (sets / all)
  and, for NeuN, the same after removing the glia index from both blocks.
Nulls
  "all mice"    : rows of one block shuffled across all mice;
  "within group": rows shuffled only within each group, so group
                  differences are kept and only individual variation is
                  tested (the question that small groups cannot answer by
                  single correlations).
Output   : results/tables/adv_multiblock_association.csv
"""
import numpy as np
import pandas as pd
import config
from features.protein_sets import resolve_members, set_scores, zmatrix
from features.dataset import get_dataset, ihc_columns
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from utils.permutation import within_group_permutation
from utils.stats_helpers import residualise_matrix, rng, standardise

MEMORY = ["T12_15", "T12_60", "NOR12"]
ANXIETY = ["EPMopen12", "EPMcentre12", "EPMentries12"]


def modified_rv(X, Y):
    """Smilde et al. 2009 modified RV (diagonal of XX' and YY' removed)."""
    A, B = X @ X.T, Y @ Y.T
    np.fill_diagonal(A, 0)
    np.fill_diagonal(B, 0)
    return np.sum(A * B) / np.sqrt(np.sum(A * A) * np.sum(B * B))


def _dcenter(M):
    D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
    return D - D.mean(0) - D.mean(1)[:, None] + D.mean()


def dcor(X, Y):
    A, B = _dcenter(X), _dcenter(Y)
    v = np.sqrt(np.mean(A * A) * np.mean(B * B))
    return np.sqrt(max(np.mean(A * B), 0) / v) if v > 0 else 0.0


def perm_test(X, Y, groups, n_perm, gen):
    obs = {"mod RV": modified_rv(X, Y), "dCor": dcor(X, Y)}
    out = {}
    for label, within in [("all mice", False), ("within group", True)]:
        null = {k: np.empty(n_perm) for k in obs}
        for i in range(n_perm):
            if within:
                idx = within_group_permutation(groups, gen)
            else:
                idx = gen.permutation(len(X))
            null["mod RV"][i] = modified_rv(X, Y[idx])
            null["dCor"][i] = dcor(X, Y[idx])
        for k in obs:
            out[f"{k} p ({label})"] = (1 + np.sum(null[k] >= obs[k])) / (1 + n_perm)
    return obs, out


def run(n_perm=None):
    n_perm = n_perm or config.N_PERM_BLOCK
    header(f"ADVANCED 5  Block association: modified RV and distance correlation ({n_perm} permutations)")
    df = get_dataset()
    gen = rng()
    ihc = ihc_columns()
    glia = [c for c in ihc if c.startswith(("GFAP", "IBA1"))]
    neun = [c for c in ihc if c.startswith("NeuN")]
    L, _, keep = get_proteomics()
    sets = set_scores(zmatrix(L), resolve_members(L))
    full = [c for c in keep if L[c].notna().all()]
    pairs = [("glia stains (10)", glia, "memory wk12 (3)", MEMORY, None),
             ("NeuN stains (5)", neun, "memory wk12 (3)", MEMORY, None),
             ("NeuN stains (5) | glia index", neun, "memory wk12 (3) | glia index", MEMORY, "Glia_idx"),
             ("glia stains (10) | NeuN index", glia, "memory wk12 (3) | NeuN index", MEMORY, "NeuN_idx"),
             ("glia stains (10)", glia, "anxiety EPM wk12 (3)", ANXIETY, None),
             ("NeuN stains (5)", neun, "anxiety EPM wk12 (3)", ANXIETY, None),
             ("all IHC (15)", ihc, "protein set scores (14)", "SETS", None),
             ("all IHC (15)", ihc, f"proteome ({len(full)} complete proteins)", "FULL", None),
             ("protein set scores (14)", "SETS", "memory wk12 (3)", MEMORY, None)]
    rows = []
    for xn, xc, yn, yc, cov in pairs:
        base = df.join(sets) if "SETS" in (xc, yc) else df
        base = base.join(L[full]) if yc == "FULL" else base
        xcols = list(sets.columns) if xc == "SETS" else xc
        ycols = list(sets.columns) if yc == "SETS" else (full if yc == "FULL" else yc)
        need = xcols + ycols + ["group"] + ([cov] if cov else [])
        d = base[need].dropna()
        X, Y = standardise(d[xcols]), standardise(d[ycols])
        if cov:
            X, Y = standardise(residualise_matrix(X, d[cov].to_numpy())), standardise(residualise_matrix(Y, d[cov].to_numpy()))
        obs, ps = perm_test(X, Y, d.group.astype(str).to_numpy(), n_perm if yc != "FULL" else min(n_perm, 999), gen)
        rows.append({"block X": xn, "block Y": yn, "n": len(d), **obs, **ps})
        log(f"  {xn} <-> {yn}: n={len(d)} modRV={obs['mod RV']:.3f} dCor={obs['dCor']:.3f} "
            + " ".join(f"{k}={v:.3f}" for k, v in ps.items()))
    t = pd.DataFrame(rows)
    log(t.round(3).to_string(index=False))
    save_table(t, "adv_multiblock_association")
    return t


if __name__ == "__main__":
    run()
