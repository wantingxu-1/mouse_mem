"""
ADVANCED 6 - PLS integration of stains and behaviour (idea from mixOmics sPLS / DIABLO).

Literature
  mixOmics (Rohart et al. 2017, PLoS Comput Biol): sPLS links two blocks
  measured on the same samples through latent components; DIABLO extends
  this to several blocks with a design that weights block-to-block links.
  Widely used for small n, many features; selection is tuned by CV.
Here
  1 Canonical PLS, IHC (15) <-> behaviour (T-maze 15/60 s, NOR, EPM open,
    EPM crossings; wk12). Component 1 pairs the stain pattern and the
    behaviour pattern that covary most. Permutation test of its covariance
    (all mice / within group). Loadings show WHICH stains (glia or NeuN)
    and WHICH behaviours define the shared axis; bootstrap 95% CI of
    loadings (sign-aligned).
  2 PLS regression IHC -> memory composite: VIP (variable importance in
    projection) with bootstrap CI. VIP > 1 = important.
  3 Multi-block (DIABLO-style, block-scaled concatenation) IHC + protein
    sets -> memory composite, nested LOOCV: does adding the proteomics
    block help, and how much weight does each block get? Protein-set
    scores are rebuilt inside each training fold (fold_set_scores).
Output   : results/tables/adv_pls_*.csv
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config
from features.protein_sets import FoldSetScores
from features.dataset import get_dataset, ihc_columns
from features.proteomics_prep import get_proteomics
from utils.io import header, log, save_table
from utils.permutation import within_group_permutation
from ml.nested_cv import mae, train_mean_predict
from utils.stats_helpers import loocv_r2, rng, standardise

BEHAVIOUR = ["T12_15", "T12_60", "NOR12", "EPMopen12", "EPMentries12"]


class BlockWeight(BaseEstimator, TransformerMixin):
    """Divide a (standardised) block by sqrt(number of columns), so each block has equal total weight."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X) / np.sqrt(np.asarray(X).shape[1])


class PLS1(PLSRegression):
    """PLSRegression with a 1-D prediction."""

    def predict(self, X, copy=True):
        return super().predict(X, copy=copy).ravel()


def comp1(X, Y):
    m = PLSCanonical(n_components=1, scale=False).fit(X, Y)
    t, u = m.transform(X, Y)
    return m, np.cov(t.ravel(), u.ravel())[0, 1], np.corrcoef(t.ravel(), u.ravel())[0, 1]


def vip(m):
    t, w, q = m.x_scores_, m.x_weights_, m.y_loadings_
    p = w.shape[0]
    ss = (q ** 2).sum(0) * (t ** 2).sum(0)
    return np.sqrt(p * ((w / np.linalg.norm(w, axis=0)) ** 2 @ ss) / ss.sum())


def run(n_perm=None):
    n_perm = n_perm or config.N_PERM_BLOCK
    header("ADVANCED 6  PLS integration (canonical PLS, VIP, multi-block)")
    df = get_dataset()
    gen = rng()
    ihc = ihc_columns()

    # 1 canonical PLS
    d = df[ihc + BEHAVIOUR + ["group"]].dropna()
    X, Y = standardise(d[ihc]), standardise(d[BEHAVIOUR])
    m, cov, cor = comp1(X, Y)
    groups = d.group.astype(str).to_numpy()
    p = {}
    for label, within in [("all mice", False), ("within group", True)]:
        null = []
        for _ in range(n_perm if not within else min(n_perm, 1999)):
            if within:
                idx = within_group_permutation(groups, gen)
            else:
                idx = gen.permutation(len(X))
            null.append(comp1(X, Y[idx])[1])
        p[label] = (1 + np.sum(np.array(null) >= cov)) / (1 + len(null))
    sx = np.sign(m.x_weights_[:, 0].sum())
    xw, yw = m.x_weights_[:, 0] * sx, m.y_weights_[:, 0] * sx
    bx, by = [], []
    for _ in range(config.N_BOOT_PLS):
        i = gen.integers(0, len(X), len(X))
        mb = PLSCanonical(n_components=1, scale=False).fit(standardise(X[i]), standardise(Y[i]))
        s = np.sign(mb.x_weights_[:, 0] @ xw)
        bx.append(mb.x_weights_[:, 0] * s)
        by.append(mb.y_weights_[:, 0] * s)
    bx, by = np.array(bx), np.array(by)
    load = pd.DataFrame({"block": ["IHC"] * len(ihc) + ["behaviour"] * len(BEHAVIOUR),
                         "variable": ihc + [config.LABELS[b] for b in BEHAVIOUR],
                         "weight": np.r_[xw, yw],
                         "95% low": np.r_[np.percentile(bx, 2.5, 0), np.percentile(by, 2.5, 0)],
                         "95% high": np.r_[np.percentile(bx, 97.5, 0), np.percentile(by, 97.5, 0)]})
    load["CI excludes 0"] = (load["95% low"] > 0) | (load["95% high"] < 0)
    summ = pd.DataFrame([{"n": len(d), "comp1 covariance": cov, "comp1 score correlation": cor,
                          "perm p (all mice)": p["all mice"], "perm p (within group)": p["within group"]}])
    log("Canonical PLS IHC <-> behaviour, component 1:\n" + summ.round(4).to_string(index=False))
    log("Weights (bootstrap 95% CI):\n" + load.round(3).to_string(index=False))
    save_table(summ, "adv_pls_canonical_summary")
    save_table(load, "adv_pls_canonical_weights")

    # 2 PLS regression VIP
    d2 = df[ihc + ["Mem12"]].dropna()
    X2, y2 = standardise(d2[ihc]), d2.Mem12.to_numpy(float)
    m2 = PLSRegression(n_components=1, scale=False).fit(X2, y2)
    v = vip(m2)
    vb = []
    for _ in range(config.N_BOOT_PLS):
        i = gen.integers(0, len(X2), len(X2))
        if np.var(y2[i]) > 0:
            vb.append(vip(PLSRegression(n_components=1, scale=False).fit(standardise(X2[i]), y2[i])))
    vb = np.array(vb)
    vt = pd.DataFrame({"stain": ihc, "VIP": v, "VIP 95% low": np.percentile(vb, 2.5, 0),
                       "VIP 95% high": np.percentile(vb, 97.5, 0),
                       "weight sign": np.sign(m2.x_weights_[:, 0] * np.sign(m2.y_loadings_[0, 0]))})
    vt = vt.sort_values("VIP", ascending=False)
    log(f"PLS regression IHC -> memory composite wk12 (n={len(d2)}), VIP:\n" + vt.round(3).to_string(index=False))
    save_table(vt, "adv_pls_vip")

    # 3 multi-block IHC + protein sets -> Mem12, nested LOOCV.
    # One pipeline per model; GridSearchCV (inner LOO) picks the number of
    # components and refits set scores, scaling and block weights in every
    # inner training fold.
    L, _, _ = get_proteomics()
    d3 = df.loc[L.index, ihc + ["Mem12"]].dropna()
    y3 = d3.Mem12.to_numpy(float)
    X3 = pd.concat([d3[ihc], L.loc[d3.index]], axis=1)
    prot = list(L.columns)
    tm = train_mean_predict(y3)
    rows = [{"model": "training mean (baseline)", "n": len(d3), "LOOCV R2": loocv_r2(y3, tm), "MAE": mae(y3, tm),
             "IHC share of comp-1 weight": np.nan}]
    blocks = {"ihc": ("ihc", make_pipeline(StandardScaler(), BlockWeight()), ihc),
              "sets": ("sets", make_pipeline(FoldSetScores(), StandardScaler(), BlockWeight()), prot)}
    for name, use in [("IHC only", ["ihc"]), ("protein sets only", ["sets"]),
                      ("IHC + protein sets (block-scaled)", ["ihc", "sets"])]:
        pipe = make_pipeline(ColumnTransformer([blocks[b] for b in use]), PLS1(scale=False))
        search = GridSearchCV(pipe, {"pls1__n_components": [1, 2]}, cv=LeaveOneOut(), scoring="neg_mean_squared_error")
        pred, share = np.empty(len(d3)), []
        for tr, te in LeaveOneOut().split(X3):
            fit = search.fit(X3.iloc[tr], y3[tr]).best_estimator_
            pred[te] = fit.predict(X3.iloc[te])
            if len(use) == 2:
                w2 = fit[-1].x_weights_[:, 0] ** 2
                share.append(w2[:len(ihc)].sum() / w2.sum())
        rows.append({"model": name, "n": len(d3), "LOOCV R2": loocv_r2(y3, pred), "MAE": mae(y3, pred),
                     "IHC share of comp-1 weight": np.mean(share) if share else np.nan})
    mb = pd.DataFrame(rows)
    log("Multi-block PLS -> memory composite wk12 (nested LOOCV):\n" + mb.round(3).to_string(index=False))
    save_table(mb, "adv_pls_multiblock")
    return summ, load, vt, mb


if __name__ == "__main__":
    run()
