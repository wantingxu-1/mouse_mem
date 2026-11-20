"""Inner-loop isolation: in a tuned search, preprocessing is refitted on each inner training fold only."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from features.dataset import get_dataset, ihc_columns
from features.ihc_indices import FoldIHCIndices
from ml.nested_cv import mae, train_mean_predict, tuned
from utils.stats_helpers import loocv_r2


def test_tuned_search_wraps_every_fitted_step():
    s = tuned(Ridge(), {"alpha": [1.0]}, FoldIHCIndices())
    assert isinstance(s, GridSearchCV)
    assert [n for n, _ in s.estimator.steps] == ["foldihcindices", "simpleimputer", "standardscaler", "ridge"]
    assert list(s.param_grid) == ["ridge__alpha"]


def test_inner_scores_use_training_fold_preprocessing_only():
    """The search's inner score equals a manual loop where indices, imputer and scaler see only inner-train mice."""
    d = get_dataset()[ihc_columns() + ["Mem12"]].dropna()
    X, y = d[ihc_columns()], d.Mem12.to_numpy(float)
    cv = KFold(5, shuffle=True, random_state=0)
    s = tuned(Ridge(), {"alpha": [3.0]}, FoldIHCIndices(), cv=cv).fit(X, y)
    manual = []
    for tr, va in cv.split(X):
        t = FoldIHCIndices().fit(X.iloc[tr])
        imp = SimpleImputer(strategy="median").fit(t.transform(X.iloc[tr]))
        sc = StandardScaler().fit(imp.transform(t.transform(X.iloc[tr])))
        f = lambda rows: sc.transform(imp.transform(t.transform(X.iloc[rows])))
        m = Ridge(alpha=3.0).fit(f(tr), y[tr])
        manual.append(-mean_squared_error(y[va], m.predict(f(va))))
    assert np.isclose(s.cv_results_["mean_test_score"][0], np.mean(manual))


def test_training_mean_baseline():
    y = np.arange(21, dtype=float)
    assert np.isclose(loocv_r2(y, train_mean_predict(y)), 1 - (21 / 20) ** 2)
    assert mae([1, 2], [2, 4]) == 1.5


def test_detection_filter_uses_training_mice_only():
    """FoldDetectionFilter keeps a protein by training-mouse detection, threshold scaled to group size."""
    import numpy as np
    import pandas as pd
    from features.proteomics_prep import FoldDetectionFilter
    idx = [f"m{i}" for i in range(12)]
    g = pd.Series(["A"] * 6 + ["B"] * 6, index=idx)
    X = pd.DataFrame({"p": [1.0] * 4 + [np.nan] * 2 + [1.0] * 6,          # 4/6 in A -> kept
                      "q": [1.0] * 3 + [np.nan] * 3 + [1.0] * 6}, index=idx)  # 3/6 in A -> dropped
    assert FoldDetectionFilter(g).fit(X).columns_ == ["p"]
    # leave out a detected A mouse: 3 of 5 left, threshold ceil(4/6*5)=4 -> p dropped
    assert FoldDetectionFilter(g).fit(X.drop(index="m0")).columns_ == []
