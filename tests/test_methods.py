"""Multi-method selector tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from feature_selector.selector import FeatureSelector, normalize_method


@pytest.fixture
def clf():
    X, y = make_classification(
        n_samples=120, n_features=10, n_informative=4, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def reg():
    X, y = make_regression(n_samples=120, n_features=8, n_informative=3, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y)


@pytest.mark.parametrize(
    "method",
    ["anova", "mutual_info", "random_forest", "lasso", "rfe", "filter"],
)
def test_methods_classification(clf, method):
    X, y = clf
    sel = FeatureSelector(k=4, task="classification", method=method, random_state=0)
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape == (len(X), 4)
    assert len(sel.selected_features_) == 4
    assert sel.scores_ is not None
    assert len(sel.scores_) == X.shape[1]


@pytest.mark.parametrize("method", ["anova", "random_forest", "lasso", "rfe"])
def test_methods_regression(reg, method):
    X, y = reg
    sel = FeatureSelector(k=3, task="regression", method=method, random_state=0)
    out = sel.fit_transform(X, y)
    assert out.shape[1] == 3
    # transform on holdout-shaped data
    out2 = sel.transform(X.iloc[:10])
    assert list(out2.columns) == sel.selected_features_


def test_normalize_method_aliases():
    assert normalize_method("rf") == ("random_forest", None)
    assert normalize_method("anova") == ("filter", "f_score")
    assert normalize_method("mi") == ("filter", "mutual_info")
    with pytest.raises(ValueError):
        normalize_method("boruta")


def test_method_scores_align_with_names(clf):
    X, y = clf
    sel = FeatureSelector(k=3, method="random_forest", task="classification")
    sel.fit(X, y)
    table = sel.get_feature_scores()
    assert table["selected"].sum() == 3
    assert set(table.loc[table["selected"], "feature"]) == set(sel.selected_features_)
