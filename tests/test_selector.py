"""Tests for leakage-safe FeatureSelector behaviour."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import f_classif, f_regression
from sklearn.model_selection import train_test_split

from feature_selector.selector import FeatureSelector, SCORE_FUNCS


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        random_state=0,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


@pytest.fixture
def reg_data():
    X, y = make_regression(
        n_samples=200, n_features=8, n_informative=3, random_state=0
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def test_classification_uses_f_classif(clf_data):
    X, y = clf_data
    sel = FeatureSelector(k=3, task="classification", score="f_score")
    assert SCORE_FUNCS[(sel.task, sel.score)] is f_classif
    sel.fit(X, y)
    assert len(sel.feature_names_out_) == 3
    assert sel.scores_ is not None
    assert len(sel.scores_) == X.shape[1]


def test_regression_uses_f_regression(reg_data):
    X, y = reg_data
    sel = FeatureSelector(k=3, task="regression", score="f_score")
    assert SCORE_FUNCS[(sel.task, sel.score)] is f_regression
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape == (len(X), 3)
    assert list(X_sel.columns) == sel.feature_names_out_


def test_preserves_feature_names(clf_data):
    X, y = clf_data
    sel = FeatureSelector(k=2, task="classification")
    X_sel = sel.fit_transform(X, y)
    for name in X_sel.columns:
        assert name in X.columns


def test_transform_matches_fit_columns(clf_data):
    X, y = clf_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )
    sel = FeatureSelector(k=4, task="classification")
    X_tr = sel.fit_transform(X_train, y_train)
    X_te = sel.transform(X_test)
    assert list(X_tr.columns) == list(X_te.columns)
    assert X_te.shape[1] == 4
    assert X_te.shape[0] == len(X_test)


def test_does_not_impute_target():
    """Rows with missing y are dropped during fit, not mean-filled."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.5, 0.5, 1.5, 1.5]})
    y = pd.Series([0.0, np.nan, 1.0, 0.0])
    sel = FeatureSelector(k=1, task="classification")
    X_out = sel.fit_transform(X, y)
    # 3 rows with valid labels
    assert len(X_out) == 3


def test_imputer_fit_on_train_only():
    """Transforming test data with unseen NaNs should not refit imputer."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 2.0]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"a": [np.nan], "b": [1.0]})
    sel = FeatureSelector(k=1, task="classification")
    sel.fit(X_train, y_train)
    out = sel.transform(X_test)
    assert out.shape == (1, 1)
    assert not out.isna().any().any()


def test_missing_feature_imputation_and_categoricals():
    X = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", None, "b"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0])
    sel = FeatureSelector(k=2, task="classification")
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape == (5, 2)
    assert not X_sel.isna().any().any()


def test_k_none_keeps_all(clf_data):
    X, y = clf_data
    sel = FeatureSelector(k=None, task="classification")
    X_sel = sel.fit_transform(X, y)
    assert X_sel.shape[1] == X.shape[1]


def test_get_feature_scores_sorted(clf_data):
    X, y = clf_data
    sel = FeatureSelector(k=3, task="classification")
    sel.fit(X, y)
    table = sel.get_feature_scores()
    assert "feature" in table.columns and "score" in table.columns
    assert table["selected"].sum() == 3
    scores = table["score"].dropna().to_numpy()
    assert np.all(scores[:-1] >= scores[1:] - 1e-12)


def test_plot_importance_name_score_alignment(clf_data):
    X, y = clf_data
    sel = FeatureSelector(k=3, task="classification")
    sel.fit(X, y)
    # Must not raise when names and scores align with selected set
    importances = np.array([0.5, 0.3, 0.2])
    sel.plot_feature_importance(
        feature_names=sel.feature_names_out_,
        feature_scores=importances,
        show=False,
    )


def test_check_correlation_pairs():
    X = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5.0],
            "b": [1, 2, 3, 4, 5.0],  # perfect corr with a
            "c": [5, 4, 3, 2, 1.0],
        }
    )
    sel = FeatureSelector(k=2, task="regression")
    pairs = sel.check_correlation(X, threshold=0.9)
    assert len(pairs) >= 1
    dropped = sel.drop_correlated(X, threshold=0.9)
    assert "a" in dropped.columns or "b" in dropped.columns
    assert dropped.shape[1] < X.shape[1]


def test_invalid_task():
    with pytest.raises(ValueError):
        FeatureSelector(task="clustering")


def test_unfitted_transform_raises(clf_data):
    X, _ = clf_data
    sel = FeatureSelector(k=2, task="classification")
    with pytest.raises(RuntimeError):
        sel.transform(X)
