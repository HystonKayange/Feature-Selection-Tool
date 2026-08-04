"""Next-level polish: sklearn API, MI seeding, chi2, fast mode."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_selector.selector import FeatureSelector, make_l1_logistic


@pytest.fixture
def clf():
    X, y = make_classification(
        n_samples=120, n_features=12, n_informative=4, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]), pd.Series(y)


def test_l1_logistic_no_future_warning():
    X, y = make_classification(n_samples=80, n_features=6, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", UserWarning)
        model = make_l1_logistic(random_state=0, max_iter=800)
        model.fit(X, y)
    assert model.coef_.shape[1] == 6


def test_lasso_method_no_penalty_warning(clf):
    X, y = clf
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", UserWarning)
        sel = FeatureSelector(
            k=4, method="lasso", task="classification", random_state=0, fast=True
        )
        out = sel.fit_transform(X, y)
    assert out.shape == (len(X), 4)


def test_mutual_info_seeded_reproducible(clf):
    X, y = clf
    a = FeatureSelector(k=5, method="mutual_info", task="classification", random_state=7)
    b = FeatureSelector(k=5, method="mutual_info", task="classification", random_state=7)
    a.fit(X, y)
    b.fit(X, y)
    assert a.selected_features_ == b.selected_features_
    np.testing.assert_allclose(a.scores_, b.scores_, rtol=1e-6, atol=1e-6)


def test_chi2_classification(clf):
    X, y = clf
    # shift some negatives — chi2 path should still work
    X = X - 2.0
    sel = FeatureSelector(k=4, method="chi2", task="classification", random_state=0)
    out = sel.fit_transform(X, y)
    assert out.shape[1] == 4
    assert sel.scores_ is not None


def test_chi2_rejects_regression(clf):
    X, y = clf
    y_reg = pd.Series(np.linspace(0, 1, len(y)))
    sel = FeatureSelector(k=3, method="chi2", task="regression")
    with pytest.raises(ValueError, match="chi2"):
        sel.fit(X, y_reg)


def test_fast_mode_fewer_estimators(clf):
    X, y = clf
    sel = FeatureSelector(k=3, method="rf", task="classification", fast=True)
    assert sel.n_estimators == 50
    out = sel.fit_transform(X, y)
    assert out.shape[1] == 3


def test_nested_cv_fast_flag(clf):
    from feature_selector.nested_cv import nested_cv_feature_selection

    X, y = clf
    result = nested_cv_feature_selection(
        X, y, method="anova", k=3, outer_splits=3, fast=True, random_state=0
    )
    assert "accuracy" in result["mean_metrics"]
