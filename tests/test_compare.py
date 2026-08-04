"""Comparison, stability, evaluation, and preprocess tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_selector.compare import compare_methods
from feature_selector.evaluate import evaluate_before_after
from feature_selector.preprocess import (
    drop_constant_features,
    drop_correlated_features,
    outlier_report,
)
from feature_selector.stability import pairwise_jaccard, stability_selection


@pytest.fixture
def clf_frame():
    X, y = make_classification(
        n_samples=100,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        random_state=1,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    # add a constant + a near-duplicate
    df["const"] = 1.0
    df["f0_dup"] = df["f0"] + 1e-8
    return df, pd.Series(y, name="y")


def test_drop_constant_and_correlated(clf_frame):
    X, _ = clf_frame
    X2, dropped = drop_constant_features(X)
    assert "const" in dropped
    X3, dropped_c, pairs = drop_correlated_features(X2, threshold=0.99)
    assert len(dropped_c) >= 1
    assert X3.shape[1] < X2.shape[1]


def test_outlier_report(clf_frame):
    X, _ = clf_frame
    rep = outlier_report(X)
    assert "outlier_rate" in rep.columns
    assert len(rep) >= 1


def test_evaluate_before_after(clf_frame):
    X, y = clf_frame
    # pick a few informative-looking columns
    selected = list(X.columns[:4])
    result = evaluate_before_after(
        X.drop(columns=["const"]), y, selected, task="classification", cv=3, random_state=0
    )
    assert "all_features" in result and "selected" in result
    assert "accuracy" in result["selected"]
    assert result["n_features_selected"] == 4


def test_stability_selection(clf_frame):
    X, y = clf_frame
    X = X.drop(columns=["const"])
    stab = stability_selection(
        X, y, method="anova", k=4, task="classification", n_splits=3, random_state=0
    )
    assert stab.n_splits == 3
    assert len(stab.fold_selections) == 3
    assert 0 <= stab.mean_stability <= 1
    assert stab.frequencies["frequency"].max() <= 1.0


def test_pairwise_jaccard():
    j = pairwise_jaccard({"a": ["f1", "f2"], "b": ["f2", "f3"], "c": ["f1", "f2"]})
    assert j.loc["a", "c"] == 1.0
    assert 0 < j.loc["a", "b"] < 1


def test_compare_methods_export(clf_frame, tmp_path: Path):
    X, y = clf_frame
    X = X.drop(columns=["const"])
    result = compare_methods(
        X,
        y,
        methods=["anova", "random_forest", "lasso"],
        k=4,
        task="classification",
        cv=3,
        stability_splits=3,
        random_state=0,
        run_stability=True,
    )
    assert len(result.selections) == 3
    assert "method" in result.summary.columns
    assert result.jaccard.shape == (3, 3)

    written = result.export(tmp_path / "cmp")
    assert (tmp_path / "cmp" / "comparison_summary.csv").is_file()
    assert (tmp_path / "cmp" / "jaccard.csv").is_file()
    assert (tmp_path / "cmp" / "comparison_report.html").is_file()
    assert "report" in written


def test_cli_compare(clf_frame, tmp_path: Path):
    X, y = clf_frame
    df = X.copy()
    df["target"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out"
    from feature_selector.cli import main

    code = main(
        [
            "compare",
            str(path),
            "-k",
            "3",
            "--methods",
            "anova,rf",
            "--cv",
            "3",
            "--stability-splits",
            "3",
            "--out",
            str(out),
            "-q",
        ]
    )
    assert code == 0
    assert (out / "comparison_summary.csv").is_file()
