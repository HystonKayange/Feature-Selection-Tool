"""End-to-end pipeline: split first, then select, then evaluate."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from feature_selector.app import run_pipeline
from feature_selector.data import load_dataset
from feature_selector.selector import FeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def test_run_pipeline_no_plots(tmp_path: Path):
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=3, random_state=7
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    path = tmp_path / "clf.csv"
    df.to_csv(path, index=False)

    result = run_pipeline(
        str(path),
        task="classification",
        k=3,
        show_plots=False,
    )
    assert result["task"] == "classification"
    assert result["k"] == 3
    assert len(result["selected_features"]) == 3
    assert "accuracy" in result["metrics"]


def test_manual_leakage_safe_flow(tmp_path: Path):
    """Documented correct usage: split → fit selector on train → transform both."""
    X, y = make_classification(
        n_samples=150, n_features=8, n_informative=3, random_state=3
    )
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(8)])
    df["label"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    ds = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.features, ds.labels, test_size=0.2, random_state=0, stratify=ds.labels
    )
    sel = FeatureSelector(k=3, task="classification")
    X_tr = sel.fit_transform(X_train, y_train)
    X_te = sel.transform(X_test)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_tr_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te_s))
    assert 0.0 <= acc <= 1.0
    assert X_tr.shape[1] == X_te.shape[1] == 3
