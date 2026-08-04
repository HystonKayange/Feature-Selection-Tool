"""Dataset catalog and nested CV tests."""

from pathlib import Path

import pytest

from feature_selector.datasets import (
    DATASET_CATALOG,
    list_datasets,
    load_benchmark,
    recommend_for_goal,
)
from feature_selector.nested_cv import (
    nested_cv_compare_methods,
    nested_cv_feature_selection,
)
from feature_selector.cli import main


def test_catalog_nonempty():
    assert "breast_cancer" in DATASET_CATALOG
    df = list_datasets()
    assert len(df) >= 5
    assert set(df["difficulty"]) <= {"starter", "standard", "research"}


def test_recommend_goals():
    med = recommend_for_goal("medical")
    assert any(d.id == "heart_disease" for d in med)
    first = recommend_for_goal("first run")
    assert any(d.id == "breast_cancer" for d in first)


def test_load_breast_cancer(tmp_path: Path):
    X, y, info = load_benchmark(
        "breast_cancer", data_dir=tmp_path, force_download=True, save=True
    )
    assert X.shape[0] == len(y)
    assert X.shape[1] == 30
    assert info.task == "classification"
    assert (tmp_path / "breast_cancer.csv").is_file()


def test_load_cached(tmp_path: Path):
    load_benchmark("wine", data_dir=tmp_path, save=True)
    X, y, _ = load_benchmark("wine", data_dir=tmp_path, force_download=False)
    assert X.shape[1] == 13


def test_nested_cv_breast(tmp_path: Path):
    X, y, _ = load_benchmark("breast_cancer", data_dir=tmp_path, save=False)
    result = nested_cv_feature_selection(
        X, y, method="anova", k=5, task="classification", outer_splits=3, random_state=0
    )
    assert "accuracy" in result["mean_metrics"]
    assert len(result["fold_selections"]) == 3
    assert result["selection_frequency"]["frequency"].max() <= 1.0


def test_nested_cv_compare(tmp_path: Path):
    X, y, _ = load_benchmark("wine", data_dir=tmp_path, save=False)
    table = nested_cv_compare_methods(
        X, y, methods=["anova", "rf"], k=4, outer_splits=3, random_state=0
    )
    assert len(table) == 2
    assert "mean_accuracy" in table.columns


def test_cli_datasets_list():
    assert main(["datasets", "list"]) == 0


def test_cli_datasets_download_sklearn(tmp_path: Path):
    code = main(
        [
            "datasets",
            "download",
            "breast_cancer",
            "wine",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "breast_cancer.csv").is_file()
    assert (tmp_path / "wine.csv").is_file()


def test_cli_nested_cv(tmp_path: Path):
    data_dir = tmp_path / "data"
    load_benchmark("breast_cancer", data_dir=data_dir, save=True)
    out = tmp_path / "ncv"
    code = main(
        [
            "nested-cv",
            str(data_dir / "breast_cancer.csv"),
            "-k",
            "5",
            "--method",
            "anova",
            "--cv",
            "3",
            "--out",
            str(out),
            "-q",
        ]
    )
    assert code == 0
    assert (out / "nested_cv_result.json").is_file()
    assert (out / "selection_frequency.csv").is_file()
