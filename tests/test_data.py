"""Tests for dataset loading."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_selector.data import TabularDataset, infer_task, load_dataset


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": ["x", "y", "x", "y"],
            "target": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


def test_load_dataset_defaults(csv_path: Path):
    ds = load_dataset(csv_path)
    assert isinstance(ds, TabularDataset)
    assert ds.n_samples == 4
    assert ds.n_features == 2
    assert list(ds.features.columns) == ["a", "b"]
    assert ds.labels.name == "target"
    # Categoricals are left non-numeric — selector encodes after split
    assert not pd.api.types.is_numeric_dtype(ds.features["b"])


def test_load_dataset_target_by_name(csv_path: Path):
    ds = load_dataset(csv_path, target_column="target")
    assert ds.n_features == 2
    assert (ds.labels == [0, 1, 0, 1]).all()


def test_drop_missing_labels(tmp_path: Path):
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "y": [0.0, np.nan, 1.0],
        }
    )
    path = tmp_path / "miss.csv"
    df.to_csv(path, index=False)
    ds = load_dataset(path, drop_missing_labels=True)
    assert ds.n_samples == 2
    assert not ds.labels.isna().any()


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(path)


def test_infer_task():
    assert infer_task(pd.Series(["a", "b", "a"])) == "classification"
    assert infer_task(pd.Series([0, 1, 0, 1, 0, 1])) == "classification"
    assert infer_task(pd.Series(np.linspace(0, 1, 50))) == "regression"
