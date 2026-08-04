"""Dataset loading and light preprocessing for tabular CSV/TXT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class TabularDataset:
    """In-memory tabular dataset with features and labels.

    Parameters
    ----------
    features : pd.DataFrame
        Input feature matrix (rows = samples).
    labels : pd.Series
        Target vector aligned with ``features``.
    source_path : str or None
        Optional path the data was loaded from.
    """

    features: pd.DataFrame
    labels: pd.Series
    source_path: Optional[str] = None

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.features.iloc[index].to_numpy(), self.labels.iloc[index]


def load_dataset(
    path: PathLike,
    *,
    delimiter: str = ",",
    header: bool = True,
    target_column: Optional[Union[str, int]] = None,
    drop_missing_labels: bool = True,
) -> TabularDataset:
    """Load a CSV or TXT file into a :class:`TabularDataset`.

    By default the **last column** is treated as the target. Categorical
    columns are left as-is so that :class:`~feature_selector.selector.FeatureSelector`
    can encode them after the train/test split (avoids leakage and double encoding).

    Parameters
    ----------
    path :
        Path to a ``.csv`` or ``.txt`` file.
    delimiter :
        Field delimiter passed to ``pandas.read_csv``.
    header :
        If True, the first row is column names; otherwise integer columns.
    target_column :
        Column name or integer position of the target. ``None`` means last column.
    drop_missing_labels :
        If True, rows with a missing target are removed (recommended).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Please provide a CSV or TXT file."
        )

    data = pd.read_csv(
        path,
        header=0 if header else None,
        delimiter=delimiter,
    )
    if data.empty:
        raise ValueError(f"Dataset at '{path}' is empty.")

    if target_column is None:
        y = data.iloc[:, -1].copy()
        X = data.iloc[:, :-1].copy()
    elif isinstance(target_column, int):
        y = data.iloc[:, target_column].copy()
        X = data.drop(data.columns[target_column], axis=1).copy()
    else:
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(data.columns)}"
            )
        y = data[target_column].copy()
        X = data.drop(columns=[target_column]).copy()

    # Stable string feature names for plotting / export
    X.columns = X.columns.astype(str)
    y.name = str(y.name) if y.name is not None else "target"

    if drop_missing_labels:
        valid = y.notna()
        n_dropped = int((~valid).sum())
        if n_dropped:
            X = X.loc[valid].reset_index(drop=True)
            y = y.loc[valid].reset_index(drop=True)
        else:
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    if len(X) == 0:
        raise ValueError("No samples left after dropping missing labels.")

    return TabularDataset(features=X, labels=y, source_path=str(path))


def infer_task(y: pd.Series) -> str:
    """Heuristic task inference: classification vs regression.

    - Non-numeric targets → classification
    - Integer-like with few unique values → classification
    - Otherwise → regression
    """
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    n_unique = y.nunique(dropna=True)
    n = len(y)
    # Classic heuristic used by many AutoML tools
    if pd.api.types.is_integer_dtype(y) and n_unique <= max(10, int(0.05 * n)):
        return "classification"
    if n_unique <= 10 and n_unique / max(n, 1) < 0.05:
        return "classification"
    return "regression"
