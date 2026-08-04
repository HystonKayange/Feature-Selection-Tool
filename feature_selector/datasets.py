"""Recommended public datasets for feature-selection research.

Most loaders write a tidy CSV under ``data/`` with the **target as the last
column**, ready for ``feature-select`` / ``load_dataset``.

Quick start (no manual download)::

    from feature_selector.datasets import load_benchmark
    X, y, meta = load_benchmark("breast_cancer")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a recommended benchmark dataset."""

    id: str
    name: str
    task: str  # classification | regression
    n_features_approx: int
    n_samples_approx: int
    source: str
    why: str
    difficulty: str  # starter | standard | research
    download: bool  # False = pure sklearn, no network


# Curated list — pick by research goal, not by hype.
DATASET_CATALOG: Dict[str, DatasetInfo] = {
    "breast_cancer": DatasetInfo(
        id="breast_cancer",
        name="Breast Cancer Wisconsin (Diagnostic)",
        task="classification",
        n_features_approx=30,
        n_samples_approx=569,
        source="sklearn",
        why="Best first run: clean numeric features, classic medical binary task.",
        difficulty="starter",
        download=False,
    ),
    "wine": DatasetInfo(
        id="wine",
        name="Wine recognition",
        task="classification",
        n_features_approx=13,
        n_samples_approx=178,
        source="sklearn",
        why="Small multiclass problem; good for smoke tests and demos.",
        difficulty="starter",
        download=False,
    ),
    "diabetes_sklearn": DatasetInfo(
        id="diabetes_sklearn",
        name="Diabetes (sklearn regression)",
        task="regression",
        n_features_approx=10,
        n_samples_approx=442,
        source="sklearn",
        why="Standard regression baseline; few features so selection is subtle.",
        difficulty="starter",
        download=False,
    ),
    "heart_disease": DatasetInfo(
        id="heart_disease",
        name="Heart Disease (Statlog)",
        task="classification",
        n_features_approx=13,
        n_samples_approx=270,
        source="OpenML id=53",
        why="Matches your original research README; mixed clinical predictors.",
        difficulty="standard",
        download=True,
    ),
    "pima_diabetes": DatasetInfo(
        id="pima_diabetes",
        name="Pima Indians Diabetes",
        task="classification",
        n_features_approx=8,
        n_samples_approx=768,
        source="OpenML id=37",
        why="Classic binary medical dataset; zeros may encode missing values.",
        difficulty="standard",
        download=True,
    ),
    "ionosphere": DatasetInfo(
        id="ionosphere",
        name="Ionosphere",
        task="classification",
        n_features_approx=34,
        n_samples_approx=351,
        source="OpenML id=59",
        why="More features than samples ratio is interesting; radar returns.",
        difficulty="standard",
        download=True,
    ),
    "sonar": DatasetInfo(
        id="sonar",
        name="Connectionist Bench (Sonar)",
        task="classification",
        n_features_approx=60,
        n_samples_approx=208,
        source="OpenML id=40",
        why="High feature/sample ratio — feature selection usually helps.",
        difficulty="standard",
        download=True,
    ),
    "wine_quality_red": DatasetInfo(
        id="wine_quality_red",
        name="Wine Quality (red)",
        task="regression",
        n_features_approx=11,
        n_samples_approx=1599,
        source="OpenML id=40691",
        why="Larger regression set; quality scores as continuous target.",
        difficulty="standard",
        download=True,
    ),
    "madelon": DatasetInfo(
        id="madelon",
        name="Madelon (NIPS FS challenge)",
        task="classification",
        n_features_approx=500,
        n_samples_approx=2600,
        source="OpenML id=1485",
        why="Designed for feature-selection research (many probes/noise features).",
        difficulty="research",
        download=True,
    ),
}


def list_datasets(difficulty: Optional[str] = None) -> pd.DataFrame:
    """Return the catalog as a DataFrame for display / docs."""
    rows = [asdict(info) for info in DATASET_CATALOG.values()]
    df = pd.DataFrame(rows)
    if difficulty:
        df = df.loc[df["difficulty"] == difficulty].reset_index(drop=True)
    return df[
        [
            "id",
            "name",
            "task",
            "n_samples_approx",
            "n_features_approx",
            "difficulty",
            "source",
            "why",
            "download",
        ]
    ]


def _default_data_dir() -> Path:
    # Prefer repo-local data/ when running from source; else ~/.cache
    cwd = Path.cwd() / "data"
    return cwd


def _save_xy(
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
    target_name: str = "target",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = y.copy()
    y.name = target_name
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df.to_csv(path, index=False)
    meta = {
        "path": str(path),
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "target": target_name,
        "feature_names": list(X.columns.astype(str)),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _from_sklearn_bunch(bunch, prefix: str = "f") -> Tuple[pd.DataFrame, pd.Series]:
    frame = getattr(bunch, "frame", None)
    if frame is not None:
        df = frame.copy()
        # sklearn frames use 'target' column
        if "target" in df.columns:
            y = df["target"]
            X = df.drop(columns=["target"])
        else:
            y = pd.Series(np.asarray(bunch.target).ravel(), name="target")
            X = df
    else:
        raw_names = getattr(bunch, "feature_names", None)
        if raw_names is None:
            names = [f"{prefix}{i}" for i in range(bunch.data.shape[1])]
        else:
            names = [str(n) for n in list(raw_names)]
        X = pd.DataFrame(bunch.data, columns=names)
        y = pd.Series(np.asarray(bunch.target).ravel(), name="target")
    X.columns = X.columns.astype(str)
    return X, y


def _load_sklearn(dataset_id: str) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

    info = DATASET_CATALOG[dataset_id]
    if dataset_id == "breast_cancer":
        X, y = _from_sklearn_bunch(load_breast_cancer())
    elif dataset_id == "wine":
        X, y = _from_sklearn_bunch(load_wine())
    elif dataset_id == "diabetes_sklearn":
        X, y = _from_sklearn_bunch(load_diabetes())
    else:
        raise KeyError(dataset_id)
    return X, y, info


def _load_openml(dataset_id: str) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    from sklearn.datasets import fetch_openml

    info = DATASET_CATALOG[dataset_id]
    openml_ids = {
        "heart_disease": 53,
        "pima_diabetes": 37,
        "ionosphere": 59,
        "sonar": 40,
        "wine_quality_red": 40691,
        "madelon": 1485,
    }
    data_id = openml_ids[dataset_id]
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = bunch.data.copy()
    y = bunch.target.copy()
    # Flatten multiindex / categorical targets
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = ["_".join(map(str, c)) for c in X.columns]
    X.columns = X.columns.astype(str)
    if hasattr(y, "dtype") and str(y.dtype) == "category":
        # keep codes for binary/multiclass strings if needed later — store as-is
        pass
    y = pd.Series(y.to_numpy(), name="target")
    # Drop rows with missing target
    mask = pd.notna(y)
    X = X.loc[mask.to_numpy()].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y, info


def load_benchmark(
    dataset_id: str,
    *,
    data_dir: Optional[PathLike] = None,
    force_download: bool = False,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
    """Load a recommended dataset as ``(X, y, info)``.

    Parameters
    ----------
    dataset_id :
        Key from :data:`DATASET_CATALOG` (e.g. ``\"breast_cancer\"``, ``\"sonar\"``).
    data_dir :
        Where to cache CSV files (default: ``./data``).
    force_download :
        Re-fetch even if a cached CSV exists.
    save :
        Write ``{id}.csv`` under ``data_dir`` for CLI use.
    """
    if dataset_id not in DATASET_CATALOG:
        known = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(f"Unknown dataset '{dataset_id}'. Choose from: {known}")

    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    csv_path = data_dir / f"{dataset_id}.csv"

    if csv_path.is_file() and not force_download:
        df = pd.read_csv(csv_path)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        X.columns = X.columns.astype(str)
        return X, y, DATASET_CATALOG[dataset_id]

    info = DATASET_CATALOG[dataset_id]
    if info.source == "sklearn":
        X, y, info = _load_sklearn(dataset_id)
    else:
        X, y, info = _load_openml(dataset_id)

    # Light cleanup: replace infinite
    X = X.replace([np.inf, -np.inf], np.nan)
    if save:
        _save_xy(X, y, csv_path)
    return X, y, info


def download_datasets(
    ids: Optional[List[str]] = None,
    *,
    data_dir: Optional[PathLike] = None,
    force: bool = False,
    include_research: bool = False,
) -> Dict[str, Path]:
    """Download/cache recommended datasets as CSV files.

    Parameters
    ----------
    ids :
        Dataset ids to fetch. ``None`` = all starter+standard (not madelon unless
        ``include_research=True``).
    include_research :
        When ``ids`` is None, also fetch large research sets (e.g. Madelon).
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    if ids is None:
        ids = [
            d.id
            for d in DATASET_CATALOG.values()
            if d.difficulty != "research" or include_research
        ]

    written: Dict[str, Path] = {}
    for dataset_id in ids:
        X, y, _ = load_benchmark(
            dataset_id, data_dir=data_dir, force_download=force, save=True
        )
        path = data_dir / f"{dataset_id}.csv"
        written[dataset_id] = path
        print(
            f"  ✓ {dataset_id:18s}  →  {path}  "
            f"({len(X)} samples × {X.shape[1]} features)"
        )
    return written


def recommend_for_goal(goal: str) -> List[DatasetInfo]:
    """Suggest datasets for a research goal keyword."""
    goal = goal.lower()
    mapping = {
        "first": ["breast_cancer", "wine", "diabetes_sklearn"],
        "starter": ["breast_cancer", "wine", "diabetes_sklearn"],
        "medical": ["breast_cancer", "heart_disease", "pima_diabetes"],
        "high-dim": ["sonar", "ionosphere", "madelon"],
        "highdim": ["sonar", "ionosphere", "madelon"],
        "regression": ["diabetes_sklearn", "wine_quality_red"],
        "paper": ["breast_cancer", "heart_disease", "sonar", "madelon"],
        "research": ["sonar", "ionosphere", "madelon", "heart_disease"],
    }
    # fuzzy match
    keys = [k for k in mapping if k in goal]
    if not keys:
        keys = ["first"]
    seen = []
    for k in keys:
        for i in mapping[k]:
            if i not in seen:
                seen.append(i)
    return [DATASET_CATALOG[i] for i in seen]
