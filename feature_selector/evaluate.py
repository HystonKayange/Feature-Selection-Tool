"""Before/after and cross-validated evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_selector.data import infer_task


ArrayLike = Union[pd.DataFrame, np.ndarray]


def resolve_task(y, task: str = "auto") -> str:
    if task == "auto":
        if isinstance(y, pd.Series):
            return infer_task(y)
        return infer_task(pd.Series(np.asarray(y).ravel()))
    if task in {"classification", "regression"}:
        return task
    from feature_selector.selector import normalize_task

    t = normalize_task(task)
    if t == "auto":
        return resolve_task(y, "auto")
    return t


def make_baseline_estimator(task: str, random_state: int = 42) -> Pipeline:
    """Simple, strong-enough baseline used for research comparisons."""
    if task == "classification":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=random_state),
                ),
            ]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _scoring_for_task(task: str) -> Dict[str, str]:
    if task == "classification":
        return {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
        }
    return {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }


def _cv_splitter(task: str, n_splits: int, random_state: int, y):
    if task == "classification":
        y_arr = np.asarray(y).ravel()
        # Stratified needs at least n_splits samples per class ideally
        try:
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        except Exception:
            return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_val_metrics(
    X: ArrayLike,
    y,
    *,
    task: str = "auto",
    cv: int = 5,
    random_state: int = 42,
    estimator: Optional[Pipeline] = None,
) -> Dict[str, float]:
    """Cross-validated metrics for a feature matrix."""
    task_r = resolve_task(y, task)
    if isinstance(X, pd.DataFrame):
        X_use = X.copy()
        # numeric-only for baseline linear models; encode objects lightly
        obj_cols = X_use.select_dtypes(exclude=[np.number]).columns
        if len(obj_cols):
            X_use = pd.get_dummies(X_use, columns=list(obj_cols), drop_first=True)
        X_arr = X_use.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)

    y_arr = np.asarray(y).ravel()
    est = estimator or make_baseline_estimator(task_r, random_state=random_state)
    scoring = _scoring_for_task(task_r)
    splitter = _cv_splitter(task_r, cv, random_state, y_arr)

    try:
        results = cross_validate(
            clone(est),
            X_arr,
            y_arr,
            cv=splitter,
            scoring=scoring,
            n_jobs=None,
            error_score="raise",
        )
    except ValueError:
        # Fall back to non-stratified if needed
        results = cross_validate(
            clone(est),
            X_arr,
            y_arr,
            cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
            scoring=scoring,
            n_jobs=None,
            error_score="raise",
        )

    metrics: Dict[str, float] = {}
    for key, values in results.items():
        if not key.startswith("test_"):
            continue
        name = key[len("test_") :]
        mean = float(np.mean(values))
        if name.startswith("neg_"):
            metrics[name[4:]] = -mean  # mse, mae as positive
        else:
            metrics[name] = mean
    metrics["cv_folds"] = float(cv)
    return metrics


def evaluate_before_after(
    X: ArrayLike,
    y,
    selected_features: Sequence[str],
    *,
    task: str = "auto",
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compare baseline CV metrics with all features vs selected subset.

    ``X`` must be a DataFrame (or convertible) with named columns so that
    ``selected_features`` can be applied.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            np.asarray(X),
            columns=[f"feature_{i}" for i in range(np.asarray(X).shape[1])],
        )
    else:
        X = X.copy()
        X.columns = X.columns.astype(str)

    missing = [f for f in selected_features if f not in X.columns]
    if missing:
        raise ValueError(f"selected_features not in X: {missing}")

    task_r = resolve_task(y, task)
    metrics_all = cross_val_metrics(
        X, y, task=task_r, cv=cv, random_state=random_state
    )
    metrics_sel = cross_val_metrics(
        X.loc[:, list(selected_features)],
        y,
        task=task_r,
        cv=cv,
        random_state=random_state,
    )

    delta = {}
    for key in metrics_sel:
        if key == "cv_folds":
            continue
        if key in metrics_all and isinstance(metrics_all[key], (int, float)):
            delta[key] = float(metrics_sel[key] - metrics_all[key])

    return {
        "task": task_r,
        "n_features_all": int(X.shape[1]),
        "n_features_selected": len(selected_features),
        "selected_features": list(selected_features),
        "all_features": metrics_all,
        "selected": metrics_sel,
        "delta_selected_minus_all": delta,
    }


def metrics_to_row(method: str, before_after: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a before/after dict into one comparison-table row."""
    row: Dict[str, Any] = {
        "method": method,
        "n_selected": before_after["n_features_selected"],
        "n_all": before_after["n_features_all"],
    }
    for split_name, prefix in (
        ("all_features", "all"),
        ("selected", "sel"),
    ):
        block = before_after[split_name]
        for k, v in block.items():
            if k == "cv_folds":
                continue
            row[f"{prefix}_{k}"] = v
    for k, v in before_after["delta_selected_minus_all"].items():
        row[f"delta_{k}"] = v
    return row
