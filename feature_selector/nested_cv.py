"""Nested cross-validation for unbiased selection + evaluation.

Outer loop estimates generalization; inner training fold fits the selector
so feature choice never peeks at the outer test fold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from feature_selector.evaluate import make_baseline_estimator, resolve_task
from feature_selector.selector import FeatureSelector, normalize_method


ArrayLike = Union[pd.DataFrame, np.ndarray]


def nested_cv_feature_selection(
    X: ArrayLike,
    y,
    *,
    method: str = "anova",
    k: int = 10,
    task: str = "auto",
    outer_splits: int = 5,
    random_state: int = 42,
    score: Optional[str] = None,
) -> Dict[str, Any]:
    """Nested CV: select features on each outer train fold, score on outer test.

    Returns
    -------
    dict with
      - task
      - method
      - k
      - fold_metrics : list of per-fold metric dicts
      - mean_metrics / std_metrics
      - fold_selections : selected features per outer fold
      - selection_frequency : DataFrame of how often each feature was chosen
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(
            np.asarray(X),
            columns=[f"feature_{i}" for i in range(np.asarray(X).shape[1])],
        )
    else:
        X = X.copy()
        X.columns = X.columns.astype(str)

    y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y).ravel())
    y_series = y_series.reset_index(drop=True)
    X = X.reset_index(drop=True)
    mask = y_series.notna()
    X = X.loc[mask.to_numpy()].reset_index(drop=True)
    y_series = y_series.loc[mask].reset_index(drop=True)

    task_r = resolve_task(y_series, task)
    method_c, score_override = normalize_method(method)
    score_c = score_override or score or "f_score"
    k_eff = min(int(k), X.shape[1])

    if task_r == "classification":
        try:
            outer = StratifiedKFold(
                n_splits=outer_splits, shuffle=True, random_state=random_state
            )
            splits = list(outer.split(X, y_series))
        except ValueError:
            outer = KFold(
                n_splits=outer_splits, shuffle=True, random_state=random_state
            )
            splits = list(outer.split(X))
    else:
        outer = KFold(
            n_splits=outer_splits, shuffle=True, random_state=random_state
        )
        splits = list(outer.split(X))

    fold_metrics: List[Dict[str, float]] = []
    fold_selections: List[List[str]] = []
    counts = {c: 0 for c in X.columns}

    for fold_id, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y_series.iloc[train_idx], y_series.iloc[test_idx]

        sel = FeatureSelector(
            k=k_eff,
            task=task_r,
            score=score_c,
            method=method_c,
            random_state=random_state + fold_id,
        )
        X_tr_s = sel.fit_transform(X_tr, y_tr)
        X_te_s = sel.transform(X_te)
        chosen = list(sel.selected_features_)
        fold_selections.append(chosen)
        for name in chosen:
            counts[name] = counts.get(name, 0) + 1

        est = make_baseline_estimator(task_r, random_state=random_state)
        # Align y with fit_transform drops (missing labels already cleaned)
        est.fit(X_tr_s, y_tr.to_numpy())
        pred = est.predict(X_te_s)

        metrics: Dict[str, float] = {"fold": float(fold_id)}
        if task_r == "classification":
            from sklearn.metrics import accuracy_score, f1_score

            metrics["accuracy"] = float(accuracy_score(y_te, pred))
            metrics["f1_weighted"] = float(
                f1_score(y_te, pred, average="weighted", zero_division=0)
            )
        else:
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            metrics["r2"] = float(r2_score(y_te, pred))
            metrics["mse"] = float(mean_squared_error(y_te, pred))
            metrics["mae"] = float(mean_absolute_error(y_te, pred))
        fold_metrics.append(metrics)

    metric_keys = [k for k in fold_metrics[0] if k != "fold"]
    mean_metrics = {
        k: float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys
    }
    std_metrics = {
        k: float(np.std([m[k] for m in fold_metrics], ddof=1))
        if len(fold_metrics) > 1
        else 0.0
        for k in metric_keys
    }

    freq = (
        pd.DataFrame(
            {
                "feature": list(counts.keys()),
                "n_selected": list(counts.values()),
                "frequency": [c / outer_splits for c in counts.values()],
            }
        )
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "task": task_r,
        "method": method if method_c == "filter" else method_c,
        "canonical_method": method_c,
        "score": score_c,
        "k": k_eff,
        "outer_splits": outer_splits,
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "fold_selections": fold_selections,
        "selection_frequency": freq,
    }


def nested_cv_compare_methods(
    X: ArrayLike,
    y,
    methods: Optional[Sequence[str]] = None,
    *,
    k: int = 10,
    task: str = "auto",
    outer_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run nested CV for several methods; return a summary table."""
    from feature_selector.compare import DEFAULT_METHODS

    methods = list(methods) if methods is not None else list(DEFAULT_METHODS)
    rows = []
    for m in methods:
        result = nested_cv_feature_selection(
            X,
            y,
            method=m,
            k=k,
            task=task,
            outer_splits=outer_splits,
            random_state=random_state,
        )
        row: Dict[str, Any] = {
            "method": m,
            "k": result["k"],
            "task": result["task"],
            "outer_splits": outer_splits,
        }
        for key, val in result["mean_metrics"].items():
            row[f"mean_{key}"] = val
            row[f"std_{key}"] = result["std_metrics"][key]
        # stability proxy: mean frequency of features selected at least once
        freq = result["selection_frequency"]
        positive = freq.loc[freq["frequency"] > 0, "frequency"]
        row["mean_selection_frequency"] = (
            float(positive.mean()) if len(positive) else 0.0
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    for col in ("mean_accuracy", "mean_r2", "mean_f1_weighted"):
        if col in summary.columns:
            return summary.sort_values(col, ascending=False).reset_index(drop=True)
    return summary
