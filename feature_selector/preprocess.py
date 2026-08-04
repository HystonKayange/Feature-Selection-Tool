"""Train-safe preprocessing helpers for research workflows."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[pd.DataFrame, np.ndarray]


def _as_frame(X: ArrayLike) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        df.columns = df.columns.astype(str)
        return df.reset_index(drop=True)
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be 2D")
    return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])


def drop_constant_features(
    X: ArrayLike,
    *,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop features with variance <= threshold (numeric) or a single unique value.

    Parameters
    ----------
    threshold :
        Numeric columns with variance ``<= threshold`` are dropped. Use ``0`` for
        true constants only.

    Returns
    -------
    X_reduced, dropped_names
    """
    X_df = _as_frame(X)
    dropped: List[str] = []
    keep: List[str] = []
    for col in X_df.columns:
        series = X_df[col]
        if pd.api.types.is_numeric_dtype(series):
            var = float(series.var(skipna=True)) if series.notna().sum() > 1 else 0.0
            if var <= threshold or series.nunique(dropna=True) <= 1:
                dropped.append(col)
            else:
                keep.append(col)
        else:
            if series.nunique(dropna=True) <= 1:
                dropped.append(col)
            else:
                keep.append(col)
    return X_df.loc[:, keep], dropped


def drop_correlated_features(
    X: ArrayLike,
    *,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Drop one feature from each highly correlated numeric pair.

    Keeps the first feature in column order; drops later partners with
    |corr| > threshold.

    Returns
    -------
    X_reduced, dropped_names, pairs_df
    """
    X_df = _as_frame(X)
    num = X_df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return X_df, [], pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    to_drop = set()
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold:
                pairs.append(
                    {"feature_a": idx, "feature_b": col, "correlation": float(val)}
                )
                # drop the later column
                to_drop.add(col)

    dropped = [c for c in X_df.columns if c in to_drop]
    kept = [c for c in X_df.columns if c not in to_drop]
    return X_df.loc[:, kept], dropped, pd.DataFrame(pairs)


def outlier_report(
    X: ArrayLike,
    *,
    iqr_multiplier: float = 1.5,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """IQR-based outlier summary per numeric column (report only; does not mutate).

    Returns a DataFrame with columns:
    feature, n, n_outliers, outlier_rate, lower_fence, upper_fence, q1, q3
    """
    X_df = _as_frame(X)
    if columns is None:
        cols = list(X_df.select_dtypes(include=[np.number]).columns)
    else:
        cols = [c for c in columns if c in X_df.columns]

    rows = []
    for col in cols:
        s = X_df[col].dropna()
        if s.empty:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "feature": col,
                "n": int(len(s)),
                "n_outliers": n_out,
                "outlier_rate": n_out / max(len(s), 1),
                "lower_fence": lower,
                "upper_fence": upper,
                "q1": q1,
                "q3": q3,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "outlier_rate", ascending=False, ignore_index=True
    )


def prepare_features(
    X: ArrayLike,
    *,
    drop_constant: bool = True,
    correlation_threshold: Optional[float] = None,
    constant_variance: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Optional cleaning pipeline before selection (apply on train only in CV)."""
    info: Dict[str, object] = {"dropped_constant": [], "dropped_correlated": [], "pairs": None}
    X_df = _as_frame(X)
    if drop_constant:
        X_df, dropped = drop_constant_features(X_df, threshold=constant_variance)
        info["dropped_constant"] = dropped
    if correlation_threshold is not None:
        X_df, dropped, pairs = drop_correlated_features(
            X_df, threshold=correlation_threshold
        )
        info["dropped_correlated"] = dropped
        info["pairs"] = pairs
    return X_df, info
