"""Cross-validated stability selection for research comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from feature_selector.data import infer_task
from feature_selector.evaluate import resolve_task
from feature_selector.selector import FeatureSelector, normalize_method


ArrayLike = Union[pd.DataFrame, np.ndarray]


@dataclass
class StabilityResult:
    """Result of cross-validated stability selection."""

    method: str
    task: str
    k: int
    n_splits: int
    frequencies: pd.DataFrame  # feature, frequency, n_selected, selected_consensus
    consensus_features: List[str]
    fold_selections: List[List[str]] = field(default_factory=list)
    threshold: float = 0.6

    @property
    def mean_stability(self) -> float:
        """Mean selection frequency across features that were selected at least once."""
        freq = self.frequencies["frequency"]
        positive = freq[freq > 0]
        if positive.empty:
            return 0.0
        return float(positive.mean())

    def to_frame(self) -> pd.DataFrame:
        return self.frequencies.copy()


def stability_selection(
    X: ArrayLike,
    y,
    *,
    method: str = "anova",
    k: int = 10,
    task: str = "auto",
    n_splits: int = 5,
    threshold: float = 0.6,
    random_state: int = 42,
    score: Optional[str] = None,
    fast: bool = False,
    n_jobs: int = -1,
) -> StabilityResult:
    """Select features on each CV fold; report selection frequencies.

    Parameters
    ----------
    method :
        Selection method (anova, mutual_info, random_forest, lasso, rfe, ...).
    k :
        Features to select per fold.
    threshold :
        Consensus if selected in at least this fraction of folds.
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

    # Align missing labels
    mask = y_series.notna()
    X = X.loc[mask.to_numpy()].reset_index(drop=True)
    y_series = y_series.loc[mask].reset_index(drop=True)

    task_r = resolve_task(y_series, task)
    method_c, score_override = normalize_method(method)
    score_c = score_override or score or "f_score"

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    if task_r == "classification":
        try:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = list(splitter.split(X, y_series))
        except ValueError:
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = list(splitter.split(X))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(X))

    counts: Dict[str, int] = {c: 0 for c in X.columns}
    fold_selections: List[List[str]] = []

    for train_idx, _ in splits:
        X_tr = X.iloc[train_idx]
        y_tr = y_series.iloc[train_idx]
        k_eff = min(k, X_tr.shape[1])
        sel = FeatureSelector(
            k=k_eff,
            task=task_r,
            score=score_c,
            method=method_c,
            random_state=random_state,
            fast=fast,
            n_jobs=n_jobs,
        )
        sel.fit(X_tr, y_tr)
        chosen = list(sel.selected_features_)
        fold_selections.append(chosen)
        for name in chosen:
            counts[name] = counts.get(name, 0) + 1

    rows = []
    for feat, cnt in counts.items():
        freq = cnt / n_splits
        rows.append(
            {
                "feature": feat,
                "n_selected": cnt,
                "frequency": freq,
                "selected_consensus": freq >= threshold,
            }
        )
    frequencies = (
        pd.DataFrame(rows)
        .sort_values(["frequency", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    consensus = frequencies.loc[
        frequencies["selected_consensus"], "feature"
    ].tolist()

    # If threshold yields fewer than k, take top-k by frequency as soft consensus
    label = f"{method_c}" if method_c != "filter" else f"filter:{score_c}"

    return StabilityResult(
        method=label,
        task=task_r,
        k=k,
        n_splits=n_splits,
        frequencies=frequencies,
        consensus_features=consensus,
        fold_selections=fold_selections,
        threshold=threshold,
    )


def pairwise_jaccard(selections: Dict[str, Sequence[str]]) -> pd.DataFrame:
    """Pairwise Jaccard similarity between method selection sets."""
    names = list(selections.keys())
    mat = np.zeros((len(names), len(names)), dtype=float)
    sets = {n: set(selections[n]) for n in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            mat[i, j] = inter / union if union else 1.0
    return pd.DataFrame(mat, index=names, columns=names)
