"""Leakage-safe multi-method feature selection with sklearn-compatible API."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from feature_selector.data import infer_task
from feature_selector.report import (
    export_html_report,
    export_run_artifacts,
    export_scores,
    export_selected_features,
)


SCORE_FUNCS = {
    ("classification", "f_score"): f_classif,
    ("classification", "mutual_info"): mutual_info_classif,
    ("regression", "f_score"): f_regression,
    ("regression", "mutual_info"): mutual_info_regression,
}

SCORE_ALIASES = {
    "f_score": "f_score",
    "anova": "f_score",
    "f": "f_score",
    "mutual_info": "mutual_info",
    "mi": "mutual_info",
    "mutual-info": "mutual_info",
    "mutual_information": "mutual_info",
}

TASK_ALIASES = {
    "auto": "auto",
    "classification": "classification",
    "clf": "classification",
    "class": "classification",
    "regression": "regression",
    "reg": "regression",
}

# User method string → (canonical method family, optional score override)
METHOD_ALIASES = {
    "filter": ("filter", None),
    "anova": ("filter", "f_score"),
    "f_score": ("filter", "f_score"),
    "mutual_info": ("filter", "mutual_info"),
    "mi": ("filter", "mutual_info"),
    "mutual-info": ("filter", "mutual_info"),
    "random_forest": ("random_forest", None),
    "rf": ("random_forest", None),
    "lasso": ("lasso", None),
    "rfe": ("rfe", None),
}

SUPPORTED_METHODS = ("filter", "random_forest", "lasso", "rfe")


def normalize_score(score: str) -> str:
    key = score.strip().lower().replace(" ", "_")
    if key not in SCORE_ALIASES:
        raise ValueError(
            f"Unknown score '{score}'. "
            f"Choose from: anova/f_score, mutual_info/mi"
        )
    return SCORE_ALIASES[key]


def normalize_task(task: str) -> str:
    key = task.strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(
            f"Unknown task '{task}'. Choose from: auto, classification, regression"
        )
    return TASK_ALIASES[key]


def normalize_method(method: str) -> Tuple[str, Optional[str]]:
    """Return (canonical_method, score_override_or_None)."""
    key = method.strip().lower().replace(" ", "_").replace("-", "_")
    # allow mutual_info style already handled
    if key not in METHOD_ALIASES:
        # try score-like names already in SCORE_ALIASES
        if key in SCORE_ALIASES:
            return "filter", SCORE_ALIASES[key]
        raise ValueError(
            f"Unknown method '{method}'. Choose from: "
            f"filter, anova, mutual_info, random_forest/rf, lasso, rfe"
        )
    return METHOD_ALIASES[key]


class FeatureSelector:
    """Multi-method feature selector with fit/transform (train-only fitting).

    Preprocessing (imputation + ordinal encoding of categoricals) is fit on
    training data only and reapplied in ``transform`` to avoid leakage.

    Parameters
    ----------
    k :
        Number of top features to select. If None, preprocessing runs but all
        features are kept.
    task :
        ``'classification'``, ``'regression'``, or ``'auto'`` (inferred at fit).
    score :
        For ``method='filter'``: ``'f_score'`` / ``'anova'`` or ``'mutual_info'``.
    method :
        ``filter``, ``random_forest`` (``rf``), ``lasso``, ``rfe``,
        or aliases ``anova`` / ``mutual_info``.
    random_state :
        Seed for stochastic methods (RF, Lasso path, RFE solvers).
    """

    def __init__(
        self,
        k: Optional[int] = None,
        task: str = "auto",
        score: str = "f_score",
        method: str = "filter",
        random_state: int = 42,
    ):
        self.k = k
        self.task = normalize_task(task)
        method_c, score_override = normalize_method(method)
        self.method = method_c
        self.score = normalize_score(score_override or score)
        self.random_state = random_state

        self.numeric_imputer_: Optional[SimpleImputer] = None
        self.categorical_imputer_: Optional[SimpleImputer] = None
        self.encoder_: Optional[OrdinalEncoder] = None

        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.scores_: Optional[np.ndarray] = None
        self.task_: Optional[str] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def selected_features_(self) -> List[str]:
        """Names of features retained after selection."""
        self._check_is_fitted()
        return list(self.feature_names_out_)

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y) -> "FeatureSelector":
        """Fit preprocessors and selector on training data only."""
        X_df = self._as_dataframe(X)
        y_series = self._prepare_target_series(y)

        if len(X_df) != len(y_series):
            raise ValueError(
                f"X and y length mismatch: {len(X_df)} vs {len(y_series)}"
            )

        mask = y_series.notna()
        if not mask.all():
            X_df = X_df.loc[mask.to_numpy()].reset_index(drop=True)
            y_series = y_series.loc[mask].reset_index(drop=True)

        if len(X_df) == 0:
            raise ValueError("No samples left after dropping missing labels.")

        if self.task == "auto":
            self.task_ = infer_task(y_series)
        else:
            self.task_ = self.task

        y_arr = y_series.to_numpy()

        self.feature_names_in_ = list(X_df.columns)
        self.numeric_columns_ = list(
            X_df.select_dtypes(include=[np.number]).columns
        )
        self.categorical_columns_ = [
            c for c in X_df.columns if c not in self.numeric_columns_
        ]

        X_proc = self._fit_preprocess(X_df)
        processed_names = list(X_proc.columns)

        if self.k is None:
            self.feature_names_out_ = processed_names
            self.scores_ = None
            self._is_fitted = True
            return self

        k = min(int(self.k), X_proc.shape[1])
        if k < 1:
            raise ValueError("k must be at least 1 when feature selection is enabled.")

        scores, selected = self._fit_method(X_proc, y_arr, k, processed_names)
        self.scores_ = scores
        self.feature_names_out_ = selected
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Apply fitted preprocessing and selection; returns a DataFrame with names."""
        self._check_is_fitted()
        X_df = self._as_dataframe(X, columns=self.feature_names_in_)
        X_proc = self._transform_preprocess(X_df)
        return X_proc.loc[:, self.feature_names_out_].reset_index(drop=True)

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y
    ) -> pd.DataFrame:
        """Fit on X, y and return transformed features for rows with valid labels."""
        X_df = self._as_dataframe(X)
        y_series = self._prepare_target_series(y)
        mask = y_series.notna()
        X_clean = X_df.loc[mask.to_numpy()].reset_index(drop=True)
        y_clean = y_series.loc[mask].reset_index(drop=True)
        self.fit(X_clean, y_clean)
        return self.transform(X_clean)

    def get_support(self, indices: bool = False):
        """Mask or indices of selected features among preprocessed columns."""
        self._check_is_fitted()
        support = np.array(
            [name in set(self.feature_names_out_) for name in self.feature_names_in_],
            dtype=bool,
        )
        if indices:
            return np.where(support)[0]
        return support

    def get_feature_scores(self) -> pd.DataFrame:
        """Scores for all preprocessed features (after fit)."""
        self._check_is_fitted()
        names = self._preprocessed_feature_names()
        if self.scores_ is None:
            return pd.DataFrame(
                {
                    "feature": names,
                    "score": np.nan,
                    "selected": [True] * len(names),
                }
            )
        scores = np.asarray(self.scores_, dtype=float)
        if len(scores) != len(names):
            names = [f"feature_{i}" for i in range(len(scores))]
        selected_set = set(self.feature_names_out_)
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "score": scores,
                    "selected": [n in selected_set for n in names],
                }
            )
            .sort_values("score", ascending=False, na_position="last")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------
    def _fit_method(
        self,
        X_proc: pd.DataFrame,
        y_arr: np.ndarray,
        k: int,
        names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        if self.method == "filter":
            return self._fit_filter(X_proc, y_arr, k, names)
        if self.method == "random_forest":
            return self._fit_random_forest(X_proc, y_arr, k, names)
        if self.method == "lasso":
            return self._fit_lasso(X_proc, y_arr, k, names)
        if self.method == "rfe":
            return self._fit_rfe(X_proc, y_arr, k, names)
        raise ValueError(f"Unsupported method: {self.method}")

    def _fit_filter(self, X_proc, y_arr, k, names):
        score_func = SCORE_FUNCS[(self.task_, self.score)]
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_proc, y_arr)
        scores = np.asarray(selector.scores_, dtype=float)
        # NaN scores → treat as worst
        scores_rank = np.where(np.isfinite(scores), scores, -np.inf)
        support = selector.get_support()
        selected = [n for n, keep in zip(names, support) if keep]
        # If SelectKBest failed on ties, fall back to top-k by score
        if len(selected) != k:
            order = np.argsort(scores_rank)[::-1][:k]
            selected = [names[i] for i in order]
        return scores, selected

    def _fit_random_forest(self, X_proc, y_arr, k, names):
        if self.task_ == "classification":
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1,
            )
        model.fit(X_proc, y_arr)
        scores = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(scores)[::-1][:k]
        selected = [names[i] for i in order]
        return scores, selected

    def _fit_lasso(self, X_proc, y_arr, k, names):
        if self.task_ == "classification":
            model = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=4000,
                random_state=self.random_state,
            )
            model.fit(X_proc, y_arr)
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim > 1:
                scores = np.mean(np.abs(coef), axis=0)
            else:
                scores = np.abs(coef)
        else:
            model = LassoCV(
                cv=min(5, max(2, len(y_arr) // 10 or 2)),
                random_state=self.random_state,
                max_iter=4000,
                n_jobs=-1,
            )
            model.fit(X_proc, y_arr)
            scores = np.abs(np.asarray(model.coef_, dtype=float))

        order = np.argsort(scores)[::-1][:k]
        selected = [names[i] for i in order]
        return scores, selected

    def _fit_rfe(self, X_proc, y_arr, k, names):
        if self.task_ == "classification":
            base = LogisticRegression(
                max_iter=2000, random_state=self.random_state
            )
        else:
            base = LinearRegression()
        # step: drop more features at once for wider matrices
        n_features = X_proc.shape[1]
        step = 1 if n_features <= 40 else max(1, n_features // 20)
        rfe = RFE(estimator=base, n_features_to_select=k, step=step)
        rfe.fit(X_proc, y_arr)
        # Higher score = better: invert ranking (1 is best)
        ranking = np.asarray(rfe.ranking_, dtype=float)
        scores = 1.0 / ranking
        selected = [n for n, keep in zip(names, rfe.support_) if keep]
        return scores, selected

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def export_scores(self, path: Union[str, Path]) -> Path:
        return export_scores(self.get_feature_scores(), path)

    def export_selected_features(self, path: Union[str, Path]) -> Path:
        return export_selected_features(self.selected_features_, path)

    def export_report(
        self,
        path: Union[str, Path],
        *,
        metrics: Optional[dict] = None,
        summary: Optional[dict] = None,
        image_paths: Optional[dict] = None,
        title: str = "Feature Selection Report",
    ) -> Path:
        self._check_is_fitted()
        base_summary = {
            "task": self.task_,
            "score": self.score,
            "method": self.method,
            "k": self.k,
            "n_features_in": len(self.feature_names_in_),
            "n_features_out": len(self.feature_names_out_),
        }
        if summary:
            base_summary.update(summary)
        return export_html_report(
            path,
            title=title,
            summary=base_summary,
            selected_features=self.selected_features_,
            score_table=self.get_feature_scores(),
            metrics=metrics,
            image_paths=image_paths,
        )

    def export_artifacts(
        self,
        out_dir: Union[str, Path],
        *,
        metrics: Optional[dict] = None,
        summary: Optional[dict] = None,
        image_paths: Optional[dict] = None,
    ) -> dict:
        self._check_is_fitted()
        base_summary = {
            "task": self.task_,
            "score": self.score,
            "method": self.method,
            "k": self.k,
            "n_features_in": len(self.feature_names_in_),
            "n_features_out": len(self.feature_names_out_),
        }
        if summary:
            base_summary.update(summary)
        return export_run_artifacts(
            out_dir,
            selected_features=self.selected_features_,
            score_table=self.get_feature_scores(),
            metrics=metrics,
            summary=base_summary,
            image_paths=image_paths,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot_feature_importance(
        self,
        feature_names: Optional[Sequence[str]] = None,
        feature_scores: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        title: str = "Top Features by Importance",
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        if feature_scores is None:
            self._check_is_fitted()
            if self.scores_ is None:
                raise ValueError("No scores available; fit with k set first.")
            scores = np.asarray(self.scores_, dtype=float)
            names = np.asarray(
                feature_names
                if feature_names is not None
                else self._preprocessed_feature_names()
            )
        else:
            scores = np.asarray(feature_scores, dtype=float)
            if feature_names is None:
                names = np.array([f"Feature_{i+1}" for i in range(len(scores))])
            else:
                names = np.asarray(feature_names)

        if len(names) != len(scores):
            raise ValueError(
                f"feature_names ({len(names)}) and feature_scores "
                f"({len(scores)}) must have the same length."
            )

        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[: int(top_k)]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(order)), scores[order], align="center")
        plt.xticks(range(len(order)), names[order], rotation=45, ha="right")
        plt.title(title)
        plt.xlabel("Feature")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_importance(self, **kwargs):
        return self.plot_feature_importance(**kwargs)

    def plot_feature_correlation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        X_df = self._as_dataframe(X)
        corr_matrix = X_df.corr(numeric_only=True)
        plt.figure(figsize=(9, 5))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_distribution(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        X_df = self._as_dataframe(X)
        n_cols = min(len(X_df.columns), 12)
        cols = list(X_df.columns)[:n_cols]
        n = len(cols)
        n_rows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, column in zip(axes, cols):
            sns.histplot(X_df[column], kde=True, ax=ax)
            ax.set_title(str(column))
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.suptitle("Feature Distributions", y=1.02)
        fig.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Correlation helpers
    # ------------------------------------------------------------------
    def check_correlation(
        self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.9
    ) -> pd.DataFrame:
        X_df = self._as_dataframe(X)
        corr = X_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and val > threshold:
                    pairs.append(
                        {"feature_a": idx, "feature_b": col, "correlation": val}
                    )
        return pd.DataFrame(pairs)

    def drop_correlated(
        self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.9
    ) -> pd.DataFrame:
        X_df = self._as_dataframe(X).copy()
        corr = X_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > threshold)
        ]
        return X_df.drop(columns=to_drop)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureSelector is not fitted. Call fit() or fit_transform() first."
            )

    def _preprocessed_feature_names(self) -> List[str]:
        return list(self.feature_names_in_)

    @staticmethod
    def _as_dataframe(
        X: Union[pd.DataFrame, np.ndarray],
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df.columns = df.columns.astype(str)
            if columns is not None:
                missing = set(columns) - set(df.columns)
                if missing:
                    raise ValueError(
                        f"X is missing columns seen during fit: {sorted(missing)}"
                    )
                df = df.loc[:, list(columns)]
            return df.reset_index(drop=True)

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D array-like.")
        if columns is not None:
            if len(columns) != arr.shape[1]:
                raise ValueError(
                    f"Expected {len(columns)} columns, got {arr.shape[1]}"
                )
            cols = list(columns)
        else:
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    @staticmethod
    def _prepare_target_series(y) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y must be 1-dimensional.")
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            return y.reset_index(drop=True)
        return pd.Series(np.asarray(y).ravel())

    def _fit_preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_columns_:
            self.numeric_imputer_ = SimpleImputer(strategy="mean")
            num = self.numeric_imputer_.fit_transform(X[self.numeric_columns_])
            parts.append(
                pd.DataFrame(num, columns=self.numeric_columns_, index=X.index)
            )
        else:
            self.numeric_imputer_ = None

        if self.categorical_columns_:
            self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
            cat = self.categorical_imputer_.fit_transform(
                X[self.categorical_columns_].astype(str)
            )
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            cat_enc = self.encoder_.fit_transform(cat)
            parts.append(
                pd.DataFrame(
                    cat_enc,
                    columns=self.categorical_columns_,
                    index=X.index,
                )
            )
        else:
            self.categorical_imputer_ = None
            self.encoder_ = None

        if not parts:
            raise ValueError("No features available after preprocessing.")

        combined = pd.concat(parts, axis=1)
        return combined.loc[:, self.feature_names_in_]

    def _transform_preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.numeric_columns_:
            num = self.numeric_imputer_.transform(X[self.numeric_columns_])
            parts.append(
                pd.DataFrame(num, columns=self.numeric_columns_, index=X.index)
            )

        if self.categorical_columns_:
            cat = self.categorical_imputer_.transform(
                X[self.categorical_columns_].astype(str)
            )
            cat_enc = self.encoder_.transform(cat)
            parts.append(
                pd.DataFrame(
                    cat_enc,
                    columns=self.categorical_columns_,
                    index=X.index,
                )
            )

        combined = pd.concat(parts, axis=1)
        return combined.loc[:, self.feature_names_in_]
