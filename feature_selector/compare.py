"""Multi-method feature-selection comparison for research workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_selector.evaluate import (
    evaluate_before_after,
    metrics_to_row,
    resolve_task,
)
from feature_selector.preprocess import outlier_report, prepare_features
from feature_selector.report import _json_default, export_html_report
from feature_selector.selector import FeatureSelector, normalize_method
from feature_selector.stability import pairwise_jaccard, stability_selection


ArrayLike = Union[pd.DataFrame, np.ndarray]

DEFAULT_METHODS = [
    "anova",
    "mutual_info",
    "random_forest",
    "lasso",
    "rfe",
]

# Include chi2 for classification-heavy exploratory runs (optional caller list)
EXTENDED_METHODS = DEFAULT_METHODS + ["chi2"]


@dataclass
class ComparisonResult:
    """Side-by-side comparison of feature selection methods."""

    task: str
    k: int
    cv: int
    summary: pd.DataFrame
    selections: Dict[str, List[str]]
    jaccard: pd.DataFrame
    stability: Dict[str, pd.DataFrame] = field(default_factory=dict)
    before_after: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    preprocess_info: Dict[str, Any] = field(default_factory=dict)
    outlier_summary: Optional[pd.DataFrame] = None

    def export(self, out_dir: Union[str, Path]) -> Dict[str, Path]:
        """Write comparison tables, heatmaps, and an HTML report."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        summary_path = out / "comparison_summary.csv"
        self.summary.to_csv(summary_path, index=False)
        written["summary"] = summary_path

        sel_path = out / "selections.json"
        sel_path.write_text(
            json.dumps(self.selections, indent=2, default=_json_default),
            encoding="utf-8",
        )
        written["selections"] = sel_path

        jac_path = out / "jaccard.csv"
        self.jaccard.to_csv(jac_path)
        written["jaccard"] = jac_path

        # Stability tables
        stab_dir = out / "stability"
        stab_dir.mkdir(exist_ok=True)
        for name, frame in self.stability.items():
            safe = name.replace(":", "_").replace("/", "_")
            p = stab_dir / f"{safe}.csv"
            frame.to_csv(p, index=False)
            written[f"stability_{safe}"] = p

        # Jaccard heatmap
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            self.jaccard,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Pairwise Jaccard similarity of selected feature sets")
        fig.tight_layout()
        heat_path = out / "jaccard_heatmap.png"
        fig.savefig(heat_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["jaccard_heatmap"] = heat_path

        # Metric bar chart (prefer accuracy or r2)
        metric_col = None
        for cand in ("sel_accuracy", "sel_r2", "sel_f1_weighted"):
            if cand in self.summary.columns:
                metric_col = cand
                break
        img_paths = {"Jaccard heatmap": heat_path}
        if metric_col and not self.summary.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_df = self.summary.sort_values(metric_col, ascending=False)
            ax.bar(plot_df["method"], plot_df[metric_col], color="#3b6ea5")
            ax.set_ylabel(metric_col)
            ax.set_title("Selected-feature CV metric by method")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            bar_path = out / "metric_by_method.png"
            fig.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written["metric_by_method"] = bar_path
            img_paths["CV metric by method"] = bar_path

        if self.outlier_summary is not None and not self.outlier_summary.empty:
            o_path = out / "outlier_report.csv"
            self.outlier_summary.to_csv(o_path, index=False)
            written["outliers"] = o_path

        # HTML
        summary_dict = {
            "task": self.task,
            "k": self.k,
            "cv": self.cv,
            "n_methods": len(self.selections),
            "methods": ", ".join(self.selections.keys()),
        }
        # Attach preprocess notes
        if self.preprocess_info:
            summary_dict["dropped_constant"] = len(
                self.preprocess_info.get("dropped_constant") or []
            )
            summary_dict["dropped_correlated"] = len(
                self.preprocess_info.get("dropped_correlated") or []
            )

        # Use first method selection for the "selected features" section + full table
        first_method = next(iter(self.selections), None)
        feats = self.selections.get(first_method, []) if first_method else []

        # Extra HTML: embed comparison table
        html_path = out / "comparison_report.html"
        # Build a richer report by writing custom HTML extending export_html_report
        base = export_html_report(
            html_path,
            title="Feature Selection Method Comparison",
            summary=summary_dict,
            selected_features=feats,
            score_table=self.summary,
            metrics=None,
            image_paths=img_paths,
        )
        # Append selections block
        extra = ["<h2>Selections by method</h2>"]
        for m, flist in self.selections.items():
            items = "".join(f"<li><code>{_esc(f)}</code></li>" for f in flist)
            extra.append(f"<h3>{_esc(m)}</h3><ol>{items}</ol>")
        extra.append("<h2>Before / after (CV)</h2>")
        extra.append(self.summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
        text = base.read_text(encoding="utf-8")
        text = text.replace("</body>", "\n".join(extra) + "\n</body>")
        base.write_text(text, encoding="utf-8")
        written["report"] = base
        return written


def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def compare_methods(
    X: ArrayLike,
    y,
    *,
    methods: Optional[Sequence[str]] = None,
    k: int = 10,
    task: str = "auto",
    cv: int = 5,
    stability_splits: int = 5,
    stability_threshold: float = 0.6,
    random_state: int = 42,
    drop_constant: bool = True,
    correlation_threshold: Optional[float] = 0.95,
    include_outlier_report: bool = True,
    run_stability: bool = True,
    fast: bool = False,
    n_jobs: int = -1,
) -> ComparisonResult:
    """Compare feature-selection methods on the same dataset.

    For each method:
      1. Fit selector on the full provided matrix (caller should pass **train** data
         or accept that this is exploratory; for final claims use CV stability).
      2. Evaluate before/after with CV metrics (selection fitted inside is separate
         from this exploratory fit — before/after uses fixed selected set).
      3. Optionally estimate selection stability across CV folds.

    Notes for research honesty
    --------------------------
    The primary selection for the summary table is fit on all of ``X``.
    Stability folds re-select on each train split. Prefer reporting both.
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

    outlier_df = outlier_report(X) if include_outlier_report else None

    X_prep, prep_info = prepare_features(
        X,
        drop_constant=drop_constant,
        correlation_threshold=correlation_threshold,
    )
    if X_prep.shape[1] == 0:
        raise ValueError("No features left after preprocessing.")

    task_r = resolve_task(y_series, task)
    methods = list(methods) if methods is not None else list(DEFAULT_METHODS)
    k_eff = min(int(k), X_prep.shape[1])

    selections: Dict[str, List[str]] = {}
    before_after: Dict[str, Dict[str, Any]] = {}
    stability_tables: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []

    for raw_method in methods:
        method_c, score_override = normalize_method(raw_method)
        score_c = score_override or "f_score"
        label = (
            f"filter:{score_c}" if method_c == "filter" else method_c
        )
        # Prefer user-facing raw name if distinct
        display = raw_method if raw_method not in selections else label

        sel = FeatureSelector(
            k=k_eff,
            task=task_r,
            score=score_c,
            method=method_c,
            random_state=random_state,
            fast=fast,
            n_jobs=n_jobs,
        )
        sel.fit(X_prep, y_series)
        feats = list(sel.selected_features_)
        selections[display] = feats

        ba = evaluate_before_after(
            X_prep,
            y_series,
            feats,
            task=task_r,
            cv=cv,
            random_state=random_state,
        )
        before_after[display] = ba
        row = metrics_to_row(display, ba)

        if run_stability:
            stab = stability_selection(
                X_prep,
                y_series,
                method=raw_method,
                k=k_eff,
                task=task_r,
                n_splits=stability_splits,
                threshold=stability_threshold,
                random_state=random_state,
                fast=fast,
                n_jobs=n_jobs,
            )
            stability_tables[display] = stab.frequencies
            row["mean_stability"] = stab.mean_stability
            row["n_consensus"] = len(stab.consensus_features)
        rows.append(row)

    summary = pd.DataFrame(rows)
    # Sort by primary metric if present
    for col in ("sel_accuracy", "sel_r2", "sel_f1_weighted"):
        if col in summary.columns:
            summary = summary.sort_values(col, ascending=False).reset_index(drop=True)
            break

    jaccard = pairwise_jaccard(selections)

    return ComparisonResult(
        task=task_r,
        k=k_eff,
        cv=cv,
        summary=summary,
        selections=selections,
        jaccard=jaccard,
        stability=stability_tables,
        before_after=before_after,
        preprocess_info=prep_info,
        outlier_summary=outlier_df,
    )
