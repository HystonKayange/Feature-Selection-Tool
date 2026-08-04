"""Interactive runner and high-level evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_selector.data import infer_task, load_dataset
from feature_selector.selector import (
    FeatureSelector,
    normalize_method,
    normalize_score,
    normalize_task,
)


def open_file_dialog() -> Optional[str]:
    """Open a file picker if a display is available; otherwise return None."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(
            "FeatureSelectorTool",
            "Welcome to the Feature Selection Tool for Machine Learning!\n\n"
            "Select a dataset to get started.",
        )
        file_path = filedialog.askopenfilename(
            title="Select a dataset file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("TXT files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path or None
    except Exception:
        return None


def prompt_path() -> str:
    path = open_file_dialog()
    if path:
        return path
    print("No GUI file dialog available. Enter the path to a CSV/TXT dataset.")
    path = input("Dataset path: ").strip().strip('"').strip("'")
    if not path:
        print("No file provided. Exiting.")
        sys.exit(1)
    if not Path(path).is_file():
        print(f"File not found: {path}")
        sys.exit(1)
    return path


def prompt_task(default: str) -> str:
    print("\nChoose a task:")
    print(f"  1. Classification{'  (suggested)' if default == 'classification' else ''}")
    print(f"  2. Regression{'  (suggested)' if default == 'regression' else ''}")
    print("  3. Auto-detect")
    print("  4. Exit")
    choice = input("Enter the number of your choice: ").strip()
    if choice == "1":
        return "classification"
    if choice == "2":
        return "regression"
    if choice == "3":
        return "auto"
    if choice == "4":
        print("Exiting.")
        sys.exit(0)
    print("Invalid choice. Exiting.")
    sys.exit(1)


def prompt_k(n_features: int) -> int:
    raw = input(
        f"Enter the number of features to select (1–{n_features}): "
    ).strip()
    try:
        k = int(raw)
    except ValueError:
        print("Invalid input. Please enter an integer. Exiting.")
        sys.exit(1)
    if k < 1 or k > n_features:
        print(f"k must be between 1 and {n_features}. Exiting.")
        sys.exit(1)
    return k


def evaluate_classification(y_true, y_pred) -> dict:
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    except Exception:
        pass
    return metrics


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_pipeline(
    file_path: str,
    *,
    task: Optional[str] = None,
    k: Optional[int] = None,
    score: str = "f_score",
    method: str = "filter",
    target_column: Optional[Union[str, int]] = None,
    delimiter: str = ",",
    header: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    show_plots: bool = True,
    out_dir: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run a leakage-safe select + baseline-evaluate pipeline.

    Parameters
    ----------
    file_path :
        Path to CSV/TXT dataset.
    task :
        ``classification``, ``regression``, ``auto``, or None (prompt).
    k :
        Number of features to keep, or None (prompt).
    score :
        Filter score: ``f_score`` / ``anova`` or ``mutual_info`` / ``mi``.
    method :
        ``filter``, ``random_forest``, ``lasso``, or ``rfe``.
    out_dir :
        If set, write selected features, scores, metrics, plots, and HTML report.
    """
    dataset = load_dataset(
        file_path,
        delimiter=delimiter,
        header=header,
        target_column=target_column,
    )
    suggested = infer_task(dataset.labels)

    def log(msg: str = "") -> None:
        if not quiet:
            print(msg)

    log("_" * 80)
    log("Dataset preview (features):")
    log("_" * 80)
    log(str(dataset.features.head()))
    log("_" * 80)
    log(f"Samples: {dataset.n_samples} | Features: {dataset.n_features}")
    log(f"Suggested task (from target): {suggested}")
    log("_" * 80)

    if task is None:
        task = prompt_task(suggested)
    task = normalize_task(task)
    method_c, score_override = normalize_method(method)
    score = normalize_score(score_override or score)

    if k is None:
        k = prompt_k(dataset.n_features)
    k = int(k)
    if k < 1 or k > dataset.n_features:
        raise ValueError(f"k must be between 1 and {dataset.n_features}, got {k}")

    X = dataset.features
    y = dataset.labels

    task_for_split = suggested if task == "auto" else task

    # --- Leakage-safe: split BEFORE fitting the selector ---
    stratify = (
        y if task_for_split == "classification" and y.nunique() > 1 else None
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    selector = FeatureSelector(
        k=k,
        task=task,
        score=score,
        method=method_c,
        random_state=random_state,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    resolved_task = selector.task_

    log(
        f"\nResolved task: {resolved_task} | method: {method_c} "
        f"| score: {score} | k: {k}"
    )
    log("\nSelected features:")
    for name in selector.selected_features_:
        log(f"  - {name}")

    score_table = selector.get_feature_scores()
    log("\nFeature scores (train):")
    log(score_table.to_string(index=False))

    image_paths: Dict[str, Path] = {}
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    # Plot selector scores (method-native) when available
    filter_save = str(out_path / "method_scores.png") if out_path else None
    if selector.scores_ is not None:
        selector.plot_feature_importance(
            title=f"Feature scores ({method_c})",
            save_path=filter_save,
            show=show_plots,
            top_k=min(k, len(selector.feature_names_in_)),
        )
        if filter_save:
            image_paths["Method scores"] = Path(filter_save)

    # Also RF importances on selected features for interpretability
    if resolved_task == "classification":
        importance_model = RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
    else:
        importance_model = RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
    importance_model.fit(X_train_sel, y_train)
    importances = importance_model.feature_importances_

    imp_save = str(out_path / "importance.png") if out_path else None
    dist_save = str(out_path / "distributions.png") if out_path else None

    selector.plot_feature_importance(
        feature_names=selector.selected_features_,
        feature_scores=importances,
        title="Selected Feature Importances (Random Forest)",
        save_path=imp_save,
        show=show_plots,
    )
    if imp_save:
        image_paths["Feature importance"] = Path(imp_save)

    selector.plot_feature_distribution(
        X_train_sel, save_path=dist_save, show=show_plots
    )
    if dist_save:
        image_paths["Feature distributions"] = Path(dist_save)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    if resolved_task == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_classification(y_test, y_pred)
        log("\nTest metrics (Logistic Regression on selected features):")
    else:
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate_regression(y_test, y_pred)
        log("\nTest metrics (Linear Regression on selected features):")

    for key, value in metrics.items():
        log(
            f"  {key}: {value:.4f}"
            if isinstance(value, float)
            else f"  {key}: {value}"
        )

    summary = {
        "source": str(file_path),
        "target": str(y.name),
        "task": resolved_task,
        "method": method_c,
        "score": score,
        "k": k,
        "test_size": test_size,
        "random_state": random_state,
        "n_samples": dataset.n_samples,
        "n_features_in": dataset.n_features,
        "n_features_out": len(selector.selected_features_),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    artifacts = None
    if out_path:
        artifacts = selector.export_artifacts(
            out_path,
            metrics=metrics,
            summary=summary,
            image_paths=image_paths,
        )
        log(f"\nArtifacts written to: {out_path.resolve()}")
        for name, p in artifacts.items():
            log(f"  - {name}: {p}")

    return {
        "task": resolved_task,
        "method": method_c,
        "score": score,
        "k": k,
        "selected_features": list(selector.selected_features_),
        "metrics": metrics,
        "score_table": score_table,
        "summary": summary,
        "selector": selector,
        "artifacts": artifacts,
    }


def run_interactive() -> None:
    """Legacy interactive entry (file dialog + prompts)."""
    file_path = prompt_path()
    run_pipeline(file_path, show_plots=True)
