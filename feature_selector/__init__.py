"""Feature Selector Tool — leakage-safe multi-method feature selection."""

from feature_selector.compare import ComparisonResult, compare_methods
from feature_selector.data import TabularDataset, infer_task, load_dataset
from feature_selector.datasets import (
    DATASET_CATALOG,
    download_datasets,
    list_datasets,
    load_benchmark,
    recommend_for_goal,
)
from feature_selector.evaluate import cross_val_metrics, evaluate_before_after
from feature_selector.nested_cv import (
    nested_cv_compare_methods,
    nested_cv_feature_selection,
)
from feature_selector.preprocess import (
    drop_constant_features,
    drop_correlated_features,
    outlier_report,
    prepare_features,
)
from feature_selector.selector import (
    FeatureSelector,
    make_l1_logistic,
    normalize_method,
    normalize_score,
    normalize_task,
)
from feature_selector.stability import StabilityResult, pairwise_jaccard, stability_selection

__version__ = "0.5.0"
__all__ = [
    "FeatureSelector",
    "make_l1_logistic",
    "TabularDataset",
    "load_dataset",
    "infer_task",
    "normalize_score",
    "normalize_task",
    "normalize_method",
    "compare_methods",
    "ComparisonResult",
    "stability_selection",
    "StabilityResult",
    "pairwise_jaccard",
    "evaluate_before_after",
    "cross_val_metrics",
    "nested_cv_feature_selection",
    "nested_cv_compare_methods",
    "drop_constant_features",
    "drop_correlated_features",
    "outlier_report",
    "prepare_features",
    "DATASET_CATALOG",
    "list_datasets",
    "load_benchmark",
    "download_datasets",
    "recommend_for_goal",
    "__version__",
]
