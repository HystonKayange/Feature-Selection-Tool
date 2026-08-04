# API overview

## Core

```python
from feature_selector import FeatureSelector, load_dataset, load_benchmark

X, y, info = load_benchmark("breast_cancer")
sel = FeatureSelector(k=10, method="anova", task="auto")
X_s = sel.fit_transform(X, y)
sel.selected_features_
sel.get_feature_scores()
sel.export_artifacts("out/")
```

## Comparison & stability

```python
from feature_selector import compare_methods, stability_selection, evaluate_before_after

cmp = compare_methods(X, y, k=10, methods=["anova", "rf", "lasso"])
stab = stability_selection(X, y, method="rf", k=10, n_splits=5)
ba = evaluate_before_after(X, y, sel.selected_features_, cv=5)
```

## Nested CV

```python
from feature_selector import nested_cv_feature_selection, nested_cv_compare_methods

result = nested_cv_feature_selection(X, y, method="rf", k=10, outer_splits=5)
table = nested_cv_compare_methods(X, y, k=10)
```

## Datasets

```python
from feature_selector import list_datasets, download_datasets, recommend_for_goal

list_datasets()
recommend_for_goal("medical")
download_datasets(["heart_disease", "sonar"], data_dir="data")
```

## Methods

`anova`, `mutual_info` / `mi`, `random_forest` / `rf`, `lasso`, `rfe`
