# Feature Selector Tool

Leakage-safe **multi-method feature selection** for tabular machine learning, with research comparisons, stability analysis, and nested CV.

## Install

```bash
pip install -e ".[dev]"
# optional UI
pip install -e ".[ui]"
```

## Quick start

```python
from feature_selector import FeatureSelector, load_benchmark, compare_methods

X, y, info = load_benchmark("breast_cancer")  # no manual download
print(info.why)

cmp = compare_methods(X, y, k=10, task="classification", cv=5)
print(cmp.summary)
cmp.export("results/breast_cancer")
```

## CLI

```bash
feature-select datasets list
feature-select datasets download breast_cancer heart_disease sonar --out data
feature-select compare data/sonar.csv -k 15 --out results/sonar
feature-select nested-cv data/breast_cancer.csv -k 10 --method rf --out results/ncv
```

## Docs

- [Datasets to download](datasets.md)
- [Research workflow](research.md)
- [API overview](api.md)
