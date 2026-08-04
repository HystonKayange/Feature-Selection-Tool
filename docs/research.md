# Research workflow

## Honest evaluation ladder

1. **Exploratory compare** (`compare_methods`) on the training set  
   - before/after CV metrics  
   - stability frequencies  
   - Jaccard agreement across methods  

2. **Nested CV** (`nested_cv_feature_selection`)  
   - selection fit only on outer train folds  
   - use mean ± std for tables in papers  

3. **Final hold-out** (optional)  
   - freeze method + k from nested CV  
   - fit once on train, report test once  

## Example

```python
from sklearn.model_selection import train_test_split
from feature_selector import (
    load_benchmark,
    compare_methods,
    nested_cv_compare_methods,
    FeatureSelector,
)

X, y, info = load_benchmark("sonar")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1) explore on train
cmp = compare_methods(X_tr, y_tr, k=15, task="classification", cv=5)
cmp.export("results/sonar_explore")

# 2) nested CV ranking
ncv = nested_cv_compare_methods(
    X_tr, y_tr, methods=["anova", "rf", "lasso"], k=15, outer_splits=5
)
print(ncv)

# 3) final model
best = "random_forest"
sel = FeatureSelector(k=15, method=best, task="classification")
X_tr_s = sel.fit_transform(X_tr, y_tr)
X_te_s = sel.transform(X_te)
```

## Interpreting results

| Pattern | Meaning |
|---------|---------|
| High `sel_accuracy`, high `mean_stability` | Strong candidate subset |
| High score, low stability | Brittle — different folds pick different features |
| High Jaccard across methods | Method-robust features |
| Nested CV ≪ exploratory score | Leakage / optimism in the exploratory table |

## Artifacts to keep for a paper

- `comparison_summary.csv`
- `jaccard_heatmap.png`
- `stability/*.csv`
- `nested_cv_result.json` / `selection_frequency.csv`
- short note of dataset id, k, seed, methods
