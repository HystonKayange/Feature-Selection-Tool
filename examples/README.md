# Examples

Paper-style results in the root README were produced from:

| Path | Contents |
|------|----------|
| `benchmark_out/` | Exploratory multi-method compare (9 datasets) |
| `nested_cv_out/` | Nested 5-fold CV on heart / sonar / madelon |
| `../Images/results/` | Figures embedded in the README |

## 0. Get datasets

```bash
feature-select datasets list --goal paper
feature-select datasets download heart_disease sonar madelon --out data
```

## 1. Exploratory benchmark

```bash
python examples/benchmark_datasets.py \
  --datasets heart_disease,sonar,madelon \
  --k 10 --out examples/benchmark_out
```

## 2. Nested CV (use for claims)

```bash
feature-select nested-cv data/heart_disease.csv -k 10 --method mutual_info --cv 5 --out results/heart_ncv
feature-select nested-cv data/sonar.csv -k 15 --method lasso --cv 5 --out results/sonar_ncv
feature-select nested-cv data/madelon.csv -k 20 --method random_forest --cv 5 --out results/madelon_ncv
```

## 3. Synthetic smoke test

```bash
python examples/compare_synthetic.py --k 5 --out examples/output
```
