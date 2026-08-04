# Which dataset should you download?

You do **not** need to hunt randomly on Kaggle. This project ships a curated catalog.

## Decision guide

| Your goal | Download these | Why |
|-----------|----------------|-----|
| **First successful run today** | `breast_cancer`, `wine` | Clean, sklearn-backed, no network issues |
| **Match your original thesis topic (medical)** | `heart_disease`, `pima_diabetes`, `breast_cancer` | Clinical tabular predictors |
| **Show that feature selection helps** | `sonar`, `ionosphere` | Many features / fewer samples |
| **Regression** | `diabetes_sklearn`, `wine_quality_red` | Continuous targets |
| **Paper / stress test** | `madelon` (+ sonar) | Designed for FS challenges (500 features) |

### Avoid for *this* tool

- **MovieLens / pure recommender** data — not a classic supervised feature matrix  
- **Raw images / text** — tabular only  
- **Huge multi-GB tables** first — start small, then scale  

---

## One-command download

```bash
# offline starters (sklearn)
feature-select datasets download breast_cancer wine diabetes_sklearn --out data

# standard public sets (needs network; OpenML via sklearn)
feature-select datasets download heart_disease pima_diabetes ionosphere sonar wine_quality_red --out data

# research-scale (larger)
feature-select datasets download madelon --out data

# recommendations
feature-select datasets list --goal medical
feature-select datasets list --goal high-dim
feature-select datasets list --goal paper
```

Each file is written as `data/{id}.csv` with the **target in the last column**.

---

## Catalog (built-in)

| id | Task | ~size | Difficulty | Source |
|----|------|-------|------------|--------|
| `breast_cancer` | classification | 569×30 | starter | sklearn |
| `wine` | classification | 178×13 | starter | sklearn |
| `diabetes_sklearn` | regression | 442×10 | starter | sklearn |
| `heart_disease` | classification | 270×13 | standard | OpenML 53 |
| `pima_diabetes` | classification | 768×8 | standard | OpenML 37 |
| `ionosphere` | classification | 351×34 | standard | OpenML 59 |
| `sonar` | classification | 208×60 | standard | OpenML 40 |
| `wine_quality_red` | regression | 1599×11 | standard | OpenML 40691 |
| `madelon` | classification | 2600×500 | research | OpenML 1485 |

---

## Suggested run order (research)

1. **`breast_cancer`** — sanity check all methods  
2. **`heart_disease`** — domain story for a paper/thesis  
3. **`sonar`** — where selection often improves or compresses well  
4. **`madelon`** (optional) — high-dimensional noise features  

```bash
# compare methods on sonar
feature-select compare data/sonar.csv -k 15 \
  --methods anova,mutual_info,random_forest,lasso,rfe \
  --out results/sonar

# nested CV for publication-style estimate
feature-select nested-cv data/sonar.csv -k 15 --method random_forest --cv 5 --out results/sonar_ncv

# multi-dataset benchmark script
python examples/benchmark_datasets.py \
  --datasets breast_cancer,heart_disease,sonar \
  --k 10 --out examples/benchmark_out --download
```

---

## Python API

```python
from feature_selector import load_benchmark, download_datasets, list_datasets

print(list_datasets())
download_datasets(["breast_cancer", "sonar"], data_dir="data")
X, y, meta = load_benchmark("sonar")
print(meta.why, X.shape)
```

---

## Manual downloads (if OpenML is blocked)

If `fetch_openml` fails (firewall / offline):

1. Use sklearn-only sets (always work).  
2. Or download CSVs from [UCI](https://archive.ics.uci.edu/) / [OpenML](https://www.openml.org/) in a browser and place them in `data/` with the target as the **last column**.  
3. Run: `feature-select compare data/myfile.csv -k 10 --out results/myfile`
