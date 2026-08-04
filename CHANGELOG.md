# Changelog

## 0.5.0 — Next-level polish

- **sklearn 1.8+ L1 logistic** via `l1_ratio=1.0` (no deprecated `penalty='l1'` warnings)
- **Seeded mutual information** (`random_state`) for reproducible nested CV
- New filter method: **`chi2`** (classification; auto non-negative shift)
- Performance knobs: `fast=`, `n_jobs=`, `n_estimators=` on `FeatureSelector`
- Nested CV defaults to **fast mode** (much quicker on wide data like Madelon)
- PyPI readiness: `MANIFEST.in`, `py.typed`, `PUBLISHING.md`, publish workflow
- Version **0.5.0**

## 0.4.0 — Phase 3

- Curated **dataset catalog** with download helpers (`feature-select datasets`)
- Recommended public benchmarks (sklearn + OpenML)
- **Nested CV** API + CLI (`feature-select nested-cv`)
- Multi-dataset **benchmark script** (`examples/benchmark_datasets.py`)
- MkDocs documentation (`docs/`)
- GitHub issue templates
- Version bump for distribution readiness

## 0.3.0 — Phase 2

- Multi-method selection: ANOVA, MI, Random Forest, Lasso, RFE
- Before/after CV metrics, stability selection, Jaccard agreement
- Comparison export + Streamlit research UI

## 0.2.0 — Phase 1

- Installable package, CLI, artifact export, CI

## 0.1.0 — Phase 0

- Leakage-safe filter selection, tests, package layout
