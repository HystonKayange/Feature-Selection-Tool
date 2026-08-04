#!/usr/bin/env python3
"""Benchmark selection methods on recommended public datasets.

Examples
--------
  # starter sets only (sklearn, offline)
  python examples/benchmark_datasets.py --datasets breast_cancer,wine --k 10

  # download standard sets then benchmark
  feature-select datasets download --out data
  python examples/benchmark_datasets.py --datasets heart_disease,sonar,pima_diabetes \\
      --k 10 --out examples/benchmark_out

  # nested CV (slower, better for papers)
  python examples/benchmark_datasets.py --datasets breast_cancer --nested --k 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from feature_selector.compare import compare_methods
from feature_selector.datasets import DATASET_CATALOG, download_datasets, load_benchmark
from feature_selector.nested_cv import nested_cv_compare_methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="breast_cancer,wine,diabetes_sklearn",
        help="Comma-separated dataset ids from the catalog",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--methods",
        default="anova,mutual_info,random_forest,lasso,rfe",
        help="Comma-separated methods",
    )
    parser.add_argument("--out", type=str, default="examples/benchmark_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Use nested CV (unbiased) instead of exploratory compare_methods",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Force refresh/cache CSVs under ./data before running",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset cache directory",
    )
    args = parser.parse_args()

    ids = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    unknown = [i for i in ids if i not in DATASET_CATALOG]
    if unknown:
        raise SystemExit(
            f"Unknown datasets: {unknown}. Known: {sorted(DATASET_CATALOG)}"
        )

    if args.download:
        print("Caching datasets…")
        download_datasets(ids, data_dir=args.data_dir, force=False)

    all_rows = []
    for dataset_id in ids:
        print("\n" + "=" * 72)
        print(f"Dataset: {dataset_id}")
        print("=" * 72)
        X, y, info = load_benchmark(dataset_id, data_dir=args.data_dir, save=True)
        print(f"{info.name} | task={info.task} | {X.shape[0]}×{X.shape[1]}")
        print(f"Why: {info.why}")

        k_eff = min(args.k, X.shape[1])
        ds_out = out_root / dataset_id
        ds_out.mkdir(parents=True, exist_ok=True)

        if args.nested:
            summary = nested_cv_compare_methods(
                X,
                y,
                methods=methods,
                k=k_eff,
                task=info.task,
                outer_splits=args.cv,
                random_state=args.seed,
            )
            summary.insert(0, "dataset", dataset_id)
            summary.to_csv(ds_out / "nested_cv_summary.csv", index=False)
            print(summary.to_string(index=False))
            all_rows.append(summary)
        else:
            result = compare_methods(
                X,
                y,
                methods=methods,
                k=k_eff,
                task=info.task,
                cv=args.cv,
                stability_splits=min(5, args.cv),
                random_state=args.seed,
                # madelon is large — skip heavy stability if many features
                run_stability=X.shape[1] <= 100,
            )
            result.export(ds_out)
            summary = result.summary.copy()
            summary.insert(0, "dataset", dataset_id)
            summary.to_csv(ds_out / "summary_with_dataset.csv", index=False)
            print(summary.to_string(index=False))
            all_rows.append(summary)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = out_root / "all_datasets_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined summary → {combined_path.resolve()}")

    meta = {
        "datasets": ids,
        "methods": methods,
        "k": args.k,
        "cv": args.cv,
        "nested": args.nested,
    }
    (out_root / "benchmark_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Done.")


if __name__ == "__main__":
    main()
