#!/usr/bin/env python3
"""Research-style comparison on a synthetic classification problem.

Run from repo root (with package installed):

  python examples/compare_synthetic.py
  python examples/compare_synthetic.py --out examples/output --k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from feature_selector import compare_methods, stability_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--out", type=str, default="examples/output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=400,
        n_features=20,
        n_informative=5,
        n_redundant=5,
        n_repeated=0,
        random_state=args.seed,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")

    print("Running multi-method comparison…")
    result = compare_methods(
        X,
        y,
        methods=["anova", "mutual_info", "random_forest", "lasso", "rfe"],
        k=args.k,
        task="classification",
        cv=args.cv,
        stability_splits=5,
        random_state=args.seed,
    )

    out = Path(args.out)
    written = result.export(out)

    print("\n=== Summary (sorted by selected-feature CV metric) ===")
    print(result.summary.to_string(index=False))
    print("\n=== Jaccard similarity of selected sets ===")
    print(result.jaccard.round(3).to_string())

    print("\n=== Per-method selections ===")
    for method, feats in result.selections.items():
        print(f"{method}: {feats}")

    print("\n=== ANOVA stability (top 10 frequencies) ===")
    stab = stability_selection(
        X, y, method="anova", k=args.k, n_splits=5, random_state=args.seed
    )
    print(stab.frequencies.head(10).to_string(index=False))
    print(f"Consensus (freq≥{stab.threshold}): {stab.consensus_features}")

    print(f"\nArtifacts: {out.resolve()}")
    for name, path in written.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
