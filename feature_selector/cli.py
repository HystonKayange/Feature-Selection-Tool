"""Command-line interface for Feature Selector Tool.

Subcommands
-----------
  run         Single-method selection + baseline hold-out metrics (default)
  compare     Multi-method comparison (before/after CV, stability, Jaccard)
  stability   Cross-validated stability selection for one method
  ui          Launch Streamlit research UI (requires streamlit extra)
  datasets    List / download recommended public benchmarks
  nested-cv   Unbiased nested cross-validation for one method

Examples
--------
  feature-select data.csv -k 10 --task classification --out results/
  feature-select compare data.csv -k 10 --methods anova,mi,rf,lasso,rfe --out cmp/
  feature-select stability data.csv -k 8 --method random_forest --cv 5 --out stab/
  feature-select datasets list
  feature-select datasets download --out data
  feature-select nested-cv data/breast_cancer.csv -k 10 --method rf --out ncv/
  feature-select ui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from feature_selector import __version__


def _parse_target(raw: Optional[str]):
    if raw is None:
        return None
    if raw.isdigit():
        return int(raw)
    return raw


def _parse_methods(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("data", help="Path to CSV/TXT dataset")
    p.add_argument(
        "-t",
        "--target",
        default=None,
        help="Target column name or index (default: last column)",
    )
    p.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression", "clf", "reg"],
        help="Learning task",
    )
    p.add_argument(
        "--delimiter",
        default=",",
        help="Field delimiter",
    )
    p.add_argument(
        "--no-header",
        action="store_true",
        help="File has no header row",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-select",
        description="Leakage-safe feature selection and research comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    # ---- run (also legacy default) ----
    run_p = sub.add_parser(
        "run",
        help="Single-method selection + hold-out evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(run_p)
    run_p.add_argument("-k", "--k", type=int, required=True, help="Features to keep")
    run_p.add_argument(
        "--method",
        default="anova",
        help="Method: anova, mutual_info/mi, random_forest/rf, lasso, rfe",
    )
    run_p.add_argument(
        "--score",
        default=None,
        help="Filter score override (f_score/mutual_info); usually set via --method",
    )
    run_p.add_argument("-o", "--out", default=None, help="Output directory")
    run_p.add_argument("--test-size", type=float, default=0.2)
    run_p.add_argument("--no-plots", action="store_true")
    run_p.add_argument("-q", "--quiet", action="store_true")

    # ---- compare ----
    cmp_p = sub.add_parser(
        "compare",
        help="Compare multiple selection methods (research)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(cmp_p)
    cmp_p.add_argument("-k", "--k", type=int, required=True, help="Features to keep")
    cmp_p.add_argument(
        "--methods",
        default="anova,mutual_info,random_forest,lasso,rfe",
        help="Comma-separated methods",
    )
    cmp_p.add_argument("--cv", type=int, default=5, help="CV folds for metrics")
    cmp_p.add_argument(
        "--stability-splits",
        type=int,
        default=5,
        help="Folds for stability selection",
    )
    cmp_p.add_argument(
        "--stability-threshold",
        type=float,
        default=0.6,
        help="Consensus frequency threshold",
    )
    cmp_p.add_argument(
        "--no-stability",
        action="store_true",
        help="Skip stability estimation (faster)",
    )
    cmp_p.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Drop correlated features above this |r| (0 to disable)",
    )
    cmp_p.add_argument(
        "--keep-constant",
        action="store_true",
        help="Do not drop constant features",
    )
    cmp_p.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output directory for comparison artifacts",
    )
    cmp_p.add_argument("-q", "--quiet", action="store_true")

    # ---- stability ----
    stab_p = sub.add_parser(
        "stability",
        help="CV stability selection for one method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(stab_p)
    stab_p.add_argument("-k", "--k", type=int, required=True)
    stab_p.add_argument(
        "--method",
        default="anova",
        help="Selection method",
    )
    stab_p.add_argument("--cv", type=int, default=5, help="Number of folds")
    stab_p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Consensus frequency threshold",
    )
    stab_p.add_argument("-o", "--out", required=True, help="Output directory")
    stab_p.add_argument("-q", "--quiet", action="store_true")

    # ---- ui ----
    ui_p = sub.add_parser("ui", help="Launch Streamlit research UI")
    ui_p.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port",
    )

    # ---- datasets ----
    ds_p = sub.add_parser(
        "datasets",
        help="List or download recommended benchmark datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ds_sub = ds_p.add_subparsers(dest="datasets_command")
    list_p = ds_sub.add_parser("list", help="Show catalog")
    list_p.add_argument(
        "--difficulty",
        choices=["starter", "standard", "research"],
        default=None,
    )
    list_p.add_argument(
        "--goal",
        default=None,
        help="Recommend for goal: first, medical, high-dim, regression, paper",
    )
    dl_p = ds_sub.add_parser("download", help="Cache datasets as CSV under --out")
    dl_p.add_argument(
        "ids",
        nargs="*",
        help="Dataset ids (default: starter+standard, not madelon)",
    )
    dl_p.add_argument(
        "-o",
        "--out",
        default="data",
        help="Output directory",
    )
    dl_p.add_argument(
        "--research",
        action="store_true",
        help="When no ids given, also include research sets (madelon)",
    )
    dl_p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if CSV exists",
    )

    # ---- nested-cv ----
    ncv_p = sub.add_parser(
        "nested-cv",
        help="Nested CV (selection inside outer train folds)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(ncv_p)
    ncv_p.add_argument("-k", "--k", type=int, required=True)
    ncv_p.add_argument("--method", default="anova")
    ncv_p.add_argument("--cv", type=int, default=5, help="Outer folds")
    ncv_p.add_argument("-o", "--out", required=True, help="Output directory")
    ncv_p.add_argument("-q", "--quiet", action="store_true")

    # ---- interactive ----
    sub.add_parser("interactive", help="Interactive file-dialog mode")

    return parser


def _cmd_run(args) -> int:
    from feature_selector.app import run_pipeline
    from feature_selector.selector import normalize_method, normalize_score

    method_c, score_override = normalize_method(args.method)
    score = score_override or (
        normalize_score(args.score) if args.score else "f_score"
    )

    # Map method for FeatureSelector: pass family name
    method_for_sel = method_c
    # For filter aliases user may have passed anova — method_c is filter

    result = run_pipeline(
        args.data,
        task=args.task,
        k=args.k,
        score=score,
        method=method_for_sel,
        target_column=_parse_target(args.target),
        delimiter=args.delimiter,
        header=not args.no_header,
        test_size=args.test_size,
        random_state=args.seed,
        show_plots=not args.no_plots,
        out_dir=args.out,
        quiet=args.quiet,
    )
    if args.quiet:
        print(",".join(result["selected_features"]))
    return 0


def _cmd_compare(args) -> int:
    from feature_selector.compare import compare_methods
    from feature_selector.data import load_dataset

    ds = load_dataset(
        args.data,
        delimiter=args.delimiter,
        header=not args.no_header,
        target_column=_parse_target(args.target),
    )
    methods = _parse_methods(args.methods)
    corr = args.corr_threshold
    if corr is not None and corr <= 0:
        corr = None

    if not args.quiet:
        print(
            f"Comparing methods={methods} k={args.k} task={args.task} "
            f"cv={args.cv} on {ds.n_samples}×{ds.n_features} ..."
        )

    result = compare_methods(
        ds.features,
        ds.labels,
        methods=methods,
        k=args.k,
        task=args.task,
        cv=args.cv,
        stability_splits=args.stability_splits,
        stability_threshold=args.stability_threshold,
        random_state=args.seed,
        drop_constant=not args.keep_constant,
        correlation_threshold=corr,
        run_stability=not args.no_stability,
    )
    written = result.export(args.out)

    if not args.quiet:
        print("\n=== Comparison summary ===")
        print(result.summary.to_string(index=False))
        print("\n=== Jaccard similarity ===")
        print(result.jaccard.round(3).to_string())
        print(f"\nArtifacts written to: {Path(args.out).resolve()}")
        for name, path in written.items():
            print(f"  - {name}: {path}")
    return 0


def _cmd_stability(args) -> int:
    from feature_selector.data import load_dataset
    from feature_selector.stability import stability_selection

    ds = load_dataset(
        args.data,
        delimiter=args.delimiter,
        header=not args.no_header,
        target_column=_parse_target(args.target),
    )
    result = stability_selection(
        ds.features,
        ds.labels,
        method=args.method,
        k=args.k,
        task=args.task,
        n_splits=args.cv,
        threshold=args.threshold,
        random_state=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    freq_path = out / "stability_frequencies.csv"
    result.frequencies.to_csv(freq_path, index=False)
    consensus_path = out / "consensus_features.json"
    import json

    consensus_path.write_text(
        json.dumps(
            {
                "method": result.method,
                "threshold": result.threshold,
                "n_splits": result.n_splits,
                "mean_stability": result.mean_stability,
                "consensus_features": result.consensus_features,
                "fold_selections": result.fold_selections,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"Method: {result.method} | task: {result.task} | folds: {result.n_splits}")
        print(f"Mean stability (features selected ≥1 fold): {result.mean_stability:.3f}")
        print(f"Consensus features (freq ≥ {result.threshold}):")
        for f in result.consensus_features:
            print(f"  - {f}")
        print("\nTop frequencies:")
        print(result.frequencies.head(15).to_string(index=False))
        print(f"\nWrote {freq_path} and {consensus_path}")
    return 0


def _cmd_ui(args) -> int:
    import shutil
    import subprocess

    if shutil.which("streamlit") is None:
        print(
            "Streamlit is not installed. Install with:\n"
            '  pip install -e ".[ui]"\n'
            "or: pip install streamlit",
            file=sys.stderr,
        )
        return 1
    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    return subprocess.call(cmd)


def _cmd_datasets(args) -> int:
    from feature_selector.datasets import (
        download_datasets,
        list_datasets,
        recommend_for_goal,
    )

    if args.datasets_command == "list" or args.datasets_command is None:
        if getattr(args, "goal", None):
            recs = recommend_for_goal(args.goal)
            print(f"Recommended for goal '{args.goal}':\n")
            for info in recs:
                print(
                    f"  {info.id:18s}  [{info.difficulty:8s}]  "
                    f"{info.task:14s}  ~{info.n_samples_approx}×{info.n_features_approx}"
                )
                print(f"    {info.why}")
            print(
                "\nDownload: feature-select datasets download "
                + " ".join(r.id for r in recs)
                + " --out data"
            )
            return 0
        df = list_datasets(difficulty=getattr(args, "difficulty", None))
        # readable console table
        cols = [
            "id",
            "task",
            "n_samples_approx",
            "n_features_approx",
            "difficulty",
            "why",
        ]
        print(df[cols].to_string(index=False))
        print(
            "\nDownload starters:  feature-select datasets download "
            "breast_cancer wine diabetes_sklearn --out data"
        )
        print(
            "Download standard:  feature-select datasets download "
            "heart_disease pima_diabetes ionosphere sonar --out data"
        )
        return 0

    if args.datasets_command == "download":
        ids = list(args.ids) if args.ids else None
        print(f"Downloading to {Path(args.out).resolve()} …")
        written = download_datasets(
            ids,
            data_dir=args.out,
            force=args.force,
            include_research=args.research,
        )
        print(f"Done ({len(written)} files).")
        return 0

    print("Usage: feature-select datasets {list,download}", file=sys.stderr)
    return 1


def _cmd_nested_cv(args) -> int:
    import json

    from feature_selector.data import load_dataset
    from feature_selector.nested_cv import nested_cv_feature_selection

    ds = load_dataset(
        args.data,
        delimiter=args.delimiter,
        header=not args.no_header,
        target_column=_parse_target(args.target),
    )
    result = nested_cv_feature_selection(
        ds.features,
        ds.labels,
        method=args.method,
        k=args.k,
        task=args.task,
        outer_splits=args.cv,
        random_state=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result["selection_frequency"].to_csv(
        out / "selection_frequency.csv", index=False
    )
    payload = {
        k: v
        for k, v in result.items()
        if k != "selection_frequency"
    }
    # fold metrics as JSON-serializable
    (out / "nested_cv_result.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            f"Nested CV | method={result['method']} task={result['task']} "
            f"k={result['k']} folds={result['outer_splits']}"
        )
        print("Mean metrics:", result["mean_metrics"])
        print("Std metrics:", result["std_metrics"])
        print("Top selection frequencies:")
        print(result["selection_frequency"].head(15).to_string(index=False))
        print(f"Wrote {out.resolve()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Legacy / bare invocation compatibility:
    #   feature-select                 → interactive
    #   feature-select data.csv -k 5   → run
    #   feature-select compare ...     → compare
    known = {
        "run",
        "compare",
        "stability",
        "ui",
        "interactive",
        "datasets",
        "nested-cv",
        "help",
    }
    if not argv:
        from feature_selector.app import run_interactive

        run_interactive()
        return 0

    if argv[0] not in known and not argv[0].startswith("-"):
        # treat as legacy "run" with positional data
        argv = ["run"] + argv
    elif argv[0] in {"-h", "--help"}:
        pass
    elif argv[0].startswith("-") and argv[0] not in {"--version"}:
        # flags only without subcommand → inject run if looks like old CLI
        if any(a == "-k" or a.startswith("-k") or a == "--k" for a in argv):
            # need data though
            argv = ["run"] + argv

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "compare":
            return _cmd_compare(args)
        if args.command == "stability":
            return _cmd_stability(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "datasets":
            return _cmd_datasets(args)
        if args.command == "nested-cv":
            return _cmd_nested_cv(args)
        if args.command == "interactive":
            from feature_selector.app import run_interactive

            run_interactive()
            return 0
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
