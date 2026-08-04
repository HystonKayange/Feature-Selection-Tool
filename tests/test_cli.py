"""CLI and export tests."""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from feature_selector.cli import main
from feature_selector.selector import FeatureSelector, normalize_score


@pytest.fixture
def clf_csv(tmp_path: Path) -> Path:
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=3, random_state=1
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y
    path = tmp_path / "clf.csv"
    df.to_csv(path, index=False)
    return path


def test_normalize_score_aliases():
    assert normalize_score("anova") == "f_score"
    assert normalize_score("MI") == "mutual_info"
    assert normalize_score("mutual-info") == "mutual_info"
    with pytest.raises(ValueError):
        normalize_score("shap")


def test_cli_run_writes_artifacts(clf_csv: Path, tmp_path: Path):
    out = tmp_path / "results"
    code = main(
        [
            "run",
            str(clf_csv),
            "--target",
            "label",
            "--task",
            "classification",
            "--k",
            "3",
            "--method",
            "anova",
            "--out",
            str(out),
            "--no-plots",
            "-q",
        ]
    )
    assert code == 0
    assert (out / "selected_features.json").is_file()
    assert (out / "scores.csv").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "report.html").is_file()

    payload = json.loads((out / "selected_features.json").read_text())
    assert payload["n_selected"] == 3


def test_cli_legacy_positional(clf_csv: Path, tmp_path: Path):
    """Legacy: feature-select data.csv -k 3 --no-plots still works."""
    out = tmp_path / "legacy"
    code = main(
        [
            str(clf_csv),
            "-k",
            "2",
            "--method",
            "rf",
            "--out",
            str(out),
            "--no-plots",
            "-q",
        ]
    )
    assert code == 0
    assert (out / "selected_features.json").is_file()


def test_cli_method_mi(clf_csv: Path, tmp_path: Path):
    out = tmp_path / "m"
    code = main(
        [
            "run",
            str(clf_csv),
            "--method",
            "mutual_info",
            "--task",
            "clf",
            "-k",
            "2",
            "--out",
            str(out),
            "--no-plots",
            "-q",
        ]
    )
    assert code == 0
    assert (out / "report.html").is_file()


def test_cli_stability(clf_csv: Path, tmp_path: Path):
    out = tmp_path / "stab"
    code = main(
        [
            "stability",
            str(clf_csv),
            "-k",
            "3",
            "--method",
            "anova",
            "--cv",
            "3",
            "--out",
            str(out),
            "-q",
        ]
    )
    assert code == 0
    assert (out / "stability_frequencies.csv").is_file()
    assert (out / "consensus_features.json").is_file()


def test_cli_missing_k(clf_csv: Path):
    with pytest.raises(SystemExit):
        main(["run", str(clf_csv), "--task", "classification", "--no-plots"])


def test_export_api(clf_csv: Path, tmp_path: Path):
    df = pd.read_csv(clf_csv)
    X = df.drop(columns=["label"])
    y = df["label"]
    sel = FeatureSelector(k=2, task="auto", method="anova")
    sel.fit(X, y)
    assert sel.task_ == "classification"
    assert len(sel.selected_features_) == 2

    out = tmp_path / "api_out"
    written = sel.export_artifacts(out, metrics={"accuracy": 0.9})
    assert written["report"].is_file()
    sel.plot_importance(show=False, save_path=str(out / "filter_scores.png"))
    assert (out / "filter_scores.png").is_file()


def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
