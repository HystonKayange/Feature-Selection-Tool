"""Export selection results: JSON, CSV, and a simple HTML report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd


PathLike = Union[str, Path]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def export_selected_features(
    features: Sequence[str],
    path: PathLike,
) -> Path:
    """Write selected feature names to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"selected_features": list(features), "n_selected": len(features)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_scores(score_table: pd.DataFrame, path: PathLike) -> Path:
    """Write feature score table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(path, index=False)
    return path


def export_metrics(metrics: Dict[str, Any], path: PathLike) -> Path:
    """Write evaluation metrics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def export_html_report(
    path: PathLike,
    *,
    title: str = "Feature Selection Report",
    summary: Optional[Dict[str, Any]] = None,
    selected_features: Optional[Sequence[str]] = None,
    score_table: Optional[pd.DataFrame] = None,
    metrics: Optional[Dict[str, Any]] = None,
    image_paths: Optional[Dict[str, PathLike]] = None,
) -> Path:
    """Write a self-contained-enough HTML report (images referenced by relative path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    selected_features = list(selected_features or [])
    image_paths = image_paths or {}

    def esc(text: Any) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    rows_summary = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in summary.items()
    )
    feat_list = "".join(f"<li><code>{esc(f)}</code></li>" for f in selected_features)

    if score_table is not None and not score_table.empty:
        # Prefer selected first already sorted by score
        display = score_table.copy()
        scores_html = display.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    else:
        scores_html = "<p>No score table available.</p>"

    if metrics:
        metrics_rows = "".join(
            f"<tr><th>{esc(k)}</th><td>{esc(v if not isinstance(v, float) else f'{v:.4f}')}</td></tr>"
            for k, v in metrics.items()
        )
        metrics_html = f"<table class='kv'>{metrics_rows}</table>"
    else:
        metrics_html = "<p>No metrics recorded.</p>"

    images_html_parts: List[str] = []
    for label, img in image_paths.items():
        img_path = Path(img)
        # Prefer path relative to the report file
        try:
            rel = img_path.resolve().relative_to(path.parent.resolve())
        except Exception:
            rel = img_path.name if img_path.exists() else img_path
        if img_path.exists() or Path(path.parent, rel).exists():
            images_html_parts.append(
                f"<figure><figcaption>{esc(label)}</figcaption>"
                f'<img src="{esc(rel)}" alt="{esc(label)}" style="max-width:100%;"/></figure>'
            )
    images_html = "\n".join(images_html_parts) or "<p>No plots saved.</p>"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{esc(title)}</title>
  <style>
    :root {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: #1a1a1a; }}
    body {{ max-width: 920px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
    h1, h2 {{ color: #0f2744; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.45rem 0.6rem; text-align: left; }}
    th {{ background: #f4f7fb; }}
    table.kv th {{ width: 40%; }}
    code {{ background: #f0f3f7; padding: 0.1rem 0.35rem; border-radius: 4px; }}
    .meta {{ color: #666; font-size: 0.9rem; }}
    figure {{ margin: 1.5rem 0; }}
    figcaption {{ font-weight: 600; margin-bottom: 0.4rem; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <p class="meta">Generated {esc(generated)}</p>

  <h2>Summary</h2>
  <table class="kv">{rows_summary or "<tr><td>No summary</td></tr>"}</table>

  <h2>Selected features ({len(selected_features)})</h2>
  <ol>{feat_list or "<li>None</li>"}</ol>

  <h2>Test metrics</h2>
  {metrics_html}

  <h2>Feature scores</h2>
  {scores_html}

  <h2>Plots</h2>
  {images_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def export_run_artifacts(
    out_dir: PathLike,
    *,
    selected_features: Sequence[str],
    score_table: pd.DataFrame,
    metrics: Optional[Dict[str, Any]] = None,
    summary: Optional[Dict[str, Any]] = None,
    image_paths: Optional[Dict[str, PathLike]] = None,
) -> Dict[str, Path]:
    """Write the standard artifact bundle into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    written["selected_features"] = export_selected_features(
        selected_features, out / "selected_features.json"
    )
    written["scores"] = export_scores(score_table, out / "scores.csv")
    if metrics is not None:
        written["metrics"] = export_metrics(metrics, out / "metrics.json")
    written["report"] = export_html_report(
        out / "report.html",
        summary=summary,
        selected_features=selected_features,
        score_table=score_table,
        metrics=metrics,
        image_paths=image_paths,
    )
    return written
