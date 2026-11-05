"""Global SHAP report for one model: what drives its predictions across customers.

Kept apart from the explainer because plotting needs matplotlib, which the serving
image does not install. Uses a sample of the training split, so the test split stays
untouched. Values are in log-odds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display: this runs in pipelines and CI

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churnai.config import Settings
from churnai.data.schema import FEATURE_COLUMNS
from churnai.explain.explainer import ChurnExplainer
from churnai.models.training import sample_background

SHAP_ARTIFACT_PATH = "shap"
IMPORTANCE_FILE = "shap_importance.csv"
BAR_FILE = "shap_bar.png"
BEESWARM_FILE = "shap_beeswarm.png"
IMPORTANCE_DECIMALS = 4
PLOT_DPI = 150


def _save_plot(values: pd.DataFrame, features: pd.DataFrame, path: Path, plot_type: str, title: str) -> Path:
    # Text features such as Geography have no colour scale, so SHAP draws their dots in grey.
    shap.summary_plot(
        values.to_numpy(), features, feature_names=list(values.columns),
        plot_type=plot_type, max_display=len(values.columns), show=False,
    )
    plt.title(title)
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    return path


def write_shap_report(
    pipeline: Pipeline, X_train: pd.DataFrame, settings: Settings, directory: Path
) -> list[Path]:
    """Write the importance table and both plots to ``directory``; returns their paths."""
    seed = settings.project.random_seed
    explainer = ChurnExplainer.from_pipeline(
        pipeline, list(FEATURE_COLUMNS),
        sample_background(X_train, settings.explain.background_size, seed), settings.explain.top_k,
    )
    sample = sample_background(X_train, settings.explain.report_sample_size, seed)
    values, features = explainer.contributions(sample)

    importance = (
        values.abs().mean().sort_values(ascending=False).round(IMPORTANCE_DECIMALS)
        .rename_axis("feature").reset_index(name="mean_abs_shap")
    )
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / IMPORTANCE_FILE
    importance.to_csv(csv_path, index=False)
    return [
        csv_path,
        _save_plot(values, features, directory / BAR_FILE, "bar", "Average impact on churn log-odds"),
        _save_plot(values, features, directory / BEESWARM_FILE, "dot", "Churn log-odds impact per customer"),
    ]
