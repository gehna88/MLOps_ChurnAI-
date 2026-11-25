"""Sequential drift study: slice the dataset, treat each slice as newly arrived data.

The procedure, for slices S1..Sn of the validated dataset:

* step k (k = 1..n-1): a model trained on S1..Sk meets S(k+1) as "new data"
* data drift of S(k+1) is measured against S1 (the initial data) and against
  S1..Sk (everything the current model was trained on): PSI for every feature,
  the Kolmogorov-Smirnov test and summary statistics for the numeric ones
* prediction drift compares the model's out-of-fold probabilities on its own
  training data with its probabilities on the new slice
* performance on the new slice is recorded for the retrained model and for the
  stale model trained on S1 only, so the cost of *not* retraining is visible
* the next step retrains on S1..S(k+1)

This module is pure computation (no plotting, no MLflow) so it can be unit-tested
and imported by the API. :mod:`churnai.pipelines.drift_study` adds graphs,
the report and experiment tracking around :func:`iterate_steps`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline

from churnai.config import DriftScenario, Settings
from churnai.data.preparation import drop_columns, select_model_columns
from churnai.data.schema import FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET
from churnai.data.validation import DataValidator
from churnai.exceptions import ConfigError
from churnai.features.preprocessing import build_pipeline
from churnai.models.bundle import ModelBundle
from churnai.models.evaluation import compute_metrics, tune_threshold
from churnai.models.factory import BuildContext, ModelSpec, build_estimator, load_model_specs
from churnai.models.training import ModelTrainer, predict_positive
from churnai.monitoring.drift import PREDICTION_KEY, PSI_DECIMALS, FeatureReference

ROW_ORDER_COLUMN = "RowNumber"
# Files the pipeline writes to drift_study.output_dir; the API reads them.
RESULTS_FILE = "results.json"
REPORT_FILE = "drift_study.md"
COMPARISON_PLOT = "scenario_comparison.png"
CHAMPION = "champion"
STAT_DECIMALS = 4
MIN_SLICES = 2
QUANTILES = {"p05": 0.05, "p50": 0.50, "p95": 0.95}

LEVEL_STABLE = "stable"
LEVEL_MODERATE = "moderate"
LEVEL_SIGNIFICANT = "significant"


# ------------------------------------------------------------------ data ---
def load_study_data(settings: Settings) -> pd.DataFrame:
    """The validated, leakage-free dataset in its original row order."""
    from churnai.data import io

    raw = io.read_csv(settings.resolve(settings.paths.raw_data))
    if ROW_ORDER_COLUMN in raw.columns:
        raw = raw.sort_values(ROW_ORDER_COLUMN, kind="stable")
    valid = DataValidator(require_target=True).validate(raw).valid
    dropped = [*settings.data.identifier_columns, *settings.data.leakage_columns]
    return select_model_columns(drop_columns(valid, dropped)).reset_index(drop=True)


def order_rows(
    df: pd.DataFrame, scenario: DriftScenario, features: list[str], seed: int
) -> pd.DataFrame:
    """Arrange the rows for ``scenario``. Each row appears exactly once.

    A drift scenario scores every row as ``mean_shift * z + spread * |z|`` averaged
    over the drift features (z = standardised value), adds Gaussian noise scaled by
    ``noise`` and sorts by the result. Low scores land in the early slices, so the
    later slices hold higher values (mean shift) and more extreme ones (more variety).
    """
    if scenario.is_natural:
        return df.reset_index(drop=True)
    values = df[features].astype(float)
    z = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
    signal = (scenario.mean_shift * z + scenario.spread * (z.abs() - z.abs().mean())).mean(axis=1)
    signal = (signal - signal.mean()) / (signal.std(ddof=0) or 1.0)
    noise = np.random.default_rng(seed).standard_normal(len(df)) * scenario.noise
    order = np.argsort(signal.to_numpy() + noise, kind="stable")
    return df.iloc[order].reset_index(drop=True)


def make_slices(df: pd.DataFrame, n_slices: int) -> list[pd.DataFrame]:
    """Consecutive, near-equal slices (sizes differ by at most one row)."""
    if n_slices < MIN_SLICES:
        raise ValueError("A drift study needs at least two slices")
    if len(df) < n_slices:
        raise ValueError(f"Cannot cut {len(df)} rows into {n_slices} slices")
    bounds = np.linspace(0, len(df), n_slices + 1).round().astype(int)
    return [df.iloc[a:b].reset_index(drop=True) for a, b in pairwise(bounds)]


# --------------------------------------------------------------- measures ---
def psi_level(value: float, warning: float, alert: float) -> str:
    if value > alert:
        return LEVEL_SIGNIFICANT
    return LEVEL_MODERATE if value > warning else LEVEL_STABLE


def feature_psi(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], bins: int
) -> dict[str, float]:
    """PSI of every feature in ``current`` against ``reference`` (bins fitted on the reference)."""
    return {f: FeatureReference.fit(f, reference[f], bins).psi_against(current[f]) for f in features}


def ks_tests(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str]
) -> dict[str, dict[str, float]]:
    """Two-sample Kolmogorov-Smirnov test: largest gap between the two cumulative distributions."""
    out = {}
    for f in features:
        # Asymptotic p-values: slices hold hundreds of rows, where the exact method is slow and fails.
        result = ks_2samp(reference[f].astype(float), current[f].astype(float), method="asymp")
        out[f] = {"statistic": round(float(result.statistic), STAT_DECIMALS),
                  "pvalue": float(f"{result.pvalue:.4g}")}
    return out


def describe(values: pd.Series) -> dict[str, float]:
    values = values.astype(float)
    stats = {"mean": values.mean(), "std": values.std(ddof=0), "min": values.min(), "max": values.max()}
    stats |= {name: values.quantile(q) for name, q in QUANTILES.items()}
    return {k: round(float(v), STAT_DECIMALS) for k, v in stats.items()}


# ------------------------------------------------------------------ model ---
def resolve_model_spec(settings: Settings) -> ModelSpec:
    """The model configuration retrained at each step."""
    specs = {s.name: s for s in load_model_specs(settings.resolve(settings.training.models_config))}
    name = settings.drift_study.model
    if name != CHAMPION:
        if name not in specs:
            raise ConfigError(
                f"drift_study.model '{name}' is not a candidate in models.yaml: {sorted(specs)}"
            )
        return specs[name]
    if not settings.model_path.is_file():
        raise ConfigError(
            "drift_study.model is 'champion' but no trained model exists. "
            "Run `make train` first, or set drift_study.model to a candidate name."
        )
    meta = ModelBundle.load(settings.model_path).metadata
    # Prefer the catalogue entry: its "auto" params are re-resolved for every training set.
    return specs.get(meta.model_name) or ModelSpec(meta.model_name, meta.estimator, dict(meta.params))


@dataclass
class StepModel:
    pipeline: Pipeline
    threshold: float
    oof_proba: np.ndarray


def fit_step_model(
    spec: ModelSpec, X: pd.DataFrame, y: pd.Series, settings: Settings
) -> StepModel:
    """Same protocol as training: OOF probabilities tune the threshold, then refit on all rows."""
    seed = settings.project.random_seed
    metric = settings.training.threshold_metric
    trainer = ModelTrainer(settings.training.cv_folds, seed, metric)
    oof = trainer.cv_proba(spec, X, y)
    threshold = tune_threshold(y.to_numpy(), oof, metric)
    pipeline = build_pipeline(build_estimator(spec, BuildContext.from_target(y.to_numpy(), seed)))
    return StepModel(pipeline.fit(X, y), threshold, oof)


# ---------------------------------------------------------------- results ---
@dataclass
class SliceSummary:
    index: int  # 1-based
    rows: int
    churn_rate: float
    stats: dict[str, dict[str, float]]


@dataclass
class StepResult:
    step: int
    train_slices: list[int]
    new_slice: int
    train_rows: int
    new_rows: int
    psi_vs_initial: dict[str, float]
    psi_vs_training: dict[str, float]
    psi_levels: dict[str, str]  # level of psi_vs_initial
    ks_vs_initial: dict[str, dict[str, float]]
    ks_vs_training: dict[str, dict[str, float]]
    drifted_features: list[str]
    prediction_psi: float
    mean_predicted: float
    actual_churn_rate: float
    threshold: float
    retrained_metrics: dict[str, float]
    stale_metrics: dict[str, float]
    plots: list[str] = field(default_factory=list)
    mlflow_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepData:
    """Frames behind one step, handed to the plotting code alongside the result."""

    initial: pd.DataFrame
    training: pd.DataFrame
    new: pd.DataFrame


def summarise_slices(slices: list[pd.DataFrame], features: list[str]) -> list[SliceSummary]:
    return [
        SliceSummary(i, len(s), round(float(s[TARGET].mean()), STAT_DECIMALS),
                     {f: describe(s[f]) for f in features})
        for i, s in enumerate(slices, start=1)
    ]


def iterate_steps(
    slices: list[pd.DataFrame], spec: ModelSpec, settings: Settings
) -> Iterator[tuple[StepResult, StepData]]:
    """Run the study one step at a time, so callers can plot and log each iteration."""
    features = list(FEATURE_COLUMNS)
    numeric = [f for f in features if f in NUMERIC_COLUMNS]
    bins = settings.monitoring.psi_bins
    alert = settings.monitoring.psi_alert_threshold
    warning = settings.drift_study.psi_warning_threshold

    initial = slices[0]
    stale: StepModel | None = None
    for k in range(1, len(slices)):
        training = pd.concat(slices[:k], ignore_index=True)
        new = slices[k]
        X_train, y_train = training[features], training[TARGET]
        X_new, y_new = new[features], new[TARGET]

        model = fit_step_model(spec, X_train, y_train, settings)
        stale = stale or model  # the step-1 model, never retrained
        proba = predict_positive(model.pipeline, X_new)
        stale_proba = predict_positive(stale.pipeline, X_new)

        psi_initial = feature_psi(initial, new, features, bins)
        prediction_ref = FeatureReference.fit(PREDICTION_KEY, pd.Series(model.oof_proba), bins)
        result = StepResult(
            step=k,
            train_slices=list(range(1, k + 1)),
            new_slice=k + 1,
            train_rows=len(training),
            new_rows=len(new),
            psi_vs_initial=psi_initial,
            psi_vs_training=feature_psi(training, new, features, bins),
            psi_levels={f: psi_level(v, warning, alert) for f, v in psi_initial.items()},
            ks_vs_initial=ks_tests(initial, new, numeric),
            ks_vs_training=ks_tests(training, new, numeric),
            drifted_features=sorted(f for f, v in psi_initial.items() if v > alert),
            prediction_psi=prediction_ref.psi_against(pd.Series(proba)),
            mean_predicted=round(float(proba.mean()), PSI_DECIMALS),
            actual_churn_rate=round(float(y_new.mean()), PSI_DECIMALS),
            threshold=model.threshold,
            retrained_metrics=compute_metrics(y_new.to_numpy(), proba, model.threshold),
            stale_metrics=compute_metrics(y_new.to_numpy(), stale_proba, stale.threshold),
        )
        yield result, StepData(initial, training, new)
