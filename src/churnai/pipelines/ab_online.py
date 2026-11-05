"""Simulated online A/B test on the held-out test split.

Run with ``make ab-online`` or ``python -m churnai.pipelines.ab_online``, after ``make train``.

1. Fix the decision rule: a power calculation for the smallest recall gain worth having.
2. Replay each test customer through the real ``/predict`` and ``/feedback`` routes.
3. Join the logs and compare the arms: recall (two-proportion z-test) and cost per
   customer (bootstrap). B is promoted only if the test had enough power, recall is
   significantly higher, and cost is no worse.

Outputs: MLflow experiment ``churnai-ab`` and ``reports/ab_online.md``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churnai.api.ab import CUSTOMER_ID_HEADER, VARIANT_A, VARIANT_B
from churnai.config import Settings
from churnai.data import io
from churnai.data.preparation import stratified_split
from churnai.data.schema import API_TO_COLUMN, TARGET
from churnai.data.validation import DataValidator
from churnai.exceptions import DataValidationError
from churnai.logging_utils import configure_logging, get_logger
from churnai.models.bundle import ModelBundle
from churnai.models.comparison import (
    bootstrap_mean_diff,
    minimum_detectable_diff,
    required_n_per_arm,
    two_proportion_ztest,
)
from churnai.models.tracking import MlflowTracker, make_tracker
from churnai.pipelines import fetch_ab_bundles
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

ID_COLUMN = "CustomerId"
ALIGNMENT_COLUMNS = ["CreditScore", "Age", TARGET]
VERDICT_UNDERPOWERED = "inconclusive_underpowered"
VERDICT_PROMOTE = "promote_B"
VERDICT_KEEP = "keep_A"
METRIC_DECIMALS = 4


@dataclass(frozen=True)
class PowerPlan:
    baseline_recall: float
    min_detectable_diff: float
    required_churners_per_arm: int
    required_customers_per_arm: int


@dataclass(frozen=True)
class ArmStats:
    variant: str
    model_version: str
    customers: int
    churners: int
    true_positives: int
    false_positives: int
    false_negatives: int
    recall: float
    precision: float
    roc_auc: float
    cost_per_customer: float


def plan_power(settings: Settings, baseline_recall: float, churn_rate: float) -> PowerPlan:
    ab = settings.ab_test
    churners = required_n_per_arm(baseline_recall, ab.min_detectable_recall_diff, ab.alpha, ab.power)
    customers = int(np.ceil(churners / churn_rate))
    return PowerPlan(baseline_recall, ab.min_detectable_recall_diff, churners, customers)


def traffic_with_ids(settings: Settings) -> pd.DataFrame:
    """The test split with customer IDs, which prepare_data drops before writing it.

    Re-splitting with the same seed gives the same rows in the same order; the
    alignment check fails loudly if that ever stops being true.
    """
    raw = io.read_csv(settings.resolve(settings.paths.raw_data))
    valid = DataValidator(require_target=True, unique_key=ID_COLUMN).validate(raw).valid
    test = stratified_split(valid, settings.data.test_size, settings.project.random_seed).test
    prepared = io.read_csv(settings.test_path)
    if not test[ALIGNMENT_COLUMNS].equals(prepared[ALIGNMENT_COLUMNS]):
        raise DataValidationError("Re-derived test split does not match data/processed. Run `make data`.")
    return test


def replay(settings: Settings, traffic: pd.DataFrame) -> None:
    """Send every customer through the real API routes, then report the true outcome."""
    from fastapi.testclient import TestClient  # needs httpx, a dev-only dependency

    from churnai.api.app import create_app

    live = settings.model_copy(
        update={"ab_test": settings.ab_test.model_copy(update={"enabled": True})}
    )
    for path in (settings.ab_test.predictions_log, settings.ab_test.feedback_log):
        settings.resolve(path).unlink(missing_ok=True)

    payloads = traffic[list(API_TO_COLUMN.values())].rename(
        columns={column: api for api, column in API_TO_COLUMN.items()}
    ).to_dict(orient="records")
    ids = traffic[ID_COLUMN].astype(str).tolist()
    outcomes = traffic[TARGET].astype(int).tolist()

    with TestClient(create_app(live)) as client:
        # One log line per replayed request is just noise.
        for noisy in ("churnai.api", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        for customer_id, payload, exited in zip(ids, payloads, outcomes, strict=True):
            headers = {CUSTOMER_ID_HEADER: customer_id}
            client.post("/predict", json=payload, headers=headers).raise_for_status()
            client.post("/feedback", json={"customer_id": customer_id, "exited": exited}).raise_for_status()
    configure_logging(fmt="text")


def load_outcomes(settings: Settings) -> pd.DataFrame:
    """Join each customer's first exposure with their reported outcome."""
    predictions = pd.read_json(
        settings.resolve(settings.ab_test.predictions_log), lines=True, dtype={"customer_id": str}
    ).drop_duplicates("customer_id", keep="first")
    feedback = pd.read_json(
        settings.resolve(settings.ab_test.feedback_log), lines=True, dtype={"customer_id": str}
    ).drop_duplicates("customer_id", keep="last")
    return predictions.merge(feedback, on="customer_id", how="inner")


def _customer_costs(arm: pd.DataFrame, settings: Settings) -> np.ndarray:
    ab = settings.ab_test
    missed = (arm["exited"] == 1) & ~arm["churn_prediction"]
    wasted = (arm["exited"] == 0) & arm["churn_prediction"]
    return (missed * ab.cost_false_negative + wasted * ab.cost_false_positive).to_numpy(dtype=float)


def arm_stats(arm: pd.DataFrame, variant: str, settings: Settings) -> ArmStats:
    predicted, actual = arm["churn_prediction"], arm["exited"] == 1
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    return ArmStats(
        variant=variant,
        model_version=str(arm["model_version"].iloc[0]),
        customers=len(arm),
        churners=int(actual.sum()),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        recall=tp / max(tp + fn, 1),
        precision=tp / max(tp + fp, 1),
        roc_auc=float(roc_auc_score(actual, arm["churn_probability"])),
        cost_per_customer=float(_customer_costs(arm, settings).mean()),
    )


def analyze(outcomes: pd.DataFrame, plan: PowerPlan, settings: Settings) -> dict[str, Any]:
    ab = settings.ab_test
    arm_a = outcomes[outcomes["model_variant"] == VARIANT_A]
    arm_b = outcomes[outcomes["model_variant"] == VARIANT_B]
    a, b = arm_stats(arm_a, VARIANT_A, settings), arm_stats(arm_b, VARIANT_B, settings)

    z, p_value = two_proportion_ztest(a.true_positives, a.churners, b.true_positives, b.churners)
    cost = bootstrap_mean_diff(
        _customer_costs(arm_a, settings), _customer_costs(arm_b, settings),
        ab.bootstrap_iterations, settings.project.random_seed,
    )
    powered = min(a.churners, b.churners) >= plan.required_churners_per_arm
    if not powered:
        verdict = VERDICT_UNDERPOWERED
    elif p_value < ab.alpha and b.recall > a.recall and cost.mean_diff <= 0:
        verdict = VERDICT_PROMOTE
    else:
        verdict = VERDICT_KEEP
    return {
        "a": a, "b": b, "recall_z": z, "recall_p_value": p_value, "cost_diff": cost, "powered": powered,
        "detectable_diff": minimum_detectable_diff(
            plan.baseline_recall, min(a.churners, b.churners), ab.alpha, ab.power
        ),
        "verdict": verdict,
    }


def _write_report(result: dict[str, Any], plan: PowerPlan, settings: Settings):
    a, b, cost = result["a"], result["b"], result["cost_diff"]
    arms = pd.DataFrame([asdict(a), asdict(b)]).round(METRIC_DECIMALS).T
    path = settings.resolve(settings.paths.reports_dir) / "ab_online.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Online A/B test (simulated replay of the test split)\n\n"
        f"**Verdict: {result['verdict']}**\n\n"
        f"Planned before replay: detecting a {plan.min_detectable_diff:.0%} recall change from a "
        f"baseline of {plan.baseline_recall:.1%} needs {plan.required_churners_per_arm} churners per "
        f"arm (about {plan.required_customers_per_arm} customers).\n"
        f"Achieved: {a.churners} and {b.churners} churners, enough to detect about "
        f"{result['detectable_diff']} recall.\n\n"
        + arms.to_markdown(headers=["", "A", "B"]) + "\n\n"
        f"- Recall difference (B - A): {b.recall - a.recall:+.4f}, z = {result['recall_z']:.3f}, "
        f"p = {result['recall_p_value']:.4f}\n"
        f"- Cost per customer difference (B - A): {cost.mean_diff:+.2f}, "
        f"95% CI [{cost.ci_low:.2f}, {cost.ci_high:.2f}]\n",
        encoding="utf-8",
    )
    return path


def _log(tracker: MlflowTracker, result: dict[str, Any], plan: PowerPlan, settings: Settings, report) -> None:
    ab = settings.ab_test
    a, b, cost = result["a"], result["b"], result["cost_diff"]
    with tracker.run("online_ab_replay", tags={"type": "online_ab", "verdict": result["verdict"]}):
        tracker.log_params({
            "model_a": a.model_version, "model_b": b.model_version, "split_pct": ab.split_pct,
            "alpha": ab.alpha, "power": ab.power, "min_detectable_recall_diff": ab.min_detectable_recall_diff,
            "cost_false_negative": ab.cost_false_negative, "cost_false_positive": ab.cost_false_positive,
        })
        for arm, prefix in ((a, "a_"), (b, "b_")):
            numbers = {k: float(v) for k, v in asdict(arm).items() if k not in {"variant", "model_version"}}
            tracker.log_metrics(numbers, prefix=prefix)
        tracker.log_metrics({
            "recall_z": result["recall_z"], "recall_p_value": result["recall_p_value"],
            "cost_diff": cost.mean_diff, "cost_diff_ci_low": cost.ci_low, "cost_diff_ci_high": cost.ci_high,
            "required_churners_per_arm": plan.required_churners_per_arm,
            "required_customers_per_arm": plan.required_customers_per_arm,
            "powered": float(result["powered"]),
        })
        tracker.log_artifact(report)


def run(settings: Settings, tracker: MlflowTracker | None = None) -> dict[str, Any]:
    tracker = tracker or make_tracker(settings, settings.mlflow.ab_experiment)
    fetch_ab_bundles.run(settings, tracker)

    baseline = ModelBundle.load(settings.ab_bundle_path(VARIANT_A)).metadata.cv_metrics["recall"]
    churn_rate = float(io.read_csv(settings.train_path)[TARGET].mean())
    plan = plan_power(settings, baseline, churn_rate)
    logger.info("Power plan: %s", asdict(plan))

    replay(settings, traffic_with_ids(settings))
    result = analyze(load_outcomes(settings), plan, settings)
    report = _write_report(result, plan, settings)
    _log(tracker, result, plan, settings, report)
    logger.info("Online A/B verdict: %s (report: %s)", result["verdict"], report)
    return result


def main() -> None:
    exit_with(run_cli("Simulate an online A/B test by replaying the test split.", run))


if __name__ == "__main__":
    main()
