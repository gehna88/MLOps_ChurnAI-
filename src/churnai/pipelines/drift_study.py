"""Sequential data drift study: train on the first slice, meet each next slice as new data.

Run with ``make drift-study`` or ``python -m churnai.pipelines.drift_study``, after
``make train`` (the study retrains the champion's configuration). Options:

    --slices N          cut the dataset into N slices instead of drift_study.n_slices
    --scenario NAME     run only this scenario (repeatable)
    --no-mlflow         skip MLflow logging

The procedure is in :mod:`churnai.monitoring.drift_study`. This pipeline adds, for
every scenario in ``drift_study.scenarios``:

* graphs at every step (iteration) and summary graphs - :mod:`churnai.monitoring.drift_plots`
* MLflow runs in the ``data-drift`` experiment: one run per study, a nested run per
  scenario (metrics logged per step, so MLflow draws them as lines over time) and a
  nested run per step holding that step's metrics and graphs

Outputs, written to ``reports/drift_study/`` only once the whole study succeeded:

* ``results.json``    - every number, read by the API's ``/drift/study`` endpoints
* ``drift_study.md``  - the report, with tables and the graphs
* ``<scenario>/*.png`` and ``scenario_comparison.png``
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from churnai.config import DriftScenario, Settings
from churnai.data.schema import FEATURE_COLUMNS, NUMERIC_COLUMNS
from churnai.exceptions import ConfigError
from churnai.logging_utils import get_logger
from churnai.models.factory import ModelSpec
from churnai.models.tracking import ExperimentTracker, NullTracker, make_tracker
from churnai.monitoring import drift_plots as plots
from churnai.monitoring.drift_study import (
    COMPARISON_PLOT,
    MIN_SLICES,
    REPORT_FILE,
    RESULTS_FILE,
    StepResult,
    iterate_steps,
    load_study_data,
    make_slices,
    order_rows,
    resolve_model_spec,
    summarise_slices,
)
from churnai.pipelines._cli import exit_with, run_cli

logger = get_logger(__name__)

SUMMARY_PLOTS = ("psi_over_time.png", "psi_heatmap.png", "feature_trends.png",
                 "performance.png", "prediction_drift.png")
PLOTS_ARTIFACT_PATH = "plots"
MAX_SLICES = 20
_METRIC_UNSAFE = re.compile(r"[^0-9A-Za-z_]")


@dataclass(frozen=True)
class DriftStudyOutcome:
    output_dir: Path
    results: dict[str, Any]


def _key(name: str) -> str:
    return _METRIC_UNSAFE.sub("_", name)


def step_metrics(result: StepResult) -> dict[str, float]:
    """Flat metric dict for MLflow (names without spaces)."""
    metrics: dict[str, float] = {}
    for feature, value in result.psi_vs_initial.items():
        metrics[f"psi_initial_{_key(feature)}"] = value
    for feature, value in result.psi_vs_training.items():
        metrics[f"psi_training_{_key(feature)}"] = value
    for feature, ks in result.ks_vs_initial.items():
        metrics[f"ks_initial_{_key(feature)}"] = ks["statistic"]
    metrics |= {f"retrained_{k}": v for k, v in result.retrained_metrics.items()}
    metrics |= {f"stale_{k}": v for k, v in result.stale_metrics.items()}
    metrics |= {
        "prediction_psi": result.prediction_psi,
        "mean_predicted": result.mean_predicted,
        "actual_churn_rate": result.actual_churn_rate,
        "threshold": result.threshold,
        "n_drifted_features": float(len(result.drifted_features)),
    }
    return metrics


def _select_scenarios(settings: Settings, names: list[str] | None) -> list[DriftScenario]:
    cfg = settings.drift_study
    if not names:
        return list(cfg.scenarios)
    try:
        return [cfg.scenario(name) for name in names]
    except KeyError as exc:
        raise ConfigError(
            f"Unknown drift scenario {exc}. Available: {[s.name for s in cfg.scenarios]}"
        ) from exc


def _run_scenario(
    settings: Settings, scenario: DriftScenario, spec: ModelSpec, n_slices: int,
    tracker: ExperimentTracker, out_dir: Path,
) -> dict[str, Any]:
    cfg = settings.drift_study
    focus = list(cfg.drift_features)
    warning, alert = cfg.psi_warning_threshold, settings.monitoring.psi_alert_threshold
    scenario_dir = out_dir / scenario.name

    data = order_rows(load_study_data(settings), scenario, focus, settings.project.random_seed)
    slices = make_slices(data, n_slices)
    slice_summaries = [asdict(s) for s in summarise_slices(slices, list(NUMERIC_COLUMNS))]

    steps: list[dict[str, Any]] = []
    with tracker.run(scenario.name, tags={"scenario": scenario.name, "model": spec.name},
                     nested=True) as scenario_run_id:
        tracker.log_params({"scenario": scenario.name, "mean_shift": scenario.mean_shift,
                            "spread": scenario.spread, "noise": scenario.noise,
                            "n_slices": n_slices, "model": spec.name})
        for result, frames in iterate_steps(slices, spec, settings):
            k, new = result.step, result.new_slice
            step_plots = [
                plots.plot_step_distributions(
                    frames.initial, frames.training, frames.new, focus, result.psi_vs_initial,
                    result.ks_vs_initial, k, new, scenario.name, scenario_dir / f"step{k}_distributions.png"),
                plots.plot_step_psi(
                    result.psi_vs_initial, result.psi_vs_training, warning, alert, k, new,
                    scenario.name, scenario_dir / f"step{k}_psi.png"),
            ]
            result.plots = [p.relative_to(out_dir).as_posix() for p in step_plots]
            metrics = step_metrics(result)
            with tracker.run(f"{scenario.name} step {k}: S1-S{k} -> S{new}",
                             tags={"scenario": scenario.name, "step": str(k)}, nested=True) as step_run_id:
                tracker.log_params({"step": k, "train_slices": f"S1-S{k}", "new_slice": f"S{new}",
                                    "train_rows": result.train_rows, "new_rows": result.new_rows})
                tracker.log_metrics(metrics)
                tracker.log_dict(result.to_dict(), "step_result.json")
                for path in step_plots:
                    tracker.log_artifact(path, PLOTS_ARTIFACT_PATH)
            result.mlflow_run_id = step_run_id
            tracker.log_metrics(metrics, step=k)  # scenario run: one line per metric over the steps
            steps.append(result.to_dict())
            logger.info(
                "[%s] step %d: trained on S1-S%d (%d rows), new slice S%d | max PSI %.3f | "
                "drifted %s | retrained AUC %.3f vs stale %.3f",
                scenario.name, k, k, result.train_rows, new, max(result.psi_vs_initial.values()),
                result.drifted_features or "none", result.retrained_metrics["roc_auc"],
                result.stale_metrics["roc_auc"],
            )

        summary = [
            plots.plot_psi_over_time(steps, focus, warning, alert, scenario.name,
                                     scenario_dir / SUMMARY_PLOTS[0]),
            plots.plot_psi_heatmap(steps, warning, alert, scenario.name, scenario_dir / SUMMARY_PLOTS[1]),
            plots.plot_feature_trends(slice_summaries, focus, scenario.name, scenario_dir / SUMMARY_PLOTS[2]),
            plots.plot_performance(steps, scenario.name, scenario_dir / SUMMARY_PLOTS[3]),
            plots.plot_prediction_drift(steps, warning, alert, scenario.name,
                                        scenario_dir / SUMMARY_PLOTS[4]),
        ]
        for path in summary:
            tracker.log_artifact(path, PLOTS_ARTIFACT_PATH)
        first_drift = next((s["step"] for s in steps if s["drifted_features"]), None)
        tracker.log_metrics({"first_drift_step": float(first_drift or 0),
                             "max_psi": max(max(s["psi_vs_initial"].values()) for s in steps)})

    return {
        "name": scenario.name,
        "description": scenario.description,
        "params": {"mean_shift": scenario.mean_shift, "spread": scenario.spread, "noise": scenario.noise},
        "first_drift_step": first_drift,
        "slices": slice_summaries,
        "steps": steps,
        "plots": [p.relative_to(out_dir).as_posix() for p in summary],
        "mlflow_run_id": scenario_run_id,
    }


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(results: dict[str, Any], path: Path) -> Path:
    """Markdown report: what was done, the tables, and every graph."""
    focus = results["drift_features"]
    warning, alert = results["psi_warning_threshold"], results["psi_alert_threshold"]
    lines = [
        "# Data drift study",
        "",
        f"Generated {results['generated_at']} with model `{results['model']['name']}` "
        f"(`{results['model']['estimator']}`).",
        "",
        f"The {results['rows']:,} validated customers are cut into {results['n_slices']} slices of about "
        f"{results['slice_size']:,} rows. Slice S1 trains the first model. At each step the next slice "
        "arrives as new data: its drift is measured against S1 and against all data trained on so far, "
        "the model is scored on it, and the model is then retrained on every slice seen so far.",
        "",
        f"**Reading PSI:** below {warning:g} is stable, {warning:g}-{alert:g} is a moderate shift, above "
        f"{alert:g} is drift. KS is the largest gap between the two cumulative distributions "
        "(0 = identical).",
        "",
        f"![Scenarios compared]({COMPARISON_PLOT})",
        "",
    ]
    for scenario in results["scenarios"]:
        name = scenario["name"]
        first = scenario["first_drift_step"]
        lines += [
            f"## Scenario: {name}",
            "",
            scenario["description"],
            "",
            f"**Result:** drift (PSI > {alert:g}) first detected at step {first}." if first
            else f"**Result:** no feature crossed PSI {alert:g} at any step.",
            "",
            "### Slices",
            "",
            "| Slice | Rows | Churn | " + " | ".join(f"{f} mean (std)" for f in focus) + " |",
            "|---|---|---|" + "---|" * len(focus),
        ]
        for s in scenario["slices"]:
            cells = [f"{s['stats'][f]['mean']:,.1f} ({s['stats'][f]['std']:,.1f})" for f in focus]
            row = f"| S{s['index']} | {s['rows']:,} | {_fmt_pct(s['churn_rate'])} | "
            lines.append(row + " | ".join(cells) + " |")
        lines += [
            "",
            "### Steps",
            "",
            "| Step | Trained on | New | " + " | ".join(f"PSI {f}" for f in focus)
            + " | Drifted features | Prediction PSI | Churn actual / predicted | AUC retrained / stale |",
            "|---|---|---|" + "---|" * len(focus) + "---|---|---|---|",
        ]
        for s in scenario["steps"]:
            psi_cells = [f"**{s['psi_vs_initial'][f]:.3f}**" if s["psi_vs_initial"][f] > alert
                         else f"{s['psi_vs_initial'][f]:.3f}" for f in focus]
            lines.append(
                f"| {s['step']} | S1-S{s['step']} ({s['train_rows']:,}) | S{s['new_slice']} | "
                + " | ".join(psi_cells)
                + f" | {', '.join(s['drifted_features']) or '-'} | {s['prediction_psi']:.3f} | "
                f"{_fmt_pct(s['actual_churn_rate'])} / {_fmt_pct(s['mean_predicted'])} | "
                f"{s['retrained_metrics']['roc_auc']:.3f} / {s['stale_metrics']['roc_auc']:.3f} |"
            )
        lines += ["", "### Summary graphs", ""]
        lines += [f"![{Path(p).stem}]({p})" for p in scenario["plots"]]
        lines += ["", "### Graphs at each step", ""]
        for s in scenario["steps"]:
            lines += [f"**Step {s['step']}** (new slice S{s['new_slice']})", ""]
            lines += [f"![step {s['step']}]({p})" for p in s["plots"]]
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _publish(staging: Path, final: Path) -> None:
    """Swap the finished study in, so readers never see a half-written one."""
    backup = final.with_name(f".{final.name}.old")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        try:
            final.rename(backup)
        except OSError:
            # Docker's overlay filesystem cannot rename a directory that came from an image
            # layer (EXDEV), e.g. the study shipped in the image: copy it aside instead.
            shutil.copytree(final, backup)
            shutil.rmtree(final)
    try:
        staging.rename(final)
    except OSError:
        if backup.exists() and not final.exists():
            backup.rename(final)  # put the previous study back
        raise
    shutil.rmtree(backup, ignore_errors=True)


def run(
    settings: Settings,
    tracker: ExperimentTracker | None = None,
    n_slices: int | None = None,
    scenarios: list[str] | None = None,
    log_to_mlflow: bool = True,
) -> DriftStudyOutcome:
    cfg = settings.drift_study
    n_slices = n_slices or cfg.n_slices
    if not MIN_SLICES <= n_slices <= MAX_SLICES:
        raise ConfigError(f"n_slices must be between {MIN_SLICES} and {MAX_SLICES}, got {n_slices}")
    chosen = _select_scenarios(settings, scenarios)
    spec = resolve_model_spec(settings)
    if tracker is None:
        tracker = make_tracker(settings, cfg.experiment_name) if log_to_mlflow else NullTracker()
    rows = len(load_study_data(settings))

    final_dir = settings.drift_study_dir
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}-", dir=final_dir.parent))
    started = time.perf_counter()
    try:
        stamp = datetime.now(UTC)
        with tracker.run(f"drift-study {stamp:%Y-%m-%d %H:%M}",
                         tags={"kind": "drift_study", "model": spec.name}) as study_run_id:
            tracker.log_params({
                "n_slices": n_slices, "slice_size": rows // n_slices, "rows": rows, "model": spec.name,
                "estimator": spec.estimator, "psi_bins": settings.monitoring.psi_bins,
                "psi_alert_threshold": settings.monitoring.psi_alert_threshold,
                "psi_warning_threshold": cfg.psi_warning_threshold,
                "scenarios": ",".join(s.name for s in chosen),
                "drift_features": ",".join(cfg.drift_features),
            })
            scenario_results = [
                _run_scenario(settings, scenario, spec, n_slices, tracker, staging) for scenario in chosen
            ]
            plots.plot_scenario_comparison(
                scenario_results, list(cfg.drift_features), cfg.psi_warning_threshold,
                settings.monitoring.psi_alert_threshold, staging / COMPARISON_PLOT)
            results = {
                "generated_at": stamp.isoformat(timespec="seconds"),
                "duration_seconds": round(time.perf_counter() - started, 1),
                "rows": rows,
                "n_slices": n_slices,
                "slice_size": rows // n_slices,
                "model": {"name": spec.name, "estimator": spec.estimator},
                "features": list(FEATURE_COLUMNS),
                "drift_features": list(cfg.drift_features),
                "psi_bins": settings.monitoring.psi_bins,
                "psi_warning_threshold": cfg.psi_warning_threshold,
                "psi_alert_threshold": settings.monitoring.psi_alert_threshold,
                "mlflow": {"experiment": cfg.experiment_name, "run_id": study_run_id},
                "comparison_plot": COMPARISON_PLOT,
                "report": REPORT_FILE,
                "scenarios": scenario_results,
            }
            (staging / RESULTS_FILE).write_text(json.dumps(results, indent=2), encoding="utf-8")
            write_report(results, staging / REPORT_FILE)
            for name in (COMPARISON_PLOT, RESULTS_FILE, REPORT_FILE):
                tracker.log_artifact(staging / name)
        _publish(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("Drift study written to %s (%.0fs)", final_dir, time.perf_counter() - started)
    return DriftStudyOutcome(final_dir, results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slices", type=int, default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    def runner(settings: Settings) -> DriftStudyOutcome:
        return run(settings, n_slices=args.slices, scenarios=args.scenario, log_to_mlflow=not args.no_mlflow)

    runner.__module__ = __name__
    exit_with(run_cli(
        "Sequential data drift study (options: --slices N, --scenario NAME, --no-mlflow).", runner, rest))


if __name__ == "__main__":
    main()
