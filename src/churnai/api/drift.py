"""Data drift study endpoints, under ``/drift``.

Read the latest study (written by ``make drift-study`` or by ``POST /drift/study/run``):

* ``GET  /drift/study``                              every number, as JSON
* ``GET  /drift/study/report``                       the Markdown report
* ``GET  /drift/study/plots``                        URLs of every graph, per scenario and step
* ``GET  /drift/study/files/{path}``                 one graph (PNG)
* ``GET  /drift/study/scenarios/{name}``             one scenario
* ``GET  /drift/study/scenarios/{name}/steps/{k}``   one step (iteration)

Run a study and follow it:

* ``POST /drift/study/run``         starts a study in the background (202), one at a time
* ``GET  /drift/study/status``      idle / running / succeeded / failed
* ``GET  /drift/study/mlflow-runs`` the runs in the ``data-drift`` MLflow experiment

Ad hoc comparison:

* ``POST /drift/compare``  PSI (and KS, with explicit reference rows) between two sets of customers

The study pipeline needs matplotlib and MLflow, so it is imported only when a study runs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from churnai.api.schemas import CustomerFeatures, to_frame
from churnai.config import Settings
from churnai.data.schema import FEATURE_COLUMNS, NUMERIC_COLUMNS
from churnai.logging_utils import get_logger
from churnai.monitoring.drift import FeatureReference
from churnai.monitoring.drift_study import (
    COMPARISON_PLOT,
    REPORT_FILE,
    RESULTS_FILE,
    describe,
    ks_tests,
    psi_level,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/drift", tags=["drift"])

FILES_ROUTE = "/drift/study/files"
MIN_COMPARE_ROWS = 20
MAX_COMPARE_ROWS = 10_000
DEFAULT_RUN_LIMIT = 50
MAX_RUN_LIMIT = 500
MS_PER_SECOND = 1000

JobStatus = Literal["idle", "running", "succeeded", "failed"]


# ----------------------------------------------------------- background job ---
@dataclass
class DriftStudyJob:
    """Runs at most one drift study at a time in a background thread."""

    settings: Settings
    status: JobStatus = "idle"
    params: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self.status, "params": self.params, "started_at": self.started_at,
                    "finished_at": self.finished_at, "error": self.error}

    def start(self, n_slices: int | None, scenarios: list[str] | None, log_to_mlflow: bool) -> dict[str, Any]:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("A drift study is already running")
            self.status = "running"
            self.params = {"n_slices": n_slices or self.settings.drift_study.n_slices,
                           "scenarios": scenarios or [s.name for s in self.settings.drift_study.scenarios],
                           "log_to_mlflow": log_to_mlflow}
            self.started_at, self.finished_at, self.error = _now(), None, None
            self._thread = threading.Thread(
                target=self._run, args=(n_slices, scenarios, log_to_mlflow), name="drift-study", daemon=True
            )
            self._thread.start()
        return self.snapshot()

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current study finishes (used by tests and scripts)."""
        if self._thread:
            self._thread.join(timeout)

    def _run(self, n_slices: int | None, scenarios: list[str] | None, log_to_mlflow: bool) -> None:
        try:
            from churnai.pipelines import drift_study  # matplotlib + MLflow: load only when needed

            drift_study.run(
                self.settings, n_slices=n_slices, scenarios=scenarios, log_to_mlflow=log_to_mlflow
            )
        except Exception as exc:  # report every failure through /drift/study/status
            logger.exception("Drift study failed")
            outcome: tuple[JobStatus, str | None] = ("failed", f"{type(exc).__name__}: {exc}")
        else:
            outcome = ("succeeded", None)
        with self._lock:
            self.status, self.error = outcome
            self.finished_at = _now()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# -------------------------------------------------------------- request models ---
class DriftStudyRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_slices: int | None = Field(default=None, ge=2, le=20, description="Defaults to drift_study.n_slices")
    scenarios: list[str] | None = Field(default=None, min_length=1, description="Defaults to all scenarios")
    log_to_mlflow: bool = True


class DriftCompareRequest(BaseModel):
    """Two sets of customers. Without ``reference`` the served model's training profile is used."""

    model_config = ConfigDict(extra="forbid")

    current: list[CustomerFeatures] = Field(  # type: ignore[valid-type]
        min_length=MIN_COMPARE_ROWS, max_length=MAX_COMPARE_ROWS)
    reference: list[CustomerFeatures] | None = Field(  # type: ignore[valid-type]
        default=None, min_length=MIN_COMPARE_ROWS, max_length=MAX_COMPARE_ROWS)


# -------------------------------------------------------------------- helpers ---
def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_job(request: Request) -> DriftStudyJob:
    return request.app.state.drift_job


def _study_dir(settings: Settings) -> Path:
    return settings.drift_study_dir


def load_results(settings: Settings = Depends(get_settings_dep)) -> dict[str, Any]:
    path = _study_dir(settings) / RESULTS_FILE
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            "No drift study yet. Run `make drift-study` or POST /drift/study/run.")
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario(results: dict[str, Any], name: str) -> dict[str, Any]:
    for scenario in results["scenarios"]:
        if scenario["name"] == name:
            return scenario
    raise HTTPException(status.HTTP_404_NOT_FOUND,
                        f"Unknown scenario '{name}'. Available: {[s['name'] for s in results['scenarios']]}")


def _plot_files(results: dict[str, Any]) -> set[str]:
    files = {results.get("comparison_plot", COMPARISON_PLOT)}
    for scenario in results["scenarios"]:
        files.update(scenario["plots"])
        for step in scenario["steps"]:
            files.update(step["plots"])
    return files


def _url(path: str) -> str:
    return f"{FILES_ROUTE}/{path}"


# --------------------------------------------------------------- study: read ---
@router.get("/study")
def study_results(results: dict[str, Any] = Depends(load_results)) -> dict[str, Any]:
    """The full results of the latest drift study."""
    return results


@router.get("/study/report", response_class=PlainTextResponse)
def study_report(settings: Settings = Depends(get_settings_dep)) -> PlainTextResponse:
    load_results(settings)
    text = (_study_dir(settings) / REPORT_FILE).read_text(encoding="utf-8")
    return PlainTextResponse(text, media_type="text/markdown")


@router.get("/study/plots")
def study_plots(results: dict[str, Any] = Depends(load_results)) -> dict[str, Any]:
    """Where every graph is: the comparison, each scenario's summaries and each step's graphs."""
    return {
        "comparison": _url(results.get("comparison_plot", COMPARISON_PLOT)),
        "scenarios": {
            s["name"]: {
                "summary": [_url(p) for p in s["plots"]],
                "steps": {str(step["step"]): [_url(p) for p in step["plots"]] for step in s["steps"]},
            }
            for s in results["scenarios"]
        },
    }


@router.get("/study/files/{path:path}", response_class=FileResponse)
def study_file(path: str, settings: Settings = Depends(get_settings_dep)) -> FileResponse:
    """A graph from the latest study. Only files the study listed are served."""
    results = load_results(settings)
    if path not in _plot_files(results):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No study file '{path}'")
    return FileResponse(_study_dir(settings) / path, media_type="image/png")


@router.get("/study/scenarios/{name}")
def study_scenario(name: str, results: dict[str, Any] = Depends(load_results)) -> dict[str, Any]:
    return _scenario(results, name)


@router.get("/study/scenarios/{name}/steps/{step}")
def study_step(name: str, step: int, results: dict[str, Any] = Depends(load_results)) -> dict[str, Any]:
    for item in _scenario(results, name)["steps"]:
        if item["step"] == step:
            return item
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"Scenario '{name}' has no step {step}")


# ---------------------------------------------------------------- study: run ---
@router.post("/study/run", status_code=status.HTTP_202_ACCEPTED)
def run_study(
    payload: DriftStudyRunRequest,
    settings: Settings = Depends(get_settings_dep),
    job: DriftStudyJob = Depends(get_job),
) -> dict[str, Any]:
    """Start a drift study in the background; follow it with GET /drift/study/status."""
    known = {s.name for s in settings.drift_study.scenarios}
    unknown = sorted(set(payload.scenarios or []) - known)
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"Unknown scenarios {unknown}. Available: {sorted(known)}")
    try:
        return job.start(payload.n_slices, payload.scenarios, payload.log_to_mlflow)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/study/status")
def study_status(job: DriftStudyJob = Depends(get_job)) -> dict[str, Any]:
    return job.snapshot()


@router.get("/study/mlflow-runs")
def study_mlflow_runs(
    limit: int = Query(DEFAULT_RUN_LIMIT, ge=1, le=MAX_RUN_LIMIT),
    settings: Settings = Depends(get_settings_dep),
) -> dict[str, Any]:
    """Runs in the drift study's MLflow experiment, newest first."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - MLflow ships in every image now
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "MLflow is not installed") from exc

    name = settings.drift_study.experiment_name
    try:
        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(name)
        runs = [] if experiment is None else client.search_runs(
            [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=limit)
    except Exception as exc:  # a broken or unwritable store must not become a 500
        logger.exception("MLflow store unavailable")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"MLflow store unavailable: {type(exc).__name__}: {exc}") from exc
    return {
        "experiment": name,
        "tracking_uri": settings.mlflow_tracking_uri,
        "runs": [
            {
                "run_id": r.info.run_id,
                "name": r.info.run_name,
                "parent_run_id": r.data.tags.get("mlflow.parentRunId"),
                "status": r.info.status,
                "start_time": datetime.fromtimestamp(r.info.start_time / MS_PER_SECOND, UTC)
                .isoformat(timespec="seconds"),
                "params": r.data.params,
                "metrics": r.data.metrics,
            }
            for r in runs
        ],
    }


# ------------------------------------------------------------------- compare ---
@router.post("/compare")
def compare(payload: DriftCompareRequest, request: Request) -> dict[str, Any]:
    """Drift of ``current`` against ``reference`` rows, or against the served model's training data."""
    settings: Settings = request.app.state.settings
    bins = settings.monitoring.psi_bins
    alert = settings.monitoring.psi_alert_threshold
    warning = settings.drift_study.psi_warning_threshold
    current = to_frame(payload.current)[list(FEATURE_COLUMNS)]

    if payload.reference is not None:
        reference = to_frame(payload.reference)[list(FEATURE_COLUMNS)]
        source, n_reference = "request", len(reference)
        psi = {f: FeatureReference.fit(f, reference[f], bins).psi_against(current[f])
               for f in FEATURE_COLUMNS}
        numeric = [f for f in FEATURE_COLUMNS if f in NUMERIC_COLUMNS]
        ks = ks_tests(reference, current, numeric)
        stats = {f: {"reference": describe(reference[f]), "current": describe(current[f])} for f in numeric}
    else:
        service = request.app.state.service
        if service is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "No model loaded: send 'reference' rows to compare against")
        profile = service.bundle.reference
        source, n_reference = "model_training_data", profile.n_samples
        psi = {f: profile.features[f].psi_against(current[f])
               for f in FEATURE_COLUMNS if f in profile.features}
        ks = {}
        stats = {f: {"current": describe(current[f])} for f in FEATURE_COLUMNS if f in NUMERIC_COLUMNS}

    features = {
        f: {"psi": value, "level": psi_level(value, warning, alert),
            **({"ks": ks[f]} if f in ks else {}), **({"stats": stats[f]} if f in stats else {})}
        for f, value in psi.items()
    }
    drifted = sorted(f for f, value in psi.items() if value > alert)
    return {
        "reference_source": source,
        "n_reference": n_reference,
        "n_current": len(current),
        "psi_warning_threshold": warning,
        "psi_alert_threshold": alert,
        "drift_detected": bool(drifted),
        "drifted_features": drifted,
        "features": features,
    }


def make_job(settings: Settings) -> DriftStudyJob:
    return DriftStudyJob(settings)

