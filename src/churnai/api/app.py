"""Application factory.

``create_app`` builds a fresh, fully configured app - there is no module-level
global, so tests can create isolated apps with their own settings and model.

Run locally:  ``uvicorn churnai.api.app:create_app --factory --port 8000``
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from starlette.routing import Match

from churnai import __version__
from churnai.api.ab import VARIANT_A, VARIANT_B, ABExperiment
from churnai.api.routes import router
from churnai.api.service import ModelService
from churnai.config import Settings, get_settings
from churnai.exceptions import ChurnAIError
from churnai.logging_utils import configure_logging, get_logger
from churnai.models.bundle import ModelBundle
from churnai.monitoring.metrics import RequestMetrics

logger = get_logger("churnai.api")

REQUEST_ID_HEADER = "X-Request-ID"
# Set by the Docker build to the image tag (e.g. sha-1a2b3c4), so /health shows exactly
# which image is live; useful after a deploy or a rollback.
ENV_BUILD = "CHURNAI_BUILD"
LOCAL_BUILD = "local"
MS_PER_SECOND = 1000.0
UNMATCHED_ROUTE = "unmatched"


def _route_template(request: Request) -> str:
    """Use the route template (e.g. /predict) rather than the raw URL to bound cardinality."""
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match is Match.FULL:
            return getattr(route, "path", UNMATCHED_ROUTE)
    return UNMATCHED_ROUTE


def _load_service(
    settings: Settings, bundle: ModelBundle | None, path: Path, variant: str = VARIANT_A
) -> ModelService | None:
    try:
        bundle = bundle or ModelBundle.load(path)
        service = ModelService(bundle, settings, variant)
    except ChurnAIError as exc:
        # Start anyway so /health can report the problem instead of crash-looping.
        logger.error("Model unavailable: %s", exc)
        return None
    logger.info("Model loaded", extra={"fields": {"model_version": service.version, "variant": variant}})
    return service


def _load_experiment(
    settings: Settings, control: ModelService | None, challenger: ModelBundle | None
) -> ABExperiment | None:
    """Both variants or no experiment: without a challenger the API serves the champion alone."""
    treatment = _load_service(settings, challenger, settings.ab_bundle_path(VARIANT_B), VARIANT_B)
    if control is None or treatment is None:
        logger.error("A/B test disabled: a variant is unavailable. Run `make ab-fetch`.")
        return None
    return ABExperiment.from_settings(settings, control, treatment)


def create_app(
    settings: Settings | None = None,
    bundle: ModelBundle | None = None,
    challenger: ModelBundle | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ab_enabled = settings.ab_test.enabled
        path = settings.ab_bundle_path(VARIANT_A) if ab_enabled else settings.model_path
        app.state.service = _load_service(settings, bundle, path)
        if ab_enabled:
            app.state.ab = _load_experiment(settings, app.state.service, challenger)
        yield

    app = FastAPI(
        title="ChurnAI",
        description="Bank customer churn prediction with per-prediction SHAP explanations.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.build = os.environ.get(ENV_BUILD, LOCAL_BUILD)
    app.state.service = None
    app.state.ab = None
    app.state.metrics = RequestMetrics(settings.monitoring.latency_window_size)
    app.state.started_at = time.monotonic()
    app.include_router(router)

    @app.middleware("http")
    async def observe_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            latency_ms = (time.perf_counter() - start) * MS_PER_SECOND
            route = _route_template(request)
            app.state.metrics.record(route, status_code, latency_ms)
            variant = getattr(request.state, "variant", None)
            if app.state.ab and variant:
                app.state.ab.metrics[variant].record(route, status_code, latency_ms)
            logger.info(
                "request",
                extra={"fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": status_code,
                    "latency_ms": round(latency_ms, 2),
                }},
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    return app
