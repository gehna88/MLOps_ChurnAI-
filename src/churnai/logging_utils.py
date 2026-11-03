"""Structured JSON logging.

Every log line is a single JSON object so that Render (or any log shipper) can
index fields such as ``request_id``, ``latency_ms`` and ``model_version``.
Pass structured fields with ``logger.info("msg", extra={"fields": {...}})``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

ENV_LOG_LEVEL = "CHURNAI_LOG_LEVEL"
ENV_LOG_FORMAT = "CHURNAI_LOG_FORMAT"  # "json" (default) or "text"
DEFAULT_LEVEL = "INFO"
TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger once; safe to call repeatedly."""
    level = (level or os.environ.get(ENV_LOG_LEVEL) or DEFAULT_LEVEL).upper()
    fmt = (fmt or os.environ.get(ENV_LOG_FORMAT) or "json").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


PROJECT_LOGGER_PREFIX = "churnai"


def reenable_project_loggers() -> None:
    """Undo third-party ``logging.config.fileConfig`` calls that disable our loggers.

    MLflow runs Alembic DB migrations on first use, and Alembic's logging config
    disables every existing logger - which silently swallows the pipeline's output.
    """
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (
            name.startswith(PROJECT_LOGGER_PREFIX) or name == "__main__"
        ):
            logger.disabled = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
