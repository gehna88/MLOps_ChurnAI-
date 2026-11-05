"""Online A/B test: sticky customer routing, exposure and outcome logs, per-variant stats."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from churnai.api.service import ModelService, Prediction
from churnai.config import Settings
from churnai.monitoring.metrics import RequestMetrics

VARIANT_A = "A"  # champion
VARIANT_B = "B"  # challenger
CUSTOMER_ID_HEADER = "X-Customer-ID"
BUCKETS = 100


def bucket(customer_id: str) -> int:
    """Stable bucket in [0, 100). Python's hash() is randomised per process, so it can't be used."""
    return int(hashlib.sha256(customer_id.encode("utf-8")).hexdigest(), 16) % BUCKETS


class JsonlLog:
    """Thread-safe append-only log, one JSON record per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


@dataclass(frozen=True)
class ABExperiment:
    services: dict[str, ModelService]
    metrics: dict[str, RequestMetrics]
    split_pct: int
    predictions: JsonlLog
    feedback: JsonlLog

    @classmethod
    def from_settings(
        cls, settings: Settings, control: ModelService, treatment: ModelService
    ) -> ABExperiment:
        window = settings.monitoring.latency_window_size
        return cls(
            services={VARIANT_A: control, VARIANT_B: treatment},
            metrics={VARIANT_A: RequestMetrics(window), VARIANT_B: RequestMetrics(window)},
            split_pct=settings.ab_test.split_pct,
            predictions=JsonlLog(settings.resolve(settings.ab_test.predictions_log)),
            feedback=JsonlLog(settings.resolve(settings.ab_test.feedback_log)),
        )

    def assign(self, customer_id: str) -> str:
        return VARIANT_B if bucket(customer_id) < self.split_pct else VARIANT_A

    def record_prediction(self, customer_id: str, prediction: Prediction) -> None:
        self.predictions.append({"customer_id": customer_id, **asdict(prediction)})

    def record_feedback(self, customer_id: str, exited: int) -> None:
        self.feedback.append({"customer_id": customer_id, "exited": exited})

    def snapshot(self) -> dict[str, Any]:
        return {
            "split_pct": self.split_pct,
            "variants": {
                variant: {
                    "model_version": service.version,
                    "requests": self.metrics[variant].snapshot(),
                    "drift": asdict(service.drift.report()),
                }
                for variant, service in self.services.items()
            },
        }
