"""In-process request metrics: counts, error rates and latency percentiles."""

from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any

import numpy as np

HTTP_ERROR_STATUS = 400
PERCENTILES = (50, 95, 99)
LATENCY_DECIMALS = 2


class RequestMetrics:
    """Thread-safe collector. Latency stats cover a rolling window of requests."""

    def __init__(self, latency_window: int) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=latency_window)

    def record(self, route: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._requests[route] += 1
            if status_code >= HTTP_ERROR_STATUS:
                self._errors[route] += 1
            self._latencies.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = dict(self._requests)
            errors = dict(self._errors)
            latencies = np.array(self._latencies)

        latency: dict[str, float | None] = {"avg_ms": None}
        latency.update({f"p{p}_ms": None for p in PERCENTILES})
        if latencies.size:
            latency["avg_ms"] = round(float(latencies.mean()), LATENCY_DECIMALS)
            for p in PERCENTILES:
                latency[f"p{p}_ms"] = round(float(np.percentile(latencies, p)), LATENCY_DECIMALS)
        return {
            "requests_total": sum(requests.values()),
            "errors_total": sum(errors.values()),
            "requests_by_route": requests,
            "errors_by_route": errors,
            "latency": latency,
        }
