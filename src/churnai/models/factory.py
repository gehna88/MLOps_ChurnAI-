"""Model catalogue: turn declarative YAML specs into estimators.

Adding a new model family means registering one builder function; nothing in
the training loop changes (open/closed principle).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from churnai.config import load_yaml
from churnai.exceptions import ConfigError

AUTO = "auto"
SCALE_POS_WEIGHT = "scale_pos_weight"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    """Information about the training data that some builders need."""

    random_seed: int
    positive_ratio: float  # negatives / positives

    @classmethod
    def from_target(cls, y: np.ndarray, random_seed: int) -> BuildContext:
        positives = float(np.sum(y))
        negatives = float(len(y) - positives)
        return cls(random_seed=random_seed, positive_ratio=negatives / max(positives, 1.0))


Builder = Callable[[dict[str, Any], BuildContext], BaseEstimator]
_REGISTRY: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def decorator(builder: Builder) -> Builder:
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_estimators() -> list[str]:
    return sorted(_REGISTRY)


def _resolve_auto(params: dict[str, Any], ctx: BuildContext) -> dict[str, Any]:
    resolved = dict(params)
    if resolved.get(SCALE_POS_WEIGHT) == AUTO:
        resolved[SCALE_POS_WEIGHT] = round(ctx.positive_ratio, 4)
    return resolved


@register("logistic_regression")
def _logistic_regression(params: dict[str, Any], ctx: BuildContext) -> BaseEstimator:
    return LogisticRegression(random_state=ctx.random_seed, **params)


@register("xgboost")
def _xgboost(params: dict[str, Any], ctx: BuildContext) -> BaseEstimator:
    return XGBClassifier(
        random_state=ctx.random_seed,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        **_resolve_auto(params, ctx),
    )


@register("lightgbm")
def _lightgbm(params: dict[str, Any], ctx: BuildContext) -> BaseEstimator:
    return LGBMClassifier(random_state=ctx.random_seed, verbose=-1, n_jobs=-1, **params)


def build_estimator(spec: ModelSpec, ctx: BuildContext) -> BaseEstimator:
    try:
        builder = _REGISTRY[spec.estimator]
    except KeyError as exc:
        raise ConfigError(
            f"Unknown estimator '{spec.estimator}' for '{spec.name}'. "
            f"Available: {available_estimators()}"
        ) from exc
    return builder(spec.params, ctx)


def resolved_params(spec: ModelSpec, ctx: BuildContext) -> dict[str, Any]:
    """Parameters as actually used (``auto`` substituted) - for experiment logs."""
    return _resolve_auto(spec.params, ctx)


def load_model_specs(path: Path) -> list[ModelSpec]:
    raw = load_yaml(path).get("candidates")
    if not raw:
        raise ConfigError(f"No model candidates defined in {path}")
    specs = [ModelSpec(name=c["name"], estimator=c["estimator"], params=c.get("params") or {})
             for c in raw]
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ConfigError(f"Duplicate candidate names in {path}: {names}")
    return specs
