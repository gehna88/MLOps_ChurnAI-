"""Training, bundle and explainer tests (use the session-trained project)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from churnai.config import Settings
from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.exceptions import LeakageSuspectedError, ModelNotLoadedError
from churnai.explain.explainer import DECREASES, INCREASES, ChurnExplainer
from churnai.features.engineering import ENGINEERED_BINARY, ENGINEERED_NUMERIC
from churnai.features.preprocessing import build_pipeline
from churnai.models.bundle import ModelBundle
from churnai.models.factory import BuildContext, ModelSpec, build_estimator
from churnai.models.training import CandidateResult, guard_against_leakage, select_champion
from tests.conftest import FAST_MODELS

pytestmark = pytest.mark.slow

MIN_ACCEPTABLE_AUC = 0.75
TOP_K = 5
EXPLAINABLE = {*FEATURE_COLUMNS, *ENGINEERED_BINARY, *ENGINEERED_NUMERIC}


def test_training_outputs_exist(trained_project: Settings) -> None:
    assert trained_project.model_path.is_file()
    assert (trained_project.model_dir / "metadata.json").is_file()
    leaderboard = pd.read_csv(trained_project.resolve("reports") / "leaderboard.csv")
    assert len(leaderboard) == len(FAST_MODELS["candidates"])


def test_champion_is_best_by_cv_auc_and_reasonable(trained_project: Settings) -> None:
    leaderboard = pd.read_csv(trained_project.resolve("reports") / "leaderboard.csv")
    bundle = ModelBundle.load(trained_project.model_path)
    assert bundle.metadata.model_name == leaderboard.iloc[0]["model"]
    assert bundle.metadata.test_metrics["roc_auc"] > MIN_ACCEPTABLE_AUC


def test_bundle_predicts_probabilities(bundle: ModelBundle, sample_raw_df: pd.DataFrame) -> None:
    proba = bundle.predict_proba(sample_raw_df.head(20))  # extra columns are ignored
    assert proba.shape == (20,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(bundle.predict_label(proba)) <= {0, 1}


def test_bundle_version_without_mlflow_is_local(bundle: ModelBundle) -> None:
    assert "-local-" in bundle.metadata.version


def test_missing_bundle_raises(tmp_path) -> None:
    with pytest.raises(ModelNotLoadedError, match="make train"):
        ModelBundle.load(tmp_path / "missing.joblib")


def _fake_result(bundle: ModelBundle, cv_auc: float, test_auc: float) -> CandidateResult:
    return CandidateResult(
        spec=ModelSpec("fake", "xgboost"), pipeline=bundle.pipeline, params={}, threshold=0.5,
        cv_metrics={"roc_auc": cv_auc}, test_metrics={"roc_auc": test_auc}, test_confusion={},
    )


def test_leakage_guard(bundle: ModelBundle) -> None:
    with pytest.raises(LeakageSuspectedError):
        guard_against_leakage(_fake_result(bundle, 0.99, 0.99), auc_threshold=0.95)
    guard_against_leakage(_fake_result(bundle, 0.85, 0.85), auc_threshold=0.95)


def test_select_champion_uses_cv_metric(bundle: ModelBundle) -> None:
    # Highest *test* score must not win: selection is on CV only.
    results = [_fake_result(bundle, 0.7, 0.99), _fake_result(bundle, 0.9, 0.6)]
    assert select_champion(results, "roc_auc").cv_metrics["roc_auc"] == 0.9


def test_explainer_returns_top_features(bundle: ModelBundle, sample_raw_df: pd.DataFrame) -> None:
    (explanation,) = ChurnExplainer(bundle, top_k=TOP_K).explain(sample_raw_df.head(1))
    assert len(explanation.contributions) == TOP_K
    shap_abs = [abs(c.shap_value) for c in explanation.contributions]
    assert shap_abs == sorted(shap_abs, reverse=True)
    assert {c.feature for c in explanation.contributions} <= EXPLAINABLE


ADDITIVITY_TOLERANCE = 1e-3  # contributions are rounded to 4 d.p.


def _assert_additive(bundle: ModelBundle, rows: pd.DataFrame) -> None:
    """base_value + all contributions must reproduce the model's log-odds exactly."""
    explainer = ChurnExplainer(bundle, top_k=len(EXPLAINABLE))
    proba = bundle.predict_proba(rows)
    for explanation, p in zip(explainer.explain(rows), proba, strict=True):
        total = explanation.base_value + sum(c.shap_value for c in explanation.contributions)
        assert total == pytest.approx(np.log(p / (1 - p)), abs=ADDITIVITY_TOLERANCE)


def test_tree_explanations_are_additive(bundle: ModelBundle, sample_raw_df: pd.DataFrame) -> None:
    _assert_additive(bundle, sample_raw_df.head(10))


@pytest.mark.parametrize("estimator", ["xgboost", "lightgbm", "logistic_regression"])
def test_explanations_additive_for_every_model_family(
    estimator: str, bundle: ModelBundle, sample_raw_df: pd.DataFrame
) -> None:
    spec = next(ModelSpec(c["name"], c["estimator"], c["params"])
                for c in FAST_MODELS["candidates"] if c["estimator"] == estimator)
    X, y = sample_raw_df[list(FEATURE_COLUMNS)], sample_raw_df[TARGET]
    ctx = BuildContext.from_target(y.to_numpy(), random_seed=0)
    pipeline = build_pipeline(build_estimator(spec, ctx)).fit(X, y)
    _assert_additive(replace(bundle, pipeline=pipeline), sample_raw_df.head(10))


def test_linear_explainer_path(bundle: ModelBundle, sample_raw_df: pd.DataFrame) -> None:
    """Swap in a logistic regression to exercise the LinearExplainer branch."""
    pipeline = clone(bundle.pipeline).set_params(model=LogisticRegression(max_iter=1000))
    pipeline.fit(sample_raw_df[list(FEATURE_COLUMNS)], sample_raw_df[TARGET])
    linear_bundle = replace(bundle, pipeline=pipeline)
    explanations = ChurnExplainer(linear_bundle, top_k=3).explain(sample_raw_df.head(2))
    assert len(explanations) == 2
    assert all(len(e.contributions) == 3 for e in explanations)
    assert explanations[0].contributions[0].direction in {INCREASES, DECREASES}
