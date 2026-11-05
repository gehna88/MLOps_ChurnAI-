import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnai.data.schema import FEATURE_COLUMNS, TARGET
from churnai.exceptions import ConfigError
from churnai.features.engineering import (
    BALANCE_IS_ZERO,
    BALANCE_SALARY_RATIO,
    TENURE_AGE_RATIO,
    ChurnFeatureEngineer,
)
from churnai.features.preprocessing import PREPROCESS_STEP, build_pipeline, source_features
from churnai.models.evaluation import compute_metrics, confusion, tune_threshold
from churnai.models.factory import (
    BuildContext,
    ModelSpec,
    build_estimator,
    load_model_specs,
    resolved_params,
)


@pytest.fixture
def features(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    return sample_raw_df[list(FEATURE_COLUMNS)]


def test_feature_engineer_adds_expected_columns(features: pd.DataFrame) -> None:
    out = ChurnFeatureEngineer().fit_transform(features)
    for col in (BALANCE_IS_ZERO, BALANCE_SALARY_RATIO, TENURE_AGE_RATIO):
        assert col in out.columns
    assert set(out[BALANCE_IS_ZERO].unique()) <= {0, 1}
    assert (out.loc[features["Balance"] == 0, BALANCE_IS_ZERO] == 1).all()


def test_source_features_align_with_transformed_width(sample_raw_df: pd.DataFrame) -> None:
    features, y = sample_raw_df[list(FEATURE_COLUMNS)], sample_raw_df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(features, y)
    preprocess = pipe.named_steps[PREPROCESS_STEP]
    width = preprocess.transform(pipe[0].transform(features)).shape[1]
    sources = source_features(preprocess)
    assert len(sources) == width
    assert sources.count("Geography") == 3


def test_scale_pos_weight_auto_is_resolved() -> None:
    ctx = BuildContext.from_target(np.array([0, 0, 0, 1]), random_seed=0)
    spec = ModelSpec("x", "xgboost", {"scale_pos_weight": "auto"})
    assert resolved_params(spec, ctx)["scale_pos_weight"] == 3.0
    assert build_estimator(spec, ctx).get_params()["scale_pos_weight"] == 3.0


def test_unknown_estimator_raises() -> None:
    with pytest.raises(ConfigError, match="Unknown estimator"):
        build_estimator(ModelSpec("x", "random_forest_9000"), BuildContext(0, 1.0))


def test_duplicate_candidate_names_rejected(tmp_path) -> None:
    path = tmp_path / "m.yaml"
    path.write_text(
        "candidates:\n"
        "  - {name: a, estimator: xgboost}\n"
        "  - {name: a, estimator: lightgbm}\n"
    )
    with pytest.raises(ConfigError, match="Duplicate"):
        load_model_specs(path)


def test_metrics_and_threshold_tuning() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.3, 0.35, 0.9])
    assert compute_metrics(y, proba, 0.5)["roc_auc"] == 1.0
    best = tune_threshold(y, proba, "f1")
    assert 0.3 < best <= 0.35
    assert confusion(y, proba, best) == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_unsupported_threshold_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        tune_threshold(np.array([0, 1]), np.array([0.2, 0.8]), "roc_auc")
