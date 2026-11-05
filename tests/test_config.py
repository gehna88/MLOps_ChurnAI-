from pathlib import Path

import pytest

from churnai.config import load_settings
from churnai.exceptions import ConfigError
from tests.conftest import CONFIG_FILE


def test_settings_load_and_resolve_relative_to_root(tmp_path: Path) -> None:
    settings = load_settings(config_path=CONFIG_FILE, root=tmp_path)
    assert settings.model_path == tmp_path.resolve() / "models" / "model.joblib"
    assert settings.train_path.is_relative_to(tmp_path.resolve())


def test_sqlite_tracking_uri_is_made_absolute(tmp_path: Path) -> None:
    settings = load_settings(config_path=CONFIG_FILE, root=tmp_path)
    assert settings.mlflow_tracking_uri.startswith("sqlite:///")
    assert tmp_path.resolve().as_posix() in settings.mlflow_tracking_uri


def test_env_overrides_take_precedence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CHURNAI_MLFLOW_TRACKING_URI", "http://mlflow.internal:5000")
    monkeypatch.setenv("CHURNAI_MODEL_DIR", str(tmp_path / "elsewhere"))
    settings = load_settings(config_path=CONFIG_FILE, root=tmp_path)
    assert settings.mlflow_tracking_uri == "http://mlflow.internal:5000"
    assert settings.model_path.parent == (tmp_path / "elsewhere").resolve()


def test_ab_test_can_be_enabled_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    assert load_settings(config_path=CONFIG_FILE, root=tmp_path).ab_test.enabled is False
    monkeypatch.setenv("CHURNAI_AB_ENABLED", "true")
    assert load_settings(config_path=CONFIG_FILE, root=tmp_path).ab_test.enabled is True


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(config_path=tmp_path / "nope.yaml", root=tmp_path)


def test_invalid_config_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: {name: x}\n")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_settings(config_path=bad, root=tmp_path)
