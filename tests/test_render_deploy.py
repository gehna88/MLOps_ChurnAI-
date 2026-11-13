"""Tests for scripts/render_deploy.py (loaded by path: scripts/ is not a package)."""

import importlib.util
import urllib.parse

import pytest

from tests.conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location("render_deploy", REPO_ROOT / "scripts" / "render_deploy.py")
render_deploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_deploy)

HOOK = "https://api.render.com/deploy/srv-abc123?key=secret"
IMAGE = "ghcr.io/gehna88/churnai:sha-1a2b3c4"


def test_hook_keeps_key_and_adds_encoded_image() -> None:
    url = render_deploy.hook_with_image(HOOK, IMAGE)
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert query == {"key": "secret", "imgURL": IMAGE}
    assert url.startswith("https://api.render.com/deploy/srv-abc123?")


def test_existing_image_parameter_is_replaced() -> None:
    url = render_deploy.hook_with_image(f"{HOOK}&imgURL=old", IMAGE)
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["imgURL"] == [IMAGE]


@pytest.mark.parametrize(
    ("image", "build"),
    [(IMAGE, "sha-1a2b3c4"), ("ghcr.io/gehna88/churnai:v2.0-milestone2", "v2.0-milestone2"),
     ("localhost:5000/churnai:latest", "latest")],
)
def test_image_build_is_the_tag(image: str, build: str) -> None:
    assert render_deploy.image_build(image) == build


def test_image_without_tag_is_rejected() -> None:
    with pytest.raises(ValueError, match="no tag"):
        render_deploy.image_build("localhost:5000/churnai")


def test_missing_hook_skips_without_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(render_deploy.ENV_HOOK, raising=False)
    assert render_deploy.main(["--image", IMAGE]) == render_deploy.EXIT_OK


def test_deploy_triggers_hook_and_waits_for_build(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv(render_deploy.ENV_HOOK, HOOK)
    monkeypatch.setenv(render_deploy.ENV_SERVICE, "https://churnai.example")
    monkeypatch.setattr(render_deploy, "trigger", lambda hook, image: calls.append(image))
    monkeypatch.setattr(render_deploy, "current_build", lambda url: "sha-1a2b3c4")
    assert render_deploy.main(["--image", IMAGE, "--wait"]) == render_deploy.EXIT_OK
    assert calls == [IMAGE]


def test_deploy_fails_if_build_never_goes_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(render_deploy.ENV_HOOK, HOOK)
    monkeypatch.setenv(render_deploy.ENV_SERVICE, "https://churnai.example")
    monkeypatch.setattr(render_deploy, "trigger", lambda hook, image: None)
    monkeypatch.setattr(render_deploy, "current_build", lambda url: "sha-0000000")
    monkeypatch.setattr(render_deploy, "POLL_INTERVAL_SECONDS", 0)
    assert render_deploy.main(["--image", IMAGE, "--wait", "--timeout", "0.05"]) == render_deploy.EXIT_FAILURE
