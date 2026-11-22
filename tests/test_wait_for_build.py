import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "wait_for_build.py"
_spec = importlib.util.spec_from_file_location("wait_for_build", SCRIPT)
wfb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wfb)


def test_returns_immediately_when_build_matches():
    sleeps = []
    assert wfb.wait_for_build("http://x", "sha-1", fetch=lambda u: "sha-1", sleep=sleeps.append)
    assert sleeps == []


def test_waits_until_new_build_is_live():
    seen = iter([None, "sha-old", "sha-new"])
    sleeps = []
    assert wfb.wait_for_build(
        "http://x", "sha-new", attempts=5, delay=1, fetch=lambda u: next(seen), sleep=sleeps.append
    )
    assert sleeps == [1, 1]


def test_gives_up_after_all_attempts():
    sleeps = []
    assert not wfb.wait_for_build(
        "http://x", "sha-new", attempts=3, delay=2, fetch=lambda u: "sha-old", sleep=sleeps.append
    )
    assert sleeps == [2, 2]


def test_fetch_build_returns_none_when_unreachable():
    assert wfb.fetch_build("http://127.0.0.1:9", timeout=0.5) is None


def test_main_exit_codes(monkeypatch):
    monkeypatch.setattr(wfb, "fetch_build", lambda url: "sha-1")
    assert wfb.main(["--url", "http://x", "--build", "sha-1", "--attempts", "1"]) == 0
    assert wfb.main(["--url", "http://x", "--build", "sha-2", "--attempts", "1"]) == 1
