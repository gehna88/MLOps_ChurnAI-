"""Wait until the live service reports an expected build on /health.

Used by CI and the Rollback workflow after asking Azure to run a new image:
the deploy only counts as done once /health returns {"build": "<expected>"}.

    python scripts/wait_for_build.py --url https://<app>.azurewebsites.net --build sha-1a2b3c4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections.abc import Callable


def fetch_build(url: str, timeout: float = 10.0) -> str | None:
    """Return the "build" reported by <url>/health, or None if it can't be read."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as resp:
            return json.load(resp).get("build")
    except (OSError, ValueError, AttributeError):
        return None


def wait_for_build(
    url: str,
    expected: str,
    attempts: int = 40,
    delay: float = 15.0,
    fetch: Callable[[str], str | None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> bool:
    """Poll /health until it reports `expected`. Returns False if it never does."""
    fetch = fetch or fetch_build
    sleep = sleep or time.sleep
    for attempt in range(1, attempts + 1):
        live = fetch(url)
        print(f"attempt {attempt}/{attempts}: live={live} expected={expected}", flush=True)
        if live == expected:
            return True
        if attempt < attempts:
            sleep(delay)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wait until /health reports a given build.")
    parser.add_argument("--url", required=True, help="Base URL of the live service")
    parser.add_argument("--build", required=True, help="Expected build, e.g. sha-1a2b3c4")
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--delay", type=float, default=15.0, help="Seconds between attempts")
    args = parser.parse_args(argv)

    if wait_for_build(args.url, args.build, args.attempts, args.delay):
        print(f"{args.build} is live at {args.url}")
        return 0
    print(f"::error::{args.build} did not go live at {args.url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
