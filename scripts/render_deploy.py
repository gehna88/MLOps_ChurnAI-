"""Deploy a specific Docker image tag to Render, and optionally wait until it is live.

Used by CI after every push to main, and by the Rollback workflow to redeploy an older
tag. Standard library only, so it runs anywhere without installing the project.

Environment:
    RENDER_DEPLOY_HOOK_URL  the service's deploy hook (secret; Render > Settings > Deploy Hook)
    RENDER_SERVICE_URL      optional public URL; with --wait, poll its /health until the
                            new build is serving

    python scripts/render_deploy.py --image ghcr.io/gehna88/churnai:sha-1a2b3c4 --wait
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ENV_HOOK = "RENDER_DEPLOY_HOOK_URL"
ENV_SERVICE = "RENDER_SERVICE_URL"
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 15
DEFAULT_WAIT_SECONDS = 900  # free-tier deploys and cold starts can take several minutes

EXIT_OK = 0
EXIT_FAILURE = 1


def hook_with_image(hook_url: str, image: str) -> str:
    """Add Render's imgURL parameter so the hook deploys exactly this image tag."""
    parts = urllib.parse.urlsplit(hook_url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, v) for k, v in query if k != "imgURL"] + [("imgURL", image)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def image_build(image: str) -> str:
    """The tag part of an image reference: ghcr.io/x/churnai:sha-1a2b3c4 -> sha-1a2b3c4."""
    name = image.rsplit("/", 1)[-1]
    if ":" not in name:
        raise ValueError(f"Image '{image}' has no tag; pass an explicit tag such as :sha-1a2b3c4")
    return name.rsplit(":", 1)[1]


def trigger(hook_url: str, image: str) -> None:
    request = urllib.request.Request(hook_with_image(hook_url, image), method="POST")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        print(f"Render accepted the deploy of {image} (HTTP {response.status})")


def current_build(service_url: str) -> str | None:
    try:
        url = service_url.rstrip("/") + "/health"
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            health = json.load(response)
    except (OSError, ValueError):
        return None
    return health.get("build") if health.get("model_loaded") else None


def wait_until_live(service_url: str, build: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        live = current_build(service_url)
        print(f"  live build: {live or 'not ready'} (waiting for {build})")
        if live == build:
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="full image reference including the tag")
    parser.add_argument("--wait", action="store_true", help="poll /health until this build is live")
    parser.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)
    args = parser.parse_args(argv)

    hook = os.environ.get(ENV_HOOK)
    if not hook:
        print(f"{ENV_HOOK} is not set; skipping the Render deploy.")
        return EXIT_OK

    build = image_build(args.image)
    try:
        trigger(hook, args.image)
    except urllib.error.URLError as exc:
        print(f"Render deploy hook failed: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    service = os.environ.get(ENV_SERVICE)
    if not args.wait:
        return EXIT_OK
    if not service:
        print(f"{ENV_SERVICE} is not set; not waiting for the deploy to finish.")
        return EXIT_OK
    if wait_until_live(service, build, args.timeout):
        print(f"{build} is live at {service}")
        return EXIT_OK
    print(f"{build} did not go live within {args.timeout:.0f}s", file=sys.stderr)
    return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
