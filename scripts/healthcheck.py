"""Container health check: exit 0 if /health reports a loaded model, else 1.

Uses only the standard library so it adds nothing to the image.
"""

import json
import os
import sys
import urllib.request

DEFAULT_PORT = "8000"
TIMEOUT_SECONDS = 3


def main() -> int:
    url = f"http://127.0.0.1:{os.environ.get('PORT', DEFAULT_PORT)}/health"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            healthy = json.load(response).get("model_loaded") is True
    except (OSError, ValueError):
        healthy = False
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
