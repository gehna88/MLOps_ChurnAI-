"""Shared CLI scaffolding so every pipeline handles args and errors the same way."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from churnai.config import Settings, load_settings
from churnai.exceptions import ChurnAIError
from churnai.logging_utils import configure_logging, get_logger

EXIT_OK = 0
EXIT_FAILURE = 1


def run_cli(
    description: str, runner: Callable[[Settings], object], argv: list[str] | None = None
) -> int:
    """Parse args, run the pipeline and map expected failures to an exit code.

    ``argv`` defaults to ``sys.argv[1:]``; pass it explicitly in tests.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    parser.add_argument("--log-format", choices=["json", "text"], default="text")
    args = parser.parse_args(argv)

    configure_logging(fmt=args.log_format)
    logger = get_logger(runner.__module__)
    try:
        runner(load_settings(config_path=args.config))
    except ChurnAIError as exc:
        logger.error("Pipeline failed: %s", exc)
        return EXIT_FAILURE
    return EXIT_OK


def exit_with(code: int) -> None:
    sys.exit(code)
