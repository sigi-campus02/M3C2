"""Tests for :mod:`m3c2.io.logging_utils`."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from m3c2.io.logging_utils import setup_logging


def test_setup_logging_idempotent(tmp_path: Path) -> None:
    logger = logging.getLogger()
    # ensure clean logger state
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_file = tmp_path / "test.log"

    setup_logging(log_file=str(log_file))
    handlers_before = list(logger.handlers)

    setup_logging(log_file=str(log_file))
    handlers_after = list(logger.handlers)

    assert handlers_after == handlers_before
    assert len(logger.handlers) == 2

    # cleanup
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def test_fatal_alias() -> None:
    """Ensure that the ``FATAL`` level name maps to ``CRITICAL``."""
    assert logging.getLevelName("FATAL") == logging.CRITICAL
