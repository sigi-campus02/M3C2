"""Tests for :mod:`m3c2.io.logging_utils`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import m3c2.io.logging_utils as logging_utils


def test_setup_logging_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Verify that repeated logging configuration calls have no side effects.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by ``pytest`` to store the log file.

    Returns
    -------
    None
        This function performs assertions and returns nothing.

    Examples
    --------
    >>> from pathlib import Path
    >>> from m3c2.io.logging_utils import setup_logging
    >>> setup_logging()
    >>> setup_logging()
    """
    logger = logging.getLogger()
    # ensure clean logger state
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_file = tmp_path / "test.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))

    logging_utils.setup_logging()
    handlers_before = list(logger.handlers)

    logging_utils.setup_logging()
    handlers_after = list(logger.handlers)

    assert handlers_after == handlers_before
    assert len(logger.handlers) == 2

    # cleanup
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    monkeypatch.delenv("LOG_FILE", raising=False)


def test_fatal_alias() -> None:
    """Ensure that the ``FATAL`` level name maps to ``CRITICAL``."""
    assert logging.getLevelName("FATAL") == logging.CRITICAL


def test_resolve_log_level_warning(monkeypatch, caplog) -> None:
    """Warn when configuration loading fails and use default level."""
    # ensure environment variable does not interfere
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    def bad_load(handle) -> None:  # type: ignore[unused-ignore]
        raise json.JSONDecodeError("boom", "", 0)

    monkeypatch.setattr(logging_utils.json, "load", bad_load)

    with caplog.at_level(logging.WARNING):
        level = logging_utils.resolve_log_level()

    assert level == "INFO"
    assert any(
        "Failed to load logging configuration" in record.message
        for record in caplog.records
    )
