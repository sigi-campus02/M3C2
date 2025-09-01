"""Helpers for configuring application-wide logging."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging level aliases
# ---------------------------------------------------------------------------
# ``logging`` already exposes ``FATAL`` as a constant equal to ``CRITICAL`` but
# the string "FATAL" is not always recognised when converting from names.
# Register an explicit alias so that ``logging.getLevelName("FATAL")`` yields
# ``logging.CRITICAL``.
logging._nameToLevel["FATAL"] = logging.CRITICAL


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def resolve_log_level(level: str | None = None) -> str:
    """Determine the logging level using config defaults and ``LOG_LEVEL``."""
    if level is not None:
        return level

    config_level = "INFO"
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config_level = data.get("logging", {}).get("level", config_level)
    except Exception:
        pass

    return os.getenv("LOG_LEVEL", config_level)


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level name to use. When ``None`` the level from ``config.json``
        is used, and the ``LOG_LEVEL`` environment variable can override it.
    log_file:
        Optional path to a log file.  When provided, a file handler is added in
        addition to the console handler.  Repeated calls update the existing
        handlers rather than adding duplicates.
    """

    level = resolve_log_level(level)
    numeric_level = logging.getLevelName(level.upper())
    if isinstance(numeric_level, str):  # unknown level names return a string
        numeric_level = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_FORMAT)

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    stream_handler = next(
        (
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ),
        None,
    )
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        root.addHandler(stream_handler)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # File handler
    # ------------------------------------------------------------------
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler: logging.FileHandler | None = None
        for handler in root.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and Path(getattr(handler, "baseFilename", "")).resolve()
                == path.resolve()
            ):
                file_handler = handler
                break

        if file_handler is None:
            file_handler = logging.FileHandler(path)
            root.addHandler(file_handler)

        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

