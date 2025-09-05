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


def resolve_log_level() -> str:
    """Determine the logging level using config defaults and ``LOG_LEVEL``."""

    config_level = "INFO"
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config_level = data.get("logging", {}).get("level", config_level)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger(__name__).warning(
            "Failed to load logging configuration, using defaults: %s", exc
        )

    return os.getenv("LOG_LEVEL", config_level)

def resolve_log_file(file: str | None = None) -> str:
    """Determine the logging file path using config defaults and ``LOG_FILE``."""
    if file is not None:
        return file

    config_file = "logs/orchestration.log"
    config_path = Path(__file__).resolve().parents[2] / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        config_file = data.get("logging", {}).get("file", config_file)
    except (OSError, json.JSONDecodeError) as exc:
        logging.getLogger(__name__).warning(
            "Failed to load logging configuration, using defaults: %s", exc
        )

    return os.getenv("LOG_FILE", config_file)


def setup_logging() -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level name to use. When ``None`` the level from ``config.json``
        is used, and the ``LOG_LEVEL`` environment variable can override it.
    """

    level = resolve_log_level()
    file = resolve_log_file()

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
    if file:
        path = Path(file)
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
    
    # Suppress excessive matplotlib DEBUG output
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

