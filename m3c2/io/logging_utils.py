"""Utility functions for configuring application-wide logging.

This module exposes a :func:`setup_logging` function that installs both
console and rotating file handlers on the root logger.  The function is
idempotent and can safely be called multiple times without creating
duplicate handlers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure the root logger with console and file handlers.

    Configuration defaults are read from ``config.json``.  The logging level
    can be overridden either by the ``LOG_LEVEL`` environment variable or by
    explicitly passing ``level``.
    """

    # Load configuration from config.json if no config dict is supplied
    if config is None:
        cfg_path = Path(__file__).resolve().parents[2] / "config.json"
        if cfg_path.exists():
            try:
                config = json.loads(cfg_path.read_text())
            except Exception:  # pragma: no cover - reading config should not fail hard
                config = {}

    log_cfg = (config or {}).get("logging", {})

    if log_file is None:
        log_file = log_cfg.get("file")

    # Precedence: function argument > environment variable > config default
    level = level or os.getenv("LOG_LEVEL") or log_cfg.get("level", "INFO")
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    fmt = log_cfg.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    date_fmt = log_cfg.get("date_format")
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Console handler
    if console:
        if not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        ):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(numeric_level)
            root.addHandler(ch)
        else:
            for h in root.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, logging.FileHandler
                ):
                    h.setFormatter(formatter)
                    h.setLevel(numeric_level)

    # File handler
    if file and log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == path.resolve()
            for h in root.handlers
        ):
            fh = logging.FileHandler(path)
            fh.setFormatter(formatter)
            fh.setLevel(numeric_level)
            root.addHandler(fh)
        else:
            for h in root.handlers:
                if isinstance(h, logging.FileHandler) and Path(
                    getattr(h, "baseFilename", "")
                ).resolve() == path.resolve():
                    h.setFormatter(formatter)
                    h.setLevel(numeric_level)

    # Ensure at least the level is set even when no handlers requested
    if not root.handlers:
        logging.basicConfig(level=numeric_level, format=fmt, datefmt=date_fmt)

