"""Utility functions for configuring application-wide logging.

This module exposes a :func:`setup_logging` function that installs both
console and rotating file handlers on the root logger.  The function is
idempotent and can safely be called multiple times without creating
duplicate handlers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
        config: Optional[Dict[str, Any]] = None, 
        log_file: str = "orchestration.log", 
        level: Optional[str] = None,
        console: bool = True,
        file: bool = True,
    ) -> None:
    """Configure the root logger with console and file handlers.

    Parameters
    ----------
    log_file:
        Path to the log file. The file and its parent directories are
        created if they do not yet exist.
    level:
        Logging level to configure on the root logger.
    """

    log_cfg = (config or {}).get("logging", {})

    if log_file is None:
        log_file = log_cfg.get("file")
    if level is None:
        level = log_cfg.get("level", "INFO")

    handlers = []
    if console:
        handlers.append(logging.StreamHandler())
    if file and log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_cfg.get(
            "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ),
        handlers=handlers or None,
    )

