"""Utility functions for configuring application-wide logging.

This module exposes a :func:`setup_logging` function that installs both
console and rotating file handlers on the root logger.  The function is
idempotent and can safely be called multiple times without creating
duplicate handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = "orchestration.log", level: int = logging.INFO) -> None:
    """Configure the root logger with console and file handlers.

    Parameters
    ----------
    log_file:
        Path to the log file. The file and its parent directories are
        created if they do not yet exist.
    level:
        Logging level to configure on the root logger.
    """

    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate log entries when the
    # function is invoked repeatedly.
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


__all__ = ["setup_logging"]

