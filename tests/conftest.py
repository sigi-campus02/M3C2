"""Pytest configuration and shared fixtures for the test suite.

This module ensures that the project's root directory is available on
``sys.path`` so that tests can import the local ``m3c2`` package without
requiring an installed distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
