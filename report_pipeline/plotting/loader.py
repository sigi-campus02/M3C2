from __future__ import annotations

"""Simple utilities for reading numeric distance data from text files.

This module provides a lightweight :func:`load_distance_series` function that
reads a one-dimensional series of floating point values from a path on disk.
Only finite values are returned; non-numeric entries trigger a ``ValueError``.
The implementation is intentionally small and does not depend on any project
specific infrastructure which makes it easy to test in isolation.
"""

from pathlib import Path

import numpy as np


def load_distance_series(path: Path) -> np.ndarray:
    """Return a series of floats contained in ``path``.

    Parameters
    ----------
    path:
        Location of a text or CSV file containing numeric values.  The file
        may contain a single column of numbers or a comma separated list.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing only finite floating point values.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file content cannot be interpreted as numeric data.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Distance file not found: {path}")

    try:
        if path.suffix.lower() == ".csv":
            data = np.genfromtxt(path, delimiter=",", dtype=float)
        else:
            data = np.loadtxt(path, dtype=float)
    except OSError as exc:  # pragma: no cover - file reading error
        raise FileNotFoundError(path) from exc
    except ValueError as exc:
        raise ValueError(f"Could not read numeric data from {path}") from exc

    data = np.asarray(data, dtype=float).ravel()
    return data[np.isfinite(data)]
