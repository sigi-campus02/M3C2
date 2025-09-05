"""Utilities for loading and masking comparison distance data.

This module contains small helpers used by the visualization tooling to
discover distance files for different reference variants, read them into
NumPy arrays and return paired arrays with invalid values removed.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def _resolve(fid: str, filename: str) -> str:
    """Return the candidate path for *filename* within *fid*.

    Parameters
    ----------
    fid:
        Folder identifier used as the first search location.
    filename:
        Name of the file to resolve.

    Returns
    -------
    str
        Either ``<fid>/<filename>`` (if it exists) or
        ``data/<fid>/<filename>``.  The path is returned even when the file
        is missing so the caller can decide how to handle it.

    Notes
    -----
    The function performs only an existence check and never raises an
    exception.
    """
    p1 = os.path.join(fid, filename)
    if os.path.exists(p1):
        return p1
    return os.path.join("data", fid, filename)


def _load_reference_variant_data(fid: str, variant: str) -> np.ndarray | None:
    """Load distance data for a specific reference variant.

    Parameters
    ----------
    fid:
        Folder identifier used to look up the data file.
    variant:
        Name of the reference variant that forms part of the filename.

    Returns
    -------
    numpy.ndarray or None
        The loaded distance values.  ``None`` is returned if the file is not
        found or could not be read.

    Error Handling
    --------------
    Missing files or loading errors are logged and result in ``None`` rather
    than an exception being raised.
    """
    basename = f"python_{variant}_m3c2_distances.txt"
    path = _resolve(fid, basename)
    if not os.path.exists(path):
        logger.warning("File not found: %s", path)
        return None
    try:
        return np.loadtxt(path)
    except (OSError, ValueError):  # pragma: no cover - logging only
        logger.exception("Failed to load %s", path)
        return None


def _load_and_mask(fid: str, reference_variants: List[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """Load two reference variants and remove NaN values.

    Parameters
    ----------
    fid:
        Folder identifier that contains the reference distance files.
    reference_variants:
        List containing the names of two variants to be compared.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray] or None
        Two equally sized arrays with NaN values removed.  ``None`` is
        returned if either variant cannot be loaded or if no valid data
        remains after masking.

    Error Handling
    --------------
    Any issues (missing files, only NaN values) are logged and result in
    ``None`` instead of an exception.
    """
    data = [_load_reference_variant_data(fid, v) for v in reference_variants]
    if any(d is None for d in data):
        return None
    a_raw, b_raw = data
    mask = ~np.isnan(a_raw) & ~np.isnan(b_raw)
    a = np.asarray(a_raw[mask], dtype=float)
    b = np.asarray(b_raw[mask], dtype=float)
    if a.size == 0 or b.size == 0:
        logger.warning("Empty values in %s, skipped", fid)
        return None
    return a, b
