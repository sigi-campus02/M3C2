"""Utility functions for loading and preparing comparison data."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _resolve(fid: str, filename: str) -> str:
    """Return the path to *filename* for the given folder ID.

    The helper searches first in ``<fid>/`` and then in ``data/<fid>/`` to mirror
    the behaviour of other services in this repository.
    """

    p1 = os.path.join(fid, filename)
    if os.path.exists(p1):
        return p1
    return os.path.join("data", fid, filename)


def _load_ref_variant_data(fid: str, variant: str) -> np.ndarray | None:
    """Load distance values for a specific reference variant."""

    basename = f"python_{variant}_m3c2_distances.txt"
    path = _resolve(fid, basename)
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    try:
        return np.loadtxt(path)
    except Exception as e:  # pragma: no cover - numpy may raise various errors
        logger.error(f"Failed to load {path}: {e}")
        return None


def _load_and_mask(
    fid: str, ref_variants: List[str]
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Load two reference variant arrays and mask ``NaN`` values."""

    data = [_load_ref_variant_data(fid, v) for v in ref_variants]
    if any(d is None for d in data):
        return None
    a_raw, b_raw = data
    mask = ~np.isnan(a_raw) & ~np.isnan(b_raw)
    a = np.asarray(a_raw[mask], dtype=float)
    b = np.asarray(b_raw[mask], dtype=float)
    if a.size == 0 or b.size == 0:
        logger.warning(f"Empty values in {fid}, skipped")
        return None
    return a, b


__all__ = ["_resolve", "_load_ref_variant_data", "_load_and_mask"]

