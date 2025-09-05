"""Utilities for loading M3C2 distance data for visualisation."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_1col_distances(path: str) -> np.ndarray:
    """Load a distance file containing a single numeric column."""
    arr = np.loadtxt(path, ndmin=2)
    vals = arr[:, 0].astype(float)
    return vals[np.isfinite(vals)]


def load_coordinates_inlier_distances(path: str) -> np.ndarray:
    """Load a coordinates_inlier file and return only the distance column."""
    arr = np.loadtxt(path, ndmin=2, skiprows=1)
    if arr.shape[1] < 4:
        raise ValueError(f"Erwarte 4 Spalten (x y z distance) in: {path}")
    vals = arr[:, -1].astype(float)
    return vals[np.isfinite(vals)]


def scan_distance_files_by_index(data_dir: str, versions=("python", "CC")) -> Tuple[Dict[int, Dict[str, Dict[str, np.ndarray]]], Dict[str, str]]:
    """Scan *data_dir* for distance files and group them by index.

    The function recognises files matching the naming scheme used in the
    project and returns a mapping ``index -> data`` as well as a mapping of
    case identifiers to colours that remain stable across parts.
    """
    logger.info("[Scan] Scanne Distanzdateien in %s für Versionen: %s", data_dir, versions)

    pat_with = re.compile(
        r'^(?P<ver>(?:' + "|".join(versions) + r'))_'
        r'(?P<comparison>[ab]-\d+(?:-AI)?)'
        r'-'
        r'(?P<reference>[ab]-\d+(?:-AI)?)'
        r'_m3c2_distances\.txt$',
        re.IGNORECASE,
    )
    pat_inl = re.compile(
        r'^(?P<ver>(?:' + "|".join(versions) + r'))_'
        r'(?P<comparison>[ab]-\d+(?:-AI)?)'
        r'-'
        r'(?P<reference>[ab]-\d+(?:-AI)?)'
        r'_m3c2_distances_coordinates_inlier_(?P<meth>[a-zA-Z0-9_]+)\.txt$',
        re.IGNORECASE,
    )

    def idx_of(tag: str) -> int:
        """Return the numerical index encoded in a tag.

        Tags follow the pattern ``'<prefix>-<num>'`` where ``<prefix>`` is
        either ``a`` or ``b`` and ``<num>`` is an integer.  An optional
        ``-AI`` suffix may be present.  This helper extracts the ``<num>``
        portion and returns it as an integer, or ``-1`` if the tag does not
        match the expected format.
        """
        m = re.match(r'^[ab]-(\d+)(?:-AI)?$', tag, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    def to_case_and_label(comparison: str, reference: str, i: int) -> tuple[str, str]:
        """Derive case identifier and label for a dataset comparison.

        Both *comparison* and *reference* contain tags such as ``"a-1"`` or
        ``"b-2-AI"``.  The presence of the ``"-AI"`` suffix determines the
        classification into one of four cases:

        * ``CASE1`` – neither tag includes ``"-AI"``.
        * ``CASE2`` – only the reference tag includes ``"-AI"``.
        * ``CASE3`` – only the comparison tag includes ``"-AI"``.
        * ``CASE4`` – both tags include ``"-AI"``.

        The returned label mirrors the underlying tags using the pattern
        ``"a-{i}... vs b-{i}..."``.
        """

        comparison_ai = "-AI" in comparison
        reference_ai = "-AI" in reference
        if not comparison_ai and not reference_ai:
            return "CASE1", f"a-{i} vs b-{i}"
        if not comparison_ai and reference_ai:
            return "CASE2", f"a-{i} vs b-{i}-AI"
        if comparison_ai and not reference_ai:
            return "CASE3", f"a-{i}-AI vs b-{i}"
        if comparison_ai and reference_ai:
            return "CASE4", f"a-{i}-AI vs b-{i}-AI"
        return "CASE1", f"a-{i} vs b-{i}"

    per_index: Dict[int, Dict[str, Dict[str, np.ndarray]]] = defaultdict(
        lambda: {"WITH": {}, "INLIER": {}, "CASE_WITH": {}, "CASE_INLIER": {}}
    )

    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name)
        if not os.path.isfile(p):
            continue

        mW = pat_with.match(name)
        if mW:
            comparison, reference = mW.group("comparison"), mW.group("reference")
            i_comparison, i_reference = idx_of(comparison), idx_of(reference)
            if i_comparison == i_reference and i_comparison != -1:
                i = i_comparison
                cas, label = to_case_and_label(comparison, reference, i)
                try:
                    arr = load_1col_distances(p)
                    per_index[i]["WITH"][label] = arr
                    per_index[i]["CASE_WITH"][label] = cas
                except (OSError, ValueError, StopIteration):
                    logger.warning(
                        "[Scan] Laden fehlgeschlagen (WITH: %s)", name, exc_info=True
                    )
            continue

        mI = pat_inl.match(name)
        if mI:
            comparison, reference = mI.group("comparison"), mI.group("reference")
            i_comparison, i_reference = idx_of(comparison), idx_of(reference)
            if i_comparison == i_reference and i_comparison != -1:
                i = i_comparison
                cas, label = to_case_and_label(comparison, reference, i)
                try:
                    arr = load_coordinates_inlier_distances(p)
                    per_index[i]["INLIER"][label] = arr
                    per_index[i]["CASE_INLIER"][label] = cas
                except (OSError, ValueError, StopIteration):
                    logger.warning(
                        "[Scan] Laden fehlgeschlagen (INLIER: %s)", name, exc_info=True
                    )
            continue

    case_colors = {
        "CASE1": "#1f77b4",
        "CASE2": "#ff7f0e",
        "CASE3": "#2ca02c",
        "CASE4": "#9467bd",
    }
    return per_index, case_colors
