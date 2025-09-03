"""Helpers for locating and loading distance data for reports.

This module provides small utility functions that resolve the location of
M3C2 distance files and load them into :mod:`numpy` arrays.  The functions
were extracted from :mod:`report_service` to keep the report building logic
focused on orchestration rather than file handling.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..loaders.distance_loader import load_1col_distances

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------

def resolve_path(fid: str, filename: str) -> str:
    """Return the path to *filename* for the given folder ID.

    The function first checks whether ``filename`` exists relative to
    ``fid``.  If not, it falls back to the legacy ``data/Multi-illumination``
    directory structure used in older test data sets.
    """

    candidate = os.path.join(fid, filename)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(
        "data", "Multi-illumination", "Job_0378_8400-110", "1-3_2-3", fid, filename
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    fid: str, filenames: List[str], versions: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Load distance data for a folder and compute Gaussian parameters.

    Parameters
    ----------
    fid:
        Folder identifier from which the data should be loaded.
    filenames, versions:
        Currently unused but kept for API compatibility with the former
        implementation.  ``versions`` is iterated to assemble the expected
        file names which are then resolved via :func:`resolve_path`.

    Returns
    -------
    tuple(dict, dict)
        Two dictionaries: one mapping the resolved labels to the loaded
        distance arrays and another containing the Gaussian ``(mu, sigma)``
        parameters estimated from these arrays.
    """

    data_with: Dict[str, np.ndarray] = {}
    gauss_with: Dict[str, Tuple[float, float]] = {}

    for v in versions:
        base_with = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
        path_with = resolve_path(fid, base_with)
        logger.info("[Report] Lade WITH: %s", path_with)
        if not os.path.exists(path_with):
            logger.warning("[Report] Datei fehlt (WITH): %s", path_with)
            continue
        try:
            if v.lower() == "cc":
                try:
                    arr = load_1col_distances(path_with)
                except (OSError, ValueError) as e:
                    logger.warning(
                        "[Report] Standard-Loader fehlgeschlagen (WITH: %s): %s – versuche CC-Fallback",
                        path_with,
                        e,
                    )
                    try:
                        df = pd.read_csv(path_with, sep=";")
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) == 0:
                            raise ValueError("Keine numerische Spalte gefunden (CC).")
                        arr = df[num_cols[0]].astype(float).to_numpy()
                        arr = arr[np.isfinite(arr)]
                    except (OSError, ValueError) as e:
                        logger.error(
                            "[Report] CC-Fallback fehlgeschlagen (WITH: %s): %s",
                            path_with,
                            e,
                        )
                        continue
            else:
                arr = load_1col_distances(path_with)
        except (OSError, ValueError) as e:
            logger.error("[Report] Laden fehlgeschlagen (WITH: %s): %s", path_with, e)
            continue

        if arr.size:
            label = f"{v}_{fid}"
            data_with[label] = arr
            mu, std = norm.fit(arr)
            gauss_with[label] = (float(mu), float(std))

    return data_with, gauss_with
