from __future__ import annotations

"""Helpers for loading distance measurement data used in reports."""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..loaders.distance_loader import load_1col_distances

logger = logging.getLogger(__name__)


def resolve_path(fid: str, filename: str) -> str:
    """Return the path to *filename* for the given folder ID.

    The function first looks for the file inside the folder ``fid``. If it is
    not present, it falls back to the default ``data`` directory used in tests
    and examples.
    """
    p1 = os.path.join(fid, filename)
    if os.path.exists(p1):
        return p1
    return os.path.join(
        "data", "Multi-illumination", "Job_0378_8400-110", "1-3_2-3", fid, filename
    )


def load_distance_data(
    fid: str, filenames: List[str], versions: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Load distance measurements for a folder.

    Parameters
    ----------
    fid:
        Folder identifier.
    filenames, versions:
        Kept for API compatibility with the previous implementation. The
        ``filenames`` parameter is currently unused.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]]]
        Loaded distance arrays keyed by ``<version>_<fid>`` and corresponding
        Gaussian parameters ``(mu, sigma)``.
    """
    data_with: Dict[str, np.ndarray] = {}
    gauss_with: Dict[str, Tuple[float, float]] = {}

    for v in versions:
        base_with = (
            f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
        )
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
