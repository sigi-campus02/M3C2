"""Utilities for excluding outliers from M3C2 distance files.

The original implementation mixed reading, processing and writing logic in a
single function.  This module now separates these concerns and exposes a small
typed API that is easier to test.  Distances are loaded from a file, processed
into inliers/outliers and can then optionally be written back to disk.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class OutlierResult:
    """Container holding split distance rows."""

    inliers: np.ndarray
    outliers: np.ndarray


def _load_distances(file_path: str) -> np.ndarray:
    """Load distance rows ignoring the first header line."""

    return np.loadtxt(file_path, skiprows=1)


def _detect_outlier_mask(distances: np.ndarray, method: str, factor: float) -> Tuple[np.ndarray, float]:
    """Return a boolean mask of outliers and the threshold used."""

    if method == "rmse":
        metric = float(np.sqrt(np.mean(distances**2)))
        threshold = factor * metric
        mask = np.abs(distances) > threshold
        logger.info("[Exclude Outliers] RMS: %.6f", metric)
        logger.info("[Exclude Outliers] Outlier-Schwelle: %.6f", threshold)
    elif method == "iqr":
        q1, q3 = np.percentile(distances, [25, 75])
        metric = q3 - q1
        lower = q1 - 1.5 * metric
        upper = q3 + 1.5 * metric
        mask = (distances < lower) | (distances > upper)
        threshold = float(max(abs(lower), abs(upper)))
        logger.info("[Exclude Outliers] IQR: %.6f", metric)
        logger.info("[Exclude Outliers] Outlier-Schwellen: %.6f bis %.6f", lower, upper)
    elif method == "std":
        mean = float(np.mean(distances))
        metric = float(np.std(distances))
        threshold = factor * metric
        mask = np.abs(distances - mean) > threshold
        logger.info("[Exclude Outliers] STD: %.6f", metric)
        logger.info("[Exclude Outliers] Outlier-Schwelle: %.6f", threshold)
    elif method == "nmad":
        median = float(np.median(distances))
        metric = 1.4826 * float(np.median(np.abs(distances - median)))
        threshold = factor * metric
        mask = np.abs(distances - median) > threshold
    else:
        raise ValueError("Unknown outlier detection method")

    return mask, threshold


def exclude_outliers(
    file_path: str,
    method: str,
    outlier_multiplicator: float = 3.0,
) -> OutlierResult:
    """Split the given distance file into inliers and outliers.

    Parameters
    ----------
    file_path:
        Path to a ``python_*_m3c2_distances_coordinates.txt`` file.
    method:
        Outlier detection method (``rmse``, ``iqr``, ``std`` or ``nmad``).
    outlier_multiplicator:
        Factor applied to the respective metric.
    """

    distances_all = _load_distances(file_path)
    valid_mask = ~np.isnan(distances_all[:, 3])
    distances_valid = distances_all[valid_mask]
    mask, _ = _detect_outlier_mask(distances_valid[:, 3], method, outlier_multiplicator)

    result = OutlierResult(inliers=distances_valid[~mask], outliers=distances_valid[mask])

    base = Path(file_path).with_suffix("")
    inlier_path = f"{base}_inlier_{method}.txt"
    outlier_path = f"{base}_outlier_{method}.txt"
    header = "x y z distance"
    np.savetxt(inlier_path, result.inliers, fmt="%.6f", header=header)
    np.savetxt(outlier_path, result.outliers, fmt="%.6f", header=header)

    logger.info("[Exclude Outliers] Gesamt: %d", distances_all.shape[0])
    logger.info("[Exclude Outliers] NaN: %d", int(np.isnan(distances_all[:, 3]).sum()))
    logger.info("[Exclude Outliers] Valid (ohne NaN): %d", distances_valid.shape[0])
    logger.info("[Exclude Outliers] Methode: %s", method)
    logger.info("[Exclude Outliers] Outlier: %d", result.outliers.shape[0])
    logger.info("[Exclude Outliers] Inlier: %d", result.inliers.shape[0])
    logger.info("[Exclude Outliers] Inlier gespeichert: %s", inlier_path)
    logger.info("[Exclude Outliers] Outlier gespeichert: %s", outlier_path)

    return result

