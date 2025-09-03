"""Outlier detection and analysis utilities.

This module provides helpers to compute boolean masks identifying outliers
based on various statistical measures (RMSE, interquartile range, standard
deviation and NMAD) as well as functions to summarize inlier and outlier
distributions.
"""

from __future__ import annotations

from typing import Callable, Dict

import logging
import numpy as np


logger = logging.getLogger(__name__)


def _mask_rmse(clipped: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    rmse = float(np.sqrt(np.mean(clipped**2)))
    threshold = factor * rmse
    mask = np.abs(clipped) > threshold
    logger.info("[Outliers] RMSE: %.6f, Threshold: %.6f", rmse, threshold)
    return mask, threshold


def _mask_iqr(clipped: np.ndarray, factor: float) -> tuple[np.ndarray, str]:
    q1 = np.percentile(clipped, 25)
    q3 = np.percentile(clipped, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    threshold = f"({lower_bound:.3f}, {upper_bound:.3f})"
    mask = (clipped < lower_bound) | (clipped > upper_bound)
    logger.info("[Outliers] IQR: %.6f", iqr)
    logger.info("[Outliers] Thresholds: %.6f to %.6f", lower_bound, upper_bound)
    return mask, threshold


def _mask_std(clipped: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    mu = np.mean(clipped)
    std = np.std(clipped)
    threshold = factor * std
    mask = np.abs(clipped - mu) > threshold
    logger.info("[Outliers] STD: %.6f, Threshold: %.6f", std, threshold)
    return mask, threshold


def _mask_nmad(clipped: np.ndarray, factor: float) -> tuple[np.ndarray, float]:
    med = np.median(clipped)
    nmad = 1.4826 * np.median(np.abs(clipped - med))
    threshold = factor * nmad
    mask = np.abs(clipped - med) > threshold
    logger.info("[Outliers] NMAD: %.6f, Threshold: %.6f", nmad, threshold)
    return mask, threshold


_MASK_DISPATCH: Dict[str, Callable[[np.ndarray, float], tuple[np.ndarray, float | str]]] = {
    "rmse": _mask_rmse,
    "iqr": _mask_iqr,
    "std": _mask_std,
    "nmad": _mask_nmad,
}


def get_outlier_mask(
    clipped: np.ndarray, method: str, factor: float
) -> tuple[np.ndarray, float | str]:
    """Create a boolean mask marking elements considered outliers.

    Parameters
    ----------
    clipped : np.ndarray
        Array of numeric values whose outliers should be detected.
    method : str
        Outlier detection algorithm to use; one of ``"rmse"``, ``"iqr"``,
        ``"std"`` or ``"nmad"``.
    factor : float
        Multiplier applied to the base statistic of the selected method to
        derive the outlier threshold.

    Returns
    -------
    tuple[np.ndarray, float | str]
        A tuple containing the boolean mask of detected outliers and the
        threshold used for classification. For the ``"iqr"`` method the
        threshold is returned as a formatted string ``"(lower, upper)"``.
    """
    logger.info("[Outliers] Method: %s", method)
    try:
        func = _MASK_DISPATCH[method]
    except KeyError as exc:
        raise ValueError(
            "Unknown method for outlier detection: 'rmse', 'iqr', 'std', 'nmad'"
        ) from exc
    return func(clipped, factor)


def compute_outliers(inliers: np.ndarray, outliers: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for inlier and outlier values.

    Parameters
    ----------
    inliers : np.ndarray
        Array of values that are considered *inliers* (values that met the
        filtering criteria of the outlier detection step).
    outliers : np.ndarray
        Array of values that are considered *outliers* (values that violated
        the thresholds of the detection step).

    Returns
    -------
    Dict[str, float]
        A dictionary with counts and simple statistics:

        ``outlier_count`` / ``inlier_count``
            Number of outlier and inlier elements.
        ``mean_out`` / ``std_out``
            Mean and standard deviation of the ``outliers`` array.
        ``pos_out`` / ``neg_out``
            Counts of positive and negative outliers.
        ``pos_in`` / ``neg_in``
            Counts of positive and negative inliers.

    Notes
    -----
    If either of the input arrays is empty the respective statistics are set to
    ``NaN`` (for ``mean_out`` and ``std_out``) and the counts become zero. This
    allows callers to handle cases where no outliers or inliers were found
    without raising an exception.
    """
    mean_out = float(np.mean(outliers)) if outliers.size else np.nan
    std_out = float(np.std(outliers)) if outliers.size > 0 else np.nan

    pos_out = int(np.sum(outliers > 0))
    neg_out = int(np.sum(outliers < 0))
    pos_in = int(np.sum(inliers > 0))
    neg_in = int(np.sum(inliers < 0))

    logger.info(
        "[Outliers] Outlier: +%d / -%d | Inlier: +%d / -%d",
        pos_out,
        neg_out,
        pos_in,
        neg_in,
    )

    return {
        "outlier_count": int(outliers.size),
        "inlier_count": int(inliers.size),
        "mean_out": mean_out,
        "std_out": std_out,
        "pos_out": pos_out,
        "neg_out": neg_out,
        "pos_in": pos_in,
        "neg_in": neg_in,
    }

