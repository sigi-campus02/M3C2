"""Outlier detection and analysis utilities.

This module provides helpers to compute boolean masks identifying outliers
based on various statistical measures (RMSE, interquartile range, standard
deviation and NMAD) as well as functions to summarize inlier and outlier
distributions.
"""

from __future__ import annotations

from typing import Dict

import logging
import numpy as np


logger = logging.getLogger(__name__)


def get_outlier_mask(clipped, method, outlier_multiplicator):
    """Create a boolean mask marking elements considered outliers.

    Parameters
    ----------
    clipped : np.ndarray
        Array of numeric values whose outliers should be detected.
    method : str
        Outlier detection algorithm to use; one of ``"rmse"``, ``"iqr"``,
        ``"std"`` or ``"nmad"``.
    outlier_multiplicator : float
        Multiplier applied to the base statistic of the selected method to
        derive the outlier threshold.

    Returns
    -------
    tuple[np.ndarray, float | str]
        A tuple containing the boolean mask of detected outliers and the
        threshold used for classification.  For the ``"iqr"`` method the
        threshold is returned as a formatted string ``"(lower, upper)"``.
    """
    logger.info("[Outliers] Methode: %s", method)
    if method == "rmse":
        rmse = np.sqrt(np.mean(clipped**2))
        outlier_threshold = outlier_multiplicator * rmse
        outlier_mask = np.abs(clipped) > outlier_threshold
        logger.info(
            "[Outliers] RMSE: %.6f, Outlier-Schwelle: %.6f",
            rmse,
            outlier_threshold,
        )
    elif method == "iqr":
        q1 = np.percentile(clipped, 25)
        q3 = np.percentile(clipped, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_threshold = f"({lower_bound:.3f}, {upper_bound:.3f})"
        outlier_mask = (clipped < lower_bound) | (clipped > upper_bound)
        logger.info("[Outliers] IQR: %.6f", iqr)
        logger.info(
            "[Outliers] Outlier-Schwellen: %.6f bis %.6f",
            lower_bound,
            upper_bound,
        )
    elif method == "std":
        mu = np.mean(clipped)
        std = np.std(clipped)
        outlier_threshold = outlier_multiplicator * std
        outlier_mask = np.abs(clipped - mu) > outlier_threshold
        logger.info(
            "[Outliers] STD: %.6f, Outlier-Schwelle: %.6f",
            std,
            outlier_threshold,
        )
    elif method == "nmad":
        med = np.median(clipped)
        nmad = 1.4826 * np.median(np.abs(clipped - med))
        outlier_threshold = outlier_multiplicator * nmad
        outlier_mask = np.abs(clipped - med) > outlier_threshold
        logger.info(
            "[Outliers] NMAD: %.6f, Outlier-Schwelle: %.6f",
            nmad,
            outlier_threshold,
        )
    else:
        raise ValueError(
            "Unbekannte Methode für Ausreißer-Erkennung: 'rmse', 'iqr', 'std', 'nmad'"
        )
    return outlier_mask, outlier_threshold


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
    ``NaN`` (for ``mean_out`` and ``std_out``) and the counts become zero.  This
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
