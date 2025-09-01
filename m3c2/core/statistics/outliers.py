from __future__ import annotations

from typing import Dict

import numpy as np


def get_outlier_mask(clipped, method, outlier_multiplicator):
    if method == "rmse":
        rmse = np.sqrt(np.mean(clipped ** 2))
        outlier_threshold = outlier_multiplicator * rmse
        outlier_mask = np.abs(clipped) > outlier_threshold
    elif method == "iqr":
        q1 = np.percentile(clipped, 25)
        q3 = np.percentile(clipped, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_threshold = f"({lower_bound:.3f}, {upper_bound:.3f})"
        outlier_mask = (clipped < lower_bound) | (clipped > upper_bound)
    elif method == "std":
        mu = np.mean(clipped)
        std = np.std(clipped)
        outlier_threshold = outlier_multiplicator * std
        outlier_mask = np.abs(clipped - mu) > outlier_threshold
    elif method == "nmad":
        med = np.median(clipped)
        nmad = 1.4826 * np.median(np.abs(clipped - med))
        outlier_threshold = outlier_multiplicator * nmad
        outlier_mask = np.abs(clipped - med) > outlier_threshold
    else:
        raise ValueError(
            "Unbekannte Methode für Ausreißer-Erkennung: 'rmse', 'iqr', 'std', 'nmad'"
        )
    return outlier_mask, outlier_threshold


def compute_outliers(inliers: np.ndarray, outliers: np.ndarray) -> Dict[str, float]:
    """Bestimme Kennzahlen zu Inliern und Outliern."""
    mean_out = float(np.mean(outliers)) if outliers.size else np.nan
    std_out = float(np.std(outliers)) if outliers.size > 0 else np.nan

    pos_out = int(np.sum(outliers > 0))
    neg_out = int(np.sum(outliers < 0))
    pos_in = int(np.sum(inliers > 0))
    neg_in = int(np.sum(inliers < 0))

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
