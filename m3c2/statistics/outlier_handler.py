"""Utility for detecting outliers in distance arrays."""

from __future__ import annotations

import numpy as np

from m3c2.statistics.distance_outlier_metrics import get_outlier_mask


class OutlierHandler:
    """Handle detection of outliers for distance measurements."""

    @staticmethod
    def detect(distances: np.ndarray, method: str, factor: float) -> np.ndarray:
        """Detect outliers and return a mask with uint8 flags.

        Parameters
        ----------
        distances : np.ndarray
            Array of distance values potentially containing NaNs.
        method : str
            Outlier detection method to use.
        factor : float
            Multiplicative factor for the selected method.

        Returns
        -------
        np.ndarray
            Array of uint8 values where ``1`` marks an outlier and ``0`` an inlier
            or missing value.
        """
        valid = distances[~np.isnan(distances)]
        mask_valid, _ = get_outlier_mask(valid, method, factor)
        mask = np.zeros(len(distances), dtype=np.uint8)
        mask[~np.isnan(distances)] = mask_valid.astype(np.uint8)
        return mask


__all__ = ["OutlierHandler"]
