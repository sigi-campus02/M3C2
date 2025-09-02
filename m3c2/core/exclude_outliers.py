"""Utilities for excluding outliers from M3C2 distance files."""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for outlier detection."""

    dists_path: str
    method: str
    outlier_multiplicator: float = 3.0


@dataclass
class OutlierResult:
    """Container holding split distance rows."""

    inliers: np.ndarray
    outliers: np.ndarray


class OutlierDetector:
    """Perform outlier detection based on a configuration."""

    def __init__(self, config: OutlierConfig) -> None:
        """Initialize the detector with a configuration.

        Parameters
        ----------
        config : OutlierConfig
            Configuration specifying the path to the distance file, the
            detection method to use and the outlier threshold multiplier.
        """

        self.config = config

    @staticmethod
    def _detect_outlier_mask(
        distances: np.ndarray, method: str, factor: float
    ) -> Tuple[np.ndarray, float]:
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
            logger.info(
                "[Exclude Outliers] Outlier-Schwellen: %.6f bis %.6f", lower, upper
            )
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

    def _save(self, result: OutlierResult) -> None:
        """Persist inlier and outlier rows to text files.

        Two files are created in the same directory as the source distance
        file. The file names are derived from the input path without its suffix
        and follow the pattern ``<base>_inlier_<method>.txt`` and
        ``<base>_outlier_<method>.txt``. The arrays are written with
        :func:`numpy.savetxt` using the ``"x y z distance"`` header.
        """

        base = Path(self.config.dists_path).with_suffix("")
        inlier_path = f"{base}_inlier_{self.config.method}.txt"
        outlier_path = f"{base}_outlier_{self.config.method}.txt"
        header = "x y z distance"
        np.savetxt(inlier_path, result.inliers, fmt="%.6f", header=header)
        np.savetxt(outlier_path, result.outliers, fmt="%.6f", header=header)
        logger.info("[Exclude Outliers] Inlier gespeichert: %s", inlier_path)
        logger.info("[Exclude Outliers] Outlier gespeichert: %s", outlier_path)

    def run(self) -> OutlierResult:
        """Load distances, split into inliers/outliers and write results."""

        distances_all = np.loadtxt(self.config.dists_path, skiprows=1)
        valid_mask = ~np.isnan(distances_all[:, 3])
        distances_valid = distances_all[valid_mask]
        mask, _ = self._detect_outlier_mask(
            distances_valid[:, 3], self.config.method, self.config.outlier_multiplicator
        )
        result = OutlierResult(
            inliers=distances_valid[~mask], outliers=distances_valid[mask]
        )

        self._save(result)

        logger.info("[Exclude Outliers] Gesamt: %d", distances_all.shape[0])
        logger.info(
            "[Exclude Outliers] NaN: %d",
            int(np.isnan(distances_all[:, 3]).sum()),
        )
        logger.info(
            "[Exclude Outliers] Valid (ohne NaN): %d", distances_valid.shape[0]
        )
        logger.info("[Exclude Outliers] Methode: %s", self.config.method)
        logger.info("[Exclude Outliers] Outlier: %d", result.outliers.shape[0])
        logger.info("[Exclude Outliers] Inlier: %d", result.inliers.shape[0])

        return result


def exclude_outliers(
    dists_path: str, method: str, outlier_multiplicator: float = 3.0
) -> OutlierResult:
    """Convenience wrapper around :class:`OutlierDetector`."""

    config = OutlierConfig(
        dists_path=dists_path,
        method=method,
        outlier_multiplicator=outlier_multiplicator,
    )
    detector = OutlierDetector(config)
    return detector.run()


__all__ = ["OutlierConfig", "OutlierResult", "OutlierDetector", "exclude_outliers"]
