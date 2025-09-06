"""Tools for estimating spatial parameters of point cloud scans.

This module provides the :class:`ParamEstimator` class which combines three
core tasks required before executing an M3C2 comparison:

* estimating the average spacing between points,
* scanning a range of candidate scales, and
* choosing a suitable pair of normal and projection scales.

The algorithms here rely on external strategy objects and scikit-learn's
nearest-neighbour implementation but keep the surrounding workflow lightweight
and dependency-free.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import logging
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors

from m3c2.m3c2_core.param_handler.strategies import ScaleScan, ScanStrategy


logger = logging.getLogger(__name__)


@dataclass
class ParamEstimator:
    """Estimate spacing and determine suitable scales for M3C2 processing.

    Parameters
    ----------
    strategy:
        Object implementing a ``scan(points, avg_spacing)`` method returning a
        list of :class:`ScaleScan` instances.  This abstraction allows
        different algorithms to be plugged in.
    k_neighbors:
        Number of nearest neighbours used when estimating the mean point
        spacing.  The default ``6`` roughly corresponds to the immediate
        neighbourhood in a regular grid.
    """

    strategy: ScanStrategy
    k_neighbors: int = 6

    def estimate_min_spacing(self, points: np.ndarray) -> float:
        """Return the average distance to the closest neighbours.

        Parameters
        ----------
        points:
            Array of shape ``(N, 3)`` containing the point coordinates.

        Returns
        -------
        float
            Mean spacing between points as estimated from the ``k_neighbors``
            nearest neighbours.

        Raises
        ------
        ValueError
            Propagated from :class:`~sklearn.neighbors.NearestNeighbors` if the
            input array is empty or otherwise invalid.
        """

        logger.info("[Spacing] estimate_min_spacing called with %d points", len(points))
        try:
            # Build a neighbour index including the point itself; therefore use
            # ``k_neighbors + 1`` and later ignore the first self-distance.
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(points)
            distances, _ = nbrs.kneighbors(points)

            # Compute the average of all neighbour distances, skipping the first
            # column which contains zeros for the point to itself.
            spacing = float(np.mean(distances[:, 1:]))
            logger.debug("[Spacing] average spacing %.6f", spacing)
            return spacing
        except (ValueError, RuntimeError, NotFittedError) as err:
            logger.error(
                "[Spacing] failed to estimate spacing for %d points", len(points), exc_info=err
            )
            raise

    def scan_scales(self, points: np.ndarray, avg_spacing: float) -> List[ScaleScan]:
        """Delegate to the configured strategy to evaluate candidate scales.

        Parameters
        ----------
        points:
            Point coordinates used for scanning.
        avg_spacing:
            Average point spacing as estimated by
            :meth:`estimate_min_spacing`.

        Returns
        -------
        list[ScaleScan]
            The raw scan results as produced by the strategy object.
        """

        logger.info(
            "[Scan] scan_scales called with %d points and avg_spacing %.6f",
            len(points),
            float(avg_spacing),
        )
        try:
            # The strategy encapsulates the actual scanning algorithm; this class
            # simply forwards the call.
            scans = self.strategy.scan(points, avg_spacing)
            logger.debug("[Scan] strategy returned %d scans", len(scans))
            return scans
        except (ValueError, RuntimeError, NotFittedError) as err:
            logger.error(
                "[Scan] unexpected error while scanning scales for %d points",
                len(points),
                exc_info=err,
            )
            raise

    @staticmethod
    def select_scales(scans: List[ScaleScan]) -> Tuple[float, float]:
        """Select normal and projection scales from scan results.

        Parameters
        ----------
        scans:
            List of candidate :class:`ScaleScan` objects produced by a
            strategy.

        Returns
        -------
        tuple[float, float]
            Selected ``(normal_scale, projection_scale)`` pair.

        Raises
        ------
        ValueError
            If ``scans`` is empty.
        """

        logger.info("[Select] select_scales called with %d scans", len(scans))
        try:
            if not scans:
                logger.error("[Select] no scales provided")
                raise ValueError("Keine Scales gefunden.")

            # Filter out scans that lack essential quality metrics. A scan is
            # considered valid only if roughness, curvature (``lambda3``) and the
            # number of valid normals are all present, finite and positive.
            valid_scans = [
                scan
                for scan in scans
                if (
                    scan.roughness is not None
                    and not np.isnan(scan.roughness)
                    and scan.mean_lambda3 is not None
                    and not np.isnan(scan.mean_lambda3)
                    and scan.valid_normals is not None
                    and scan.valid_normals > 0
                )
            ]

            # If no scan passes the validation step, fall back to selecting the
            # middle pair of scales from the ladder of all scanned scales. This
            # deterministic fallback avoids failures when input data are poor.
            if not valid_scans:
                scale_ladder = sorted({float(scan.scale) for scan in scans})
                middle_index = len(scale_ladder) // 2
                normal = scale_ladder[middle_index]
                projection = (
                    scale_ladder[middle_index + 1]
                    if middle_index + 1 < len(scale_ladder)
                    else scale_ladder[middle_index]
                )
                logger.debug(
                    "[Select] no valid scans; using middle scales normal=%.6f projection=%.6f",
                    normal,
                    projection,
                )
                return float(normal), float(projection)

            # Sort the validated scans so that smoother, well-defined surfaces are
            # ranked first: primarily by increasing curvature, then by decreasing
            # number of valid normals, and finally by increasing roughness.
            valid_scans.sort(
                key=lambda scan: (
                    float(scan.mean_lambda3),
                    -int(scan.valid_normals),
                    float(scan.roughness),
                )
            )

            # Choose the first scan whose scale-to-roughness ratio indicates a
            # sufficiently smooth surface; otherwise default to the best ranked
            # scan.
            selected_scan = None
            for scan in valid_scans:
                if scan.roughness > 0 and (
                    float(scan.scale) / float(scan.roughness)
                ) >= 25.0:
                    selected_scan = scan
                    logger.debug(
                        "[Select] scan %.6f chosen by ratio (roughness=%.6f, valid_normals=%d)",
                        float(scan.scale),
                        float(scan.roughness),
                        int(scan.valid_normals),
                    )
                    break
            if selected_scan is None:
                selected_scan = valid_scans[0]
                logger.debug(
                    "[Select] no scan met ratio threshold; using best ranked scale %.6f",
                    float(selected_scan.scale),
                )

            # Determine the neighbouring scales of the selected scan from the
            # ladder of unique scales. The chosen scale becomes the normal scale
            # and its next neighbour (or itself if none exists) the projection
            # scale.
            scale_ladder = sorted({float(scan.scale) for scan in scans})
            epsilon = 1e-12  # tolerance for matching floating-point scales
            try:
                scale_index = next(
                    i
                    for i, v in enumerate(scale_ladder)
                    if abs(v - float(selected_scan.scale)) <= epsilon
                )
            except StopIteration:
                scale_index = min(
                    range(len(scale_ladder)),
                    key=lambda i: abs(scale_ladder[i] - float(selected_scan.scale)),
                )
                logger.debug(
                    "[Select] exact selected scale missing; using nearest index %d",
                    scale_index,
                )

            normal = scale_ladder[scale_index]
            projection = (
                scale_ladder[scale_index + 1]
                if (scale_index + 1) < len(scale_ladder)
                else scale_ladder[scale_index]
            )
            logger.debug(
                "[Select] resulting normal=%.6f projection=%.6f", normal, projection
            )
            return float(normal), float(projection)
        except (ValueError, RuntimeError, NotFittedError) as err:
            logger.error(
                "[Select] failed to select scales from %d scans", len(scans), exc_info=err
            )
            raise
