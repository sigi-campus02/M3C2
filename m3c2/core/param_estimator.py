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

from m3c2.pipeline.strategies import ScaleScan, ScanStrategy


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

            # Filter out scans lacking essential quality metrics to ensure that
            # subsequent computations work on well-defined data only.
            valid = [
                s
                for s in scans
                if (
                    s.roughness is not None
                    and not np.isnan(s.roughness)
                    and s.mean_lambda3 is not None
                    and not np.isnan(s.mean_lambda3)
                    and s.valid_normals is not None
                    and s.valid_normals > 0
                )
            ]

            # If all scans are invalid, fall back to using the middle scale(s) from
            # the ladder of provided scales to ensure a deterministic result.
            if not valid:
                ladder = sorted({float(s.scale) for s in scans})
                mid_idx = len(ladder) // 2
                normal = ladder[mid_idx]
                projection = (
                    ladder[mid_idx + 1] if mid_idx + 1 < len(ladder) else ladder[mid_idx]
                )
                logger.debug(
                    "[Select] no valid scans; using middle scales normal=%.6f projection=%.6f",
                    normal,
                    projection,
                )
                return float(normal), float(projection)

            # Sort valid scans by increasing ``lambda3`` (curvature) while
            # favouring scans with many valid normals and low roughness.
            valid.sort(
                key=lambda s: (
                    float(s.mean_lambda3),
                    -int(s.valid_normals),
                    float(s.roughness),
                )
            )

            # Choose the first scan whose scale-to-roughness ratio indicates a
            # sufficiently smooth surface; otherwise default to the best ranked
            # scan.
            chosen = None
            for scan in valid:
                if scan.roughness > 0 and (
                    float(scan.scale) / float(scan.roughness)
                ) >= 25.0:
                    chosen = scan
                    logger.debug(
                        "[Select] scan %.6f chosen by ratio (roughness=%.6f, valid_normals=%d)",
                        float(scan.scale),
                        float(scan.roughness),
                        int(scan.valid_normals),
                    )
                    break
            if chosen is None:
                chosen = valid[0]
                logger.debug(
                    "[Select] no scan met ratio threshold; using best ranked scale %.6f",
                    float(chosen.scale),
                )

            # Determine the neighbouring scales of the chosen scan from the ladder
            # of unique scales; this yields normal and projection scale.
            ladder = sorted({float(s.scale) for s in scans})
            eps = 1e-12
            try:
                idx = next(
                    i for i, v in enumerate(ladder) if abs(v - float(chosen.scale)) <= eps
                )
            except StopIteration:
                idx = min(
                    range(len(ladder)), key=lambda i: abs(ladder[i] - float(chosen.scale))
                )
                logger.debug(
                    "[Select] exact chosen scale missing; using nearest index %d", idx
                )

            normal = ladder[idx]
            projection = ladder[idx + 1] if (idx + 1) < len(ladder) else ladder[idx]
            logger.debug(
                "[Select] resulting normal=%.6f projection=%.6f", normal, projection
            )
            return float(normal), float(projection)
        except (ValueError, RuntimeError, NotFittedError) as err:
            logger.error(
                "[Select] failed to select scales from %d scans", len(scans), exc_info=err
            )
            raise
