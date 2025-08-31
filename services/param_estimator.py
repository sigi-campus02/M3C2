"""Parameter estimation utilities for M3C2 scans.

The :class:`ParamEstimator` combines neighbourhood spacing estimation
with a pluggable scanning strategy that evaluates candidate scale
parameters.  The chosen scales are returned for use in downstream
processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from orchestration.strategies import ScaleScan


@dataclass
class ParamEstimator:
    """Combine spacing estimation, scale scanning and final selection.

    Attributes
    ----------
    strategy:
        Object providing a ``scan`` method returning :class:`ScaleScan`
        instances.
    k_neighbors:
        Number of neighbours used when estimating average point spacing.
    """

    strategy: object
    k_neighbors: int = 6

    def estimate_min_spacing(self, points: np.ndarray) -> float:
        """Estimate the mean spacing between points.

        Parameters
        ----------
        points:
            ``(N,3)`` array of point coordinates.

        Returns
        -------
        float
            Average nearest-neighbour distance.
        """

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        return float(np.mean(distances[:, 1:]))

    def scan_scales(
        self, points: np.ndarray, avg_spacing: float
    ) -> List[ScaleScan]:
        """Run the configured strategy to evaluate candidate scales.

        Parameters
        ----------
        points:
            Point cloud coordinates used for the scan.
        avg_spacing:
            Average spacing between points as returned by
            :meth:`estimate_min_spacing`.

        Returns
        -------
        list[ScaleScan]
            Results of the scan performed by ``strategy``.
        """

        return self.strategy.scan(points, avg_spacing)

    @staticmethod
    def select_scales(scans: List[ScaleScan]) -> Tuple[float, float]:
        """Select normal and projection scales from scan results.

        Parameters
        ----------
        scans:
            List of :class:`ScaleScan` results returned by
            :meth:`scan_scales`.

        Returns
        -------
        tuple[float, float]
            The selected normal and projection scale.

        Raises
        ------
        ValueError
            If the list of ``scans`` is empty.
        """

        if not scans:
            raise ValueError("Keine Scales gefunden.")

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

        if not valid:
            ladder = sorted({float(s.scale) for s in scans})
            mid_idx = len(ladder) // 2
            normal = ladder[mid_idx]
            projection = ladder[mid_idx + 1] if mid_idx + 1 < len(ladder) else ladder[mid_idx]
            return float(normal), float(projection)

        valid.sort(key=lambda s: (float(s.mean_lambda3), -int(s.valid_normals), float(s.roughness)))

        chosen = None
        for scan in valid:
            if scan.roughness > 0 and (float(scan.scale) / float(scan.roughness)) >= 25.0:
                chosen = scan
                break
        if chosen is None:
            chosen = valid[0]

        ladder = sorted({float(s.scale) for s in scans})
        eps = 1e-12
        try:
            idx = next(i for i, v in enumerate(ladder) if abs(v - float(chosen.scale)) <= eps)
        except StopIteration:
            idx = min(range(len(ladder)), key=lambda i: abs(ladder[i] - float(chosen.scale)))

        normal = ladder[idx]
        projection = ladder[idx + 1] if (idx + 1) < len(ladder) else ladder[idx]
        return float(normal), float(projection)
