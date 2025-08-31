"""Parameter estimation utilities for M3C2 scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from orchestration.strategies import ScaleScan


@dataclass
class ParamEstimator:
    """Combine spacing estimation, scale scanning and final selection."""

    strategy: object
    k_neighbors: int = 6

    def estimate_min_spacing(self, points: np.ndarray) -> float:
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        return float(np.mean(distances[:, 1:]))

    def scan_scales(self, points: np.ndarray, avg_spacing: float) -> List[ScaleScan]:
        return self.strategy.scan(points, avg_spacing)

    @staticmethod
    def select_scales(scans: List[ScaleScan]) -> Tuple[float, float]:
        """Select normal and projection scales from scan results."""

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
