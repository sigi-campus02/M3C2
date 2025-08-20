# ParamEstimator
# +estimate_avg_spacing(points, k=6) : : float
# +scan_scales(points, strategy, avg_spacing, ...) : : list
# +select_scales(results):(normal, projection)

# param_estimator.py
from __future__ import annotations

import logging
import numpy as np
from typing import List, Tuple

from sklearn.neighbors import NearestNeighbors
from strategies import ScaleScan, ScaleStrategy


class ParamEstimator:
    """
    Kapselt Spacing-Schätzung + Scale-Scan + finale Auswahl.
    """

    @staticmethod
    def estimate_min_spacing(
            points: np.ndarray,
            subsample: int = 200_000,
            trim: tuple[float, float] = (0.05, 0.95),
            use_median: bool = True
    ) -> float:
        """
        Robuste Schätzung der Punktwolkenauflösung (minSpacing) per 1-NN.
        """
        if subsample and len(points) > subsample:
            idx = np.random.choice(len(points), size=subsample, replace=False)
            points = points[idx]

        nn = NearestNeighbors(n_neighbors=2).fit(points)
        dists, _ = nn.kneighbors(points)
        nn1 = dists[:, 1]  # 1-NN (erste Spalte ist 0 = Punkt selbst)

        if trim:
            lo, hi = np.quantile(nn1, [trim[0], trim[1]])
            nn1 = nn1[(nn1 >= lo) & (nn1 <= hi)]

        min_spacing = float(np.median(nn1) if use_median else np.mean(nn1))
        return min_spacing

    @staticmethod
    def scan_scales(points: np.ndarray, strategy: ScaleStrategy, avg_spacing: float) -> List[ScaleScan]:
        scans = strategy.scan(points, avg_spacing)
        return scans


    @staticmethod
    def select_scales(scans: List[ScaleScan]) -> Tuple[float, float]:
        """
        Paper-nahe Auswahl:
        - Primär: minimale mittlere λ_min (Planarität), dann hohe Abdeckung, dann geringe σ(D)
        - D/σ-Regel: bevorzuge ersten Kandidaten mit D/sigma >= 25
        - Projektion d: größte getestete Skala < D; Fallback 0.5 * D
        """
        if not scans:
            raise ValueError("Keine Scales gefunden.")

        # Nur valide Scans verwenden
        valid = [
            s for s in scans
            if (
                s.roughness is not None and not np.isnan(s.roughness)
                and s.mean_lambda3 is not None and not np.isnan(s.mean_lambda3)
                and s.valid_normals is not None and s.valid_normals > 0
            )
        ]

        if not valid:
            # Fallback: Median-D, d = 0.5 * D (konservativ, d < D)
            scales_sorted = sorted(float(s.scale) for s in scans)
            mid = scales_sorted[len(scales_sorted) // 2]
            normal = float(mid)
            projection = float(0.5 * normal)
            return normal, projection

        # Primär Planarität (λ_min), dann Abdeckung, dann geringe σ(D)
        valid.sort(key=lambda s: (float(s.mean_lambda3), -int(s.valid_normals), float(s.roughness)))

        # D/σ-Regel: ersten Kandidaten nehmen, der sie erfüllt
        chosen = None
        for s in valid:
            if s.roughness > 0 and (float(s.scale) / float(s.roughness)) >= 25.0:
                chosen = s
                break
        if chosen is None:
            chosen = valid[0]  # bestes λ_min

        normal = float(chosen.scale)

        # größte getestete Skala < D; sonst 0.5 * D
        all_scales = sorted({float(s.scale) for s in scans})
        smaller = [x for x in all_scales if x < normal]
        projection = float(smaller[-1]) if smaller else float(0.5 * normal)

        return normal, projection