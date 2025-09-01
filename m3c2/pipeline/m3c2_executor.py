from __future__ import annotations
"""Execute the M3C2 algorithm and persist results."""

import logging
import os
import time
from typing import Tuple

import numpy as np

from m3c2.core.m3c2_runner import M3C2Runner

logger = logging.getLogger(__name__)


class M3C2Executor:
    """Run the M3C2 algorithm for a given configuration."""

    def _run_m3c2(self, cfg, mov, ref, corepoints, normal: float, projection: float, out_base: str, tag: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run M3C2 and store distances and uncertainties."""
        t0 = time.perf_counter()
        runner = M3C2Runner()
        distances, uncertainties = runner.run(mov, ref, corepoints, normal, projection)
        duration = time.perf_counter() - t0
        n = len(distances)
        nan_share = float(np.isnan(distances).sum()) / n if n else 0.0
        logger.info("[Run] Punkte=%d | NaN=%.2f%% | Zeit=%.3fs", n, 100.0 * nan_share, duration)

        dists_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances.txt")
        np.savetxt(dists_path, distances, fmt="%.6f")
        logger.info("[Run] Distanzen gespeichert: %s (%d Werte, %.2f%% NaN)", dists_path, n, 100.0 * nan_share)

        coords_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt")
        if hasattr(mov, "cloud"):
            xyz = np.asarray(mov.cloud)
        else:
            xyz = np.asarray(mov)
        if xyz.shape[0] == distances.shape[0]:
            arr = np.column_stack((xyz, distances))
            header = "x y z distance"
            np.savetxt(coords_path, arr, fmt="%.6f", header=header)
            logger.info(f"[Run] Distanzen mit Koordinaten gespeichert: {coords_path}")
        else:
            logger.warning(
                f"[Run] Anzahl Koordinaten stimmt nicht mit Distanzen überein: {xyz.shape[0]} vs {distances.shape[0]}"
            )

        uncert_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_uncertainties.txt")
        np.savetxt(uncert_path, uncertainties, fmt="%.6f")
        logger.info("[Run] Unsicherheiten gespeichert: %s", uncert_path)

        return distances, uncertainties
