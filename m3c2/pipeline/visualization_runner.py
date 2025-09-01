from __future__ import annotations
"""Generate visual representations of M3C2 results."""

import logging
import os

import numpy as np

from m3c2.visualization.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


class VisualizationRunner:
    """Create histograms and coloured point clouds for M3C2 outputs."""

    def generate_visuals(self, cfg, mov, distances: np.ndarray, out_base: str, tag: str) -> None:
        """Create visualisations for computed distances.

        This method is part of the public pipeline API.
        """
        logger.info("[Visual] Erzeuge Visualisierungen …")
        os.makedirs(out_base, exist_ok=True)
        hist_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_histogram.png")
        ply_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_includenonvalid.ply")
        ply_valid_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}.ply")

        VisualizationService.histogram(distances, path=hist_path)
        logger.info("[Visual] Histogram gespeichert: %s", hist_path)
        cloud = getattr(mov, "cloud", None)
        if cloud is None:
            logger.warning("[Visual] mov besitzt kein 'cloud'-Attribut; Visualisierung übersprungen")
            return

        colors = VisualizationService.colorize(cloud, distances, outply=ply_path)
        logger.info("[Visual] Farb-PLY gespeichert: %s", ply_path)

        try:
            VisualizationService.export_valid(cloud, colors, distances, outply=ply_valid_path)
            logger.info("[Visual] Valid-PLY gespeichert: %s", ply_valid_path)
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

    def generate_clouds_outliers(self, cfg, out_base: str, tag: str) -> None:
        """Create coloured point clouds for inliers and outliers.

        This method is part of the public pipeline API.
        """
        logger.info("[Visual] Erzeuge .ply Dateien für Outliers / Inliers …")
        os.makedirs(out_base, exist_ok=True)
        ply_valid_path_outlier = os.path.join(
            out_base, f"{cfg.process_python_CC}_{tag}_outlier_{cfg.outlier_detection_method}.ply"
        )
        ply_valid_path_inlier = os.path.join(
            out_base, f"{cfg.process_python_CC}_{tag}_inlier_{cfg.outlier_detection_method}.ply"
        )
        txt_path_outlier = os.path.join(
            out_base, f"python_{tag}_m3c2_distances_coordinates_outlier_{cfg.outlier_detection_method}.txt"
        )
        txt_path_inlier = os.path.join(
            out_base, f"python_{tag}_m3c2_distances_coordinates_inlier_{cfg.outlier_detection_method}.txt"
        )

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_outlier, outply=ply_valid_path_outlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_inlier, outply=ply_valid_path_inlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)
