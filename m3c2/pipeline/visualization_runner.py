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

        Parameters
        ----------
        cfg:
            Runtime configuration providing naming conventions such as
            ``process_python_CC``.
        mov:
            Moving point cloud object.  If ``mov.cloud`` is not available no
            point cloud visualisations are generated.
        distances:
            Array of per-point M3C2 distances used for colouring and the
            histogram.
        out_base:
            Directory in which the generated files are stored.  Created if it
            does not yet exist.
        tag:
            Identifier that is appended to the filenames.

        The following files are written into ``out_base``:

        * ``<cfg.process_python_CC>_<tag>_histogram.png`` – histogram of
          distances.
        * ``<cfg.process_python_CC>_<tag>_includenonvalid.ply`` – coloured point
          cloud containing all points.
        * ``<cfg.process_python_CC>_<tag>.ply`` – coloured point cloud containing
          only valid points.

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
        """Convert TXT distance outputs into colourised inlier/outlier point clouds.

        Parameters
        ----------
        cfg : object
            Configuration providing the ``process_python_CC`` prefix and the
            ``outlier_detection_method`` identifier used to compose file names.
        out_base : str
            Directory where the TXT inputs are located and the resulting PLY
            files will be written.
        tag : str
            Label appended to file names to distinguish different processing
            runs or datasets.

        Outputs
        -------
        Two PLY files are created in ``out_base``—one for outliers and one for
        inliers—derived from TXT files containing coordinates and M3C2 distance
        values.

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
