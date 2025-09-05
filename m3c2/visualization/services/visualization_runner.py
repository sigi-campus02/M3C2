"""Generate visual representations of M3C2 results."""
from __future__ import annotations

import logging
import os

import numpy as np

from m3c2.visualization.plot_helpers import histogram
from m3c2.visualization.ply_exporter import colorize, export_valid

logger = logging.getLogger(__name__)


class VisualizationRunner:
    """Create histograms and coloured point clouds for M3C2 outputs."""

    def generate_visuals(self, cfg, comparison, distances: np.ndarray, out_base: str, tag: str) -> None:
        """Create visualisations for computed distances.

        Parameters
        ----------
        cfg:
            Runtime configuration providing naming conventions such as
            ``process_python_CC``.
        comparison:
            Comparison point cloud object.  If ``comparison.cloud`` is not available no
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

        histogram(distances, path=hist_path)
        logger.info("[Visual] Histogram gespeichert: %s", hist_path)
        cloud = getattr(comparison, "cloud", None)
        if cloud is None:
            logger.warning("[Visual] comparison besitzt kein 'cloud'-Attribut; Visualisierung übersprungen")
            return

        colors = colorize(cloud, distances, outply=ply_path)
        logger.info("[Visual] Farb-PLY gespeichert: %s", ply_path)

        try:
            export_valid(cloud, colors, distances, outply=ply_valid_path)
            logger.info("[Visual] Valid-PLY gespeichert: %s", ply_valid_path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.warning("[Visual] Unerwarteter Fehler beim Export valid-only: %s", exc)
            raise


