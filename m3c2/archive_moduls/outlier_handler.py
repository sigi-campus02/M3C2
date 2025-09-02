"""Handle exclusion of statistical outliers."""
from __future__ import annotations

import logging
import os

from m3c2.archive_moduls.exclude_outliers import exclude_outliers

# Module-level logger for this handler
logger = logging.getLogger(__name__)


class OutlierHandler:
    """Remove statistical outliers from M3C2 results."""

    def exclude_outliers(self, cfg, out_base: str, tag: str) -> None:
        """Remove outliers based on configuration settings.

        This method is part of the public pipeline API.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration object providing ``outlier_detection_method`` and
            ``outlier_multiplicator``.
        out_base : str
            Path to the directory containing the results of the current run.
        tag : str
            Tag identifying the current dataset (usually based on the input
            filenames).

        Raises
        ------
        Exception
            Propagated if the underlying exclusion routine fails.
        """
        logger.info("[Outlier] Entferne Ausreißer …")
        method = cfg.outlier_detection_method
        outlier_multiplicator = cfg.outlier_multiplicator
        dists_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt")

        try:
            exclude_outliers(
                dists_path=dists_path,
                method=method,
                outlier_multiplicator=outlier_multiplicator,
            )
            logger.info("[Outlier] Entfernen abgeschlossen")
        except Exception:
            # Surface errors to callers while logging the stack trace
            logger.exception("[Outlier] Fehler beim Entfernen der Ausreißer")
            raise
