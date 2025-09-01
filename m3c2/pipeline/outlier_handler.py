from __future__ import annotations
"""Handle exclusion of statistical outliers."""

import logging

from m3c2.core.exclude_outliers import exclude_outliers

# Module-level logger for this handler
logger = logging.getLogger(__name__)


class OutlierHandler:
    """Remove statistical outliers from M3C2 results."""

    def exclude_outliers(self, cfg, out_base: str, tag: str) -> None:
        """Remove outliers based on configuration settings.

        This method is part of the public pipeline API.
        """
        logger.info("[Outlier] Entferne Ausreißer …")
        try:
            exclude_outliers(
                data_folder=out_base,
                ref_variant=tag,
                method=cfg.outlier_detection_method,
                outlier_multiplicator=cfg.outlier_multiplicator,
            )
            logger.info("[Outlier] Entfernen abgeschlossen")
        except Exception:
            # Surface errors to callers while logging the stack trace
            logger.exception("[Outlier] Fehler beim Entfernen der Ausreißer")
            raise
