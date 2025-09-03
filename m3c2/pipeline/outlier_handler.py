"""Minimal outlier handling for the pipeline tests.

This module provides a thin wrapper around the ``exclude_outliers`` function
so that pipeline components can delegate outlier removal.  The implementation
is intentionally lightweight and merely logs the operation; it can be replaced
with a more sophisticated approach in the future.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def exclude_outliers(data_folder: str, ref_variant: str, method: str, outlier_multiplicator: float) -> None:
    """Placeholder implementation that logs the exclusion parameters."""
    logger.info(
        "Excluding outliers in %s (%s) using %s with factor %s",
        data_folder,
        ref_variant,
        method,
        outlier_multiplicator,
    )


class OutlierHandler:
    """Delegates outlier removal to :func:`exclude_outliers`."""

    def exclude_outliers(self, cfg, out_base: str, tag: str) -> None:
        exclude_outliers(
            out_base,
            tag,
            getattr(cfg, "outlier_detection_method", ""),
            getattr(cfg, "outlier_multiplicator", 0),
        )
