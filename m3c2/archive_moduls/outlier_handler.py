"""Handle exclusion of statistical outliers."""
from __future__ import annotations

import logging
from m3c2.pipeline import outlier_handler as pipeline_outlier_handler

# Module-level logger for this handler
logger = logging.getLogger(__name__)


class OutlierHandler:
    """Remove statistical outliers from M3C2 results.

    This lightweight wrapper encapsulates the optional post-processing stage
    of the pipeline in which distance measurements considered outliers are
    discarded.  The handler merely forwards the work to
    :func:`exclude_outliers` while wiring in configuration and bookkeeping
    provided by the surrounding orchestration code.

    Parameters to :meth:`exclude_outliers` are pulled from a
    ``PipelineConfig`` instance.  It must provide the attributes
    ``outlier_detection_method`` (e.g. ``"sigma"`` or ``"iqr"``) and
    ``outlier_multiplicator`` describing the threshold to use.  The handler
    operates on the ``*_m3c2_distances_coordinates.txt`` file inside the
    ``out_base`` directory and writes the filtered version back to disk.

    The class itself does not return any value.  Its sole side effect is the
    generation of the cleaned output file; any exceptions raised by the
    underlying routine are propagated to the caller so that the pipeline can
    react accordingly.
    """

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

        try:
            pipeline_outlier_handler.exclude_outliers(
                out_base, tag, method, outlier_multiplicator
            )
            logger.info("[Outlier] Entfernen abgeschlossen")
        except Exception:
            # Surface errors to callers while logging the stack trace
            logger.exception("[Outlier] Fehler beim Entfernen der Ausreißer")
            raise
