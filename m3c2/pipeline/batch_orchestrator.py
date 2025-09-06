"""High level orchestration for running the complete M3C2 workflow.

The :class:`BatchOrchestrator` coordinates configuration handling, parameter
estimation, execution of the core algorithm and subsequent analysis steps.
"""

from __future__ import annotations

import logging
import time
from typing import List

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.component_factory import PipelineComponentFactory
from m3c2.pipeline.multicloud_processor import MulticloudProcessor
from m3c2.pipeline.param_manager import ParamManager
from m3c2.pipeline.singlecloud_processor import SinglecloudProcessor

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Run the full M3C2 pipeline for a collection of configurations.

    Parameters
    ----------
    configs : list[PipelineConfig]
        Collection of configuration objects to process.
    sample_size : int, optional
        Number of core points to sample when estimating scales.
    output_format : str
        Format for statistical outputs, ``"excel"`` or ``"json"``.
    """

    def __init__(
        self,
        configs: List[PipelineConfig],
        strategy: str = "radius",
        fail_fast: bool = False,
    ) -> None:
        """Create a new orchestrator instance.

        Parameters
        ----------
        configs : list[PipelineConfig]
            Collection of configuration objects to process.
        strategy : str, optional
            Strategy used by the pipeline components.
        fail_fast : bool, optional
            Abort the batch on unexpected errors if ``True``. If ``False``
            (default), errors are logged and processing continues with the
            next job.
        """

        self.configs = configs
        self.fail_fast = fail_fast
        output_format = configs[0].output_format if configs else "excel"
        self.factory = PipelineComponentFactory(strategy, output_format)
        self.data_loader = self.factory.create_data_loader()
        self.scale_estimator = self.factory.create_scale_estimator()
        self.m3c2_executor = self.factory.create_m3c2_executor()
        self.outlier_handler = self.factory.create_outlier_handler()
        self.statistics_runner = self.factory.create_statistics_runner()

        self.param_manager = ParamManager()
        self.multicloud_processor = MulticloudProcessor(
            self.data_loader,
            self.scale_estimator,
            self.m3c2_executor,
            self.statistics_runner,
            self.param_manager,
            outlier_handler=self.outlier_handler,
        )
        self.singlecloud_processor = SinglecloudProcessor(
            self.data_loader,
            self.scale_estimator,
            self.statistics_runner,
            self.param_manager,
        )

        logger.info("=== BatchOrchestrator initialisiert ===")
        logger.info("Konfigurationen: %d Jobs", len(self.configs))


    # Small helper to create consistent file name tags
    @staticmethod
    def _run_tag(cfg: PipelineConfig) -> str:
        """Return a concise tag representing the comparison and reference files.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for the current job.

        Returns
        -------
        str
            Combination of comparison and reference filenames.
        """
        if cfg.filename_comparison and cfg.filename_reference:
            return f"{cfg.filename_comparison}-{cfg.filename_reference}"
        else:
            return f"{cfg.filename_singlecloud}"
        
    
    def run_all(self) -> None:
        """Run the pipeline for each configured dataset.

        Returns
        -------
        None

        Notes
        -----
        Any runtime error raised during processing of a single job is caught and
        logged. When ``fail_fast`` is ``False`` (default) the batch continues
        with the next job. If ``fail_fast`` is ``True``, such errors
        abort the batch.
        """
        if not self.configs:
            logger.warning("Keine Konfigurationen – nichts zu tun.")
            return

        # Process each configuration in sequence
        for cfg in self.configs:
            try:
                self._run_single(cfg)
            except (IOError, ValueError):
                logger.exception(
                    "[Job] Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_reference,
                )
            except RuntimeError:
                logger.exception(
                    "[Job] Unerwarteter Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_reference,
                )
                if self.fail_fast:
                    raise
            except Exception:
                logger.exception(
                    "[Job] Unbekannter Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_reference,
                )
                if self.fail_fast:
                    raise

    def _run_single(self, cfg: PipelineConfig) -> None:
        """Execute the pipeline for a single configuration.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for the dataset to process.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Any exception from internal steps is caught and logged.
        """
        logger.info(
            "%s, %s, %s, %s",
            cfg.folder_id,
            cfg.filename_comparison,
            cfg.filename_reference,
            cfg.process_python_CC,
        )
        start = time.perf_counter()
        tag = self._run_tag(cfg)

        if cfg.stats_singleordistance == "distance":
            self.multicloud_processor.process(cfg, tag)
        elif cfg.stats_singleordistance == "single":
            self.singlecloud_processor.process(cfg, tag)
        else:
            raise ValueError("Unbekannter stats_singleordistance-Wert")

        logger.info(
            "[Job] %s abgeschlossen in %.3fs",
            cfg.folder_id,
            time.perf_counter() - start,
        )
