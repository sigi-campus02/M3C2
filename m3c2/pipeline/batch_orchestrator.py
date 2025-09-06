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
    def _run_tag(config: PipelineConfig) -> str:
        """Return a concise tag representing the comparison and reference files.

        Parameters
        ----------
        config : PipelineConfig
            Configuration for the current job.

        Returns
        -------
        str
            Combination of comparison and reference filenames.
        """
        if config.filename_comparison and config.filename_reference:
            return f"{config.filename_comparison}-{config.filename_reference}"
        else:
            return f"{config.filename_singlecloud}"
        
    
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

        # Process each configuration in sequence.  Each iteration logs
        # failures so subsequent jobs can continue unless ``fail_fast`` is set.
        # The heavy lifting, including tagging and branching to the appropriate
        # processor, is delegated to ``_run_single``.
        for config in self.configs:
            try:
                self._run_single(config)
            except (IOError, ValueError):
                logger.exception(
                    "[Job] Fehler in Job '%s' (Version %s)",
                    config.folder_id,
                    config.filename_reference,
                )
            except RuntimeError:
                logger.exception(
                    "[Job] Unerwarteter Fehler in Job '%s' (Version %s)",
                    config.folder_id,
                    config.filename_reference,
                )
                if self.fail_fast:
                    raise
            except Exception:
                logger.exception(
                    "[Job] Unbekannter Fehler in Job '%s' (Version %s)",
                    config.folder_id,
                    config.filename_reference,
                )
                if self.fail_fast:
                    raise

    def _run_single(self, config: PipelineConfig) -> None:
        """Execute the pipeline for a single configuration.

        Parameters
        ----------
        config : PipelineConfig
            Configuration for the dataset to process.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Any exception from internal steps is caught and logged.
        """
        # Log key identifiers about the job before processing begins
        logger.info(
            "%s, %s, %s, %s",
            config.folder_id,
            config.filename_comparison,
            config.filename_reference,
            config.process_python_CC,
        )
        start = time.perf_counter()
        # Create a concise tag based on filenames for use in output paths
        run_tag = self._run_tag(config)

        # Branch to the correct processor depending on whether we compare
        # multiple clouds or analyse a single cloud
        if config.stats_singleordistance == "distance":
            self.multicloud_processor.process(config, run_tag)
        elif config.stats_singleordistance == "single":
            self.singlecloud_processor.process(config, run_tag)
        else:
            raise ValueError("Unbekannter stats_singleordistance-Wert")

        # Log runtime after processing is complete
        logger.info(
            "[Job] %s abgeschlossen in %.3fs",
            config.folder_id,
            time.perf_counter() - start,
        )
