"""Compute and export statistical summaries for the M3C2 pipeline."""

from __future__ import annotations

import logging
import os

from m3c2.statistics.m3c2_aggregator import compute_m3c2_statistics
from m3c2.statistics.single_cloud_service import calc_single_cloud_stats

logger = logging.getLogger(__name__)


class StatisticsRunner:
    """Run various statistical evaluations depending on configuration."""

    def __init__(self, output_format: str) -> None:
        """Initialize the statistics runner.

        Args:
            output_format: Desired format for the exported statistics. Use
                ``"excel"`` to create an Excel workbook or ``"json"`` to write
                a JSON file.
        """
        self.output_format = output_format

    def compute_statistics(self, config, comparison, reference, run_tag: str) -> None:
        """Compute distance based M3C2 statistics for a job.

        Parameters
        ----------
        config
            Configuration object describing the current job.  The runner
            expects attributes such as :attr:`stats_singleordistance`,
            :attr:`folder_id`, :attr:`project` and various filenames.
        comparison, reference
            Information on the comparison and reference clouds.  The parameters are
            currently unused but kept for API compatibility with other pipeline
            components.
        run_tag : str
            Identifier for the reference cloud when evaluating distance based
            statistics.

        Notes
        -----
        This method only handles ``"distance"`` statistics.  Computation of
        statistics for individual clouds is handled separately via
        :meth:`single_cloud_statistics_handler` and should be triggered by the
        orchestrator.

        Output
        ------
        Results are stored in ``outputs/{project}_output`` with the filename
        ``{project}_m3c2_stats_distances`` and an extension of either
        ``.xlsx`` or ``.json`` depending on ``self.output_format``.

        This method is part of the public pipeline API.
        """
        if config.stats_singleordistance == "distance":
            return self._multi_cloud_handler(config, comparison, reference, run_tag)

    def _multi_cloud_handler(self, config, comparison, reference, run_tag):
        """Compute and export distance-based statistics across multiple clouds.

        Parameters
        ----------
        config
            Configuration object describing the current job. It provides the
            output directory, project name and options for the statistics
            calculation.
        comparison, reference
            Information about the comparison and reference clouds. The parameters
            are currently unused but preserved for API compatibility with the
            pipeline's public methods.
        run_tag : str
            Identifier for the reference cloud whose distance statistics are
            calculated.

        Side Effects
        ------------
        - Writes an ``.xlsx`` or ``.json`` file with distance statistics to
          ``outputs/{project}_output`` depending on :attr:`output_format`.
        - Logs progress information to the module logger.
        - Delegates the heavy lifting to
          :func:`m3c2.statistics.m3c2_aggregator.compute_m3c2_statistics`.
        """
        logger.info(
            f"[Stats on Distance] Berechne M3C2-Statistiken {config.folder_id},{config.filename_reference} ..."
        )
        if self.output_format == "excel":
            # Excel output requested: assemble path for workbook export.
            output_path = os.path.join(
                f"outputs/{config.project}_output/{config.project}_m3c2_stats_distances.xlsx"
            )
        elif self.output_format == "json":
            # JSON output requested: assemble path for JSON export.
            output_path = os.path.join(
                f"outputs/{config.project}_output/{config.project}_m3c2_stats_distances.json"
            )
        else:
            # Protect against unsupported formats to avoid silent errors.
            raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

        # Delegate the heavy lifting and write results to the derived path.
        compute_m3c2_statistics(
            folder_ids=[config.folder_id],
            filename_reference=run_tag,
            process_python_CC=config.process_python_CC,
            out_path=output_path,
            sheet_name="Results",
            output_format=self.output_format,
            outlier_multiplicator=config.outlier_multiplicator,
            outlier_method=config.outlier_detection_method,
        )

    def single_cloud_statistics_handler(self, config, singlecloud, normal):
        """Compute statistics for a single point cloud.

        Args:
            config: Configuration for the current job. Must provide
                ``folder_id``, ``filename_singlecloud`` and ``project`` so
                that results can be written to the correct output folder.
            singlecloud: The point cloud for which statistics are evaluated.
            normal: Radius used during computation of the cloud statistics.

        Writes
        ------
        ``outputs/{project}_output/{project}_m3c2_stats_clouds`` with an
        extension of ``.xlsx`` or ``.json`` depending on the selected output
        format.
        """
        logger.info(
            f"[Stats on SingleClouds] Berechne M3C2-Statistiken {config.folder_id},{config.filename_singlecloud} ...",
        )
        if self.output_format == "excel":
            # Build path for Excel workbook containing cloud statistics.
            output_path = os.path.join(
                f"outputs/{config.project}_output/{config.project}_m3c2_stats_clouds.xlsx"
            )
        elif self.output_format == "json":
            # Build path for JSON output when requested.
            output_path = os.path.join(
                f"outputs/{config.project}_output/{config.project}_m3c2_stats_clouds.json"
            )
        else:
            # Unsupported formats are rejected to prevent misconfiguration.
            raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

        # Compute statistics for the single cloud and export to the path.
        calc_single_cloud_stats(
            folder_ids=[config.folder_id],
            filename_singlecloud=config.filename_singlecloud,
            singlecloud=singlecloud,
            data_dir=config.data_dir,
            radius=normal,
            out_path=output_path,
            sheet_name="CloudStats",
            output_format=self.output_format,
        )
