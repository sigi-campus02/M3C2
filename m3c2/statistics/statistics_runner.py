"""Compute and export statistical summaries for the M3C2 pipeline."""

from __future__ import annotations

import logging
import os

from m3c2.statistics import StatisticsService

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

    def compute_statistics(self, cfg, mov, ref, tag: str) -> None:
        """Compute distance based M3C2 statistics for a job.

        Parameters
        ----------
        cfg
            Configuration object describing the current job.  The runner
            expects attributes such as :attr:`stats_singleordistance`,
            :attr:`folder_id`, :attr:`project` and various filenames.
        mov, ref
            Information on the moving and reference clouds.  The parameters are
            currently unused but kept for API compatibility with other pipeline
            components.
        tag : str
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
        if cfg.stats_singleordistance == "distance":
            return self._multi_cloud_handler(cfg, mov, ref, tag)

    def _multi_cloud_handler(self, cfg, mov, ref, tag):
        """Compute and export distance-based statistics across multiple clouds.

        Parameters
        ----------
        cfg
            Configuration object describing the current job. It provides the
            output directory, project name and options for the statistics
            calculation.
        mov, ref
            Information about the moving and reference clouds. The parameters
            are currently unused but preserved for API compatibility with the
            pipeline's public methods.
        tag : str
            Identifier for the reference cloud whose distance statistics are
            calculated.

        Side Effects
        ------------
        - Writes an ``.xlsx`` or ``.json`` file with distance statistics to
          ``outputs/{project}_output`` depending on :attr:`output_format`.
        - Logs progress information to the module logger.
        - Delegates the heavy lifting to
          :func:`m3c2.core.statistics.StatisticsService.compute_m3c2_statistics`.
        """
        logger.info(
            f"[Stats on Distance] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …"
        )
        if self.output_format == "excel":
            out_path = os.path.join(
                f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.xlsx"
            )
        elif self.output_format == "json":
            out_path = os.path.join(
                f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.json"
            )
        else:
            raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

        StatisticsService.compute_m3c2_statistics(
            folder_ids=[cfg.folder_id],
            filename_ref=tag,
            process_python_CC=cfg.process_python_CC,
            out_path=out_path,
            sheet_name="Results",
            output_format=self.output_format,
            outlier_multiplicator=cfg.outlier_multiplicator,
            outlier_method=cfg.outlier_detection_method,
        )

    def single_cloud_statistics_handler(self, cfg, singlecloud, normal):
        """Compute statistics for a single point cloud.

        Args:
            cfg: Configuration for the current job. Must provide
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
            f"[Stats on SingleClouds] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_singlecloud} …",
        )
        if self.output_format == "excel":
            out_path = os.path.join(
                f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.xlsx"
            )
        elif self.output_format == "json":
            out_path = os.path.join(
                f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.json"
            )
        else:
            raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

        StatisticsService.calc_single_cloud_stats(
            folder_ids=[cfg.folder_id],
            filename_singlecloud=cfg.filename_singlecloud,
            singlecloud=singlecloud,
            data_dir=cfg.data_dir,
            radius=normal,
            out_path=out_path,
            sheet_name="CloudStats",
            output_format=self.output_format,
        )
