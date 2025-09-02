"""Compute and export statistical summaries for the M3C2 pipeline."""

from __future__ import annotations

import logging
import os

from m3c2.core.statistics import StatisticsService

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

    def compute_statistics(self, cfg, ref, tag: str) -> None:
        """Compute M3C2 statistics for a job.

        Parameters
        ----------
        cfg
            Configuration object describing the current job.  The runner
            expects attributes such as :attr:`stats_singleordistance`,
            :attr:`folder_id`, :attr:`project` and various filenames.
        ref
            Optional reference information.  The parameter is currently not
            used but retained for API compatibility with other pipeline
            components.
        tag : str
            Identifier for the reference cloud when evaluating distance based
            statistics.

        Branching
        ---------
        If ``cfg.stats_singleordistance`` is ``"distance"`` the method
        computes statistics on the distances between two clouds using
        :func:`StatisticsService.compute_m3c2_statistics`.  When the value is
        ``"single"`` statistics for individual clouds are computed via
        :func:`StatisticsService.calc_single_cloud_stats`.

        Output
        ------
        Results are stored in ``outputs/{project}_output``.  Distance
        statistics are written to ``{project}_m3c2_stats_distances`` and
        single cloud statistics to ``{project}_m3c2_stats_clouds``.  The
        extension is either ``.xlsx`` or ``.json`` depending on
        ``self.output_format``.

        This method is part of the public pipeline API.
        """
        if cfg.stats_singleordistance == "distance":
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

        if cfg.stats_singleordistance == "single":
            logger.info(
                f"[Stats on SingleClouds] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …",
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
                filename_mov=cfg.filename_mov,
                filename_ref=cfg.filename_ref,
                out_path=out_path,
                sheet_name="CloudStats",
                output_format=self.output_format,
            )
