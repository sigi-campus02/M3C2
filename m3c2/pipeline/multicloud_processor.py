import logging
import os
from typing import Any

import numpy as np

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.param_manager import ParamManager
from m3c2.exporter.ply_exporter import export_xyz_distance

logger = logging.getLogger(__name__)


class MulticloudProcessor:
    """Handle multicloud distance computation and related outputs."""

    def __init__(
        self,
        data_loader: Any,
        scale_estimator: Any,
        m3c2_executor: Any,
        statistics_runner: Any,
        param_manager: ParamManager,
        outlier_handler: Any,
    ) -> None:
        self.data_loader = data_loader
        self.scale_estimator = scale_estimator
        self.m3c2_executor = m3c2_executor
        self.statistics_runner = statistics_runner
        self.param_manager = param_manager
        self.outlier_handler = outlier_handler

    def process(self, config: PipelineConfig, run_tag: str) -> None:
        """Process statistics for a pair of point clouds."""
        # Load data and corepoints for multicloud processing
        data_source, comparison, reference, corepoints = self.data_loader.load_data(
            config, mode="multicloud"
        )
        output_dir = data_source.config.folder

        if not config.only_stats:
            # Perform full M3C2 workflow and export
            self._process_full(
                config, comparison, reference, corepoints, output_dir, run_tag
            )
        else:
            try:
                logger.info("[Statistics] Berechne Statistiken ...")
                # Only compute statistics without running M3C2
                self.statistics_runner.compute_statistics(
                    config, comparison, reference, run_tag
                )
            except (IOError, ValueError):
                logger.exception("Fehler bei der Berechnung der Statistik")
            except RuntimeError:
                logger.exception(
                    "Unerwarteter Fehler bei der Berechnung der Statistik"
                )
                raise

    def _process_full(
        self,
        config: PipelineConfig,
        comparison: str,
        reference: str,
        corepoints: np.ndarray,
        output_dir: str,
        run_tag: str,
    ) -> None:
        # Initialize scales as NaN until estimated or loaded
        normal_scale = projection_scale = np.nan

        if config.use_existing_params:
            # Try to load previously computed scales
            normal_scale, projection_scale = self.param_manager.handle_existing_params(
                config, output_dir, run_tag
            )
            if np.isnan(normal_scale) and np.isnan(projection_scale):
                logger.info(
                    "[Params] keine vorhandenen Parameter gefunden, berechne neu"
                )
                # Estimate scales from corepoints if none were found
                normal_scale, projection_scale = self.scale_estimator.determine_scales(
                    config, corepoints
                )
        else:
            # Estimate scales from corepoints
            normal_scale, projection_scale = self.scale_estimator.determine_scales(
                config, corepoints
            )
            # Persist estimated scales for subsequent runs
            self.param_manager.save_params(
                config, normal_scale, projection_scale, output_dir, run_tag
            )

        # Execute the M3C2 algorithm to compute distances
        distances, _, _ = self.m3c2_executor.run_m3c2(
            config,
            comparison,
            reference,
            corepoints,
            normal_scale,
            projection_scale,
            output_dir,
            run_tag,
        )

        # Detect outliers in the distance results
        outliers = self.outlier_handler.detect(
            distances, config.outlier_detection_method, config.outlier_multiplicator
        )

        # Export the distances and outlier flags as a PLY file
        ply_path = os.path.join(
            output_dir, f"{config.process_python_CC}_{run_tag}_m3c2_distances.ply"
        )
        export_xyz_distance(corepoints, distances, outliers, ply_path)
        logger.info("[PLY] Distanzen als PLY gespeichert: %s", ply_path)
