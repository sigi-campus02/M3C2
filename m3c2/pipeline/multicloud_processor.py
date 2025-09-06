import logging
import os
from typing import Any, Optional

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
        outlier_handler: Optional[Any] = None,
    ) -> None:
        self.data_loader = data_loader
        self.scale_estimator = scale_estimator
        self.m3c2_executor = m3c2_executor
        self.statistics_runner = statistics_runner
        self.param_manager = param_manager
        self.outlier_handler = outlier_handler

    def process(self, cfg: PipelineConfig, tag: str) -> None:
        """Process statistics for a pair of point clouds."""
        ds, comparison, reference, corepoints = self.data_loader.load_data(cfg, mode="multicloud")
        out_base = ds.config.folder

        if not cfg.only_stats:
            self._process_full(cfg, comparison, reference, corepoints, out_base, tag)
        else:
            try:
                logger.info("[Statistics] Berechne Statistiken …")
                self.statistics_runner.compute_statistics(cfg, comparison, reference, tag)
            except (IOError, ValueError):
                logger.exception("Fehler bei der Berechnung der Statistik")
            except RuntimeError:
                logger.exception(
                    "Unerwarteter Fehler bei der Berechnung der Statistik"
                )
                raise

    def _process_full(
        self,
        cfg: PipelineConfig,
        comparison: str,
        reference: str,
        corepoints: np.ndarray,
        out_base: str,
        tag: str,
    ) -> None:
        normal = projection = np.nan

        if cfg.use_existing_params:
            normal, projection = self.param_manager.handle_existing_params(
                cfg, out_base, tag
            )
            if np.isnan(normal) and np.isnan(projection):
                logger.info(
                    "[Params] keine vorhandenen Parameter gefunden, berechne neu"
                )
                normal, projection = self.scale_estimator.determine_scales(
                    cfg, corepoints
                )
        else:
            normal, projection = self.scale_estimator.determine_scales(
                cfg, corepoints
            )
            self.param_manager.save_params(cfg, normal, projection, out_base, tag)

        distances, _, _ = self.m3c2_executor.run_m3c2(
            cfg, comparison, reference, corepoints, normal, projection, out_base, tag
        )

        ply_path = os.path.join(
            out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances.ply"
        )
        export_xyz_distance(corepoints, distances, ply_path)
        logger.info("[PLY] Distanzen als PLY gespeichert: %s", ply_path)
