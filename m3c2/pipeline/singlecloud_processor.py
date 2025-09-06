"""Processing tasks for statistics of a single point cloud."""

import logging
import os
from typing import Any

import numpy as np

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.param_manager import ParamManager

logger = logging.getLogger(__name__)


class SinglecloudProcessor:
    """Compute statistics for an individual point cloud."""

    def __init__(
        self,
        data_loader: Any,
        scale_estimator: Any,
        statistics_runner: Any,
        param_manager: ParamManager,
    ) -> None:
        self.data_loader = data_loader
        self.scale_estimator = scale_estimator
        self.statistics_runner = statistics_runner
        self.param_manager = param_manager

    def process(self, cfg: PipelineConfig, tag: str) -> None:
        single_cloud = self.data_loader.load_data(cfg, mode="singlecloud")

        if cfg.use_existing_params:
            normal, projection = self.param_manager.handle_override_params(cfg)
        elif not cfg.use_existing_params:
            logger.info("[Params] keine vorhandenen Parameter gefunden, berechne neu")
            normal, projection = self.scale_estimator.determine_scales(
                cfg, single_cloud
            )
            out_base = os.path.join(cfg.data_dir, cfg.folder_id)
            self.param_manager.save_params(cfg, normal, projection, out_base, tag)
        else:
            logger.error("[Params] Ungültige/Fehlende Parameter in Config")
            normal = projection = np.nan

        try:
            logger.info("[Statistics] Berechne Statistiken ...")
            self.statistics_runner.single_cloud_statistics_handler(
                cfg, single_cloud, normal
            )
        except (IOError, ValueError):
            logger.exception("Fehler bei der Berechnung der Statistik")
        except RuntimeError:
            logger.exception(
                "Unerwarteter Fehler bei der Berechnung der Statistik"
            )
            raise
