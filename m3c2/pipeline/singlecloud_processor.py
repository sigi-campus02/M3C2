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

    def process(self, config: PipelineConfig, run_tag: str) -> None:
        # Load the point cloud for which statistics will be computed
        cloud_points = self.data_loader.load_data(config, mode="singlecloud")

        # Step 1: Retrieve or estimate the normal and projection scales
        if config.use_existing_params:
            # Existing parameters are loaded from disk and returned
            normal_scale, projection_scale = self.param_manager.handle_override_params(
                config
            )
        elif not config.use_existing_params:
            logger.info("[Params] keine vorhandenen Parameter gefunden, berechne neu")
            # Compute scales from the point cloud when no saved values are available
            normal_scale, projection_scale = self.scale_estimator.determine_scales(
                config, cloud_points
            )
            # Step 2: Persist newly determined parameters for future runs
            out_base = os.path.join(config.data_dir, config.folder_id)
            self.param_manager.save_params(
                config, normal_scale, projection_scale, out_base, run_tag
            )
        else:
            logger.error("[Params] Ungültige/Fehlende Parameter in Config")
            normal_scale = projection_scale = np.nan

        try:
            # Step 3: Generate statistics using the determined scales
            logger.info("[Statistics] Berechne Statistiken ...")
            self.statistics_runner.single_cloud_statistics_handler(
                config, cloud_points, normal_scale
            )
        except (IOError, ValueError):
            logger.exception("Fehler bei der Berechnung der Statistik")
        except RuntimeError:
            logger.exception(
                "Unerwarteter Fehler bei der Berechnung der Statistik"
            )
            raise
