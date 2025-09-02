"""Factory for creating pipeline components used by :class:`BatchOrchestrator`."""

from __future__ import annotations

import logging

from m3c2.pipeline.data_loader import DataLoader
from m3c2.pipeline.scale_estimator import ScaleEstimator
from m3c2.pipeline.m3c2_executor import M3C2Executor
from m3c2.pipeline.outlier_handler import OutlierHandler
from m3c2.pipeline.statistics_runner import StatisticsRunner
from m3c2.pipeline.visualization_runner import VisualizationRunner


logger = logging.getLogger(__name__)


class PipelineComponentFactory:
    """Create configured instances of pipeline helper classes."""

    def __init__(self, strategy_name: str, output_format: str) -> None:
        self.strategy_name = strategy_name
        self.output_format = output_format
        logger.debug(
            "Initialized PipelineComponentFactory with strategy_name=%s, output_format=%s",
            strategy_name,
            output_format,
        )

    def create_data_loader(self) -> DataLoader:
        logger.debug("Creating %s", DataLoader.__name__)
        return DataLoader()

    def create_scale_estimator(self) -> ScaleEstimator:
        logger.debug("Creating %s", ScaleEstimator.__name__)
        return ScaleEstimator(strategy_name=self.strategy_name)

    def create_m3c2_executor(self) -> M3C2Executor:
        logger.debug("Creating %s", M3C2Executor.__name__)
        return M3C2Executor()

    def create_outlier_handler(self) -> OutlierHandler:
        logger.debug("Creating %s", OutlierHandler.__name__)
        return OutlierHandler()

    def create_statistics_runner(self) -> StatisticsRunner:
        logger.debug("Creating %s", StatisticsRunner.__name__)
        return StatisticsRunner(self.output_format)

    def create_visualization_runner(self) -> VisualizationRunner:
        logger.debug("Creating %s", VisualizationRunner.__name__)
        return VisualizationRunner()
