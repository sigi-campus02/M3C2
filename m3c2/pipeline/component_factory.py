from __future__ import annotations
"""Factory for creating pipeline components used by :class:`BatchOrchestrator`."""

from m3c2.pipeline.data_loader import DataLoader
from m3c2.pipeline.scale_estimator import ScaleEstimator
from m3c2.pipeline.m3c2_executor import M3C2Executor
from m3c2.pipeline.outlier_handler import OutlierHandler
from m3c2.pipeline.statistics_runner import StatisticsRunner
from m3c2.pipeline.visualization_runner import VisualizationRunner


class PipelineComponentFactory:
    """Create configured instances of pipeline helper classes."""

    def __init__(self, strategy_name: str, output_format: str) -> None:
        self.strategy_name = strategy_name
        self.output_format = output_format

    def create_data_loader(self) -> DataLoader:
        return DataLoader()

    def create_scale_estimator(self) -> ScaleEstimator:
        return ScaleEstimator(strategy_name=self.strategy_name)

    def create_m3c2_executor(self) -> M3C2Executor:
        return M3C2Executor()

    def create_outlier_handler(self) -> OutlierHandler:
        return OutlierHandler()

    def create_statistics_runner(self) -> StatisticsRunner:
        return StatisticsRunner(self.output_format)

    def create_visualization_runner(self) -> VisualizationRunner:
        return VisualizationRunner()
