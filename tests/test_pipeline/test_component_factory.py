from __future__ import annotations

from m3c2.pipeline.component_factory import PipelineComponentFactory
from m3c2.pipeline.data_loader import DataLoader
from m3c2.pipeline.scale_estimator import ScaleEstimator
from m3c2.pipeline.m3c2_executor import M3C2Executor
from m3c2.archive_moduls.outlier_handler import OutlierHandler
from m3c2.pipeline.statistics_runner import StatisticsRunner
from m3c2.pipeline.visualization_runner import VisualizationRunner


def test_factory_creates_configured_components():
    factory = PipelineComponentFactory(strategy_name="radius", output_format="excel")

    assert isinstance(factory.create_data_loader(), DataLoader)

    scale_estimator = factory.create_scale_estimator()
    assert isinstance(scale_estimator, ScaleEstimator)
    assert scale_estimator.strategy_name == "radius"

    assert isinstance(factory.create_m3c2_executor(), M3C2Executor)
    # assert isinstance(factory.create_outlier_handler(), OutlierHandler)

    stats_runner = factory.create_statistics_runner()
    assert isinstance(stats_runner, StatisticsRunner)
    assert stats_runner.output_format == "excel"

    assert isinstance(factory.create_visualization_runner(), VisualizationRunner)
