"""Tests for the pipeline component factory.

This module verifies that :class:`~m3c2.pipeline.component_factory.PipelineComponentFactory`
creates fully configured pipeline components.
"""

from __future__ import annotations

from m3c2.pipeline.component_factory import PipelineComponentFactory
from m3c2.importer.data_loader import DataLoader
from m3c2.m3c2_core.param_handler.scale_estimator import ScaleEstimator
from m3c2.m3c2_core.m3c2_executor import M3C2Executor
from m3c2.statistics.statistics_runner import StatisticsRunner
from m3c2.visualization.services.visualization_runner import VisualizationRunner


def test_factory_creates_configured_components():
    """Test that configured components are created correctly.

    Notes
    -----
    The assertion for the outlier handler remains commented out until the
    component implementation is complete.
    """
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
