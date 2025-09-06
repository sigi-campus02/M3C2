"""Unit tests for :mod:`m3c2.pipeline.singlecloud_processor`.

These tests validate that the processor correctly handles successful
single-cloud processing and properly reacts to failure scenarios such as
executor errors or missing input files.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.singlecloud_processor import SinglecloudProcessor


def _config(tmp_path, use_existing_params=False):
    return PipelineConfig(
        data_dir=str(tmp_path),
        folder_id="run",
        filename_comparison="comparison.xyz",
        filename_reference="reference.xyz",
        filename_singlecloud="cloud.xyz",
        use_subsampled_corepoints=1,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
        use_existing_params=use_existing_params,
    )


def test_process_single_cloud(tmp_path):
    """Process a single cloud and verify parameter handling and statistics."""

    calls: dict[str, SimpleNamespace] = {}
    cloud = np.zeros((1, 3))

    class DummyDataLoader:
        def load_data(self, config, mode="singlecloud"):
            calls["load_data"] = SimpleNamespace(config=config, mode=mode)
            return cloud

    class DummyScaleEstimator:
        def determine_scales(self, config, cloud_points):
            calls["determine_scales"] = SimpleNamespace(
                config=config, cloud_points=cloud_points
            )
            return 1.0, 2.0

    class DummyStatisticsRunner:
        def single_cloud_statistics_handler(self, config, cloud_points, normal_scale):
            calls["stats"] = SimpleNamespace(
                config=config, cloud_points=cloud_points, normal_scale=normal_scale
            )

    class DummyParamManager:
        def save_params(
            self, config, normal_scale, projection_scale, out_base, run_tag
        ):
            calls["save_params"] = SimpleNamespace(
                config=config,
                normal_scale=normal_scale,
                projection_scale=projection_scale,
                out_base=out_base,
                run_tag=run_tag,
            )

        def handle_override_params(
            self, config
        ):  # pragma: no cover - should not be called
            calls["handle_override_params"] = config
            return np.nan, np.nan

    processor = SinglecloudProcessor(
        data_loader=DummyDataLoader(),
        scale_estimator=DummyScaleEstimator(),
        statistics_runner=DummyStatisticsRunner(),
        param_manager=DummyParamManager(),
    )

    config = _config(tmp_path, use_existing_params=False)
    processor.process(config, run_tag="t")

    assert calls["load_data"].mode == "singlecloud"
    assert calls["determine_scales"].cloud_points is cloud
    assert (
        calls["save_params"].out_base == os.path.join(config.data_dir, config.folder_id)
    )
    assert calls["stats"].normal_scale == 1.0
    assert "handle_override_params" not in calls


def test_process_raises_on_executor_error(tmp_path):
    """Ensure runtime errors from the statistics runner are propagated."""

    cloud = np.zeros((1, 3))

    class DummyDataLoader:
        def load_data(self, config, mode="singlecloud"):
            return cloud

    class DummyScaleEstimator:
        def determine_scales(self, config, cloud_points):
            return 1.0, 1.0

    class FailingStatisticsRunner:
        def single_cloud_statistics_handler(
            self, config, cloud_points, normal_scale
        ):
            raise RuntimeError("fail")

    class DummyParamManager:
        def handle_override_params(self, config):
            return 1.0, 1.0

    processor = SinglecloudProcessor(
        data_loader=DummyDataLoader(),
        scale_estimator=DummyScaleEstimator(),
        statistics_runner=FailingStatisticsRunner(),
        param_manager=DummyParamManager(),
    )

    config = _config(tmp_path, use_existing_params=True)
    with pytest.raises(RuntimeError):
        processor.process(config, run_tag="t")


def test_process_missing_file(tmp_path):
    """Loading a missing file should surface the underlying exception."""

    class MissingDataLoader:
        def load_data(self, config, mode="singlecloud"):
            raise FileNotFoundError("missing")

    processor = SinglecloudProcessor(
        data_loader=MissingDataLoader(),
        scale_estimator=None,
        statistics_runner=None,
        param_manager=None,
    )

    config = _config(tmp_path, use_existing_params=True)
    with pytest.raises(FileNotFoundError):
        processor.process(config, run_tag="t")
