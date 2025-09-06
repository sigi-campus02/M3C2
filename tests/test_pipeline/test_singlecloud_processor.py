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


def _cfg(tmp_path, use_existing_params=False):
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
        def load_data(self, cfg, mode="singlecloud"):
            calls["load_data"] = SimpleNamespace(cfg=cfg, mode=mode)
            return cloud

    class DummyScaleEstimator:
        def determine_scales(self, cfg, arr):
            calls["determine_scales"] = SimpleNamespace(cfg=cfg, arr=arr)
            return 1.0, 2.0

    class DummyStatisticsRunner:
        def single_cloud_statistics_handler(self, cfg, arr, normal):
            calls["stats"] = SimpleNamespace(cfg=cfg, arr=arr, normal=normal)

    class DummyParamManager:
        def save_params(self, config, normal, projection, output_dir, tag):
            calls["save_params"] = SimpleNamespace(
                cfg=config, normal=normal, projection=projection, output_dir=output_dir, tag=tag
            )

        def handle_override_params(self, cfg):  # pragma: no cover - should not be called
            calls["handle_override_params"] = cfg
            return np.nan, np.nan

    processor = SinglecloudProcessor(
        data_loader=DummyDataLoader(),
        scale_estimator=DummyScaleEstimator(),
        statistics_runner=DummyStatisticsRunner(),
        param_manager=DummyParamManager(),
    )

    cfg = _cfg(tmp_path, use_existing_params=False)
    processor.process(cfg, tag="t")

    assert calls["load_data"].mode == "singlecloud"
    assert calls["determine_scales"].arr is cloud
    assert calls["save_params"].output_dir == os.path.join(cfg.data_dir, cfg.folder_id)
    assert calls["stats"].normal == 1.0
    assert "handle_override_params" not in calls


def test_process_raises_on_executor_error(tmp_path):
    """Ensure runtime errors from the statistics runner are propagated."""

    cloud = np.zeros((1, 3))

    class DummyDataLoader:
        def load_data(self, cfg, mode="singlecloud"):
            return cloud

    class DummyScaleEstimator:
        def determine_scales(self, cfg, arr):
            return 1.0, 1.0

    class FailingStatisticsRunner:
        def single_cloud_statistics_handler(self, cfg, arr, normal):
            raise RuntimeError("fail")

    class DummyParamManager:
        def handle_override_params(self, cfg):
            return 1.0, 1.0

    processor = SinglecloudProcessor(
        data_loader=DummyDataLoader(),
        scale_estimator=DummyScaleEstimator(),
        statistics_runner=FailingStatisticsRunner(),
        param_manager=DummyParamManager(),
    )

    cfg = _cfg(tmp_path, use_existing_params=True)
    with pytest.raises(RuntimeError):
        processor.process(cfg, tag="t")


def test_process_missing_file(tmp_path):
    """Loading a missing file should surface the underlying exception."""

    class MissingDataLoader:
        def load_data(self, cfg, mode="singlecloud"):
            raise FileNotFoundError("missing")

    processor = SinglecloudProcessor(
        data_loader=MissingDataLoader(),
        scale_estimator=None,
        statistics_runner=None,
        param_manager=None,
    )

    cfg = _cfg(tmp_path, use_existing_params=True)
    with pytest.raises(FileNotFoundError):
        processor.process(cfg, tag="t")
