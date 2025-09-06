"""Tests for the :mod:`m3c2.pipeline.multicloud_processor` module.

This test verifies that the multicloud processor exports distances to a PLY
file after running the M3C2 computation and that the file contains ``NaN``
values in the distance column.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from plyfile import PlyData

from m3c2.pipeline.multicloud_processor import MulticloudProcessor


def test_process_exports_ply_with_nan(tmp_path):
    """Run the processor and check that a PLY file with ``NaN`` is created."""

    distances = np.array([0.0, np.nan])
    comparison = reference = corepoints = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )

    class DummyDataLoader:
        def load_data(self, cfg, mode="multicloud"):
            ds = SimpleNamespace(config=SimpleNamespace(folder=str(tmp_path)))
            return ds, comparison, reference, corepoints

    class DummyScaleEstimator:
        def determine_scales(self, cfg, cps):
            return 1.0, 1.0

    class DummyM3C2Executor:
        def run_m3c2(
            self, cfg, comp, ref, cps, normal, projection, output_dir, run_tag
        ):
            return distances, None, None

    class DummyOutlierHandler:
        def detect(self, distances, method, factor):
            return np.array([0, 1], dtype=np.uint8)

    class DummyStatisticsRunner:
        def compute_statistics(self, cfg, comparison, reference, tag):
            pass

    class DummyParamManager:
        def save_params(self, cfg, normal, projection, out_base, tag):
            pass

        def handle_existing_params(self, cfg, out_base, tag):
            return np.nan, np.nan

    cfg = SimpleNamespace(
        process_python_CC="cfg",
        use_existing_params=False,
        only_stats=False,
        outlier_detection_method="rmse",
        outlier_multiplicator=3.0,
    )

    processor = MulticloudProcessor(
        data_loader=DummyDataLoader(),
        scale_estimator=DummyScaleEstimator(),
        m3c2_executor=DummyM3C2Executor(),
        statistics_runner=DummyStatisticsRunner(),
        param_manager=DummyParamManager(),
        outlier_handler=DummyOutlierHandler(),
    )

    processor.process(cfg, tag="run")

    ply_file = tmp_path / "cfg_run_m3c2_distances.ply"
    assert ply_file.is_file()

    ply = PlyData.read(ply_file)
    distances_read = ply["vertex"]["distance"]
    assert any(np.isnan(distances_read))
    np.testing.assert_array_equal(
        ply["vertex"]["outlier"], np.array([0, 1], dtype=np.uint8)
    )

