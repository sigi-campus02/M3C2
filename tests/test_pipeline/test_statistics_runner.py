"""Tests covering the :mod:`StatisticsRunner` pipeline component.

These tests validate distance statistics, single cloud statistics, and the
handling of invalid output formats.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from m3c2.pipeline.statistics_runner import StatisticsRunner
from m3c2.core.statistics import StatisticsService


def test_compute_statistics_distance(monkeypatch, caplog):
    """Validate computation and logging of distance statistics.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace ``compute_m3c2_statistics`` with a mock.
    caplog : pytest.LogCaptureFixture
        Fixture that captures log messages for assertion.

    Returns
    -------
    None
        This test asserts side effects and does not return a value.
    """

    called = {}

    def fake_compute_m3c2_statistics(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        StatisticsService,
        "compute_m3c2_statistics",
        staticmethod(fake_compute_m3c2_statistics),
    )

    cfg = SimpleNamespace(
        stats_singleordistance="distance",
        folder_id="fid",
        filename_ref="ref",
        process_python_CC="proc",
        project="proj",
        outlier_multiplicator=3.0,
        outlier_detection_method="rmse",
    )

    runner = StatisticsRunner(output_format="excel")
    caplog.set_level(logging.INFO)
    runner.compute_statistics(cfg, mov=None, ref=None, tag="ref")

    assert called["out_path"].endswith("proj_m3c2_stats_distances.xlsx")
    assert any("Stats on Distance" in rec.message for rec in caplog.records)


def test_single_cloud_statistics_handler(monkeypatch, caplog):
    """Ensure the runner handles statistics for single clouds correctly.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch ``calc_single_cloud_stats``.
    caplog : pytest.LogCaptureFixture
        Fixture capturing log output for verification.

    Returns
    -------
    None
        This test asserts side effects and does not return a value.
    """

    called = {}

    def fake_calc_single_cloud_stats(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        StatisticsService,
        "calc_single_cloud_stats",
        classmethod(lambda cls, **kwargs: fake_calc_single_cloud_stats(**kwargs)),
    )

    cfg = SimpleNamespace(
        folder_id="fid",
        filename_singlecloud="cloud",
        project="proj",
        data_dir="dd",
    )

    runner = StatisticsRunner(output_format="json")
    caplog.set_level(logging.INFO)
    runner.single_cloud_statistics_handler(cfg, singlecloud=None, normal=1.0)

    assert called["out_path"].endswith("proj_m3c2_stats_clouds.json")
    assert any("Stats on SingleClouds" in rec.message for rec in caplog.records)


def test_invalid_output_format():
    """Check that an unsupported output format raises ``ValueError``.

    Returns
    -------
    None
        This test asserts that the appropriate exception is raised.
    """

    runner = StatisticsRunner(output_format="xml")
    cfg = SimpleNamespace(stats_singleordistance="distance", folder_id="fid", filename_ref="ref")
    try:
        runner.compute_statistics(cfg, mov=None, ref=None, tag="ref")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid output format"
