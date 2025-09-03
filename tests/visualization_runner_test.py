"""Test suite for the :mod:`m3c2.pipeline.visualization_runner` module.

These tests validate how :class:`~m3c2.pipeline.visualization_runner.VisualizationRunner`
handles the generation of visualization artifacts both when a point cloud is
available and when it is not.
"""

import logging
from types import SimpleNamespace

import numpy as np

from m3c2.visualization.services.visualization_runner import VisualizationRunner
from m3c2.visualization.services.visualization_service import VisualizationService


def test_generate_visuals_with_cloud(monkeypatch, tmp_path):
    """Ensure visuals are generated when a point cloud is provided.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to replace :class:`VisualizationService` methods with fakes.
    tmp_path : pathlib.Path
        Temporary directory where output files are written.

    Returns
    -------
    None
        This test only asserts side effects.

    Examples
    --------
    >>> runner = VisualizationRunner()
    >>> cfg = SimpleNamespace(process_python_CC='foo')
    >>> mov = SimpleNamespace(cloud='dummy')
    >>> distances = np.array([1.0, 2.0])
    >>> runner.generate_visuals(cfg, mov, distances, '/tmp', 'tag')
    """
    runner = VisualizationRunner()
    cfg = SimpleNamespace(process_python_CC="foo")
    mov = SimpleNamespace(cloud="dummy")
    distances = np.array([1.0, 2.0])

    calls = {}

    def fake_histogram(dist, path):
        calls["histogram"] = (dist, path)

    def fake_colorize(cloud, dist, outply):
        calls["colorize"] = (cloud, dist, outply)
        return "colors"

    def fake_export_valid(cloud, colors, dist, outply):
        calls["export_valid"] = (cloud, colors, dist, outply)

    monkeypatch.setattr(VisualizationService, "histogram", fake_histogram)
    monkeypatch.setattr(VisualizationService, "colorize", fake_colorize)
    monkeypatch.setattr(VisualizationService, "export_valid", fake_export_valid)

    runner.generate_visuals(cfg, mov, distances, str(tmp_path), "tag")

    assert "colorize" in calls
    assert "export_valid" in calls


def test_generate_visuals_without_cloud(monkeypatch, tmp_path, caplog):
    """Check behavior when no point cloud is supplied.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces :class:`VisualizationService` methods with stubs.
    tmp_path : pathlib.Path
        Temporary directory where output files are written.
    caplog : pytest.LogCaptureFixture
        Captures log output for assertions.

    Returns
    -------
    None
        This test only asserts side effects.

    Examples
    --------
    >>> runner = VisualizationRunner()
    >>> cfg = SimpleNamespace(process_python_CC='foo')
    >>> mov = SimpleNamespace()
    >>> distances = np.array([1.0, 2.0])
    >>> runner.generate_visuals(cfg, mov, distances, '/tmp', 'tag')
    """
    runner = VisualizationRunner()
    cfg = SimpleNamespace(process_python_CC="foo")
    mov = SimpleNamespace()  # no cloud attribute
    distances = np.array([1.0, 2.0])

    calls = {}

    def fake_histogram(dist, path):
        calls["histogram"] = (dist, path)

    def fake_colorize(*args, **kwargs):
        calls["colorize"] = True

    def fake_export_valid(*args, **kwargs):
        calls["export_valid"] = True

    monkeypatch.setattr(VisualizationService, "histogram", fake_histogram)
    monkeypatch.setattr(VisualizationService, "colorize", fake_colorize)
    monkeypatch.setattr(VisualizationService, "export_valid", fake_export_valid)

    with caplog.at_level(logging.WARNING):
        runner.generate_visuals(cfg, mov, distances, str(tmp_path), "tag")

    assert "histogram" in calls
    assert "colorize" not in calls
    assert "export_valid" not in calls
    assert "kein 'cloud'-Attribut" in caplog.text
