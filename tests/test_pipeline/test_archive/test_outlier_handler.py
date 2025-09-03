"""Archived tests for the outlier handler.

These tests ensure that the archived :class:`~m3c2.archive_moduls.outlier_handler.OutlierHandler`
correctly delegates outlier exclusion to the pipeline module.
"""

from __future__ import annotations

from types import SimpleNamespace

from m3c2.archive_moduls.outlier_handler import OutlierHandler


def test_exclude_outliers(monkeypatch):
    """Ensure outliers are excluded via the pipeline module.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the pipeline's ``exclude_outliers`` function with a
        fake implementation so that the called arguments can be inspected.
    """

    called = {}

    def fake_exclude_outliers(dists_path, method, outlier_multiplicator):
        called["args"] = (dists_path, method, outlier_multiplicator)

    monkeypatch.setattr(
        "m3c2.archive_moduls.outlier_handler.exclude_outliers",
        fake_exclude_outliers,
    )

    cfg = SimpleNamespace(
        outlier_detection_method="iqr",
        outlier_multiplicator=2.5,
        process_python_CC="python",
    )
    handler = OutlierHandler()

    handler.exclude_outliers(cfg, out_base="base", tag="tag")

    expected_path = "base/python_tag_m3c2_distances_coordinates.txt"
    assert called["args"] == (expected_path, "iqr", 2.5)
