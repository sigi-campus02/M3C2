from __future__ import annotations

from types import SimpleNamespace

from m3c2.archive_moduls.outlier_handler import OutlierHandler


def test_exclude_outliers(monkeypatch):
    called = {}

    def fake_exclude_outliers(data_folder, ref_variant, method, outlier_multiplicator):
        called["args"] = (data_folder, ref_variant, method, outlier_multiplicator)

    monkeypatch.setattr(
        "m3c2.pipeline.outlier_handler.exclude_outliers", fake_exclude_outliers
    )

    cfg = SimpleNamespace(outlier_detection_method="iqr", outlier_multiplicator=2.5)
    handler = OutlierHandler()

    handler.exclude_outliers(cfg, out_base="base", tag="tag")

    assert called["args"] == ("base", "tag", "iqr", 2.5)
