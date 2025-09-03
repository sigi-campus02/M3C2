"""Tests for the :class:`ReportBuilder` orchestration class."""

import numpy as np

from m3c2.config.plot_config import PlotConfig, PlotOptions
import m3c2.visualization.services.report_service as report_service
from m3c2.visualization.services.report_service import ReportBuilder


class CallRecorder:
    """Simple callable used to record invocations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1


def test_build_delegates(monkeypatch, tmp_path):
    """Ensure :meth:`ReportBuilder.build` delegates to plot functions."""

    # Provide fake data loading
    monkeypatch.setattr(
        report_service,
        "load_data",
        lambda fid, filenames, versions: ({f"py_{fid}": np.array([1.0, 2.0])}, {}),
    )
    monkeypatch.setattr(
        report_service,
        "load_coordinates_inlier_distances",
        lambda path: np.array([1.0, 2.0]),
    )
    # Provide a dummy path that is considered existing
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("0")
    monkeypatch.setattr(report_service, "resolve_path", lambda fid, name: str(dummy))

    hist = CallRecorder()
    grouped = CallRecorder()
    monkeypatch.setattr(report_service, "plot_overlay_histogram", hist)
    monkeypatch.setattr(report_service, "plot_grouped_bar_means_stds_dual", grouped)

    cfg = PlotConfig(
        folder_ids=["f1"],
        filenames=["ref"],
        bins=10,
        outdir=str(tmp_path),
        project="P",
        versions=["py"],
    )
    opts = PlotOptions(
        plot_hist=True,
        plot_gauss=False,
        plot_weibull=False,
        plot_box=False,
        plot_qq=False,
        plot_grouped_bar=True,
        plot_violin=False,
    )

    ReportBuilder(cfg, opts).build()

    assert hist.calls == 2  # WITH and INLIER
    assert grouped.calls == 2
