"""Tests for the compare distances plot service.

These tests ensure that the :func:`PlotServiceCompareDistances.overlay_plots`
function delegates to the proper plotting helpers based on the provided
configuration and that no plots are generated when all options are disabled.
"""

from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances
from m3c2.visualization.services import plot_comparedistances_service
from m3c2.visualization.services.plot_comparedistances_service import (
    PlotServiceCompareDistances,
)


class CallRecorder:
    """Recorder for function calls during testing.

    Instances of this class act as callables and store all invocations in the
    ``calls`` list attribute.  Each call is recorded as a tuple containing the
    ``folder_ids``, ``ref_variants``, and ``outdir`` arguments.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, folder_ids, ref_variants, outdir):
        self.calls.append((tuple(folder_ids), tuple(ref_variants), outdir))


def test_overlay_plots_delegates(monkeypatch, tmp_path):
    """Ensure :func:`overlay_plots` delegates to selected plot functions.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace plotting functions with mock call
        recorders.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for output files.
    """

    ba = CallRecorder()
    pb = CallRecorder()
    lr = CallRecorder()

    monkeypatch.setattr(plot_comparedistances_service, "bland_altman_plot", ba)
    monkeypatch.setattr(plot_comparedistances_service, "passing_bablok_plot", pb)
    monkeypatch.setattr(plot_comparedistances_service, "linear_regression_plot", lr)

    cfg = PlotConfig(
        folder_ids=["f1"],
        filenames=["ref", "ref_ai"],
        bins=10,
        outdir=str(tmp_path),
        project="P",
    )
    opts = PlotOptionsComparedistances(
        plot_blandaltman=True,
        plot_passingbablok=False,
        plot_linearregression=True,
    )

    PlotServiceCompareDistances.overlay_plots(cfg, opts)

    assert len(ba.calls) == 1
    assert len(pb.calls) == 0
    assert len(lr.calls) == 1
    assert ba.calls[0][0] == ("f1",)
    assert lr.calls[0][0] == ("f1",)


def test_overlay_plots_no_options(monkeypatch, tmp_path):
    """Verify :func:`overlay_plots` does nothing when all options are false.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to mock the plotting functions.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for output files.
    """

    ba = CallRecorder()
    pb = CallRecorder()
    lr = CallRecorder()

    monkeypatch.setattr(plot_comparedistances_service, "bland_altman_plot", ba)
    monkeypatch.setattr(plot_comparedistances_service, "passing_bablok_plot", pb)
    monkeypatch.setattr(plot_comparedistances_service, "linear_regression_plot", lr)

    cfg = PlotConfig(
        folder_ids=["f1"],
        filenames=["ref", "ref_ai"],
        bins=10,
        outdir=str(tmp_path),
        project="P",
    )
    opts = PlotOptionsComparedistances(
        plot_blandaltman=False,
        plot_passingbablok=False,
        plot_linearregression=False,
    )

    PlotServiceCompareDistances.overlay_plots(cfg, opts)

    assert ba.calls == []
    assert pb.calls == []
    assert lr.calls == []
