from pathlib import Path
from unittest.mock import MagicMock

import pytest

from report_pipeline.orchestrator import ReportOrchestrator


class Job:
    def __init__(self, items, title, plot_type="histogram"):
        self.items = items
        self.page_title = title
        self.plot_type = plot_type


def test_orchestrator_flattens_figures():
    plotter = MagicMock()
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    plotter.make_overlay.side_effect = [[fig1, fig2], [fig3]]

    pdf_writer = MagicMock()
    pdf_writer.write.return_value = Path("out.pdf")

    jobs = [Job([1], "p1", "histogram"), Job([2], "p2", "gauss")]
    builder = MagicMock()
    builder.build_jobs.return_value = jobs
    builder.legend = True

    orchestrator = ReportOrchestrator(plotter, pdf_writer, builder)
    out_path = Path("out.pdf")
    result = orchestrator.run(out_path, "Title")

    builder.build_jobs.assert_called_once_with()
    pdf_writer.write.assert_called_once_with([fig1, fig2, fig3], out_path, "Title")
    plotter.make_overlay.assert_any_call(
        [1], title="p1", plot_type="histogram", show_legend=True
    )
    plotter.make_overlay.assert_any_call(
        [2], title="p2", plot_type="gauss", show_legend=True
    )
    assert result == Path("out.pdf")


@pytest.mark.parametrize(
    ("plot_type", "method_name"),
    [
        ("bland-altman", "make_bland_altman"),
        ("linear-regression", "make_linear_regression"),
        ("passing-bablok", "make_passing_bablok"),
        ("grouped-bar", "make_grouped_bar"),
    ],
)
def test_orchestrator_dispatches_special_plot_types(plot_type, method_name):
    plotter = MagicMock()
    fig = MagicMock()
    getattr(plotter, method_name).return_value = [fig]

    pdf_writer = MagicMock()
    pdf_writer.write.return_value = Path("out.pdf")

    jobs = [Job([1], "p", plot_type)]
    builder = MagicMock()
    builder.build_jobs.return_value = jobs

    orchestrator = ReportOrchestrator(plotter, pdf_writer, builder)
    out_path = Path("out.pdf")
    result = orchestrator.run(out_path, "Title")

    getattr(plotter, method_name).assert_called_once_with([1], title="p")
    pdf_writer.write.assert_called_once_with([fig], out_path, "Title")
    assert result == Path("out.pdf")


def test_orchestrator_raises_on_invalid_plot_type():
    plotter = MagicMock()
    pdf_writer = MagicMock()
    builder = MagicMock()
    builder.build_jobs.return_value = [Job([1], "p", "invalid")]
    orchestrator = ReportOrchestrator(plotter, pdf_writer, builder)
    with pytest.raises(ValueError):
        orchestrator.run(Path("out.pdf"), "Title")
