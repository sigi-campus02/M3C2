from pathlib import Path
from unittest.mock import MagicMock, call

from report_pipeline.orchestrator import ReportOrchestrator


class Job:
    def __init__(self, items, title):
        self.items = items
        self.page_title = title


def test_orchestrator_flattens_figures():
    plotter = MagicMock()
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    plotter.make_overlay.side_effect = [[fig1, fig2], [fig3]]

    pdf_writer = MagicMock()
    pdf_writer.write.return_value = Path("out.pdf")

    jobs = [Job([1], "p1"), Job([2], "p2")]
    builder = MagicMock()
    builder.build_jobs.return_value = jobs

    orchestrator = ReportOrchestrator(plotter, pdf_writer, builder, show_legend=True)
    result = orchestrator.run()

    builder.build_jobs.assert_called_once_with()
    plotter.make_overlay.assert_has_calls(
        [
            call([1], title="p1", show_legend=True),
            call([2], title="p2", show_legend=True),
        ]
    )
    pdf_writer.write.assert_called_once_with([fig1, fig2, fig3])
    assert result == Path("out.pdf")
